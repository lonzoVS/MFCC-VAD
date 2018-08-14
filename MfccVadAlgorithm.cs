public class MfccVadAlgorithm 
{
    public double[] VadList { get; set; }
    private double p = 0.99999999999999;
    public List<IMask> Masks { get; set; } = new List<IMask>();
    private int mfccSize;
    private int freq;
    private int freqMin;
    private int freqMax;
    private int frameLength;
    private string envMaskPath;

    public MfccVadAlgorithm(int mfccSize, string envMaskPath, int freq = 11025, int freqMin = 300, int freqMax = 8000, int frameLength = 256)
    {
        this.mfccSize = mfccSize;
        this.freq = freq;
        this.freqMin = freqMin;
        this.freqMax = freqMax;
        this.frameLength = frameLength;
        this.envMaskPath = envMaskPath;
    }


    public void PerformVad(double[] samples)
    {
        var mfccList = PrepareMfcc(samples);
        mfccList = AplyMasks(mfccList, envMaskPath);
        var backGroundNoise = ComputeBackgroundNoise(mfccList.ToArray());
        var avrgNoise = ComputeAverageNoise(backGroundNoise);
        VadList = ComputeSimilarity(mfccList, avrgNoise);
    }

    private double[] AplyMasks(double[] mfcc, string envPath)
    {
        double[] mfccMasked = new double[mfcc.Length];
        for (int i = 0; i < Masks.Count; i++)
        {
            mfccMasked = Masks[i].Mask(mfcc, envPath);
        }
        return mfccMasked;
    }


    //Frame = 256;
    //windowing:  Multiplied each frames by a Hamming window;
    //computing FFT;
    //computing MFFC from FTT.
    private double[] PrepareMfcc(double[] samples)
    {
        //DSP dsp = new DSP();
        MFCC mfcc = new MFCC();
        List<double> mfccList = new List<double>();
        var spec = new RawFFTVisualization(samples).Compute(true);
        foreach (double[] chunk in spec.Split(frameLength))
        {
            mfccList.AddRange(mfcc.ExecuteMFCC(chunk, mfccSize, freq, freqMin, freqMax, frameLength));
        }

        return mfccList.ToArray();

    }

    //Calculating the MFCC vector of all frames, assuming the first 10 frames as background noise.
    private double[,] ComputeBackgroundNoise(double[] mfccList)
    {
        var noiseMFCC = mfccList.Take(10 * mfccSize).ToList();
        List<double[]> backNoise = new List<double[]>();
        foreach (double[] avgSlice in noiseMFCC.ToArray().Split(32))
        {
            backNoise.Add(avgSlice);
        }
        double[,] arrBUF = MathExtensions.CreateRectangularArray(backNoise);
        return arrBUF;
    }

    //Gettin avg value of the first ten frames = initial MFCC vector of background noise.
    private double[] ComputeAverageNoise(double[,] backNoise)
    {
        List<double> avrgNoise = new List<double>();
        for (int j = 0; j < 32; j++)
        {
            avrgNoise.Add(MathExtensions.GetCol(backNoise, j).Average());
        }
        avrgNoise = avrgNoise.Select(a => a * p).ToList();
        return avrgNoise.ToArray();
    }

    //Updating background noise vector with c_no = pc_+(1-p)c_i formula;
    //computing similarity between c_no(background noise vector) and current mfcc vector;
    private double[] ComputeSimilarity(double[] mfccList, double[] avrgNoise)
    {
        List<double> normalized = new List<double>();
        List<double> similarity = new List<double>();
        foreach (double[] Ci in mfccList.ToArray().Split(32))
        {
            var newAmounts = Ci.Select(a => a * (1.0 - p)).ToList();
            var Cno = avrgNoise.Zip(newAmounts, (a, y) => a + y);
            var coeffCorr = 1.0 - MathExtensions.ComputeCoeff(Ci.ToArray(), Cno.ToArray());
            similarity.Add(coeffCorr);
        }
        double length = similarity.Sum(a => a * a);
        length = Math.Sqrt(length);
        foreach (double ddd in similarity)
        {
            normalized.Add(ddd / length);
        }
        return normalized.ToArray();

    }