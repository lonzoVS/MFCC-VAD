public class MfccVadAlgorithm
{
    public double[] VadList { get; set; }
    public List<IMask> Masks { get; set; } = new List<IMask>();
    private int mfccSize;
    private int freq;
    private int freqMin;
    private int freqMax;
    private int frameLength;
    private string envMaskPath;
    private double p = 0.9; //value should be close to 1

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


    /// <summary>
    /// Computing MFFC from FTT.
    /// </summary>
    /// <param name="samples">FFT values(Frame = 256,  each frame multiplied by Hamming window) </param>
    private double[] PrepareMfcc(double[] samples)
    {
        MFCC mfcc = new MFCC();
        List<double> mfccList = new List<double>();
        foreach (double[] chunk in samples.Split(frameLength))
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
        double[,] arrBUF = CreateRectangularArray(backNoise);
        return arrBUF;
    }

    //Gettin avg value of the first ten frames = initial MFCC vector of background noise.
    private double[] ComputeAverageNoise(double[,] backNoise)
    {
        List<double> avrgNoise = new List<double>();
        for (int j = 0; j < 32; j++)
        {
            avrgNoise.Add(GetCol(backNoise, j).Average());
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
            var coeffCorr = 1.0 - CorrelationCoefficient(Ci.ToArray(), Cno.ToArray());
            similarity.Add(coeffCorr);
        }
        double length = similarity.Sum(a => a * a);
        length = Math.Sqrt(length);
        foreach (double value in similarity)
        {
            normalized.Add(value / length);
        }
        return normalized.ToArray();
    }
}

//Helper fucntions
public static double[] Multiply(this double[] x, double[] y)
{
    double[] z = new double[x.Length];
    for (int i = 0; i < x.Length; i++)
    {
        z[i] = x[i] * y[i];
    }
    return z;
}

public static T[,] CreateRectangularArray<T>(IList<T[]> arrays)
{
    int minorLength = arrays[0].Length;
    T[,] ret = new T[arrays.Count, minorLength];
    for (int i = 0; i < arrays.Count; i++)
    {
        var array = arrays[i];
        if (array.Length != minorLength)
        {
            throw new ArgumentException
                ("All arrays must be the same length");
        }
        for (int j = 0; j < minorLength; j++)
        {
            ret[i, j] = array[j];
        }
    }
    return ret;
}
public static double[] GetCol(double[,] matrix, int col)
{
    var colLength = matrix.GetLength(0);
    var colVector = new double[colLength];

    for (var i = 0; i < colLength; i++)
    {
        colVector[i] = matrix[i, col];
    }

    return colVector;
}

public static double CorrelationCoefficient(double[] values1, double[] values2)
{
    if (values1.Length != values2.Length)
        throw new ArgumentException("Values must be the same length");

    var avg1 = values1.Average();
    var avg2 = values2.Average();

    var sum1 = values1.Zip(values2, (x1, y1) => (x1 - avg1) * (y1 - avg2)).Sum();

    var sumSqr1 = values1.Sum(x => Math.Pow((x - avg1), 2.0));
    var sumSqr2 = values2.Sum(y => Math.Pow((y - avg2), 2.0));

    var result = sum1 / Math.Sqrt(sumSqr1 * sumSqr2);

    return result;
}

public static IEnumerable<double[]> Split(this double[] value, int bufferLength)
{
    int countOfArray = 0;
    countOfArray = value.Length / bufferLength;
    for (int i = 0; i < countOfArray; i++)
    {
        yield return value.Skip(i * bufferLength).Take(bufferLength).ToArray();
    }

}