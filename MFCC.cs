public class MFCC
{
    //Tutorial link that I used to implement MFCC below:
    //http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
    public double ConvertToMel(int f)
    {
        double mel = 1125 * Math.Log(1 + (double)f / 700);
        return mel;
    }

    public double ConvertFromMel(double m)
    {
        double f = 700 * (Math.Exp(m / 1125) - 1);
        return f;
    }
    public double[][] GetMFCCFilters(int mfccSize, int filterLength, int frequency, int freqMin, int freqMax)
    {
        double[] fb = new double[mfccSize + 2];
        fb[0] = ConvertToMel(freqMin);
        fb[mfccSize + 1] = ConvertToMel(freqMax);

        for (int i = 1; i < mfccSize + 1; i++)
        {
            fb[i] = fb[0] + (double)i * (fb[mfccSize + 1] - fb[0]) / (double)(mfccSize + 1);
        }
        for (int i = 0; i < mfccSize + 2; i++)
        {

            fb[i] = ConvertFromMel(fb[i]);

            fb[i] = Math.Floor((filterLength + 1) * fb[i] / (double)frequency);

        }
        double[][] filterBanks = new double[mfccSize][];
        for (int m = 0; m < mfccSize; m++)
        {
            filterBanks[m] = new double[filterLength];
        }
        for (int i = 1; i < mfccSize + 1; i++)
        {
            for (int j = 0; j < filterLength; j++)
            {

                if (fb[i - 1] <= j && j <= fb[i])
                {
                    filterBanks[i - 1][j] = (j - fb[i - 1]) / (fb[i] - fb[i - 1]);
                }
                else if (fb[i] < j && j <= fb[i + 1])
                {
                    filterBanks[i - 1][j] = (fb[i + 1] - j) / (fb[i + 1] - fb[i]);
                }
                else
                {
                    filterBanks[i - 1][j] = 0;
                }
            }
        }

        return filterBanks;
    }

    public double[] ApplyFiltersToLogPower(double[] fftRaw, int fftSize, double[][] melFilters, int mfccCount)
    {
        double[] logPow = new double[mfccCount];

        for (int i = 0; i < mfccCount; i++)
        {
            logPow[i] = 0.0;

            for (int j = 0; j < fftSize; j++)
            {
                logPow[i] += melFilters[i][j] * Math.Pow(fftRaw[j], 2);
            }

            logPow[i] = Math.Log(logPow[i]);
            if (double.IsInfinity(logPow[i]))
                logPow[i] = 0;
        }

        return logPow;
    }


    public double[] DctTransform(double[] logData, int N)
    {
        double[] dct = new double[N];
        for (int i = 0; i < N; i++)
        {
            dct[i] = 0;
            for (int j = 0; j < N; j++)
            {
                dct[i] += logData[j] * Math.Cos(Math.PI * i * (j + 1.0 / 2.0) / N);
            }
        }

        return dct;
    }


    public double[] ExecuteMFCC(double[] fftRaw, int mfccSize, int frequency, int freqMin, int freqMax, int frameLenght)
    {
        double[][] melFilters = GetMFCCFilters(mfccSize, frameLenght, frequency, freqMin, freqMax);

        double[] logPower = ApplyFiltersToLogPower(fftRaw, fftRaw.Length, melFilters, mfccSize);
        double[] dct = DctTransform(logPower, logPower.Length);

        return dct;
    }
}