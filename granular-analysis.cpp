#include <algorithm>  // std::sort, might not need this...
#include <cmath> // sqrt, pow, etc...
#include <complex>
#include <iostream>
#include <valarray>
#include <vector>

#include "float.h" // DBL_MAX

// utility functions

// higher memory implementation via http://rosettacode.org/wiki/Fast_Fourier_transform#C.2B.2B
typedef std::complex<double> Complex;
typedef std::valarray<Complex> CArray;
void fft(CArray& x)
{
    const size_t N = x.size();
    if (N <= 1) return;
 
    // divide
    CArray even = x[std::slice(0, N/2, 2)];
    CArray  odd = x[std::slice(1, N/2, 2)];
 
    // conquer
    fft(even);
    fft(odd);
 
    // combine
    for (size_t k = 0; k < N/2; ++k)
    {
        Complex t = std::polar(1.0, -2 * M_PI * k / N) * odd[k];
        x[k    ] = even[k] + t;
        x[k+N/2] = even[k] - t;
    }
}

// fixed size for now
double* hann_window() {
    static double window[2048];

    for (int i = 0; i < 2048; i++) {
        window[i] = 0.5 * (1.0 - cos(2.0*M_PI*(i+1)/2049.0));
    }

    return window;
}

// data structure for features
// not really necessary in this file, but useful to implement this now for part 2
struct Grain {
    int begin; // index into the vector<float> of the sound file
    int end;   // index into the vector<float> of the sound file

    float peakToPeak;
    float rms;
    float zcr;
    float centroid;
    float f0;
};

int main(int argc, char *argv[]) {
    // take the data in

    std::vector<double> audio_data;
    double value;
    int n = 0;
    while (std::cin >> value) {
        audio_data.push_back(value);
        n++;
    }

    assert(argc > 1);
    int N = std::stof(argv[1]);
    assert(N % 2 == 0 && N > 0);
    // print usages?

    double* window = hann_window();
    int SAMPLING_RATE = 48000; // import?
    int hop_size = N / 2;
    int nfft = N*4; // more frequency "precision"
    int window_size = N;

    int nframes = ceil(n / float(hop_size));

    CArray fft_buf(nfft);
    int start_index = 0;

    // analysis loop
    for (int fr = 0; fr < nframes; fr++) {
        // should deal with size corner cases
        int end_index = std::min(n, start_index+hop_size);

        // PART 1: easy sliding window calculations
        // let's assume reasonable audio
        double frame_max_amp = -1.1;
        double frame_min_amp = 1.1;
        double frame_sum = 0.0;
        double frame_zcr = 0.0;
        for (int i = start_index; i < end_index; i++) {
            frame_max_amp = std::max(frame_max_amp, audio_data[i]);
            frame_min_amp = std::min(frame_min_amp, audio_data[i]);
            frame_sum += pow(audio_data[i], 2);
            if (i > start_index) {
                double val = audio_data[i] * audio_data[i-1];
                // checking for cases of all zeros here, not sure it will be very effective given floating point math
                frame_zcr = val <= 0.0 && (audio_data[i] != 0.0 && audio_data[i-1] != 0.0) ? frame_zcr + 1 : frame_zcr;
                //frame_zcr = val < 0.0 ? frame_zcr + 1 : frame_zcr;
            }
        }

        //frame_zcr /= (end_index - start_index - 1); // check for divide by zero?
        double frame_ptp = frame_max_amp - frame_min_amp;
        double frame_rms = sqrt(frame_sum / (end_index - start_index));

        std::cout << "Peak-to-peak amplitude: " << frame_ptp << "\n";
        std::cout << "RMS amplitude: " << frame_rms << "\n";
        std::cout << "ZCR: " << frame_zcr << "\n";

        // PART 2: perform FFT
        int j = 0;
        for (int i = start_index; i < end_index; i++) {
            fft_buf[j] = audio_data[i];
            j++;
        }

        // zero-pad what's left
        while (j < nfft) {
            fft_buf[j] = 0.0;
            j++;
        }
        
        // apply hann window
        for (int i = 0; i < window_size; i++) {
            fft_buf[i] *= window[i];
        }

        fft(fft_buf);

        // PART 3: FFT analyses
        


        // next frame
        start_index += hop_size;
    }
    return 0;
}
