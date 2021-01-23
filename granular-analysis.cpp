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
std::vector<double> hann_window(int window_size) {
    std::vector<double> window;

    for (int i = 0; i < window_size; i++) {
        window.push_back(0.5 * (1.0 - cos(2.0*M_PI*(i+1)/2049.0)));
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

    int SAMPLING_RATE = 48000; // import?
    int hop_size = N / 2;
    int nfft = N*4; // more frequency "precision"
    int window_size = N;

    int nframes = ceil(n / float(hop_size));

    CArray fft_buf(nfft);
    std::vector<double> window = hann_window(window_size);
    int start_index = 0;

    // analysis loop
    for (int fr = 0; fr < nframes; fr++) {
        // should deal with size corner cases
        int end_index = std::min(n, start_index+window_size);

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
        if (fr == nframes - 1) {
            for (int i = 0; i < window_size; i++) {
                fft_buf[i] *= window[i];
            }
        } else {
            // last window can be shorter than window_size, we still don't want a pop
            std::vector<double> last_window = hann_window(end_index - start_index);
            for (int i = 0; i < end_index - start_index; i++) {
                fft_buf[i] *= last_window[i];
            }
        }

        fft(fft_buf);

        // PART 3: FFT analyses
        double bin_step = double(SAMPLING_RATE) / nfft;
        double frame_sc_num = 0.0;
        double frame_sc_dem = 0.0;
        // don't bother with negative freqs
        for (int j = 0; j < nfft/2+1; j++) {
            double amp = std::abs(fft_buf[j]);
            double freq = j * bin_step;
            frame_sc_num += (amp * freq);
            frame_sc_dem += amp;
        }
        double frame_sc = frame_sc_num / frame_sc_dem;
        std::cout << "SC: " << frame_sc << "\n";

        // next frame
        start_index += hop_size;
    }
    return 0;
}
