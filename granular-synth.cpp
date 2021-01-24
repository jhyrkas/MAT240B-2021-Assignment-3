// Assignment 3
//
//
// -- Karl Yerkes / 2021-01-23 / MAT240B
//
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>  // std::sort
#include <algorithm>  // std::sort, std::min
#include <cmath>      // ::cos()
#include <complex>
#include <iostream>
#include <valarray>
#include <vector>

#include "al/app/al_App.hpp"
#include "al/ui/al_ControlGUI.hpp"
#include "al/ui/al_Parameter.hpp"

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

typedef std::complex<double> Complex;
typedef std::valarray<Complex> CArray;

const int BLOCK_SIZE = 512;
const int SAMPLE_RATE = 48000;

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

double dbtoa(double db) { return pow(10.0, db / 20.0); }

// Cooley–Tukey FFT (in-place, divide-and-conquer)
// Higher memory requirements and redundancy although more intuitive
void fft(CArray &x) {
  const size_t N = x.size();
  if (N <= 1) return;

  // divide
  CArray even = x[std::slice(0, N / 2, 2)];
  CArray odd = x[std::slice(1, N / 2, 2)];

  // conquer
  fft(even);
  fft(odd);

  // combine
  for (size_t k = 0; k < N / 2; ++k) {
    Complex t = std::polar(1.0, -2 * M_PI * k / N) * odd[k];
    x[k] = even[k] + t;
    x[k + N / 2] = even[k] - t;
  }
}

void load(std::vector<double> &input, const char *filePath) {
  unsigned int channels;
  unsigned int sampleRate;
  drwav_uint64 totalPCMFrameCount;
  float *pSampleData = drwav_open_file_and_read_pcm_frames_f32(
      filePath, &channels, &sampleRate, &totalPCMFrameCount, NULL);
  if (pSampleData == NULL) {
    printf("failed to load %s\n", filePath);
    exit(1);
  }

  //
  if (channels == 1)
    for (int i = 0; i < totalPCMFrameCount; i++) {
      input.push_back(pSampleData[i]);
    }
  else if (channels == 2) {
    for (int i = 0; i < totalPCMFrameCount; i++) {
      input.push_back((pSampleData[2 * i] + pSampleData[2 * i + 1]) / 2);
    }
  } else {
    printf("can't handle %d channels\n", channels);
    exit(1);
  }

  drwav_free(pSampleData, NULL);
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// adapted from granular-analysis.cpp

// fixed size for now
std::vector<double> hann_window(int window_size) {
    std::vector<double> window;

    for (int i = 0; i < window_size; i++) {
        window.push_back(0.5 * (1.0 - cos(2.0*M_PI*(i+1)/2049.0)));
    }

    return window;
}

// from stft-peaks.cpp
// used in fft
// adapted from: https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes

struct FFT_Pair {
  double amplitude;
  int bin;
};

bool FFT_Pair_comparator_amp ( const FFT_Pair& l, const FFT_Pair& r)
   { return l.amplitude > r.amplitude; } // sort descending

bool FFT_Pair_comparator_freq ( const FFT_Pair& l, const FFT_Pair& r)
   { return l.bin < r.bin; } // sort ascending

std::vector<FFT_Pair> stft_peaks(CArray& fft_buf, int nfft, int n_peaks) {
    std::vector<FFT_Pair> peaks;
    // avoid giving peaks at DC or Nyquist
    // don't bother with negative frequencies
    for (int j = 1; j < nfft/2; j++) {
        double amp = std::abs(fft_buf[j]);
        // making one of these >= so that only one value in a plateau is captured
        if (amp > std::abs(fft_buf[j-1]) && amp >= std::abs(fft_buf[j+1])) {
            peaks.push_back({std::abs(fft_buf[j]), j});
        }
    }

    // sort by amp
    std::sort(peaks.begin(), peaks.end(), FFT_Pair_comparator_amp);

    // only keep top n_Peaks
    if (peaks.size() > n_peaks) {
        peaks.erase(peaks.begin() + n_peaks, peaks.end());
    }

    // re-sort so that entries are sorted low to high in frequency
    // (we don't even really need to do this)
    std::sort(peaks.begin(), peaks.end(), FFT_Pair_comparator_freq);
    return peaks;
}

int get_mode(std::vector<int> arr) {
    assert(arr.size() > 0);
    std::sort(arr.begin(), arr.end());
    int last_val = arr[0];
    int mode = last_val;
    int mode_count = 1;
    int last_val_count = 1;
    for (int i = 1; i < arr.size(); i++) {
        // increment and keep going
        if (arr[i] == last_val) {
            last_val_count++;
        }
        // number changed
        else {
            // did we find a new mode?
            if (last_val_count > mode_count) {
                mode = last_val;
                mode_count = last_val_count;
            }

            last_val = arr[i];
            last_val_count = 1;
        }
    }

    return mode;
}

struct Grain {
    int begin; // index into the vector<float> of the sound file
    int end;   // index into the vector<float> of the sound file

    double peakToPeak;
    double rms;
    double zcr;
    double centroid;
    double f0;

    int size() {
        return end - begin;
    }
};

bool Grain_comparator_ptp ( const Grain& l, const Grain& r)
   { return l.peakToPeak < r.peakToPeak; }
bool Grain_comparator_rms ( const Grain& l, const Grain& r)
   { return l.rms < r.rms; }
bool Grain_comparator_zcr ( const Grain& l, const Grain& r)
   { return l.zcr < r.zcr; }
bool Grain_comparator_sc ( const Grain& l, const Grain& r)
   { return l.centroid < r.centroid; }
bool Grain_comparator_f0 ( const Grain& l, const Grain& r)
   { return l.f0 < r.f0; }

// pass in grains vector, we will just add them
// N is the grain size
// this is basically the main of granular-analysis.cpp
void create_grains(std::vector<Grain>& grains, std::vector<double> audio_data, int N) {
    assert(N % 2 == 0 && N > 0);
    // print usages?

    int hop_size = N / 2;
    int nfft = N*4; // more frequency "precision"
    int window_size = N;
    int n = audio_data.size();

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

        frame_zcr *= (double(SAMPLE_RATE) / window_size); // convert to Hz
        double frame_ptp = frame_max_amp - frame_min_amp;
        double frame_rms = sqrt(frame_sum / (end_index - start_index));

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

        // SC
        double bin_step = double(SAMPLE_RATE) / nfft;
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
        // divide by 0
        if (frame_sc_dem < 0.0001 && frame_sc_dem > -0.0001) {
            frame_sc = 0.0;
        }

        // f0

        // sorted ascending by freq
        std::vector<FFT_Pair> peaks = stft_peaks(fft_buf, nfft, 10);
        int frame_f0_hop = 0; // no peaks = silence
        if (peaks.size() == 1) {
            frame_f0_hop = peaks[0].bin;
        } else if (peaks.size() > 1) {
            std::vector<int> diffs;
            // double for-loop, but this should be very small (max 50 iterations or so)
            for (int i = 0; i < peaks.size() - 1; i++) {
                for (int j = i + 1; j < peaks.size(); j++) {
                    diffs.push_back(peaks[j].bin - peaks[i].bin);
                }
            }
            frame_f0_hop = get_mode(diffs);
        }

        grains.push_back({start_index, end_index, frame_ptp , frame_rms, frame_zcr, frame_sc, frame_f0_hop * bin_step});
        // next frame
        start_index += hop_size;
    }
}


///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

using namespace al;

struct MyApp : App {
  Parameter background{"background", "", 0.0, "", 0.0f, 1.0f};
  Parameter db{"db", "", -60.0, "", -60.0f, 0.0f};
  ControlGUI gui;

  // converting float to double...probably fine for this application
  std::vector<double> input;
  std::vector<Grain> grains;
  int N;

  MyApp(int argc, char *argv[]) {
    if (argc < 2) {
      printf("granular-synth <.wav>\n");
      exit(1);
    }

    load(input, argv[1]);
    printf("input size is %ld\n", input.size());
    fflush(stdout);

    // this is how to get a command line argument
    N = 2048;
    if (argc > 2) {
        N = std::stoi(argv[2]);
    }

    create_grains(grains, input, N);

    // making this easier for now when we have to do hann windows, etc
    if (grains[grains.size() -1].size() < N) {
        grains.erase(grains.end() - 1);
    }
  }

  void onCreate() override {
    gui << background;
    gui << db;
    gui.init();
    navControl().active(false);
  }

  void onAnimate(double dt) override {
    //
  }

  void onDraw(Graphics &g) override {
    g.clear(background);
    gui.draw(g);
  }

  void onSound(AudioIOData &io) override {
    std::vector<double> window = hann_window(this->N);
    // two special cases
    if (this->grains.size() == 0) {
        printf("no audio\n");
        return;
    } else if (this->grains.size() == 1) {
        int start = this->grains[0].begin;
        int end = this->grains[0].end;
        while(io()) {
            // grains are guananteed to be N samples long, so this is a little overkill
            for (int i = 0; start + i < end; i++) {
                float sample = dbtoa(db.get()) * this->input[start+i] * window[i];
                io.out(0) = sample;
                io.out(1) = sample;
            }
        }
        return;
    }

    // general case
    while (io()) {
        // start the first grain
        int start = this->grains[0].begin;
        int end = this->grains[0].end;
        for (int i = 0; i < N / 2; i++) {
            float sample = dbtoa(db.get()) * this->input[start+i] * window[i];
            io.out(0) = sample;
            io.out(1) = sample;   
        }
        // now for all other grains
        for (int fr = 0; fr < this->grains.size(); fr++) {
            int frame_start = this->grains[fr].begin;
            int last_frame_start = this->grains[fr-1].begin;
            for (int i = 0; i < N / 2; i++) {
                // first half of this frame
                float sample = this->input[frame_start+i] * window[i];
                // last half of last frame
                sample += this->input[last_frame_start+(N/2)+i] * window[(N/2)+i];
                sample *= dbtoa(db.get());
                io.out(0) = sample;
                io.out(1) = sample;
            }
        }

        // finish the last grain
        start = this->grains[this->grains.size()-1].begin;
        end = this->grains[this->grains.size()-1].end;
        for (int i = N/2; i < N; i++) {
            float sample = dbtoa(db.get()) * this->input[start+i] * window[i];
            io.out(0) = sample;
            io.out(1) = sample;
        }
    }
  }

  bool onKeyDown(const Keyboard &k) override {
    int ascii = k.key();

    // respond to user action to re-order the grains
    //
    switch (ascii) {
        case 49 :
            printf("Sorted by Peak-to-peak Amp\n");
            std::sort(this->grains.begin(), this->grains.end(), Grain_comparator_ptp);
            break;
        case 50 :
            printf("Sorted by RMS\n");
            std::sort(this->grains.begin(), this->grains.end(), Grain_comparator_rms);
            break;
        case 51 :
            printf("Sorted by ZCR\n");
            std::sort(this->grains.begin(), this->grains.end(), Grain_comparator_zcr);
            break;
        case 52 :
            printf("Sorted by Spectral Centroid\n");
            std::sort(this->grains.begin(), this->grains.end(), Grain_comparator_sc);
            break;
        case 53 :
            printf("Sorted by F0\n");
            std::sort(this->grains.begin(), this->grains.end(), Grain_comparator_f0);
            break;
        default :
            printf("Please press a key 1-5\n");
    }
    return true;
  }

  bool onKeyUp(const Keyboard &k) override {
    int midiNote = asciiToMIDI(k.key());
    return true;
  }
};

int main(int argc, char *argv[]) {
  MyApp app(argc, argv);
  app.configureAudio(SAMPLE_RATE, BLOCK_SIZE, 2, 1);
  app.start();
  return 0;
}
