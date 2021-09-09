#include "dsp.h"
#include <math.h>

// Creates a MFCC matrix where each row corresponds to each window.
fmat_t * DspTools::create_mfcc(fvec_t * source, uint_t sample_size, uint_t nfilters, uint_t cep_count, uint_t nfft, uint_t wlen) {
    // Initialize necessary variables.
    cvec_t * window_part = new_cvec(wlen);
    uint_t read = 0, frames = source->length / wlen, x = 0;
    fmat_t * whole = new_fmat(frames, cep_count);

    // Get frequency domain.
    auto phase_vocoder = new_aubio_pvoc(wlen, wlen);

    // Get MFCC class.
    auto mfcc = new_aubio_mfcc(wlen, nfilters, cep_count, sample_size);
    if(!mfcc) {
        std::cout << "An error occurred while creating the MFCC coefficients." << std::endl;
        if(phase_vocoder) del_aubio_pvoc(phase_vocoder);
        return nullptr;
    }

    // Apply MFCC for each passing window. Note that there is no implementation for overlap.
    fvec_t * mfcc_window_input, * mfcc_output;
    do {
        mfcc_window_input = new_fvec(wlen), mfcc_output = new_fvec(cep_count);
        for(int i = 0; i < wlen; i++) {
            mfcc_window_input->data[i] = source->data[x];
            x = x + 1;
        }
        aubio_pvoc_do(phase_vocoder, mfcc_window_input, window_part);
        aubio_mfcc_do(mfcc, window_part, mfcc_output);
        for(int i = 0; i < cep_count; i++) {
            whole->data[read][i] = mfcc_output->data[i];
        }
        read++;
    } while(read != frames);

    // Destroy unused variables.
    delete mfcc_window_input->data;
    mfcc_window_input->data = nullptr;
    delete mfcc_window_input;
    mfcc_window_input = nullptr;
    delete mfcc_output->data;
    delete mfcc_output;
    delete window_part->norm;
    delete window_part->phas;
    window_part->norm = nullptr;
    window_part->phas = nullptr;
    delete window_part;
    window_part = nullptr;
    del_aubio_mfcc(mfcc);
    del_aubio_pvoc(phase_vocoder);

    // Return the MFCC matrix.
    return whole;
}

// Creates a spectrogram matrix where each row corresponds to each window.
fmat_t * DspTools::create_spectrogram(fvec_t * source, uint_t nfft) {
    // Initialize necessary variables.
    cvec_t * window_part = new_cvec(nfft);
    uint_t frames = source->length / nfft, x = 0;
    fmat_t * whole = new_fmat(frames, nfft / 2 + 1);

    // Get FFT class.
    aubio_fft_t * fft = new_aubio_fft(nfft);
    if(!fft) {
        std::cout << "An error occurred while applying FFT." << std::endl;
        return nullptr;
    }

    // Apply FFT for each passing window. Note that there is no implementation for overlap.
    // Create a FFT window.
    fvec_t * fft_window_input = new_fvec(nfft);


    for(int i = 0; i < frames; i++) {
        // Fit the window.
        for(int j = 0; j < nfft; j++) {
            fft_window_input->data[j] = source->data[x];
            x = x + 1;
        }
        // Apply FFT.
        aubio_fft_do(fft, fft_window_input, window_part);
        for(int j = 0; j < window_part->length; j++) {
            // Compute magnitude of complex number. Round the numbers to prevent overflow.
            float round_norm = window_part->norm[j];
            float round_phas = window_part->phas[j];
            float squared = round_norm * round_norm + round_phas * round_phas;
            float magn;
            if(squared == 0)
                magn = 0;
            else
                magn = 0.5 * log10(squared);
            whole->data[i][j] = magn;
        }
    }

    // Destroy unused variables.
    delete fft_window_input->data;
    fft_window_input->data = nullptr;
    delete fft_window_input;
    fft_window_input = nullptr;
    delete window_part->norm;
    delete window_part->phas;
    window_part->norm = nullptr;
    window_part->phas = nullptr;
    delete window_part;
    window_part = nullptr;
    del_aubio_fft(fft);

    // Return the MFCC matrix.
    return whole;
}

// Creates a Mel spectrogram matrix where each row corresponds to each window.
fmat_t * DspTools::create_mel_spectrogram(fvec_t * source, uint_t nfft) {
    // Initialize necessary variables.
    cvec_t * window_part = new_cvec(nfft);
    uint_t frames = source->length / nfft, x = 0;
    fmat_t * whole = new_fmat(frames, nfft / 2 + 1);

    // Get FFT class.
    aubio_fft_t * fft = new_aubio_fft(nfft);
    if(!fft) {
        std::cout << "An error occurred while applying FFT." << std::endl;
        return nullptr;
    }

    // Apply FFT for each passing window. Note that there is no implementation for overlap.
    // Create a FFT window.
    fvec_t * fft_window_input = new_fvec(nfft);


    for(int i = 0; i < frames; i++) {
        // Fit the window.
        for(int j = 0; j < nfft; j++) {
            fft_window_input->data[j] = source->data[x];
            x = x + 1;
        }
        // Apply FFT.
        aubio_fft_do(fft, fft_window_input, window_part);
        for(int j = 0; j < window_part->length; j++) {
            // Compute magnitude of complex number. Round the numbers to prevent overflow.
            float round_norm = window_part->norm[j];
            float round_phas = window_part->phas[j];
            float squared = round_norm * round_norm + round_phas * round_phas;

            // Different from normal spectrogram, convert the value to Mel scale.
            squared = 2595 * log10(1 + squared / 700);

            float magn;

            if(squared == 0)
                magn = 0;
            else
                magn = 0.5 * log10(squared);

            whole->data[i][j] = magn;
        }
    }

    // Destroy unused variables.
    delete fft_window_input->data;
    fft_window_input->data = nullptr;
    delete fft_window_input;
    fft_window_input = nullptr;
    delete window_part->norm;
    delete window_part->phas;
    window_part->norm = nullptr;
    window_part->phas = nullptr;
    delete window_part;
    window_part = nullptr;
    del_aubio_fft(fft);

    // Return the MFCC matrix.
    return whole;
}
