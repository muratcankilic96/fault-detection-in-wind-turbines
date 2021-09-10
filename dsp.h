#include <string>
#include "read_wav.h"

#ifndef DSP_H
#define DSP_H

class DspTools
{
public:
    // Creates a MFCC matrix where each row corresponds to each window.
    static fmat_t * create_mfcc(fvec_t * source, uint_t sample_size, uint_t nfilters, uint_t cep_count, uint_t nfft, uint_t wlen);

    // Creates a spectrogram matrix where each row corresponds to each window.
    static fmat_t * create_spectrogram(fvec_t * source, uint_t nfft);

    // Creates a Mel spectrogram matrix where each row corresponds to each window.
    static fmat_t * create_mel_spectrogram(fvec_t * source, uint_t nfft);
};
#endif // DSP_H
