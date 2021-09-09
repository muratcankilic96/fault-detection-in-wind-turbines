#include <vector>
#include <string>
#include <iostream>
#include <aubio.h>

#ifndef READ_WAV_H
#define READ_WAV_H
#define HOP_SIZE 512

#endif // READ_WAV_H

fvec_t * read_wav_file(std::string filename, uint_t * sample_rate);
void write_fvec_buffer(fvec_t * write_from, fvec_t * write_to, uint_t start_pos);
std::vector<fvec_t *> slice_fvec(fvec_t * read, uint_t chunk_size);
void del_fvec_vector(std::vector<fvec_t *> * vec);
void del_fmat_vector(std::vector<fmat_t *> * mat);
