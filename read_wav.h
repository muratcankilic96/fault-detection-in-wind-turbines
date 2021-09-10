#include <vector>
#include <string>
#include <iostream>
#include <aubio.h>

#ifndef READ_WAV_H
#define READ_WAV_H
#define HOP_SIZE 512

#endif // READ_WAV_H

// Reads a WAV file and converts its contents into a float array.
fvec_t * read_wav_file(std::string filename, uint_t * sample_rate);

// Places the chunk of a smaller float vector into part of a larger float vector.
void write_fvec_buffer(fvec_t * write_from, fvec_t * write_to, uint_t start_pos);

// Slices a long fvec_t object into smaller chunks and combines them as a vector<fvec_t *> object.
std::vector<fvec_t *> slice_fvec(fvec_t * read, uint_t chunk_size);

// Memory management: Clears a vector of fvec_t object.
void del_fvec_vector(std::vector<fvec_t *> * vec);

// Memory management: Clears a vector of fmat_t object.
void del_fmat_vector(std::vector<fmat_t *> * mat);
