#include "read_wav.h"

// Reads a WAV file and converts its contents into a float array.
fvec_t * read_wav_file(std::string filename, uint_t * sample_rate) {
    // Initialize necessary variables.
    uint_t frame = 0, read = 0;
    aubio_source_t * wav_file;

    // Get source.
    wav_file = new_aubio_source(filename.c_str(), 0, HOP_SIZE);
    if(!wav_file) {
        std::cout << "An error occurred while creating the sample." << std::endl;
        return nullptr;
    }

    // Two float vectors are defined, one is for buffers and the other for the whole sound sample.
    fvec_t * wav_array = new_fvec(HOP_SIZE);
    fvec_t * whole_array = new_fvec(aubio_source_get_duration(wav_file));

    // Read from the source file into buffer, then places the buffer into the whole sound sample.
    do {
        aubio_source_do(wav_file, wav_array, &read);
        write_fvec_buffer(wav_array, whole_array, HOP_SIZE * frame++);
    } while(read == HOP_SIZE);

    // Save sample rate.
    *sample_rate = aubio_source_get_samplerate(wav_file);

    // Destroy unused objects.
    del_fvec(wav_array);
    aubio_source_close(wav_file);

    // Return the float vector.
    return whole_array;
}

// Places the chunk of a smaller float vector into part of a larger float vector.
void write_fvec_buffer(fvec_t * write_from, fvec_t * write_to, uint_t start_pos) {
    for(int i = 0; i < write_from->length && start_pos + i < write_to->length; i++)
        write_to->data[start_pos + i] = write_from->data[i];

    return;
}

// Slices a long fvec_t object into smaller chunks and combines them as a vector<fvec_t *> object.
std::vector<fvec_t *> slice_fvec(fvec_t * read, uint_t chunk_size) {
    int num_blocks = read->length / chunk_size;
    int x = 0;

    std::vector<fvec_t *> out(num_blocks);
    for(int i = 0; i < num_blocks; i++) {
        out[i] = new_fvec(chunk_size);
        for(int j = 0; j < chunk_size; j++) {
            out[i]->data[j] = read->data[x++];
        }
    }


    return out;
}
