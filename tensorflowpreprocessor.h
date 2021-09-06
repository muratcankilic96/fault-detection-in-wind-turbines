#ifndef TENSORFLOWPREPROCESSOR_H
#define TENSORFLOWPREPROCESSOR_H


#include <cppflow/cppflow.h>
#include <cppflow/tensor.h>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/serialization/vector.hpp>

#include "dsp.h"

using namespace cppflow;

class TensorflowPreprocessor
{
public:
    static tensor aubio_matrix_vector_to_tensor(std::vector<fmat_t *> vec);
    static tensor min_max_scaling(tensor t, float lower_bound, float upper_bound);
    static void print_tensor_3d(tensor t);
    static void to_json(std::vector<tensor> t, bool smote_enabled);
};

#endif // TENSORFLOWPREPROCESSOR_H

