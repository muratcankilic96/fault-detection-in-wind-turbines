#include "tensorflowpreprocessor.h"
#include <fstream>
#include <QFile>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <filesystem>

// Converts a vector of Aubio matrices (vector<fmat_t *>) to a three-dimensional Tensorflow 2 tensor.
tensor TensorflowPreprocessor::aubio_matrix_vector_to_tensor(std::vector<fmat_t *> vec) {
    int v = (int) vec.size();
    int x = 0;
    // Determine the tensor shape.
    std::vector<int64_t> t_shape({v, vec[0]->height, vec[0]->length});
    std::vector<float> index(v * vec[0]->height * vec[0]->length);

    // Create the tensor in a flat manner.
    for(int i = 0; i < v; i++) {
        for(int j = 0; j < vec[0]->height; j++) {
            for(int k = 0; k < vec[0]->length; k++) {
                index[x++] = vec[i]->data[j][k];
            }
        }
    }

    // Build the tensor from the given values and shape.
    tensor t = cppflow::tensor(index, t_shape);

    // Return the tensor.
    return t;
}

// Prints the contents of a three-dimensional tensor.
void TensorflowPreprocessor::print_tensor_3d(tensor t) {
    // Get the tensor data as a flat array.
    auto index = t.get_data<float>();
    // Initialize index.
    int x = 0;
    // Get the tensor shape.
    auto shape = t.shape();
    // Get individual sizes for each tensor dimension.
    auto d1 = shape.get_data<int64_t>()[0];
    auto d2 = shape.get_data<int64_t>()[1];
    auto d3 = shape.get_data<int64_t>()[2];
    // Print the tensor.
    for(int i = 0; i < d1; i++) {
        for(int j = 0; j < d2; j++) {
            for(int k = 0; k < d3; k++) {
                std::cout << index[x++] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

// Normalizes a three-dimensional Tensorflow 2 tensor object within the given range [lower_bound, upper_bound].
tensor TensorflowPreprocessor::min_max_scaling(tensor t, float lower_bound, float upper_bound) {
    // Get the tensor data as a flat array.
    auto index = t.get_data<float>();
    // Initialize index.
    int x = 0;
    // Get the tensor shape.
    auto shape = t.shape();
    // Get individual sizes for each tensor dimension.
    auto d1 = shape.get_data<int64_t>()[0];
    auto d2 = shape.get_data<int64_t>()[1];
    auto d3 = shape.get_data<int64_t>()[2];

    // Initialize the minimum and the maximum values.
    float max_v = index[0];
    float min_v = index[0];

    // Determine the minimum and the maximum values.
    for(int i = 0; i < index.size(); i++) {
        if(index[i] > max_v)
            max_v = index[i];
        if(index[i] < min_v)
            min_v = index[i];
    }

    // Apply the scaling.
    for(int i = 0; i < d1; i++) {
        for(int j = 0; j < d2; j++) {
            for(int k = 0; k < d3; k++) {
                index[x] = lower_bound + (((index[x] - max_v) * (upper_bound - lower_bound)) / (max_v - min_v));
                x++;
            }
        }
    }

    // Create a new three-dimensional tensor from the new values..
    t = cppflow::tensor(index, {d1, d2, d3});

    // Return the tensor.
    return t;
}

// Writes the contents of a Tensorflow 2 tensor object into a JSON file, with SMOTE application and training model information.
void TensorflowPreprocessor::to_json(std::string filename, std::vector<tensor> t, int epoch_count, bool smote_enabled, int model) {
    // Get build directory.
    auto path = std::filesystem::current_path();
    // Get filename and create a file in the parent path (source path).
    std::ofstream json_file(path.parent_path() / filename);
    // Check whether the file is successfully created or not.
    if(json_file.is_open()) {
        // Begin the JSON file.
        json_file << "{\n";
        // Create an array of tensors.
        json_file << "  \"tensors\": [\n";
        // Write the information for each tensor given in the vector.
        for(int i = 0; i < t.size(); i++) {
            json_file << "{\n";
            // Write the contents for the tensor.
            json_file << "      \"content\": [";
            // Get the tensor data as a flat array.
            std::vector<float> arr = t[i].get_data<float>();
            // Get the tensor shape.
            std::vector<int64_t> shp = t[i].shape().get_data<int64_t>();
            for(int j = 0; j < arr.size(); j++) {
                if(j == arr.size() - 1)
                    json_file << arr[j];
                else
                    json_file << arr[j] << ", ";
            }
            json_file << "],\n";
            // Write the dimensions for the tensor.
            json_file << "      \"dims\": [";
            for(int j = 0; j < shp.size(); j++) {
                if(j == shp.size() - 1)// Write the dimensions for the tensor.
                    json_file << shp[j];
                else
                    json_file << shp[j] << ", ";
            }
            json_file << "]\n";
        json_file << "}";
        if(i == t.size() - 1)
            json_file << "\n";
        else
            json_file << ",\n";
        }
        // Write SMOTE boolean.
        json_file << "],\n";
        json_file << "  \"smote\": " << (smote_enabled ? "true" : "false") << "\n";
        // Write Keras model number.
        json_file << ",\n";
        json_file << "  \"model\": " << model << "\n";
        // Write the epoch count for training.
        json_file << ",\n";
        json_file << "  \"epoch\": " << epoch_count << "\n";
        // Close the JSON file.
        json_file << "}";
        json_file.close();
    } else // Return error if opening the file ends up with failure.
        std::cout << "Error opening the file" << std::endl;
}

// Writes the contents of a Tensorflow 2 tensor object into a JSON file, with SMOTE application information.
void TensorflowPreprocessor::to_json(std::string filename, std::vector<tensor> t, bool smote_enabled) {
    // Get build directory.
    auto path = std::filesystem::current_path();
    // Get filename and create a file in the parent path (source path).
    std::ofstream json_file(path.parent_path() / filename);
    // Check whether the file is successfully created or not.
    if(json_file.is_open()) {
        // Begin the JSON file.
        json_file << "{\n";
        // Create an array of tensors.
        json_file << "  \"tensors\": [\n";
        // Write the information for each tensor given in the vector.
        for(int i = 0; i < t.size(); i++) {
            json_file << "{\n";
            // Write the contents for the tensor.
            json_file << "      \"content\": [";
            // Get the tensor data as a flat array.
            std::vector<float> arr = t[i].get_data<float>();
            // Get the tensor shape.
            std::vector<int64_t> shp = t[i].shape().get_data<int64_t>();
            for(int j = 0; j < arr.size(); j++) {
                if(j == arr.size() - 1)
                    json_file << arr[j];
                else
                    json_file << arr[j] << ", ";
            }
            json_file << "],\n";
            // Write the dimensions for the tensor.
            json_file << "      \"dims\": [";
            for(int j = 0; j < shp.size(); j++) {
                if(j == shp.size() - 1)
                    json_file << shp[j];
                else
                    json_file << shp[j] << ", ";
            }
            json_file << "]\n";
        json_file << "}";
        if(i == t.size() - 1)
            json_file << "\n";
        else
            json_file << ",\n";
        }
        // Write SMOTE boolean.
        json_file << "],\n";
        json_file << "  \"smote\": " << (smote_enabled ? "true" : "false") << "\n";
        json_file << "}";
        // Close the JSON file.
        json_file.close();
    } else // Return error if opening the file ends up with failure.
        std::cout << "Error opening the file" << std::endl;
}

// Builds a vector of tensor objects from a JSON file.
std::vector<tensor> TensorflowPreprocessor::from_json(std::string filename) {
    // Create a QT 5 file object.
    QFile json_file;
    // Initialize a vector of tensors.
    std::vector<tensor> tensors = std::vector<tensor>();
    // Get build directory.
    auto path = std::filesystem::current_path();
    // Get the desired file from the parent path.
    json_file.setFileName(path.parent_path() / filename);
    // Open the JSON file.
    json_file.open(QIODevice::ReadOnly | QIODevice::Text);
    // Check whether the file is successfully created or not.
    if(json_file.isOpen()) {
        // Load the JSON data into memory.
        QString buffer = json_file.readAll();
        QJsonDocument doc = QJsonDocument::fromJson(buffer.toUtf8());
        buffer.clear();
        QJsonObject doc_obj = doc.object();
        QJsonArray tensor_list = doc_obj["tensors"].toArray();
        for(int i = 0; i < tensor_list.size(); i++) {
            QJsonObject tensor_info = tensor_list[i].toObject();
            QJsonArray tensor_elements = tensor_info["content"].toArray();
            QJsonArray tensor_dims = tensor_info["dims"].toArray();
            std::vector<float> vec_elems = std::vector<float>();
            std::vector<int64_t> vec_dims = std::vector<int64_t>();
            for(int i = 0; i < tensor_elements.size(); i++)
                vec_elems.push_back(tensor_elements[i].toDouble());
            for(int i = 0; i < tensor_dims.size(); i++)
                vec_dims.push_back((int64_t) tensor_dims[i].toInt());
            tensor t = cppflow::tensor(vec_elems, vec_dims);
            tensors.push_back(t);
        }
        // Remove the file since it is intended to be a temporary object.
        remove(path.parent_path() / filename);
        return tensors;
    } else { // Return error if opening the file ends up with failure.
        std::cout << "Error opening the file" << std::endl;
        return {};
    }
}

// Ignores all dimensions larger than 3D.
tensor TensorflowPreprocessor::reshape_dims_to_3d(tensor t) {
    // Get the tensor data as a flat array.
    auto index = t.get_data<float>();
    // Get the tensor shape.
    auto shape = t.shape();
    // Get individual sizes for each tensor dimension.
    auto d1 = shape.get_data<int64_t>()[0];
    auto d2 = shape.get_data<int64_t>()[1];
    auto d3 = shape.get_data<int64_t>()[2];
    // Resize the flat array. [[UPDATE 13 September 2021 OUT OF BOUNDS FIX]]
    index.resize(d1 * d2 * d3);
    // Create a new tensor from start.
    tensor t_new = cppflow::tensor(index, {d1, d2, d3});
    // Return the tensor.
    return t_new;
}
