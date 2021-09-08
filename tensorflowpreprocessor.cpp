#include "tensorflowpreprocessor.h"
#include <fstream>
#include <QFile>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <filesystem>


tensor TensorflowPreprocessor::aubio_matrix_vector_to_tensor(std::vector<fmat_t *> vec) {
    int v = (int) vec.size();
    int x = 0;
    std::vector<int64_t> t_shape({v, vec[0]->height, vec[0]->length});
    std::vector<float> index(v * vec[0]->height * vec[0]->length);

    for(int i = 0; i < v; i++) {
        for(int j = 0; j < vec[0]->height; j++) {
            for(int k = 0; k < vec[0]->length; k++) {
                index[x++] = vec[i]->data[j][k];
            }
        }
    }

    tensor t = cppflow::tensor(index, t_shape);

    return t;
}

void TensorflowPreprocessor::print_tensor_3d(tensor t) {
    auto index = t.get_data<float>();
    int x = 0;
    auto shape = t.shape();
    auto d1 = shape.get_data<int64_t>()[0];
    auto d2 = shape.get_data<int64_t>()[1];
    auto d3 = shape.get_data<int64_t>()[2];
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

tensor TensorflowPreprocessor::min_max_scaling(tensor t, float lower_bound, float upper_bound) {

    auto index = t.get_data<float>();
    int x = 0;
    auto shape = t.shape();
    auto d1 = shape.get_data<int64_t>()[0];
    auto d2 = shape.get_data<int64_t>()[1];
    auto d3 = shape.get_data<int64_t>()[2];

    int max_v = index[0];
    int min_v = index[0];

    for(int i = 0; i < index.size(); i++) {
        if(index[i] > max_v)
            max_v = index[i];
        if(index[i] < min_v)
            min_v = index[i];
    }

    for(int i = 0; i < d1; i++) {
        for(int j = 0; j < d2; j++) {
            for(int k = 0; k < d3; k++) {
                index[x] = lower_bound + (((index[x] - max_v) * (upper_bound - lower_bound)) / (max_v - min_v));
                x++;
            }
        }
    }

    t = cppflow::tensor(index, {d1, d2, d3});

    return t;
}

void TensorflowPreprocessor::to_json(std::string filename, std::vector<tensor> t, int epoch_count, bool smote_enabled, int model) {
    auto path = std::filesystem::current_path();
    std::ofstream json_file(path.parent_path() / filename);
    if(json_file.is_open()) {
        json_file << "{\n";
        json_file << "  \"tensors\": [\n";
        for(int i = 0; i < t.size(); i++) {
            json_file << "{\n";
            json_file << "      \"content\": [";
            std::vector<float> arr = t[i].get_data<float>();
            std::vector<int64_t> shp = t[i].shape().get_data<int64_t>();
            for(int j = 0; j < arr.size(); j++) {
                if(j == arr.size() - 1)
                    json_file << arr[j];
                else
                    json_file << arr[j] << ", ";
            }
            json_file << "],\n";
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
        json_file << "],\n";
        json_file << "  \"smote\": " << (smote_enabled ? "true" : "false") << "\n";
        json_file << ",\n";
        json_file << "  \"model\": " << model << "\n";
        json_file << ",\n";
        json_file << "  \"epoch\": " << epoch_count << "\n";
        json_file << "}";
    } else
        std::cout << "Error opening the file" << std::endl;
}

void TensorflowPreprocessor::to_json(std::string filename, std::vector<tensor> t, bool smote_enabled) {
    auto path = std::filesystem::current_path();
    std::ofstream json_file(path.parent_path() / filename);
    if(json_file.is_open()) {
        json_file << "{\n";
        json_file << "  \"tensors\": [\n";
        for(int i = 0; i < t.size(); i++) {
            json_file << "{\n";
            json_file << "      \"content\": [";
            std::vector<float> arr = t[i].get_data<float>();
            std::vector<int64_t> shp = t[i].shape().get_data<int64_t>();
            for(int j = 0; j < arr.size(); j++) {
                if(j == arr.size() - 1)
                    json_file << arr[j];
                else
                    json_file << arr[j] << ", ";
            }
            json_file << "],\n";
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
        json_file << "],\n";
        json_file << "  \"smote\": " << (smote_enabled ? "true" : "false") << "\n";
        json_file << "}";
        json_file.close();
    } else
        std::cout << "Error opening the file" << std::endl;
}

std::vector<tensor> TensorflowPreprocessor::from_json(std::string filename) {
    QFile json_file;
    std::vector<tensor> tensors = std::vector<tensor>();
    auto path = std::filesystem::current_path();
    json_file.setFileName(path.parent_path() / filename);
    json_file.open(QIODevice::ReadOnly | QIODevice::Text);
    if(json_file.isOpen()) {
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

        remove(path.parent_path() / "WindTurbineSimulationLinux" / filename);

        return tensors;
    } else {
        std::cout << "Error opening the file" << std::endl;
        return {};
    }
}

// Ignores all dimensions larger than 3D.
tensor TensorflowPreprocessor::reshape_dims_to_3d(tensor t) {
    auto index = t.get_data<float>();
    int x = 0;
    auto shape = t.shape();
    auto d1 = shape.get_data<int64_t>()[0];
    auto d2 = shape.get_data<int64_t>()[1];
    auto d3 = shape.get_data<int64_t>()[2];
    tensor t_new = cppflow::tensor(index, {d1, d2, d3});
    return t_new;
}
