#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

std::vector<std::string> ReadLines(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    return lines;
}

std::vector<int64_t> ReadInt64Vector(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    std::vector<int64_t> values;
    int64_t v = 0;
    while (in >> v) {
        values.push_back(v);
    }

    if (values.empty()) {
        throw std::runtime_error("No integers found in file: " + path);
    }
    return values;
}

float SoftmaxScore(const float* logits, int64_t num_labels, int64_t best_index) {
    float max_logit = logits[0];
    for (int64_t i = 1; i < num_labels; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    double denom = 0.0;
    for (int64_t i = 0; i < num_labels; ++i) {
        denom += std::exp(static_cast<double>(logits[i] - max_logit));
    }
    double numer = std::exp(static_cast<double>(logits[best_index] - max_logit));
    return static_cast<float>(numer / denom);
}

#ifdef _WIN32
std::wstring Utf8ToWide(const std::string& text) {
    if (text.empty()) {
        return std::wstring();
    }
    int len = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
    if (len <= 0) {
        throw std::runtime_error("Failed to convert UTF-8 string to wide string.");
    }

    std::wstring out(static_cast<size_t>(len), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, &out[0], len);
    if (!out.empty() && out.back() == L'\0') {
        out.pop_back();
    }
    return out;
}
#endif

}  // namespace

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr
            << "Usage:\n"
            << "  ner_onnx_infer <model.onnx> <input_ids.txt> <attention_mask.txt> "
               "<token_type_ids.txt> <tokens.txt> <bio_labels.txt>\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string input_ids_path = argv[2];
    const std::string attention_mask_path = argv[3];
    const std::string token_type_ids_path = argv[4];
    const std::string tokens_path = argv[5];
    const std::string labels_path = argv[6];

    try {
        std::vector<int64_t> input_ids = ReadInt64Vector(input_ids_path);
        std::vector<int64_t> attention_mask = ReadInt64Vector(attention_mask_path);
        std::vector<int64_t> token_type_ids = ReadInt64Vector(token_type_ids_path);
        std::vector<std::string> tokens = ReadLines(tokens_path);
        std::vector<std::string> labels = ReadLines(labels_path);

        const int64_t seq_len = static_cast<int64_t>(input_ids.size());
        if (attention_mask.size() != input_ids.size() || token_type_ids.size() != input_ids.size()) {
            throw std::runtime_error("Input tensor lengths do not match.");
        }

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "koelectra_ner");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        session_options.SetIntraOpNumThreads(1);

#ifdef _WIN32
        std::wstring model_path_w = Utf8ToWide(model_path);
        Ort::Session session(env, model_path_w.c_str(), session_options);
#else
        Ort::Session session(env, model_path.c_str(), session_options);
#endif

        Ort::AllocatorWithDefaultOptions allocator;

        std::vector<Ort::AllocatedStringPtr> input_name_holders;
        std::vector<const char*> input_names;

        const size_t input_count = session.GetInputCount();
        for (size_t i = 0; i < input_count; ++i) {
            auto name = session.GetInputNameAllocated(i, allocator);
            input_names.push_back(name.get());
            input_name_holders.push_back(std::move(name));
        }

        std::unordered_map<std::string, std::vector<int64_t>*> input_data = {
            {"input_ids", &input_ids},
            {"attention_mask", &attention_mask},
            {"token_type_ids", &token_type_ids},
        };

        std::vector<Ort::Value> input_tensors;
        std::vector<int64_t> input_shape = {1, seq_len};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        for (const char* input_name : input_names) {
            auto it = input_data.find(input_name);
            if (it == input_data.end()) {
                throw std::runtime_error(std::string("Unexpected input name in ONNX model: ") + input_name);
            }
            std::vector<int64_t>* data = it->second;
            input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
                memory_info,
                data->data(),
                data->size(),
                input_shape.data(),
                input_shape.size()));
        }

        std::vector<Ort::AllocatedStringPtr> output_name_holders;
        std::vector<const char*> output_names;

        const size_t output_count = session.GetOutputCount();
        for (size_t i = 0; i < output_count; ++i) {
            auto name = session.GetOutputNameAllocated(i, allocator);
            output_names.push_back(name.get());
            output_name_holders.push_back(std::move(name));
        }

        auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names.data(),
            output_names.size());

        if (outputs.empty()) {
            throw std::runtime_error("No output tensors were produced.");
        }

        Ort::Value& logits_tensor = outputs[0];
        Ort::TensorTypeAndShapeInfo shape_info = logits_tensor.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> logits_shape = shape_info.GetShape();

        if (logits_shape.size() != 3) {
            std::ostringstream oss;
            oss << "Expected rank-3 logits tensor [B, T, C], got rank " << logits_shape.size();
            throw std::runtime_error(oss.str());
        }

        const int64_t batch = logits_shape[0];
        const int64_t out_seq_len = logits_shape[1];
        const int64_t num_labels = logits_shape[2];

        if (batch != 1) {
            throw std::runtime_error("This test tool expects batch size 1.");
        }

        if (out_seq_len != seq_len) {
            std::cerr << "Warning: model output seq_len(" << out_seq_len << ") != input seq_len(" << seq_len << ")\n";
        }

        if (shape_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("Expected float logits tensor output.");
        }

        float* logits = logits_tensor.GetTensorMutableData<float>();

        std::cout << "TokenIndex\tToken\tPredLabel\tScore\n";
        const int64_t loop_len = std::min<int64_t>(out_seq_len, static_cast<int64_t>(tokens.size()));

        for (int64_t t = 0; t < loop_len; ++t) {
            const float* row = logits + (t * num_labels);

            int64_t best_idx = 0;
            float best_logit = row[0];
            for (int64_t c = 1; c < num_labels; ++c) {
                if (row[c] > best_logit) {
                    best_logit = row[c];
                    best_idx = c;
                }
            }

            float score = SoftmaxScore(row, num_labels, best_idx);
            std::string label = (best_idx >= 0 && best_idx < static_cast<int64_t>(labels.size()))
                                    ? labels[best_idx]
                                    : ("LABEL_" + std::to_string(best_idx));

            std::cout << t << "\t" << tokens[t] << "\t" << label << "\t" << std::fixed << std::setprecision(4) << score
                      << "\n";
        }

        std::cout << "\nInference completed successfully." << std::endl;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 3;
    }

    return 0;
}
