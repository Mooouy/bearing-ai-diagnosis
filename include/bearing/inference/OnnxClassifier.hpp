#pragma once

#include "bearing/core/ErrorCode.hpp"
#include "bearing/core/Types.hpp"
#include "bearing/inference/IClassifier.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/core/mat.hpp>

#include <memory>
#include <string>
#include <vector>

namespace bearing {

// OnnxClassifier 实现 IClassifier 接口：infer() 的签名在 Task 5 设计时
// 就已经和接口要求的签名一致，这里只是补上基类声明。initialize()/
// shutdown() 不属于 IClassifier（接口只暴露 infer()），所以仍然只在
// main() 里以具体类型调用，BearingApplication 只持有接口指针。
class OnnxClassifier final : public IClassifier {
public:
    explicit OnnxClassifier(std::string model_path);

    ErrorCode initialize();
    ErrorCode infer(const cv::Mat& image, InferenceResult& result) override;
    void shutdown();

    ErrorCode prepare_input(const cv::Mat& image,
                            std::vector<float>& tensor) const;
    ErrorCode map_output(const std::vector<float>& raw_output,
                         InferenceResult& result) const;

private:
    ErrorCode run_session(const std::vector<float>& input,
                          std::vector<float>& output);

    std::string model_path_;
    Ort::Env environment_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_name_storage_;
    std::vector<std::string> output_name_storage_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
};

}  // namespace bearing
