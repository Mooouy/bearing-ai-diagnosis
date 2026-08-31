#include "bearing/inference/OnnxClassifier.hpp"
#include "bearing/processing/VibrationProcessor.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

int main() {
    bearing::OnnxClassifier classifier(TEST_MODEL_PATH);

    cv::Mat image(224, 224, CV_8UC1, cv::Scalar(112));
    std::vector<float> tensor;
    bearing::ErrorCode status = classifier.prepare_input(image, tensor);

    if (status != bearing::ErrorCode::success ||
        tensor.size() != 224U * 224U) {
        std::cerr << "input preparation failed" << std::endl;
        return EXIT_FAILURE;
    }

    float expected = (112.0F - 33.405F) / 78.795F;
    if (std::fabs(tensor.front() - expected) > 0.0001F) {
        std::cerr << "normalization does not match training" << std::endl;
        return EXIT_FAILURE;
    }

    bearing::InferenceResult mapped;
    status = classifier.map_output({0.1F, 0.2F, 0.6F, 0.1F}, mapped);
    if (status != bearing::ErrorCode::success ||
        mapped.class_id != 2 ||
        mapped.class_name != "保持架故障" ||
        std::fabs(mapped.confidence - 0.6F) > 0.0001F) {
        std::cerr << "four-class mapping is incorrect" << std::endl;
        return EXIT_FAILURE;
    }

    // ====== Step 6：用真实模型对一张真实生成的 CWT 图做一次端到端推理 ======
    // 复用 Task 4（VibrationProcessor）测试里的合成正弦信号方法，生成一张
    // 真实的 224x224 CWT 融合图，而不是像上面那样用纯色占位图。
    // 这里只断言“结果形状合理”（状态成功、类别落在 4 类范围内、置信度有限、
    // 耗时大于零），不断言具体是哪一类——和原始可执行文件的逐类比对是
    // Task 8 的工作，本任务只验证“移植后的推理流程本身能跑通”。
    bearing::RawVibrationBatch batch;
    batch.sample_rate_hz = 50000.0F;

    constexpr std::size_t sample_count = 5000;
    batch.x_axis.reserve(sample_count);
    batch.y_axis.reserve(sample_count);
    batch.z_axis.reserve(sample_count);

    for (std::size_t index = 0; index < sample_count; ++index) {
        float time = static_cast<float>(index) / batch.sample_rate_hz;
        batch.x_axis.push_back(std::sin(2.0F * 3.1415926F * 100.0F * time));
        batch.y_axis.push_back(std::sin(2.0F * 3.1415926F * 200.0F * time));
        batch.z_axis.push_back(std::sin(2.0F * 3.1415926F * 300.0F * time));
    }

    bearing::VibrationProcessor processor;
    bearing::ProcessedVibration processed;
    status = processor.process(batch, processed);
    if (status != bearing::ErrorCode::success) {
        std::cerr << "failed to build CWT fixture for real-model inference"
                   << std::endl;
        return EXIT_FAILURE;
    }

    status = classifier.initialize();
    if (status != bearing::ErrorCode::success) {
        std::cerr << "failed to load real ONNX model" << std::endl;
        return EXIT_FAILURE;
    }

    bearing::InferenceResult real_result;
    status = classifier.infer(processed.cwt_image, real_result);
    if (status != bearing::ErrorCode::success) {
        std::cerr << "real-model inference failed" << std::endl;
        return EXIT_FAILURE;
    }

    if (real_result.class_id < 0 || real_result.class_id > 3) {
        std::cerr << "class id out of the expected four-class range"
                   << std::endl;
        return EXIT_FAILURE;
    }

    if (real_result.class_name.empty()) {
        std::cerr << "class name must not be empty" << std::endl;
        return EXIT_FAILURE;
    }

    if (!std::isfinite(real_result.confidence)) {
        std::cerr << "confidence must be finite" << std::endl;
        return EXIT_FAILURE;
    }

    if (real_result.inference_time_us == 0) {
        std::cerr << "inference time must be greater than zero" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "real inference result: class_id=" << real_result.class_id
               << " class_name=" << real_result.class_name
               << " confidence=" << real_result.confidence
               << " inference_time_us=" << real_result.inference_time_us
               << std::endl;

    classifier.shutdown();

    return EXIT_SUCCESS;
}
