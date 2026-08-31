#include "bearing/inference/OnnxClassifier.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <utility>

namespace bearing {

namespace {

// ============================================================================
// 命名常量：和训练时使用的归一化参数、模型固定输入尺寸保持一致。
// 数值来源：原始项目 AIModelFramework::infer 里的
// "(pixel - 33.405) / 78.795" 归一化公式，必须和训练脚本完全一致，
// 否则模型看到的输入分布和训练时不一样，推理结果会不可靠。
// ============================================================================

// 灰度像素归一化用到的均值和标准差（训练数据集统计得到，不可更改）。
constexpr float pixel_normalization_mean = 33.405F;
constexpr float pixel_normalization_std = 78.795F;

// 模型固定接受的方形输入边长（像素）。
constexpr int model_input_size = 224;

// 模型固定输出的类别数量（轴承振动 4 分类）。
constexpr std::size_t expected_class_count = 4;

// 四分类的类别名称，顺序必须和模型训练时的类别索引一一对应：
// 0=正常, 1=滚动体故障, 2=保持架故障, 3=外圈故障。
// 这个顺序是整个项目固定的约定，不能调整。
const std::vector<std::string>& bearing_class_names() {
    static const std::vector<std::string> class_names = {
        "正常",
        "滚动体故障",
        "保持架故障",
        "外圈故障"
    };
    return class_names;
}

}  // namespace

// 构造函数只保存模型路径、构造 ONNX Runtime 的全局环境对象，
// 不在这里加载模型——真正的模型加载放到 initialize() 里，
// 这样调用方可以先用 prepare_input/map_output 做单元测试，
// 不需要每次都付出加载模型的开销。
OnnxClassifier::OnnxClassifier(std::string model_path)
    : model_path_(std::move(model_path)),
      environment_(ORT_LOGGING_LEVEL_WARNING, "BearingVibrationSystem") {}

// initialize —— 创建 ONNX Runtime 会话并加载模型文件。
// 移植自原始项目 AIModelFramework::ONNXModel::load。
// 输入/输出节点的名字必须用 std::string 单独存一份（input_name_storage_ /
// output_name_storage_），再让 const char* 数组指向这些字符串的内部缓冲区：
// 因为 Run() 接口要的是 const char* const* 数组，如果直接保存
// GetInputNameAllocated 返回的临时智能指针，函数返回后字符串就被释放了，
// 后面 run_session 用到的指针会变成悬空指针。
ErrorCode OnnxClassifier::initialize() {
    try {
        // 当前保持单线程 CPU provider：macOS 开发机和树莓派现场的
        // ONNX Runtime/XNNPACK 编译选项不一定一致，先保证可复现；若后续
        // Pi 端性能验证不足，再把 XNNPACK 作为显式性能开关引入。
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_BASIC);

        session_ = std::make_unique<Ort::Session>(
            environment_, model_path_.c_str(), session_options_);

        Ort::AllocatorWithDefaultOptions allocator;

        const std::size_t input_count = session_->GetInputCount();
        input_name_storage_.resize(input_count);
        input_names_.resize(input_count);
        for (std::size_t index = 0; index < input_count; ++index) {
            auto input_name = session_->GetInputNameAllocated(index, allocator);
            input_name_storage_[index] = input_name.get();
            input_names_[index] = input_name_storage_[index].c_str();
        }

        const std::size_t output_count = session_->GetOutputCount();
        output_name_storage_.resize(output_count);
        output_names_.resize(output_count);
        for (std::size_t index = 0; index < output_count; ++index) {
            auto output_name = session_->GetOutputNameAllocated(index, allocator);
            output_name_storage_[index] = output_name.get();
            output_names_[index] = output_name_storage_[index].c_str();
        }

        return ErrorCode::success;
    } catch (const std::exception&) {
        session_.reset();
        return ErrorCode::model_load_error;
    }
}

// infer —— 对外的主入口：预处理 -> 推理 -> 后处理 -> 记录耗时。
// 这个拆分方式直接对应需求文档里的分解：每一步都是一个独立、可单独测试的
// 私有/公开方法，infer 本身只负责把它们按顺序串起来并测量总耗时。
ErrorCode OnnxClassifier::infer(const cv::Mat& image, InferenceResult& result) {
    std::vector<float> input_tensor;
    ErrorCode status = prepare_input(image, input_tensor);
    if (status != ErrorCode::success) {
        return status;
    }

    auto start_time = std::chrono::steady_clock::now();

    std::vector<float> raw_output;
    status = run_session(input_tensor, raw_output);
    if (status != ErrorCode::success) {
        return status;
    }

    status = map_output(raw_output, result);
    if (status != ErrorCode::success) {
        return status;
    }

    auto end_time = std::chrono::steady_clock::now();
    result.inference_time_us =
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time).count());

    return ErrorCode::success;
}

void OnnxClassifier::shutdown() {
    session_.reset();
    input_names_.clear();
    output_names_.clear();
    input_name_storage_.clear();
    output_name_storage_.clear();
}

// prepare_input —— 把一张灰度 CWT 图变成模型需要的归一化浮点张量。
// 移植自原始项目 AIModelFramework::infer 里图像分支的处理逻辑
// （灰度化 -> resize 到 224x224 -> 按训练时的均值/标准差归一化），
// 去掉了原始代码里逐步打印调试信息的部分。
// 这个方法不依赖任何已加载的会话（session_ 可以是空的），所以可以在
// initialize() 之前单独调用，方便写单元测试。
ErrorCode OnnxClassifier::prepare_input(const cv::Mat& image,
                                        std::vector<float>& tensor) const {
    if (image.empty()) {
        return ErrorCode::invalid_argument;
    }

    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }

    if (gray.rows != model_input_size || gray.cols != model_input_size) {
        cv::resize(gray, gray, cv::Size(model_input_size, model_input_size));
    }

    // 归一化：与训练一致 (pixel - 33.405) / 78.795。
    // 先转成 float 再做减法/除法，避免整数运算的截断误差。
    cv::Mat float_image;
    gray.convertTo(float_image, CV_32F);
    float_image = (float_image - pixel_normalization_mean) / pixel_normalization_std;

    const int height = float_image.rows;
    const int width = float_image.cols;
    tensor.resize(static_cast<std::size_t>(height) * static_cast<std::size_t>(width));
    float* tensor_data = tensor.data();
    for (int row = 0; row < height; ++row) {
        const float* row_data = float_image.ptr<float>(row);
        for (int col = 0; col < width; ++col) {
            tensor_data[row * width + col] = row_data[col];
        }
    }

    return ErrorCode::success;
}

// map_output —— 在模型的 4 个原始输出分数里找最大值（argmax），
// 把索引、分数、类别名写入 InferenceResult。
// 移植自原始项目 AIModelFramework::postprocess_output 的类别选择逻辑，
// 类别名换成本项目固定的中文四分类名称，去掉了打印调试信息的部分。
// 同样不依赖已加载的会话，可以在 initialize() 之前单独调用。
ErrorCode OnnxClassifier::map_output(const std::vector<float>& raw_output,
                                     InferenceResult& result) const {
    if (raw_output.size() != expected_class_count) {
        return ErrorCode::invalid_argument;
    }

    std::size_t max_index = 0;
    float max_value = raw_output[0];
    for (std::size_t index = 1; index < raw_output.size(); ++index) {
        if (raw_output[index] > max_value) {
            max_value = raw_output[index];
            max_index = index;
        }
    }

    result.class_id = static_cast<int>(max_index);
    result.confidence = max_value;
    result.class_name = bearing_class_names()[max_index];

    return ErrorCode::success;
}

// run_session —— 用已加载的 ONNX Runtime 会话做一次前向推理。
// 移植自原始项目 AIModelFramework::ONNXModel::infer。
//
// 关键点：不同的模型导出/量化方式，期望的输入元素类型不一样
// （float32 原样输入；uint8/int8 量化模型则期望整数像素值）。
// 这里必须先查询会话第 0 个输入节点实际声明的元素类型，
// 再分支构造匹配类型的 Ort::Value，否则 Run() 会因为张量类型不匹配而失败。
//
// 对量化模型分支，input 参数里的浮点数已经是 prepare_input 算出的
// 归一化值，要先按公式反归一化（pixel = value * std + mean）还原成
// 0~255 的像素域，再转换成模型期望的整数类型——这是为了让同一份
// prepare_input 输出，能够正确地适配 float / uint8 / int8 三种不同的模型。
ErrorCode OnnxClassifier::run_session(const std::vector<float>& input,
                                      std::vector<float>& output) {
    if (session_ == nullptr) {
        return ErrorCode::inference_error;
    }

    try {
        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> input_shape = {1, 1, model_input_size, model_input_size};

        // 查询模型实际期望的输入元素类型，决定下面走哪条分支。
        auto input_type_info = session_->GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        const ONNXTensorElementDataType input_type =
            input_tensor_info.GetElementType();

        Ort::Value input_tensor{nullptr};

        // 这三个分支里用到的整数缓冲区必须活到 session_->Run() 调用完成，
        // 所以在分支外面声明，分支内只负责填充内容。
        std::vector<uint8_t> uint8_input;
        std::vector<int8_t> int8_input;

        if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            // float32 模型：归一化后的浮点数据可以直接喂给模型。
            input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, const_cast<float*>(input.data()), input.size(),
                input_shape.data(), input_shape.size());

        } else if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
            // uint8 量化模型：把归一化后的值反归一化回 0~255 像素域。
            uint8_input.resize(input.size());
            for (std::size_t index = 0; index < input.size(); ++index) {
                float pixel = input[index] * pixel_normalization_std +
                             pixel_normalization_mean;
                pixel = std::max(0.0F, std::min(255.0F, pixel));
                uint8_input[index] = static_cast<uint8_t>(pixel + 0.5F);
            }
            input_tensor = Ort::Value::CreateTensor<uint8_t>(
                memory_info, uint8_input.data(), uint8_input.size(),
                input_shape.data(), input_shape.size());

        } else if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
            // int8 量化模型：先还原像素值，再平移到 int8 的 [-128,127] 区间。
            int8_input.resize(input.size());
            for (std::size_t index = 0; index < input.size(); ++index) {
                float pixel = input[index] * pixel_normalization_std +
                             pixel_normalization_mean;
                float shifted = pixel - 128.0F;
                shifted = std::max(-128.0F, std::min(127.0F, shifted));
                int8_input[index] = static_cast<int8_t>(shifted);
            }
            input_tensor = Ort::Value::CreateTensor<int8_t>(
                memory_info, int8_input.data(), int8_input.size(),
                input_shape.data(), input_shape.size());

        } else {
            return ErrorCode::inference_error;
        }

        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr}, input_names_.data(), &input_tensor, 1,
            output_names_.data(), output_names_.size());

        if (output_tensors.empty()) {
            return ErrorCode::inference_error;
        }

        // 模型输出同样可能不是 float，按声明的类型转换回 float 概率。
        auto output_tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        const ONNXTensorElementDataType output_type =
            output_tensor_info.GetElementType();
        const std::size_t output_size = output_tensor_info.GetElementCount();

        if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            const float* float_data = output_tensors[0].GetTensorData<float>();
            output.assign(float_data, float_data + output_size);

        } else if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
            const uint8_t* byte_data = output_tensors[0].GetTensorData<uint8_t>();
            output.resize(output_size);
            for (std::size_t index = 0; index < output_size; ++index) {
                output[index] = static_cast<float>(byte_data[index]) / 255.0F;
            }

        } else if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
            const int8_t* byte_data = output_tensors[0].GetTensorData<int8_t>();
            output.resize(output_size);
            for (std::size_t index = 0; index < output_size; ++index) {
                output[index] =
                    (static_cast<float>(byte_data[index]) + 128.0F) / 255.0F;
            }

        } else {
            return ErrorCode::inference_error;
        }

        return ErrorCode::success;
    } catch (const std::exception&) {
        return ErrorCode::inference_error;
    }
}

}  // namespace bearing
