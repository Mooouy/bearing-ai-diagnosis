#include "bearing/processing/VibrationProcessor.hpp"

#include <opencv2/imgproc.hpp>

#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <utility>

namespace bearing {

namespace {

// ============================================================================
// CWT（连续小波变换）相关的命名常量
//
// 这些常量在算法来源（多项目框架的 DataProcessingFramework）中保存在一个
// 通用配置对象 config_ 里。本独立项目没有那个配置对象，所以把每个数值
// 抽成一个有名字的常量，数值原样保留（来源已在注释中标注）。
// ============================================================================

// 小波尺度（scale）的总数量。对应来源 config_.signal.cwt_scales = 128。
// 尺度越多，时频图在频率方向上的分辨率越细。
constexpr int cwt_scale_count = 128;

// Morlet 小波的中心频率 fc。对应来源 config_.signal.cwt_center_freq = 1.0。
constexpr double wavelet_center_frequency = 1.0;

// Morlet 小波的带宽 B（cmor1.5-1 中的 1.5）。
// 对应来源 config_.signal.bandwidth = 1.5。
// 带宽影响小波在时间和频率方向的分辨率折衷。
constexpr double wavelet_bandwidth = 1.5;

// 积分小波 int_psi 的采样点数 N_psi。
// 来源经过实测对齐 Python pywt.integrate_wavelet('cmor1.5-1', precision=10)
// 的真实输出长度（不是 2^10+1=1025，而是 1024），数值不可更改。
constexpr int integrated_wavelet_sample_count = 1024;

// 积分小波在时间轴上覆盖的范围，对应来源里的 [-8, 8] 区间，总宽度 16。
constexpr double integrated_wavelet_time_span = 16.0;

// 预加重滤波（pre-emphasis）系数 alpha，用于突出高频振动分量。
constexpr double pre_emphasis_alpha = 0.97;

// ONNX 分类模型期望的方形输入边长（像素）。
constexpr int model_input_size = 224;

// 单轴 CWT 图在 resize 之后做 CLAHE 局部对比度增强时使用的参数。
constexpr double cwt_image_clahe_clip_limit = 3.0;
constexpr int cwt_image_clahe_tile_size = 8;

// 三轴融合前的增强（CLAHE + Unsharp Masking）参数。
constexpr double pre_fusion_clahe_clip_limit = 3.0;
constexpr int pre_fusion_clahe_tile_size = 8;
constexpr double pre_fusion_unsharp_sigma = 1.0;
constexpr double pre_fusion_unsharp_strength = 1.5;

// 三轴融合之后的整体增强（CLAHE + Unsharp Masking）参数，强度比融合前的
// 增强稍弱一些，避免把融合带来的噪声也一起放大。
constexpr double post_fusion_clahe_clip_limit = 4.0;
constexpr int post_fusion_clahe_tile_size = 8;
constexpr double post_fusion_unsharp_sigma = 0.8;
constexpr double post_fusion_unsharp_strength = 1.2;

// 互相关加权融合（CCWF）时，把图像切成多大的小块分别计算权重。
constexpr int fusion_block_size = 8;

// ============================================================================
// next_fast_len：返回 >= n 的最小的 2^a * 3^b * 5^c * 7^d * 11^e
//
// 输入：n —— 需要的最小长度（例如一次卷积所需的样本数）。
// 输出：>= n 的最小“11-光滑数”（11-smooth number）。
// 为什么需要它：FFT 在这类“光滑”的长度下计算效率最高，并且这个函数的行为
// 必须与 Python 侧 scipy.fft.next_fast_len 完全一致，否则后续的 FFT 卷积
// 结果会和参考实现出现细微但会累积的偏差。
// ============================================================================
int next_fast_len(int n) {
    if (n <= 1) {
        return 1;
    }

    // 先用 2 的幂作为一个安全的上界，后面在 [n, upper_bound] 里找更小的光滑数
    int power_of_two_bound = 1;
    while (power_of_two_bound < n) {
        power_of_two_bound <<= 1;
    }
    int best_length = power_of_two_bound;

    // 枚举所有形如 2^a * 3^b * 5^c * 7^d * 11^e 的数，找到其中 >= n 的最小值
    for (long long factor_11 = 1; factor_11 <= static_cast<long long>(best_length);
         factor_11 *= 11) {
        for (long long factor_7_11 = factor_11;
             factor_7_11 <= static_cast<long long>(best_length); factor_7_11 *= 7) {
            for (long long factor_5_7_11 = factor_7_11;
                 factor_5_7_11 <= static_cast<long long>(best_length);
                 factor_5_7_11 *= 5) {
                for (long long factor_3_5_7_11 = factor_5_7_11;
                     factor_3_5_7_11 <= static_cast<long long>(best_length);
                     factor_3_5_7_11 *= 3) {
                    long long candidate = factor_3_5_7_11;
                    while (candidate < static_cast<long long>(n)) {
                        candidate <<= 1;
                    }
                    if (candidate <= static_cast<long long>(best_length) &&
                        static_cast<int>(candidate) < best_length) {
                        best_length = static_cast<int>(candidate);
                    }
                }
            }
        }
    }
    return best_length;
}

// float32 归一化互相关（normalized cross-correlation）。
// 输入：两个形状相同的 CV_32F 图像块。
// 输出：-1.0 到 1.0 之间的相关系数，用来衡量两个图像块的相似程度。
// 为什么需要它：互相关加权融合（CCWF）用这个相似度给每一路图像分配权重，
// 越和别人相似的块，说明信号越稳定，权重就越高。
float normalized_cross_correlation(const cv::Mat& patch_a, const cv::Mat& patch_b) {
    const int pixel_count = patch_a.rows * patch_a.cols;
    if (pixel_count == 0) {
        return 0.0F;
    }

    float sum_a = 0.0F;
    float sum_b = 0.0F;
    for (int row = 0; row < patch_a.rows; ++row) {
        const float* row_a = patch_a.ptr<float>(row);
        const float* row_b = patch_b.ptr<float>(row);
        for (int col = 0; col < patch_a.cols; ++col) {
            sum_a += row_a[col];
            sum_b += row_b[col];
        }
    }
    const float mean_a = sum_a / static_cast<float>(pixel_count);
    const float mean_b = sum_b / static_cast<float>(pixel_count);

    float covariance = 0.0F;
    float variance_a = 0.0F;
    float variance_b = 0.0F;
    for (int row = 0; row < patch_a.rows; ++row) {
        const float* row_a = patch_a.ptr<float>(row);
        const float* row_b = patch_b.ptr<float>(row);
        for (int col = 0; col < patch_a.cols; ++col) {
            const float deviation_a = row_a[col] - mean_a;
            const float deviation_b = row_b[col] - mean_b;
            covariance += deviation_a * deviation_b;
            variance_a += deviation_a * deviation_a;
            variance_b += deviation_b * deviation_b;
        }
    }

    const float denominator = std::sqrt(variance_a * variance_b);
    if (denominator < 1e-6F) {
        return 0.0F;
    }
    return covariance / denominator;
}

}  // namespace

ErrorCode VibrationProcessor::process(const RawVibrationBatch& batch,
                                       ProcessedVibration& processed) const {
    ErrorCode status = validate_batch(batch);
    if (status != ErrorCode::success) {
        return status;
    }

    cv::Mat x_image;
    cv::Mat y_image;
    cv::Mat z_image;

    status = compute_cwt_image(batch.x_axis, x_image);
    if (status != ErrorCode::success) {
        return status;
    }

    status = compute_cwt_image(batch.y_axis, y_image);
    if (status != ErrorCode::success) {
        return status;
    }

    status = compute_cwt_image(batch.z_axis, z_image);
    if (status != ErrorCode::success) {
        return status;
    }

    x_image = enhance_axis_image(x_image);
    y_image = enhance_axis_image(y_image);
    z_image = enhance_axis_image(z_image);

    cv::Mat fused_image = fuse_axis_images(x_image, y_image, z_image);
    processed.cwt_image = finish_fused_image(fused_image);

    if (processed.cwt_image.empty()) {
        return ErrorCode::processing_error;
    }

    return ErrorCode::success;
}

// 检查一个批次的三轴振动数据是否“形状合理”：三轴都非空，且长度一致。
// 这是后续所有 CWT 计算的前提条件，提前在这里挡掉坏数据，
// 避免后面的数组下标计算因为长度不一致而越界。
ErrorCode VibrationProcessor::validate_batch(const RawVibrationBatch& batch) const {
    if (batch.x_axis.empty() || batch.y_axis.empty() || batch.z_axis.empty()) {
        return ErrorCode::invalid_argument;
    }

    const bool lengths_match = batch.x_axis.size() == batch.y_axis.size() &&
                               batch.y_axis.size() == batch.z_axis.size();
    if (!lengths_match) {
        return ErrorCode::invalid_argument;
    }

    return ErrorCode::success;
}

// 融合前增强：CLAHE（局部自适应直方图均衡）+ Unsharp Masking（反锐化遮罩）。
// 输入：单轴 CWT 灰度图（CV_8UC1）。
// 输出：增强对比度、突出细节之后的同尺寸灰度图。
// 为什么需要它：三轴各自的 CWT 图在直接融合前先分别提升对比度，
// 这样融合阶段的互相关权重才能更准确地反映三路信号的真实相似程度。
cv::Mat VibrationProcessor::enhance_axis_image(const cv::Mat& image) const {
    CV_Assert(image.type() == CV_8U);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(
        pre_fusion_clahe_clip_limit,
        cv::Size(pre_fusion_clahe_tile_size, pre_fusion_clahe_tile_size));
    cv::Mat enhanced;
    clahe->apply(image, enhanced);

    // Unsharp Masking：先做高斯模糊，再用“原图 - 模糊图”的差异放大细节。
    // 高斯核大小由 sigma 推出，必须是奇数。
    int kernel_size = static_cast<int>(6 * pre_fusion_unsharp_sigma + 1);
    if (kernel_size % 2 == 0) {
        kernel_size += 1;
    }
    cv::Mat blurred;
    cv::GaussianBlur(enhanced, blurred, cv::Size(kernel_size, kernel_size),
                     pre_fusion_unsharp_sigma);

    // 提升到 CV_32F 再做加权叠加，避免 8 位整数运算的截断误差累积。
    cv::Mat enhanced_f32;
    cv::Mat blurred_f32;
    cv::Mat sharpened_f32;
    enhanced.convertTo(enhanced_f32, CV_32F);
    blurred.convertTo(blurred_f32, CV_32F);
    cv::addWeighted(enhanced_f32, 1.0 + pre_fusion_unsharp_strength, blurred_f32,
                    -pre_fusion_unsharp_strength, 0.0, sharpened_f32);

    cv::Mat result;
    sharpened_f32.convertTo(result, CV_8U);  // 四舍五入并截断到 [0,255]
    return result;
}

// 互相关加权融合（Cross-Correlation Weighted Fusion，CCWF）。
// 输入：三轴各自增强后的图像（已经转换为 CV_32F，便于直接累加）。
// 输出：融合后的单张 CV_8UC1 灰度图。
// 为什么需要它：三轴振动反映同一个故障，但各轴信噪比不同；
// 把图像切成小块，块内更“一致”的轴获得更高权重，
// 这样融合结果比简单平均更能突出真实的故障特征。
cv::Mat VibrationProcessor::fuse_axis_images(const cv::Mat& x_image,
                                             const cv::Mat& y_image,
                                             const cv::Mat& z_image) const {
    std::vector<cv::Mat> axis_images_f32(3);
    x_image.convertTo(axis_images_f32[0], CV_32F);
    y_image.convertTo(axis_images_f32[1], CV_32F);
    z_image.convertTo(axis_images_f32[2], CV_32F);

    const int axis_count = static_cast<int>(axis_images_f32.size());
    cv::Mat fused = cv::Mat::zeros(model_input_size, model_input_size, CV_32F);

    for (int block_top = 0; block_top < model_input_size; block_top += fusion_block_size) {
        const int block_bottom =
            std::min(block_top + fusion_block_size, model_input_size);
        const int block_height = block_bottom - block_top;

        for (int block_left = 0; block_left < model_input_size;
             block_left += fusion_block_size) {
            const int block_right =
                std::min(block_left + fusion_block_size, model_input_size);
            const int block_width = block_right - block_left;
            const cv::Rect block_region(block_left, block_top, block_width, block_height);

            // 对每一路轴，计算它和其它轴在这个小块里的平均相关程度，
            // 相关程度越高，说明这一路在这个区域越“可信”，权重越大。
            std::vector<float> axis_weights(axis_count, 0.5F);
            for (int axis_index = 0; axis_index < axis_count; ++axis_index) {
                float correlation_sum = 0.0F;
                int compared_axis_count = 0;
                for (int other_axis_index = 0; other_axis_index < axis_count;
                     ++other_axis_index) {
                    if (axis_index == other_axis_index) {
                        continue;
                    }
                    cv::Mat patch_a = axis_images_f32[axis_index](block_region).clone();
                    cv::Mat patch_b =
                        axis_images_f32[other_axis_index](block_region).clone();
                    const float correlation = normalized_cross_correlation(patch_a, patch_b);
                    // 把相关系数从 [-1,1] 映射到 [0,1]，方便和别的权重相加平均
                    correlation_sum += (correlation + 1.0F) * 0.5F;
                    ++compared_axis_count;
                }
                axis_weights[axis_index] =
                    (compared_axis_count > 0)
                        ? (correlation_sum / static_cast<float>(compared_axis_count))
                        : 0.5F;
            }

            float weight_sum = 0.0F;
            for (float weight : axis_weights) {
                weight_sum += weight;
            }
            if (weight_sum < 1e-6F) {
                weight_sum = 1.0F;
            }

            cv::Mat block_fused = cv::Mat::zeros(block_height, block_width, CV_32F);
            for (int axis_index = 0; axis_index < axis_count; ++axis_index) {
                block_fused += axis_weights[axis_index] * axis_images_f32[axis_index](block_region);
            }
            block_fused /= weight_sum;
            block_fused.copyTo(fused(block_region));
        }
    }

    cv::Mat normalized_to_byte_range;
    cv::Mat fused_u8;
    cv::normalize(fused, normalized_to_byte_range, 0, 255, cv::NORM_MINMAX);
    normalized_to_byte_range.convertTo(fused_u8, CV_8U);
    return fused_u8;
}

// 融合后处理：和 enhance_axis_image 同样的 CLAHE + Unsharp Masking 流程，
// 但强度参数稍弱，用于在融合完成之后再整体提升一次对比度和清晰度。
// 输入：融合后的灰度图（CV_8UC1）。
// 输出：最终交给分类模型使用的灰度图（CV_8UC1，同尺寸）。
cv::Mat VibrationProcessor::finish_fused_image(const cv::Mat& image) const {
    CV_Assert(image.type() == CV_8U);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(
        post_fusion_clahe_clip_limit,
        cv::Size(post_fusion_clahe_tile_size, post_fusion_clahe_tile_size));
    cv::Mat enhanced;
    clahe->apply(image, enhanced);

    int kernel_size = static_cast<int>(6 * post_fusion_unsharp_sigma + 1);
    if (kernel_size % 2 == 0) {
        kernel_size += 1;
    }
    cv::Mat blurred;
    cv::GaussianBlur(enhanced, blurred, cv::Size(kernel_size, kernel_size),
                     post_fusion_unsharp_sigma);

    cv::Mat enhanced_f32;
    cv::Mat blurred_f32;
    cv::Mat sharpened_f32;
    enhanced.convertTo(enhanced_f32, CV_32F);
    blurred.convertTo(blurred_f32, CV_32F);
    cv::addWeighted(enhanced_f32, 1.0 + post_fusion_unsharp_strength, blurred_f32,
                    -post_fusion_unsharp_strength, 0.0, sharpened_f32);

    cv::Mat result;
    sharpened_f32.convertTo(result, CV_8U);
    return result;
}

// ============================================================================
// CWT 数学实现：以下私有方法把 compute_cwt_image 拆分成若干个有名字的阶段。
// 数学公式本身保持和来源完全一致，只是把长函数拆开、把魔法数字换成命名常量，
// 方便阅读：先建尺度 → 建积分小波 → 逐尺度做卷积 → 截取系数 → 归一化为图像。
// ============================================================================
namespace {

// 一组尺度对应的小波卷积长度，逐尺度变化，所以打包成一个小结构体，
// 比起单独传 4 个参数更清楚地表达“这是同一个尺度算出来的一组数”。
struct ScaleConvolutionPlan {
    double scale = 0.0;        // 当前尺度值
    double sqrt_scale = 0.0;   // 尺度的平方根，归一化系数里要用到
    int wavelet_sample_count = 0;  // 这个尺度下小波本身占用的采样点数（j_len）
    int fft_length = 0;        // 这个尺度做卷积时使用的 FFT 长度（next_fast_len 的结果）
};

// 第一步：build_scales —— 生成 totalscal 个小波尺度。
// 输入：（无，使用命名常量 cwt_scale_count / wavelet_center_frequency）
// 输出：长度为 cwt_scale_count 的尺度数组，从大到小排列。
// 为什么需要它：CWT 需要在多个尺度上分别做卷积，尺度越大对应越低的频率，
// 这一步先把所有尺度值算出来，后面逐尺度复用。
std::vector<double> build_scales() {
    const double scale_constant =
        2.0 * wavelet_center_frequency * static_cast<double>(cwt_scale_count);

    std::vector<double> scales(cwt_scale_count);
    for (int scale_index = 0; scale_index < cwt_scale_count; ++scale_index) {
        scales[scale_index] =
            scale_constant / static_cast<double>(cwt_scale_count - scale_index);
    }
    return scales;
}

// 第二步：build_integrated_wavelet —— 生成积分小波 int_psi 的实部和虚部。
// 输入：（无，使用命名常量 integrated_wavelet_sample_count /
//        integrated_wavelet_time_span / wavelet_bandwidth /
//        wavelet_center_frequency）
// 输出：两个长度为 integrated_wavelet_sample_count 的数组，分别是
//      int_psi 的实部和虚部。
// 为什么需要它：CWT 用的是 Morlet 小波“积分”之后的版本（int_psi），
// 这一步只需要算一次，后面每个尺度都复用同一份积分小波，按尺度重新取样即可。
void build_integrated_wavelet(std::vector<double>& real_part,
                              std::vector<double>& imaginary_part) {
    const int sample_count = integrated_wavelet_sample_count;
    const double step =
        integrated_wavelet_time_span / static_cast<double>(sample_count - 1);
    const double normalization = 1.0 / std::sqrt(M_PI * wavelet_bandwidth);

    real_part.assign(sample_count, 0.0);
    imaginary_part.assign(sample_count, 0.0);

    double cumulative_real = 0.0;
    double cumulative_imaginary = 0.0;
    for (int sample_index = 0; sample_index < sample_count; ++sample_index) {
        // t 从积分小波时间窗的左端 -8 开始，按 step 递增到右端 +8
        const double t = -integrated_wavelet_time_span / 2.0 + sample_index * step;
        const double envelope = std::exp(-t * t / wavelet_bandwidth);

        // 对 Morlet 小波本身（envelope * 复指数）做累积求和，近似积分
        cumulative_real +=
            normalization * envelope * std::cos(2.0 * M_PI * wavelet_center_frequency * t) * step;
        cumulative_imaginary +=
            normalization * envelope * std::sin(2.0 * M_PI * wavelet_center_frequency * t) * step;

        real_part[sample_index] = cumulative_real;
        // 取共轭，所以虚部要取反
        imaginary_part[sample_index] = -cumulative_imaginary;
    }
}

// 对一段信号做一次复数 FFT（正向）。
// 输入：长度为 fft_length 的实数信号 padded_signal（不足的部分应已经补零）。
// 输出：FFT 结果的实部和虚部，长度同样是 fft_length。
// 为什么单独抽出来：同一个长度的信号 FFT 只需要算一次，
// 不同尺度如果用到相同的 fft_length 就可以直接复用结果（见 convolve_scale 的调用方）。
void run_forward_fft(const std::vector<double>& padded_signal,
                     std::vector<double>& fft_real,
                     std::vector<double>& fft_imaginary) {
    const int fft_length = static_cast<int>(padded_signal.size());

    fftw_complex* input_buffer =
        static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * fft_length));
    fftw_complex* output_buffer =
        static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * fft_length));

    for (int sample_index = 0; sample_index < fft_length; ++sample_index) {
        input_buffer[sample_index][0] = padded_signal[sample_index];
        input_buffer[sample_index][1] = 0.0;
    }

    fftw_plan plan = fftw_plan_dft_1d(fft_length, input_buffer, output_buffer,
                                      FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    fftw_free(input_buffer);

    fft_real.assign(fft_length, 0.0);
    fft_imaginary.assign(fft_length, 0.0);
    for (int sample_index = 0; sample_index < fft_length; ++sample_index) {
        fft_real[sample_index] = output_buffer[sample_index][0];
        fft_imaginary[sample_index] = output_buffer[sample_index][1];
    }
    fftw_free(output_buffer);
}

// 第三步：convolve_scale —— 在“一个”尺度上，把信号和（重新取样后的）小波做卷积。
// 输入：预加重之后的信号 emphasized_signal，当前尺度 scale，
//      以及信号 FFT 的缓存 signal_fft_cache（同样的 fft_length 不重复计算）。
// 输出：长度为 fft_length 的卷积结果实部、虚部，以及这次用到的 wavelet_sample_count
//      （通过 ScaleConvolutionPlan 一并返回，extract_centered_coefficients 还要用到）。
// 为什么需要它：CWT 在每个尺度上都是“信号卷积一个按尺度缩放的小波”，
// 这一步用频域乘法（FFT 卷积定理）实现卷积，比直接时域卷积快得多。
ScaleConvolutionPlan convolve_scale(
    const std::vector<double>& emphasized_signal,
    double scale,
    const std::vector<double>& wavelet_real,
    const std::vector<double>& wavelet_imaginary,
    std::map<int, std::pair<std::vector<double>, std::vector<double>>>& signal_fft_cache,
    std::vector<double>& convolution_real,
    std::vector<double>& convolution_imaginary) {
    const int signal_length = static_cast<int>(emphasized_signal.size());

    ScaleConvolutionPlan plan;
    plan.scale = scale;
    plan.sqrt_scale = std::sqrt(scale);

    // j_len：这个尺度下，小波本身占用的采样点数。
    // sd 是小波在“信号采样间隔”下覆盖的宽度，j_len 取它的整数部分再加 1。
    const double wavelet_domain_width = scale * integrated_wavelet_time_span;
    plan.wavelet_sample_count = static_cast<int>(std::floor(wavelet_domain_width)) + 1;

    // fft_length：选一个对 FFT 计算友好的长度，必须 >= 信号长度 + 小波长度 - 1，
    // 否则线性卷积的结果会发生“环绕”而出错。
    plan.fft_length =
        next_fast_len(signal_length + plan.wavelet_sample_count - 1);

    // 信号的 FFT 只取决于 fft_length，缓存起来避免不同尺度重复计算同一个长度
    auto cache_iterator = signal_fft_cache.find(plan.fft_length);
    if (cache_iterator == signal_fft_cache.end()) {
        std::vector<double> padded_signal(plan.fft_length, 0.0);
        for (int sample_index = 0; sample_index < signal_length; ++sample_index) {
            padded_signal[sample_index] = emphasized_signal[sample_index];
        }
        std::vector<double> signal_fft_real;
        std::vector<double> signal_fft_imaginary;
        run_forward_fft(padded_signal, signal_fft_real, signal_fft_imaginary);
        cache_iterator =
            signal_fft_cache
                .emplace(plan.fft_length,
                         std::make_pair(std::move(signal_fft_real),
                                       std::move(signal_fft_imaginary)))
                .first;
    }
    const std::vector<double>& signal_fft_real = cache_iterator->second.first;
    const std::vector<double>& signal_fft_imaginary = cache_iterator->second.second;

    // 把积分小波按当前尺度重新取样、并反转顺序，得到这个尺度专用的小波系数。
    // sd 决定了取样的“跨度”，wavelet_sample_count 之后的位置直接补零。
    fftw_complex* scaled_wavelet = static_cast<fftw_complex*>(
        fftw_malloc(sizeof(fftw_complex) * plan.fft_length));
    for (int position = 0; position < plan.fft_length; ++position) {
        if (position >= plan.wavelet_sample_count) {
            scaled_wavelet[position][0] = 0.0;
            scaled_wavelet[position][1] = 0.0;
            continue;
        }
        // 反转索引：小波数组要倒着读
        const int forward_index = plan.wavelet_sample_count - 1 - position;
        int wavelet_sample_index = 0;
        if (plan.wavelet_sample_count > 1) {
            wavelet_sample_index = static_cast<int>(
                static_cast<double>(forward_index) / wavelet_domain_width *
                (integrated_wavelet_sample_count - 1));
        }
        wavelet_sample_index =
            std::min(wavelet_sample_index, integrated_wavelet_sample_count - 1);
        scaled_wavelet[position][0] = wavelet_real[wavelet_sample_index];
        scaled_wavelet[position][1] = wavelet_imaginary[wavelet_sample_index];
    }

    // 小波的 FFT
    fftw_complex* wavelet_fft = static_cast<fftw_complex*>(
        fftw_malloc(sizeof(fftw_complex) * plan.fft_length));
    fftw_plan wavelet_plan = fftw_plan_dft_1d(plan.fft_length, scaled_wavelet, wavelet_fft,
                                              FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(wavelet_plan);
    fftw_destroy_plan(wavelet_plan);
    fftw_free(scaled_wavelet);

    // 频域乘法：两个频谱的复数乘法，等价于时域上的卷积
    fftw_complex* product = static_cast<fftw_complex*>(
        fftw_malloc(sizeof(fftw_complex) * plan.fft_length));
    for (int frequency_index = 0; frequency_index < plan.fft_length; ++frequency_index) {
        const double signal_real = signal_fft_real[frequency_index];
        const double signal_imaginary = signal_fft_imaginary[frequency_index];
        const double wavelet_real_part = wavelet_fft[frequency_index][0];
        const double wavelet_imaginary_part = wavelet_fft[frequency_index][1];
        product[frequency_index][0] =
            signal_real * wavelet_real_part - signal_imaginary * wavelet_imaginary_part;
        product[frequency_index][1] =
            signal_real * wavelet_imaginary_part + signal_imaginary * wavelet_real_part;
    }
    fftw_free(wavelet_fft);

    // 反向 FFT（IFFT）得到卷积结果
    fftw_complex* convolution_buffer = static_cast<fftw_complex*>(
        fftw_malloc(sizeof(fftw_complex) * plan.fft_length));
    fftw_plan inverse_plan = fftw_plan_dft_1d(plan.fft_length, product, convolution_buffer,
                                              FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(inverse_plan);
    fftw_destroy_plan(inverse_plan);
    fftw_free(product);

    convolution_real.assign(plan.fft_length, 0.0);
    convolution_imaginary.assign(plan.fft_length, 0.0);
    for (int sample_index = 0; sample_index < plan.fft_length; ++sample_index) {
        convolution_real[sample_index] = convolution_buffer[sample_index][0];
        convolution_imaginary[sample_index] = convolution_buffer[sample_index][1];
    }
    fftw_free(convolution_buffer);

    return plan;
}

// 第四步：extract_centered_coefficients —— 从卷积结果里截取出居中的一段，
// 得到这个尺度上、和原始信号等长的 CWT 系数，写入 coefficients 矩阵的一行。
// 输入：卷积结果（实部/虚部）、convolve_scale 算出的 plan、信号长度、
//      该尺度在系数矩阵中的行号 scale_index。
// 输出：无返回值，直接写入 coefficients 矩阵的第 scale_index 行。
// 为什么需要它：FFT 卷积的结果比原始信号长（长度是 fft_length），
// 真正有意义的“CWT 系数”只是中间一段（按 d = (j_len-1)/2 居中对齐），
// 这一步把这段提取出来，并按照 pywt 的约定做差分、归一化、取绝对值。
void extract_centered_coefficients(const std::vector<double>& convolution_real,
                                   const ScaleConvolutionPlan& plan,
                                   int signal_length,
                                   int scale_index,
                                   cv::Mat& coefficients) {
    // 居中偏移量：小波长度的一半（整数除法，向下取整）
    const int center_offset = (plan.wavelet_sample_count - 1) / 2;
    const double normalization_factor = static_cast<double>(plan.fft_length);

    // 防御性检查：理论上 fft_length 足够大，不会越界，但仍保留保护
    if (center_offset + signal_length >= plan.fft_length) {
        return;
    }

    float* coefficient_row = coefficients.ptr<float>(scale_index);
    for (int sample_index = 0; sample_index < signal_length; ++sample_index) {
        const int left_index = center_offset + sample_index;
        const int right_index = center_offset + sample_index + 1;

        // 对卷积结果做一阶差分并按 FFT 长度归一化（对应小波变换里的导数近似）
        const double real_difference =
            (convolution_real[right_index] - convolution_real[left_index]) /
            normalization_factor;

        // 按 -sqrt(scale) 缩放，截断到 float32 精度，再取绝对值
        // （原始信号是实数，只保留实部，这是 pywt 对实数输入的标准行为）
        const float scaled_coefficient =
            static_cast<float>(-plan.sqrt_scale * real_difference);
        coefficient_row[sample_index] = std::fabs(scaled_coefficient);
    }
}

// 第五步：normalize_cwt_matrix —— 对整张系数矩阵做 log1p 压缩动态范围。
// 输入/输出：coefficients 矩阵（原地修改）。
// 为什么需要它：CWT 系数的数值跨度可能很大（小信号和大信号差几个量级），
// 直接归一化会让小信号的细节被压扁，先用 log1p 压缩一下动态范围，
// 转成图像后细节会更清楚。
void normalize_cwt_matrix(cv::Mat& coefficients) {
    for (int row = 0; row < coefficients.rows; ++row) {
        float* coefficient_row = coefficients.ptr<float>(row);
        for (int col = 0; col < coefficients.cols; ++col) {
            coefficient_row[col] = std::log1p(coefficient_row[col]);
        }
    }
}

// 第六步：resize_to_model_input —— 把系数矩阵缩放成模型输入需要的方形图像，
// 并归一化到 [0,255]、再做一次 CLAHE 局部对比度增强。
// 输入：log1p 压缩之后的系数矩阵（cwt_scale_count 行 × 信号长度列）。
// 输出：model_input_size × model_input_size 的 CV_8UC1 灰度图。
// 为什么需要它：系数矩阵的形状是“尺度数 × 信号长度”，和模型期望的
// 224×224 灰度图尺寸不一样，这一步统一把它们变成模型可以直接使用的图像。
cv::Mat resize_to_model_input(const cv::Mat& coefficients) {
    cv::Mat resized;
    cv::resize(coefficients, resized, cv::Size(model_input_size, model_input_size), 0, 0,
              cv::INTER_LINEAR);

    cv::Mat normalized;
    cv::Mat normalized_u8;
    cv::normalize(resized, normalized, 0, 255, cv::NORM_MINMAX);
    normalized.convertTo(normalized_u8, CV_8U);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(
        cwt_image_clahe_clip_limit,
        cv::Size(cwt_image_clahe_tile_size, cwt_image_clahe_tile_size));
    cv::Mat result;
    clahe->apply(normalized_u8, result);
    return result;
}

}  // namespace

// compute_cwt_image —— 把单轴振动信号变成一张 224×224 的 CWT 灰度图。
// 输入：单轴振动信号 signal（一段时间序列）。
// 输出：写入 image，CV_8UC1，224×224。
// 整体流程：预加重 → build_scales → build_integrated_wavelet →
//          逐尺度 convolve_scale + extract_centered_coefficients →
//          normalize_cwt_matrix → resize_to_model_input。
ErrorCode VibrationProcessor::compute_cwt_image(const std::vector<float>& signal,
                                                cv::Mat& image) const {
    const int signal_length = static_cast<int>(signal.size());
    if (signal_length <= 0) {
        return ErrorCode::invalid_argument;
    }

    // ① 预加重滤波（pre-emphasis，alpha=0.97）：突出高频振动分量，
    //   抑制低频漂移。这里按 float32 截断一次，是为了和参考实现保持
    //   完全一致的浮点行为（先以 double 计算，再截断成 float，再转回 double）。
    std::vector<double> emphasized_signal(signal_length);
    emphasized_signal[0] = static_cast<double>(signal[0]);
    for (int sample_index = 1; sample_index < signal_length; ++sample_index) {
        const float emphasized_sample = static_cast<float>(
            static_cast<double>(signal[sample_index]) -
            pre_emphasis_alpha * static_cast<double>(signal[sample_index - 1]));
        emphasized_signal[sample_index] = static_cast<double>(emphasized_sample);
    }

    const std::vector<double> scales = build_scales();

    std::vector<double> wavelet_real;
    std::vector<double> wavelet_imaginary;
    build_integrated_wavelet(wavelet_real, wavelet_imaginary);

    // CWT 系数矩阵：cwt_scale_count 行（每个尺度一行）× signal_length 列
    cv::Mat coefficients(cwt_scale_count, signal_length, CV_32F, cv::Scalar(0.0F));

    // 不同尺度可能用到同样的 FFT 长度，缓存信号的 FFT 结果以避免重复计算
    std::map<int, std::pair<std::vector<double>, std::vector<double>>> signal_fft_cache;

    for (int scale_index = 0; scale_index < cwt_scale_count; ++scale_index) {
        const double scale = scales[scale_index];

        std::vector<double> convolution_real;
        std::vector<double> convolution_imaginary;
        const ScaleConvolutionPlan plan =
            convolve_scale(emphasized_signal, scale, wavelet_real, wavelet_imaginary,
                          signal_fft_cache, convolution_real, convolution_imaginary);

        extract_centered_coefficients(convolution_real, plan, signal_length, scale_index,
                                      coefficients);
    }

    normalize_cwt_matrix(coefficients);
    image = resize_to_model_input(coefficients);

    if (image.empty()) {
        return ErrorCode::processing_error;
    }
    return ErrorCode::success;
}

}  // namespace bearing
