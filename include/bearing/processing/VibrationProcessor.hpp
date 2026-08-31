#pragma once

#include "bearing/core/ErrorCode.hpp"
#include "bearing/core/Types.hpp"
#include "bearing/processing/IVibrationProcessor.hpp"

// 测试代码（tests/test_vibration_processor.cpp）只包含本头文件，
// 却需要使用 cv::minMaxLoc 检查输出图像的像素值范围，而 cv::minMaxLoc
// 声明在 opencv2/core.hpp 而不是 opencv2/core/mat.hpp 里，
// 所以这里直接包含完整的 opencv2/core.hpp，让调用方无需单独 include。
#include <opencv2/core.hpp>

#include <vector>

namespace bearing {

// VibrationProcessor 实现 IVibrationProcessor 接口：process() 的签名
// 在 Task 4 设计时就已经和接口要求的签名一致，这里只是补上基类声明，
// 让 BearingApplication 可以用接口指针持有它，方便测试时换成假实现。
class VibrationProcessor final : public IVibrationProcessor {
public:
    ErrorCode process(const RawVibrationBatch& batch,
                      ProcessedVibration& processed) const override;

private:
    ErrorCode validate_batch(const RawVibrationBatch& batch) const;
    ErrorCode compute_cwt_image(const std::vector<float>& signal,
                                cv::Mat& image) const;
    cv::Mat enhance_axis_image(const cv::Mat& image) const;
    cv::Mat fuse_axis_images(const cv::Mat& x_image,
                             const cv::Mat& y_image,
                             const cv::Mat& z_image) const;
    cv::Mat finish_fused_image(const cv::Mat& image) const;
};

}  // namespace bearing
