#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace bearing {

struct RawVibrationBatch {
    std::uint64_t timestamp_ms = 0;
    std::uint64_t sequence = 0;
    float sample_rate_hz = 50000.0F;
    std::vector<float> x_axis;
    std::vector<float> y_axis;
    std::vector<float> z_axis;
};

struct ProcessedVibration {
    cv::Mat cwt_image;
};

struct InferenceResult {
    int class_id = -1;
    std::string class_name;
    float confidence = 0.0F;
    std::uint64_t inference_time_us = 0;
};

struct ProcessingRecord {
    std::uint64_t timestamp_ms = 0;
    std::uint64_t sequence = 0;
    InferenceResult inference;
    float mapped_current_ma = 4.0F;
    bool success = false;
    std::string error_message;
};

}  // namespace bearing
