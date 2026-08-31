#include "bearing/processing/VibrationProcessor.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

int main() {
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
    bearing::ErrorCode status = processor.process(batch, processed);

    if (status != bearing::ErrorCode::success) {
        std::cerr << "CWT processing failed" << std::endl;
        return EXIT_FAILURE;
    }

    if (processed.cwt_image.rows != 224 ||
        processed.cwt_image.cols != 224 ||
        processed.cwt_image.type() != CV_8UC1) {
        std::cerr << "unexpected CWT image shape or type" << std::endl;
        return EXIT_FAILURE;
    }

    double minimum = 0.0;
    double maximum = 0.0;
    cv::minMaxLoc(processed.cwt_image, &minimum, &maximum);
    if (maximum <= minimum) {
        std::cerr << "CWT image contains no useful range" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
