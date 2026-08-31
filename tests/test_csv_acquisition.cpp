#include "bearing/acquisition/CsvAcquisitionSource.hpp"

#include <cstdlib>
#include <iostream>

int main() {
    bearing::CsvAcquisitionSource source(
        TEST_FIXTURE_DIR "/sample_vibration.csv",
        3,
        false);

    if (source.initialize() != bearing::ErrorCode::success) {
        std::cerr << "fixture initialization failed" << std::endl;
        return EXIT_FAILURE;
    }

    bearing::RawVibrationBatch first_batch;
    if (source.read(first_batch) != bearing::ErrorCode::success) {
        std::cerr << "first batch read failed" << std::endl;
        return EXIT_FAILURE;
    }

    if (first_batch.x_axis != std::vector<float>({0.10F, 0.20F, 0.30F})) {
        std::cerr << "X axis mapping is incorrect" << std::endl;
        return EXIT_FAILURE;
    }

    if (first_batch.y_axis.front() != 1.10F ||
        first_batch.z_axis.back() != 2.30F) {
        std::cerr << "Y or Z axis mapping is incorrect" << std::endl;
        return EXIT_FAILURE;
    }

    bearing::RawVibrationBatch second_batch;
    if (source.read(second_batch) != bearing::ErrorCode::success) {
        std::cerr << "second batch read failed" << std::endl;
        return EXIT_FAILURE;
    }

    bearing::RawVibrationBatch exhausted_batch;
    if (source.read(exhausted_batch) == bearing::ErrorCode::success) {
        std::cerr << "non-looping source should stop at end of file" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
