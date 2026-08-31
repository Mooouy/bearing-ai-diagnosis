#include "bearing/app/BearingApplication.hpp"

#include "bearing/acquisition/IAcquisitionSource.hpp"
#include "bearing/inference/IClassifier.hpp"
#include "bearing/logging/ResultLogger.hpp"
#include "bearing/output/IOutputSink.hpp"
#include "bearing/processing/IVibrationProcessor.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <iostream>
#include <memory>
#include <utility>
#include <thread>

namespace {

class FakeAcquisitionSource final : public bearing::IAcquisitionSource {
public:
    explicit FakeAcquisitionSource(int* shutdown_count)
        : shutdown_count_(shutdown_count) {}

    bearing::ErrorCode initialize() override {
        return bearing::ErrorCode::success;
    }

    bearing::ErrorCode read(bearing::RawVibrationBatch& batch) override {
        if (next_sequence_ >= 2) {
            return bearing::ErrorCode::file_read_error;
        }

        batch.sequence = next_sequence_;
        batch.timestamp_ms = next_sequence_ + 100;
        batch.x_axis = {1.0F, 2.0F, 3.0F};
        batch.y_axis = {4.0F, 5.0F, 6.0F};
        batch.z_axis = {7.0F, 8.0F, 9.0F};
        ++next_sequence_;
        return bearing::ErrorCode::success;
    }

    void shutdown() override {
        ++(*shutdown_count_);
    }

    std::string name() const override {
        return "fake-acquisition";
    }

private:
    int* shutdown_count_;
    std::uint64_t next_sequence_ = 0;
};

class FakeProcessor final : public bearing::IVibrationProcessor {
public:
    bearing::ErrorCode process(
        const bearing::RawVibrationBatch&,
        bearing::ProcessedVibration& output) const override {
        output.cwt_image = cv::Mat(
            224,
            224,
            CV_8UC1,
            cv::Scalar(128));
        return bearing::ErrorCode::success;
    }
};

class FakeClassifier final : public bearing::IClassifier {
public:
    bearing::ErrorCode infer(
        const cv::Mat&,
        bearing::InferenceResult& result) override {
        result.class_id = 0;
        result.class_name = "正常";
        result.confidence = 0.95F;
        result.inference_time_us = 100;
        return bearing::ErrorCode::success;
    }
};

class FakeOutputSink final : public bearing::IOutputSink {
public:
    explicit FakeOutputSink(int* shutdown_count)
        : shutdown_count_(shutdown_count) {}

    bearing::ErrorCode initialize() override {
        return bearing::ErrorCode::success;
    }

    bearing::ErrorCode publish(
        const bearing::InferenceResult&,
        float& mapped_current_ma) override {
        mapped_current_ma = 10.85F;
        ++publish_count_;
        return bearing::ErrorCode::success;
    }

    void standby() override {
    }

    void shutdown() override {
        ++(*shutdown_count_);
    }

    int publish_count() const {
        return publish_count_;
    }

private:
    int* shutdown_count_;
    int publish_count_ = 0;
};

}  // namespace

int main() {
    std::filesystem::path log_directory =
        std::filesystem::temp_directory_path() /
        "bearing_application_shutdown_test";
    std::filesystem::remove_all(log_directory);

    int acquisition_shutdown_count = 0;
    int output_shutdown_count = 0;

    auto acquisition = std::make_unique<FakeAcquisitionSource>(
        &acquisition_shutdown_count);
    auto processor = std::make_unique<FakeProcessor>();
    auto classifier = std::make_unique<FakeClassifier>();
    auto output = std::make_unique<FakeOutputSink>(&output_shutdown_count);
    auto logger = std::make_unique<bearing::ResultLogger>(
        log_directory.string());

    bearing::BearingApplication application(
        std::move(acquisition),
        std::move(processor),
        std::move(classifier),
        std::move(output),
        std::move(logger));

    if (application.initialize() != bearing::ErrorCode::success) {
        std::cerr << "application initialization failed" << std::endl;
        return EXIT_FAILURE;
    }

    if (application.start() != bearing::ErrorCode::success) {
        std::cerr << "application start failed" << std::endl;
        return EXIT_FAILURE;
    }

    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (application.diagnostics().results_pushed.load() == 0 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    auto stop_future = std::async(std::launch::async, [&application]() {
        application.stop();
    });

    if (stop_future.wait_for(std::chrono::seconds(2)) !=
        std::future_status::ready) {
        std::cerr << "application stop timed out" << std::endl;
        return EXIT_FAILURE;
    }

    stop_future.get();

    application.stop();

    if (application.is_running()) {
        std::cerr << "application still reports running" << std::endl;
        return EXIT_FAILURE;
    }

    if (application.diagnostics().results_pushed.load() == 0) {
        std::cerr << "processing thread produced no result" << std::endl;
        return EXIT_FAILURE;
    }

    if (acquisition_shutdown_count != 1) {
        std::cerr << "acquisition shutdown count should be 1, actual="
                  << acquisition_shutdown_count << std::endl;
        return EXIT_FAILURE;
    }

    if (output_shutdown_count != 1) {
        std::cerr << "output shutdown count should be 1, actual="
                  << output_shutdown_count << std::endl;
        return EXIT_FAILURE;
    }

    std::filesystem::remove_all(log_directory);
    return EXIT_SUCCESS;
}
