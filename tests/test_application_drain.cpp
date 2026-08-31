// test_application_drain —— 回归测试：专门覆盖
// test_application_shutdown.cpp 没有覆盖到的场景：“外部 stop() 和一个
// 仍在主动循环的采集线程发生竞态”。
//
// test_application_shutdown 里的 FakeAcquisitionSource 只读 2 批数据就
// 自己耗尽并自行停止（不依赖外部 stop()），所以它永远测不到“stop() 调用
// 时采集线程还在 while 循环里跑”这条路径——而这恰恰是本文件要验证的、
// 曾经真实出现过数据丢失的那条路径：如果 stop() 在采集线程还活着的时候
// 抢先 stop() 掉 raw_queue_，采集线程随后的一次 push() 会静默丢弃数据，
// 但计数器还是会增加，造成“诊断数字和真实丢失的数据不一致”的假象。
//
// 这里用一个“永不耗尽”的采集源：每次 read() 都成功返回，让采集线程
// 在测试主线程调用 stop() 的那一刻，必然还在 while 循环里持续运行，
// 从而真正触发上面说的竞态窗口。
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
#include <thread>

namespace {

// ContinuousFakeAcquisitionSource —— read() 永远成功、永不耗尽，
// 保证采集线程会一直跑在 while 循环里，直到外部调用 stop() 把
// running_ 置为 false。这正是 test_application_shutdown 里那个
// “读 2 批就自己耗尽”的 FakeAcquisitionSource 没有覆盖的情形。
class ContinuousFakeAcquisitionSource final : public bearing::IAcquisitionSource {
public:
    bearing::ErrorCode initialize() override {
        return bearing::ErrorCode::success;
    }

    bearing::ErrorCode read(bearing::RawVibrationBatch& batch) override {
        batch.sequence = next_sequence_;
        batch.timestamp_ms = next_sequence_ + 100;
        batch.x_axis = {1.0F, 2.0F, 3.0F};
        batch.y_axis = {4.0F, 5.0F, 6.0F};
        batch.z_axis = {7.0F, 8.0F, 9.0F};
        ++next_sequence_;
        return bearing::ErrorCode::success;
    }

    void shutdown() override {
    }

    std::string name() const override {
        return "continuous-fake-acquisition";
    }

private:
    std::uint64_t next_sequence_ = 0;
};

// 下面三个假实现和 test_application_shutdown.cpp 里的等价，但本文件
// 不依赖那个文件（那是 brief 给定的 verbatim 文件，不应该被修改或被
// 其它文件引用），所以这里在本文件的匿名命名空间里重新定义一份，
// 保持两个测试文件互相独立。

class FakeProcessor final : public bearing::IVibrationProcessor {
public:
    bearing::ErrorCode process(
        const bearing::RawVibrationBatch&,
        bearing::ProcessedVibration& output) const override {
        // 模拟真实 CWT/ONNX 链路不是“瞬时完成”的事实：处理端稍慢，
        // 才能暴露无界采集队列在外部 stop() 时积压过多的问题。
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
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
    bearing::ErrorCode initialize() override {
        return bearing::ErrorCode::success;
    }

    bearing::ErrorCode publish(
        const bearing::InferenceResult&,
        float& mapped_current_ma) override {
        mapped_current_ma = 10.85F;
        return bearing::ErrorCode::success;
    }

    void standby() override {
    }

    void shutdown() override {
    }
};

}  // namespace

int main() {
    std::filesystem::path log_directory =
        std::filesystem::temp_directory_path() /
        "bearing_application_drain_test";
    std::filesystem::remove_all(log_directory);

    auto acquisition = std::make_unique<ContinuousFakeAcquisitionSource>();
    auto processor = std::make_unique<FakeProcessor>();
    auto classifier = std::make_unique<FakeClassifier>();
    auto output = std::make_unique<FakeOutputSink>();
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

    // 有界等待，直到流水线已经真正产出过至少 5 条结果——确保 stop()
    // 被调用时，采集线程必然还在循环里继续读取/推送下一批数据
    // （不会因为还没开始跑而“恰好”不触发竞态窗口）。假实现都是瞬时
    // 完成的（没有真实 CWT/ONNX 那种几毫秒到几十毫秒的耗时），所以
    // 5 条结果会在很短时间内产出，2 秒上限只是防止真的卡死时测试
    // 无限挂起。
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (application.diagnostics().results_pushed.load() < 5 &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (application.diagnostics().results_pushed.load() < 5) {
        std::cerr << "pipeline did not reach steady state before stop()"
                  << std::endl;
        return EXIT_FAILURE;
    }

    // 用 std::async + 超时包一层 stop()：如果这次回归改动重新引入了
    // 死锁（例如某个队列在不该停的时候被停掉，导致某个线程永远卡在
    // pop() 里），这里能在 2 秒内发现并报错，而不是让测试无限挂起。
    auto stop_future = std::async(std::launch::async, [&application]() {
        application.stop();
    });

    if (stop_future.wait_for(std::chrono::seconds(2)) !=
        std::future_status::ready) {
        std::cerr << "application stop timed out" << std::endl;
        return EXIT_FAILURE;
    }

    stop_future.get();

    bool all_invariants_hold = true;

    if (application.is_running()) {
        std::cerr << "application still reports running" << std::endl;
        all_invariants_hold = false;
    }

    const bearing::DiagnosticCounters& counters = application.diagnostics();

    if (counters.results_pushed.load() == 0) {
        std::cerr << "processing thread produced no result" << std::endl;
        all_invariants_hold = false;
    }

    // drain 不变量 1：推送过的原始批次数必须等于处理线程收到的批次数。
    // 如果 stop() 在采集线程仍在运行时提前 stop() 了 raw_queue_，
    // 采集线程后续的 push() 会静默丢弃数据但计数器仍会增加，这条
    // 不变量就会被打破（pushed > received）。两者相等才能证明
    // “每一次成功 push 的原始批次，都被处理线程实际收到了”，
    // 没有任何一批数据在队列里被静默丢弃。
    const std::uint64_t raw_pushed = counters.raw_batches_pushed.load();
    const std::uint64_t raw_received = counters.raw_batches_received.load();
    if (raw_pushed != raw_received) {
        std::cerr << "drain invariant violated: raw_batches_pushed="
                  << raw_pushed
                  << " != raw_batches_received=" << raw_received
                  << std::endl;
        all_invariants_hold = false;
    }

    // drain 不变量 2：处理线程推送过的结果数必须等于记录线程实际写入
    // 日志的条数。如果 result_queue_ 在处理线程仍在运行时被提前停掉，
    // 处理线程产出的记录会被静默丢弃，记录线程也等不到这些记录，
    // 这条不变量就会被打破（pushed > written）。两者相等才能证明
    // “每一条算出来的诊断结果，最终都落盘了”，没有任何一条记录
    // 在内存里产生之后又凭空消失。
    const std::uint64_t results_pushed = counters.results_pushed.load();
    const std::uint64_t log_written = counters.log_records_written.load();
    if (results_pushed != log_written) {
        std::cerr << "drain invariant violated: results_pushed="
                  << results_pushed
                  << " != log_records_written=" << log_written
                  << std::endl;
        all_invariants_hold = false;
    }

    std::filesystem::remove_all(log_directory);

    if (!all_invariants_hold) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
