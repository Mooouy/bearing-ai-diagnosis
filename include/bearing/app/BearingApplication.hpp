#pragma once

#include "bearing/acquisition/IAcquisitionSource.hpp"
#include "bearing/core/ConcurrentQueue.hpp"
#include "bearing/core/DiagnosticCounters.hpp"
#include "bearing/core/ErrorCode.hpp"
#include "bearing/core/Types.hpp"
#include "bearing/inference/IClassifier.hpp"
#include "bearing/logging/ResultLogger.hpp"
#include "bearing/output/IOutputSink.hpp"
#include "bearing/processing/IVibrationProcessor.hpp"

#include <atomic>
#include <memory>
#include <string>
#include <thread>

namespace bearing {

// BearingApplication —— 把“采集 -> 处理(CWT+推理+输出) -> 记录日志”
// 三个阶段分别放到三个线程里跑，线程之间用 ConcurrentQueue 传递数据。
// 这个类只负责“组装 + 线程编排”，不实现任何 CWT/ONNX/CSV/GPIO 的具体
// 算法——那些都在构造时以接口指针的形式注入进来，方便用假实现单独
// 测试线程编排和关闭流程是否正确（参见 tests/test_application_shutdown.cpp）。
//
// 三个线程通过两个队列串联：
//   采集线程 --raw_queue_--> 处理线程 --result_queue_--> 记录线程
class BearingApplication {
public:
    BearingApplication(
        std::unique_ptr<IAcquisitionSource> acquisition_source,
        std::unique_ptr<IVibrationProcessor> processor,
        std::unique_ptr<IClassifier> classifier,
        std::unique_ptr<IOutputSink> output_sink,
        std::unique_ptr<ResultLogger> result_logger);

    // 三个线程会持有 this 指针，对象不应该被拷贝或移动，
    // 否则线程里保存的 this 会指向一个已经失效的对象。
    BearingApplication(const BearingApplication&) = delete;
    BearingApplication& operator=(const BearingApplication&) = delete;
    BearingApplication(BearingApplication&&) = delete;
    BearingApplication& operator=(BearingApplication&&) = delete;

    ~BearingApplication();

    // initialize —— 在启动线程之前，初始化“接口里本来就暴露了
    // initialize() 方法”的组件：采集源、输出端、日志记录器。
    // IClassifier 接口只暴露 infer()（不暴露 initialize()），
    // 所以分类器的模型加载必须由调用方（main()）在构造具体的
    // OnnxClassifier、并 std::move 进本类之前完成，详见 main.cpp。
    ErrorCode initialize();

    // start —— 启动采集、处理、记录三个线程；必须在 initialize() 成功之后调用。
    ErrorCode start();

    // stop —— 请求三个线程结束并等待它们退出。可以从任意线程调用，
    // 也可以安全地调用多次（ConcurrentQueue::stop() 是幂等的）。
    void stop();

    bool is_running() const;

    const DiagnosticCounters& diagnostics() const;

    // print_diagnostics —— 把诊断计数器打印成一行人类可读的中文摘要，
    // 给 main.cpp 的“每 5 秒打印一次”循环使用。
    void print_diagnostics() const;

private:
    void acquisition_loop();
    void processing_loop();
    void logging_loop();
    void shutdown_adapters_once();

    // handle_processing_failure —— processing_loop() 里
    // process()/infer()/publish() 任意一步失败时的统一处理逻辑，
    // 提取成一个私有方法是为了避免在多个失败分支里重复编写
    // “置输出端待机 + 组装失败记录 + 推入结果队列”这几行代码。
    void handle_processing_failure(const RawVibrationBatch& batch,
                                   const std::string& chinese_reason);

    std::unique_ptr<IAcquisitionSource> acquisition_source_;
    std::unique_ptr<IVibrationProcessor> processor_;
    std::unique_ptr<IClassifier> classifier_;
    std::unique_ptr<IOutputSink> output_sink_;
    std::unique_ptr<ResultLogger> result_logger_;

    ConcurrentQueue<RawVibrationBatch> raw_queue_;
    ConcurrentQueue<ProcessingRecord> result_queue_;

    std::thread acquisition_thread_;
    std::thread processing_thread_;
    std::thread logging_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> adapters_shutdown_{false};
    DiagnosticCounters counters_;
};

}  // namespace bearing
