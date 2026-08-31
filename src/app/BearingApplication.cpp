#include "bearing/app/BearingApplication.hpp"

#include <iostream>
#include <utility>

namespace bearing {

namespace {

// fault_current_ma —— process()/infer() 失败时使用的“安全电流值”。
// 和 Task 6 控制台输出里“class_id 超出范围”的哨兵电流保持一致
// （ConsoleOutputSink::map_current_ma 对越界 class_id 返回 5.0F），
// 这样无论是“分类结果不可信”还是“处理/推理过程本身失败”，
// 下游看到的都是同一个“电流环异常”信号，不需要记两套阈值。
constexpr float fault_current_ma = 5.0F;
constexpr std::size_t queue_capacity = 3;

}  // namespace

BearingApplication::BearingApplication(
    std::unique_ptr<IAcquisitionSource> acquisition_source,
    std::unique_ptr<IVibrationProcessor> processor,
    std::unique_ptr<IClassifier> classifier,
    std::unique_ptr<IOutputSink> output_sink,
    std::unique_ptr<ResultLogger> result_logger)
    : acquisition_source_(std::move(acquisition_source)),
      processor_(std::move(processor)),
      classifier_(std::move(classifier)),
      output_sink_(std::move(output_sink)),
      result_logger_(std::move(result_logger)),
      raw_queue_(queue_capacity),
      result_queue_(queue_capacity) {}

// 析构函数是最后一道安全网：如果调用方忘记调用 stop() 就让对象销毁，
// 三个 std::thread 在“仍然 joinable”的状态下被销毁会直接让程序
// std::terminate。这里补一次 stop()，stop() 内部对“线程已经停止/
// 已经 join 过”的情况是安全的（join 前会检查 joinable()），
// 所以即使调用方已经手动 stop() 过，这里再调用一次也不会出错。
BearingApplication::~BearingApplication() {
    stop();
}

// initialize —— 只初始化“接口本身暴露了 initialize() 方法”的组件：
// 采集源、输出端、日志记录器，按这个顺序逐一调用并在第一个失败处
// 立刻返回（不需要继续初始化后面的组件）。
//
// IClassifier 接口故意只暴露 infer()，不暴露 initialize()/shutdown()，
// 因为分类器加载模型文件是一次性的重操作，且需要具体类型
// （OnnxClassifier）才能调用 initialize()——一旦在这里把分类器
// std::move 进来，就只能通过接口指针访问，调不到 initialize()。
// 所以模型加载必须由 main() 在构造具体的 OnnxClassifier 之后、
// std::move 进本类之前完成，main() 那一步失败时甚至不会构造
// BearingApplication。VibrationProcessor 没有 initialize()，
// 因为它是无状态的纯计算类，构造完成即可使用。
ErrorCode BearingApplication::initialize() {
    ErrorCode status = acquisition_source_->initialize();
    if (status != ErrorCode::success) {
        return status;
    }

    status = output_sink_->initialize();
    if (status != ErrorCode::success) {
        return status;
    }

    status = result_logger_->initialize();
    if (status != ErrorCode::success) {
        return status;
    }

    return ErrorCode::success;
}

// start —— 启动三个线程；running_ 在启动线程之前先置为 true，
// 这样三个线程一开始运行就能立刻看到“应该运行”的状态。
ErrorCode BearingApplication::start() {
    running_.store(true);

    acquisition_thread_ = std::thread(&BearingApplication::acquisition_loop, this);
    processing_thread_ = std::thread(&BearingApplication::processing_loop, this);
    logging_thread_ = std::thread(&BearingApplication::logging_loop, this);

    return ErrorCode::success;
}

// stop —— 关闭流程遵循一个简单的对称模式：“谁生产，谁负责停掉自己的
// 输出队列”，stop() 本身只做两件事，绝不直接调用任何一个队列的 stop()：
//
// 1. 先把 running_ 置为 false：这一步只影响“采集线程的 while 循环
//    条件”，让采集线程在读完当前这一批数据后，自然地不再读下一批，
//    而不是被强行中断在读取数据的中间状态。
// 2. 依次 join 采集线程、处理线程、记录线程：按“采集 -> 处理 -> 记录”
//    的流水线顺序 join，保证每一步 join 完成时，它的下游队列已经处于
//    “不会再有新数据”的确定状态，可以安全地继续往下走。
//
// 为什么 stop() 不自己调用 raw_queue_.stop() 或 result_queue_.stop()？
// 因为“running_=false”只能保证采集线程不会再开始读下一批新数据，但
// 不能保证它“此刻”已经走到了循环末尾——它可能正卡在 acquisition_loop()
// 的 while 循环体内部，即将要执行 raw_queue_.push(batch)。如果 stop()
// 在这个窗口期里抢先把 raw_queue_ 停掉，这次 push 会因为队列已停止而
// 静默丢弃数据（ConcurrentQueue::push 的约定就是“stop 之后变成
// no-op”）——这是真实发生过的数据丢失，不是假设。同样的道理也适用于
// result_queue_ 和处理线程：处理线程可能正在做 CWT/推理（这两步都不是
// 瞬间完成的，真实负载下可能耗时几毫秒到几十毫秒），如果这时
// result_queue_ 被提前停掉，处理线程随后的 push 同样会静默丢弃这条
// 已经算好的诊断记录。
//
// 真正安全的做法是：每个生产者线程只在自己的 while 循环彻底结束、
// 确认“再也不会调用 push() 了”之后，才在循环末尾自己调用
// xxx_queue_.stop()——acquisition_loop() 在末尾停 raw_queue_，
// processing_loop() 在末尾停 result_queue_（分别见各自函数末尾的
// 注释）。这样任何一次 push() 发生时，对应队列必然还没有被停掉，
// 不会有任何数据落在“已停止”的队列上。stop() 这里只需要按顺序
// join 三个线程：join 完成本身就意味着对应的生产者已经走完了它的
// 收尾逻辑（包括停掉自己的输出队列），下一步可以放心地认为下游队列
// 不会再有新数据涌入。
//
// stop() 不是线程安全的，只应由主线程调用（main() 显式调用 +
// 析构函数兜底）；两处调用顺序执行、不并发，不存在竞态。
// “调用多次”的安全性：ConcurrentQueue::stop() 本身是幂等的；
// std::thread::join() 只在 joinable() 为真时才调用，第二次调用时
// 线程已经被 join 过、joinable() 为假，直接跳过，不会抛异常。
void BearingApplication::stop() {
    running_.store(false);

    if (acquisition_thread_.joinable()) {
        acquisition_thread_.join();
    }
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }

    // 到这里，处理线程已经完全退出，意味着它已经把所有处理完的记录
    // 都推入了 result_queue_，并且已经自己调用过 result_queue_.stop()
    // （见 processing_loop 末尾）。记录线程的循环现在可以安全地 drain
    // 完剩余记录并退出，直接 join 即可。
    if (logging_thread_.joinable()) {
        logging_thread_.join();
    }

    shutdown_adapters_once();
}

void BearingApplication::shutdown_adapters_once() {
    bool already_shutdown = adapters_shutdown_.exchange(true);
    if (already_shutdown) {
        return;
    }

    // stop() 可能被手动调用，也可能由析构函数兜底调用；硬件适配器通常
    // 会关闭 GPIO/SPI/电流输出等真实资源，所以只允许执行一次。
    if (output_sink_ != nullptr) {
        output_sink_->shutdown();
    }
    if (acquisition_source_ != nullptr) {
        acquisition_source_->shutdown();
    }
}

bool BearingApplication::is_running() const {
    return running_.load();
}

const DiagnosticCounters& BearingApplication::diagnostics() const {
    return counters_;
}

void BearingApplication::print_diagnostics() const {
    std::cout << "[诊断统计]"
              << " 采集循环=" << counters_.acquisition_loops.load()
              << " 已采集批次=" << counters_.batches_acquired.load()
              << " 已推送原始批次=" << counters_.raw_batches_pushed.load()
              << " 处理循环=" << counters_.processing_loops.load()
              << " 已接收原始批次=" << counters_.raw_batches_received.load()
              << " 已生成CWT图像=" << counters_.cwt_images_created.load()
              << " 已开始推理=" << counters_.inferences_started.load()
              << " 已完成推理=" << counters_.inferences_finished.load()
              << " 已发布输出=" << counters_.outputs_published.load()
              << " 已推送结果=" << counters_.results_pushed.load()
              << " 已写入日志=" << counters_.log_records_written.load()
              << std::endl;
}

// acquisition_loop —— 采集线程：循环读取一批振动数据并推入 raw_queue_。
// 源耗尽（非 --loop 模式下读到文件末尾）时，主动请求整体关闭：
// 把 running_ 置为 false 并跳出循环，让处理/记录线程也能跟着结束，
// 而不是只让采集线程自己退出、留下另外两个线程永远阻塞在 pop() 里。
void BearingApplication::acquisition_loop() {
    RawVibrationBatch batch;

    while (running_.load()) {
        ++counters_.acquisition_loops;

        ErrorCode status = acquisition_source_->read(batch);
        if (status == ErrorCode::success) {
            ++counters_.batches_acquired;
            if (raw_queue_.push(batch)) {
                ++counters_.raw_batches_pushed;
            }
        } else {
            // 数据源耗尽（CsvAcquisitionSource 在非循环模式下读到末尾
            // 会返回 file_read_error）：这不是程序错误，是“模拟数据
            // 播放完了”，应当让整个应用优雅地结束，而不是报错退出。
            running_.store(false);
            break;
        }
    }

    // 不管循环是被 running_=false（外部 stop()）打断，还是因为数据源
    // 耗尽自己跳出循环，都要在退出前停掉 raw_queue_：让处理线程的
    // "while (raw_queue_.pop())" 在drain完队列里剩余的数据后能够退出，
    // 不会永远阻塞在等待下一批数据上。
    raw_queue_.stop();
}

// processing_loop —— 处理线程：CWT -> 推理 -> 输出映射 -> 组装记录。
// 用 "while (raw_queue_.pop(batch))" 而不是 "while (running_)"，
// 是因为 stop() 之后队列里可能还残留几批已经采集但还没处理的数据，
// pop() 会先把这些数据全部处理完（drain），再因为队列已停止且为空
// 返回 false 退出循环，不会丢失采集线程已经成功推入的数据。
void BearingApplication::processing_loop() {
    RawVibrationBatch batch;

    while (raw_queue_.pop(batch)) {
        ++counters_.processing_loops;
        ++counters_.raw_batches_received;

        ProcessedVibration processed;
        ErrorCode status = processor_->process(batch, processed);
        if (status != ErrorCode::success) {
            handle_processing_failure(batch, "振动信号CWT处理失败");
            continue;
        }
        ++counters_.cwt_images_created;

        InferenceResult inference;
        ++counters_.inferences_started;
        status = classifier_->infer(processed.cwt_image, inference);
        if (status != ErrorCode::success) {
            handle_processing_failure(batch, "ONNX模型推理失败");
            continue;
        }
        ++counters_.inferences_finished;

        float mapped_current_ma = 0.0F;
        status = output_sink_->publish(inference, mapped_current_ma);
        if (status != ErrorCode::success) {
            handle_processing_failure(batch, "诊断结果输出发布失败");
            continue;
        }
        ++counters_.outputs_published;

        ProcessingRecord record;
        record.timestamp_ms = batch.timestamp_ms;
        record.sequence = batch.sequence;
        record.inference = inference;
        record.mapped_current_ma = mapped_current_ma;
        record.success = true;

        if (result_queue_.push(record)) {
            ++counters_.results_pushed;
        }
    }

    // 处理线程结束前停掉 result_queue_：让记录线程能够 drain 完剩余
    // 已经推入的记录后正常退出，原因和 acquisition_loop 里停
    // raw_queue_ 完全一样。
    result_queue_.stop();
}

// handle_processing_failure —— process()/infer()/publish() 任意一步
// 失败时的统一处理：不能让处理线程因为一批数据处理失败就崩溃退出
// （那会导致后续所有数据都得不到处理），而是要把输出端置于安全的
// 待机状态，记录一条“失败”的诊断结果（带中文错误说明和安全电流值），
// 然后继续处理下一批数据。
void BearingApplication::handle_processing_failure(
    const RawVibrationBatch& batch, const std::string& chinese_reason) {
    output_sink_->standby();

    ProcessingRecord record;
    record.timestamp_ms = batch.timestamp_ms;
    record.sequence = batch.sequence;
    record.success = false;
    record.error_message = chinese_reason;
    record.mapped_current_ma = fault_current_ma;

    if (result_queue_.push(record)) {
        ++counters_.results_pushed;
    }
}

// logging_loop —— 记录线程：把处理线程产出的每一条记录写入 CSV 日志。
// 同样用 "while (result_queue_.pop(record))" 保证 stop() 之后能
// drain 完剩余记录再退出，不丢失已经产出但还没落盘的诊断结果。
void BearingApplication::logging_loop() {
    ProcessingRecord record;

    while (result_queue_.pop(record)) {
        ErrorCode status = result_logger_->write(record);
        if (status == ErrorCode::success) {
            ++counters_.log_records_written;
        }
        // 写日志失败只是“这一条记录没能落盘”，不应该让记录线程退出
        // 导致后续所有记录都得不到处理，所以这里不 continue/return，
        // 循环本身已经会自动进入下一次 pop()。
    }
}

}  // namespace bearing
