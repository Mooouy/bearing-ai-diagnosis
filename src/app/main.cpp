// main.cpp —— 命令行入口，只做“编排”：解析参数、注册信号、选择适配器、
// 组装 BearingApplication、启动、定期打印诊断、收到信号或应用自己结束
// 时停止、返回退出码。这里不出现任何 CWT/ONNX/CSV 解析/GPIO/日志格式
// 的具体实现，那些都在各自的组件类里。
#include "bearing/acquisition/CsvAcquisitionSource.hpp"
#include "bearing/app/ApplicationOptions.hpp"
#include "bearing/app/BearingApplication.hpp"
#include "bearing/inference/OnnxClassifier.hpp"
#include "bearing/logging/ResultLogger.hpp"
#include "bearing/output/ConsoleOutputSink.hpp"
#include "bearing/processing/VibrationProcessor.hpp"

#if defined(BEARING_ENABLE_HARDWARE)
#include "bearing/hardware/Ad4134AcquisitionSource.hpp"
#include "bearing/hardware/IndustrialOutputSink.hpp"
#endif

#include <csignal>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

// g_stop_requested —— 信号处理函数（signal handler）能安全做的事情非常
// 有限：只能读写“异步信号安全”（async-signal-safe）的简单类型，不能
// 调用 std::cout、不能 new/delete、不能加锁。volatile + std::sig_atomic_t
// 正是为这种场景设计的最小安全单元，这里只设置这一个文件级标志，
// 真正“看到标志后做什么”的逻辑全部放在 main() 的主循环里完成。
volatile std::sig_atomic_t g_stop_requested = 0;

void handle_stop_signal(int /*signal_number*/) {
    g_stop_requested = 1;
}

// register_stop_signals —— 把 SIGINT（Ctrl+C）和 SIGTERM（比如
// systemd/kill 发出的终止请求）都接到同一个处理函数上，让用户用
// 这两种常见方式停止程序时，都能走到 BearingApplication::stop()
// 的优雅关闭流程，而不是被直接杀死、丢失尚未落盘的诊断记录。
void register_stop_signals() {
    std::signal(SIGINT, handle_stop_signal);
    std::signal(SIGTERM, handle_stop_signal);
}

// acquisition_batch_size —— 每一批振动数据的采样点数量。
// 取值 5000：采样率固定 50000Hz（见 Types.hpp 的 sample_rate_hz 默认值），
// 5000 个采样点对应 0.1 秒的时间窗口，和 Task 4 CWT 处理、以及项目里
// “取 CSV 前 5000 行做回归对照”所用的窗口大小一致。这里不开放成命令行
// 选项（brief 给定的 ApplicationOptions 里没有这一项），固定为命名常量，
// 比裸数字 5000 更易读、也方便整个项目统一引用同一个值。
constexpr std::size_t acquisition_batch_size = 5000;

// build_acquisition_source —— 根据命令行选项构造采集源。
// 硬件分支用 #if 在编译期整体裁剪掉：ENABLE_HARDWARE=OFF 编译出的
// 二进制里完全不包含任何硬件相关代码路径，请求 --hardware 时只能走
// #else 分支的“清晰报错”，不会假装支持硬件却实际什么都没做。
std::unique_ptr<bearing::IAcquisitionSource> build_acquisition_source(
    const bearing::ApplicationOptions& options) {
#if defined(BEARING_ENABLE_HARDWARE)
    if (options.use_hardware) {
        return std::make_unique<bearing::Ad4134AcquisitionSource>(
            acquisition_batch_size);
    }
#endif
    return std::make_unique<bearing::CsvAcquisitionSource>(
        options.simulation_path, acquisition_batch_size,
        options.loop_simulation);
}

// build_output_sink —— 和 build_acquisition_source 同样的编译期裁剪思路：
// 没有启用硬件支持时，--hardware 已经在调用本函数之前就被拒绝
// （见 main() 里的检查），所以这里始终构造控制台输出。
std::unique_ptr<bearing::IOutputSink> build_output_sink(
    const bearing::ApplicationOptions& options) {
#if defined(BEARING_ENABLE_HARDWARE)
    if (options.use_hardware) {
        return std::make_unique<bearing::IndustrialOutputSink>();
    }
#endif
    (void)options;
    return std::make_unique<bearing::ConsoleOutputSink>();
}

// reject_hardware_if_unsupported —— 本次编译没有打开 ENABLE_HARDWARE 时，
// --hardware 必须立刻给出清晰的中文报错并让程序以非零状态退出，
// 而不是静默地退化成模拟模式（用户明确要硬件却悄悄换成了模拟数据，
// 比直接报错更容易让人误判系统真的在用硬件运行）。
bool reject_hardware_if_unsupported(const bearing::ApplicationOptions& options) {
#if defined(BEARING_ENABLE_HARDWARE)
    (void)options;
    return false;
#else
    if (options.use_hardware) {
        std::cerr << "错误: 本次编译未启用硬件支持（ENABLE_HARDWARE=OFF），"
                     "无法使用 --hardware。\n"
                  << "请使用 --simulate <csv> 以模拟模式运行，"
                     "或重新用 -DENABLE_HARDWARE=ON 编译。"
                  << std::endl;
        return true;
    }
    return false;
#endif
}

// run_diagnostics_loop —— 应用运行期间的主循环：每 5 秒打印一次诊断
// 摘要，同时频繁（约 100ms 一次）检查“应用是否还在运行”和“是否收到
// 了停止信号”，让 Ctrl+C 之后能在 100ms 量级内被发现并进入停止流程，
// 而不是要等到下一个 5 秒打印点才反应过来。
void run_diagnostics_loop(bearing::BearingApplication& application) {
    const auto print_interval = std::chrono::seconds(5);
    const auto poll_interval = std::chrono::milliseconds(100);
    auto next_print_time = std::chrono::steady_clock::now() + print_interval;

    while (application.is_running() && g_stop_requested == 0) {
        std::this_thread::sleep_for(poll_interval);

        if (std::chrono::steady_clock::now() >= next_print_time) {
            application.print_diagnostics();
            next_print_time = std::chrono::steady_clock::now() + print_interval;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> arguments(argv + 1, argv + argc);

    bearing::ApplicationOptions options;
    bearing::ParseResult parse_result =
        bearing::parse_application_options(arguments, options);
    if (parse_result.should_exit) {
        if (parse_result.exit_code == 0) {
            std::cout << parse_result.message;
        } else {
            std::cerr << parse_result.message;
        }
        return parse_result.exit_code == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    if (reject_hardware_if_unsupported(options)) {
        return EXIT_FAILURE;
    }

    register_stop_signals();

    auto acquisition_source = build_acquisition_source(options);
    auto output_sink = build_output_sink(options);

    if (acquisition_source == nullptr || output_sink == nullptr) {
        std::cerr << "错误: 采集源或输出端构造失败，无法启动应用。"
                  << std::endl;
        return EXIT_FAILURE;
    }

    auto processor = std::make_unique<bearing::VibrationProcessor>();
    auto result_logger = std::make_unique<bearing::ResultLogger>(options.log_directory);

    // 分类器的模型加载（initialize()）必须在这里、用具体类型
    // OnnxClassifier 完成：IClassifier 接口故意不暴露 initialize()，
    // 一旦 std::move 进 BearingApplication 就只能通过接口指针访问，
    // 调不到 initialize() 了。详见 BearingApplication::initialize()
    // 上的注释。
    auto classifier = std::make_unique<bearing::OnnxClassifier>(options.model_path);
    if (classifier->initialize() != bearing::ErrorCode::success) {
        std::cerr << "错误: ONNX 模型加载失败，路径=" << options.model_path
                  << std::endl;
        return EXIT_FAILURE;
    }

    bearing::BearingApplication application(
        std::move(acquisition_source),
        std::move(processor),
        std::move(classifier),
        std::move(output_sink),
        std::move(result_logger));

    if (application.initialize() != bearing::ErrorCode::success) {
        std::cerr << "错误: 应用初始化失败（采集源/输出端/日志记录器之一无法初始化）"
                  << std::endl;
        return EXIT_FAILURE;
    }

    if (application.start() != bearing::ErrorCode::success) {
        std::cerr << "错误: 应用启动失败" << std::endl;
        return EXIT_FAILURE;
    }

    run_diagnostics_loop(application);

    application.stop();

    std::cout << "[退出] 应用已停止，最终诊断统计如下：" << std::endl;
    application.print_diagnostics();

    return EXIT_SUCCESS;
}
