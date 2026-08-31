#include "bearing/app/ApplicationOptions.hpp"

#include <sstream>

namespace bearing {

namespace {

// get_value_for_option —— 取出形如 "--simulate <csv>" 里 <csv> 这个值。
// 选项和值是两个独立的命令行参数（不是 "--simulate=xxx" 这种写法），
// 所以要检查“当前选项后面是否还有下一个参数”，没有就是用户少写了值。
bool get_value_for_option(const std::vector<std::string>& arguments,
                          std::size_t option_index,
                          std::string& value) {
    if (option_index + 1 >= arguments.size()) {
        return false;
    }
    value = arguments[option_index + 1];
    return true;
}

}  // namespace

std::string build_usage_text() {
    std::ostringstream text;
    text << "用法: bearing_vibration_system [选项]\n"
         << "\n"
         << "轴承振动诊断系统：从振动数据（CSV 模拟或真实硬件采集卡）读取数据，\n"
         << "经过 CWT 时频变换和 ONNX 模型推理后，输出 4-20mA 模拟电流并记录日志。\n"
         << "\n"
         << "选项:\n"
         << "  --simulate <csv>   使用 CSV 文件模拟振动数据采集（未指定 --hardware 时必填）\n"
         << "  --model <onnx>     ONNX 模型文件路径，默认 models/bearing_vibration_model.onnx\n"
         << "  --log-dir <目录>   诊断结果 CSV 日志的输出目录，默认 logs\n"
         << "  --loop             CSV 模拟数据读到末尾后从头循环，而不是结束程序\n"
         << "  --hardware         使用真实硬件采集/输出（需要 ENABLE_HARDWARE=ON 构建）\n"
         << "  --help             显示这段帮助信息并退出\n";
    return text.str();
}

// parse_application_options —— 逐个扫描命令行参数，识别已知选项并写入
// options；遇到 --help、未知选项、缺值、或缺少必填的 --simulate，
// 都通过 ParseResult.should_exit 通知调用方“不要继续启动应用”。
ParseResult parse_application_options(const std::vector<std::string>& arguments,
                                      ApplicationOptions& options) {
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        const std::string& argument = arguments[index];

        if (argument == "--help") {
            return ParseResult{true, 0, build_usage_text()};
        }

        if (argument == "--simulate") {
            std::string value;
            if (!get_value_for_option(arguments, index, value)) {
                return ParseResult{
                    true, 1,
                    "错误: --simulate 缺少 CSV 文件路径参数\n\n" +
                        build_usage_text()};
            }
            options.simulation_path = value;
            ++index;
            continue;
        }

        if (argument == "--model") {
            std::string value;
            if (!get_value_for_option(arguments, index, value)) {
                return ParseResult{
                    true, 1,
                    "错误: --model 缺少 ONNX 模型文件路径参数\n\n" +
                        build_usage_text()};
            }
            options.model_path = value;
            ++index;
            continue;
        }

        if (argument == "--log-dir") {
            std::string value;
            if (!get_value_for_option(arguments, index, value)) {
                return ParseResult{
                    true, 1,
                    "错误: --log-dir 缺少日志目录参数\n\n" +
                        build_usage_text()};
            }
            options.log_directory = value;
            ++index;
            continue;
        }

        if (argument == "--loop") {
            options.loop_simulation = true;
            continue;
        }

        if (argument == "--hardware") {
            options.use_hardware = true;
            continue;
        }

        return ParseResult{
            true, 1,
            "错误: 未知选项 \"" + argument + "\"\n\n" + build_usage_text()};
    }

    // 没有 --hardware 时，必须提供 --simulate，否则程序不知道从哪里
    // 读取振动数据。这个检查放在扫描完所有参数之后，因为用户可能把
    // --hardware 写在 --simulate 前面或后面，顺序不影响最终判断。
    if (!options.use_hardware && options.simulation_path.empty()) {
        return ParseResult{
            true, 1,
            "错误: 未指定 --hardware 时必须提供 --simulate <csv>\n\n" +
                build_usage_text()};
    }

    return ParseResult{false, 0, std::string()};
}

}  // namespace bearing
