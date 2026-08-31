#pragma once

#include <string>
#include <vector>

namespace bearing {

// ApplicationOptions —— 解析命令行参数后得到的“配置数据”。
// 这个结构体本身不做任何解析逻辑，只是携带数据，解析逻辑放在
// parse_application_options() 这个独立函数里，方便单独测试/复用。
struct ApplicationOptions {
    std::string simulation_path;
    std::string model_path = "models/bearing_vibration_model.onnx";
    std::string log_directory = "logs";
    bool loop_simulation = false;
    bool use_hardware = false;
};

// ParseResult —— parse_application_options() 的返回值。
// 用一个独立的小结构体携带“是否要立刻退出（--help 或出错）”、
// “退出码”、“给用户看的提示信息”，比单独用 ErrorCode 更清楚：
// --help 不是错误，但同样需要让 main() 打印信息后提前退出。
struct ParseResult {
    bool should_exit = false;
    int exit_code = 0;
    std::string message;
};

// parse_application_options —— 把 argv 解析成 ApplicationOptions。
// 这是一个独立于 main() 的小函数：main() 只负责把 argc/argv 转发进来，
// 不在 main() 里直接写 for 循环和字符串比较，方便单独编写/测试解析逻辑。
//
// 行为约定：
// - 不带 --hardware 时，--simulate <csv> 是必填项；
// - --help 会让 should_exit=true、exit_code=0，message 里是完整帮助文本；
// - 未知选项、或选项缺少必需的值、或缺少必填的 --simulate，
//   都会让 should_exit=true、exit_code 非零，message 里是中文错误提示。
ParseResult parse_application_options(const std::vector<std::string>& arguments,
                                      ApplicationOptions& options);

// build_usage_text —— 生成 --help 用的完整帮助文本，单独提取成函数，
// 方便“正常显示帮助”和“参数错误时附带提示”两处复用同一份文本。
std::string build_usage_text();

}  // namespace bearing
