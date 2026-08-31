#include "bearing/hardware/OledStatusDisplay.hpp"

#include <u8g2.h>

#include <iostream>
#include <sstream>

namespace bearing {

namespace {

std::string format_status_line(const InferenceResult& result,
                               float current_ma) {
    std::ostringstream stream;
    stream << "ID " << result.class_id
           << " Conf " << result.confidence
           << " I " << current_ma << "mA";
    return stream.str();
}

}  // namespace

OledStatusDisplay::~OledStatusDisplay() {
    shutdown();
}

bool OledStatusDisplay::initialize() {
    // u8g2 的具体总线回调与 OLED 模块型号有关，真实板卡验收时需要按
    // 连接方式补充 setup 回调。这里先把 OLED 作为可选观察者隔离出来：
    // 初始化失败不影响采集、CWT、ONNX 推理和工业输出主链路。
    initialized_ = true;
    std::cout << "[OLED] 状态显示模块已启用（硬件总线需在 Pi 上验证）"
              << std::endl;
    return true;
}

void OledStatusDisplay::render(const InferenceResult& result,
                               float current_ma) {
    if (!initialized_) {
        return;
    }

    std::cout << "[OLED] " << result.class_name
              << " " << format_status_line(result, current_ma)
              << std::endl;
}

void OledStatusDisplay::shutdown() {
    if (!initialized_) {
        return;
    }

    initialized_ = false;
    std::cout << "[OLED] 状态显示模块已关闭" << std::endl;
}

}  // namespace bearing
