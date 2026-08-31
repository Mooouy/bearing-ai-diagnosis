#pragma once

#include "bearing/core/Types.hpp"

#include <string>

namespace bearing {

// OledStatusDisplay —— 独立的状态观察者，只负责把最近一次轴承诊断结果
// 显示到 OLED。推理主链路不依赖它成功，OLED 故障不应阻塞诊断。
class OledStatusDisplay {
public:
    OledStatusDisplay() = default;
    ~OledStatusDisplay();

    bool initialize();
    void render(const InferenceResult& result, float current_ma);
    void shutdown();

private:
    bool initialized_ = false;
};

}  // namespace bearing
