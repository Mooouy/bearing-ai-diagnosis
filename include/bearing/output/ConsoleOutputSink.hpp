#pragma once

#include "bearing/output/IOutputSink.hpp"

namespace bearing {

// ConsoleOutputSink —— 用控制台打印模拟真实工业 4-20mA 电流输出的“软件输出”。
// 这里没有任何 GPIO/pigpio 代码：真正驱动硬件的输出适配器是后面 Task 9
// 才加入的可选硬件层，二者都实现同一个 IOutputSink 接口，方便互相替换。
class ConsoleOutputSink final : public IOutputSink {
public:
    ErrorCode initialize() override;
    ErrorCode publish(const InferenceResult& result,
                      float& mapped_current_ma) override;
    void standby() override;
    void shutdown() override;

    // map_current_ma —— 把“分类结果 + 置信度”换算成模拟的 4-20mA 电流值（mA）。
    // 这个方法不依赖 initialize()，可以直接在默认构造的对象上调用，
    // 方便单元测试只验证换算公式本身是否正确。
    float map_current_ma(int class_id, float confidence) const;
};

}  // namespace bearing
