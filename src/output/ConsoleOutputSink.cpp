#include "bearing/output/ConsoleOutputSink.hpp"

#include <algorithm>
#include <iostream>

namespace bearing {

// initialize —— 控制台输出不需要打开任何设备或文件，直接返回成功即可。
// 这个方法存在的意义是满足 IOutputSink 接口，让后面 Task 9 的硬件输出
// 适配器可以在同一个调用点上替换成真正的初始化逻辑（比如打开 pigpio）。
ErrorCode ConsoleOutputSink::initialize() {
    return ErrorCode::success;
}

// publish —— 把一次推理结果换算成模拟电流，并打印一行人类可读的诊断信息。
// 这里只做“计算 + 打印”，不写文件、不碰硬件：写 CSV 是 ResultLogger 的职责，
// 驱动真实 4-20mA 输出是 Task 9 硬件层的职责，三者职责分开方便单独测试。
ErrorCode ConsoleOutputSink::publish(const InferenceResult& result,
                                     float& mapped_current_ma) {
    mapped_current_ma = map_current_ma(result.class_id, result.confidence);

    // result.class_name 在 Task 5 的分类器里已经是中文类别名
    // （例如“正常”“滚动体故障”），这里直接打印，不再维护第二份名称列表，
    // 避免两份名称列表写法不一致带来的隐患。
    std::cout << "[诊断结果] 类别ID=" << result.class_id
               << " 类别名称=" << result.class_name
               << " 置信度=" << result.confidence
               << " 模拟电流=" << mapped_current_ma << "mA"
               << " 推理耗时=" << result.inference_time_us << "us"
               << std::endl;

    return ErrorCode::success;
}

// standby/shutdown —— 控制台输出没有需要释放的资源，保留为空操作即可。
// 打印一句提示信息方便在控制台上观察到状态切换，调试更直观。
void ConsoleOutputSink::standby() {
    std::cout << "[输出] 控制台输出进入待机状态" << std::endl;
}

void ConsoleOutputSink::shutdown() {
    std::cout << "[输出] 控制台输出已关闭" << std::endl;
}

// map_current_ma —— 工业 4-20mA 电流环的模拟换算公式：
// 1. 每个类别对应一个“满置信度时的目标电流”（rated_current），
//    数值约定：0=正常->11mA，1=滚动体故障->14mA，2=保持架故障->17mA，
//    3=外圈故障->20mA，分类越严重，电流越大，方便下游用电流阈值告警。
// 2. class_id 落在 [0,3] 之外说明分类结果不可信，返回 5.0F 作为
//    “异常/不可用”的哨兵电流值，不在 4 个正常档位范围内，便于识别。
// 3. confidence 理论上应该在 [0,1]，但模型输出可能有浮点误差，
//    用 std::clamp 夹紧到 [0,1] 区间，避免置信度异常时电流值跑出预期范围。
// 4. 电流基线是 8.0F：即使置信度是 0，也保持 8mA 而不是 4mA，
//    让“分类结果存在但不确定”和“完全没有信号（4mA）”在电流上可区分；
//    置信度越高，电流越线性逼近该类别的目标电流 target_current。
float ConsoleOutputSink::map_current_ma(int class_id,
                                        float confidence) const {
    static const float rated_current[] = {
        11.0F,
        14.0F,
        17.0F,
        20.0F
    };

    if (class_id < 0 || class_id > 3) {
        return 5.0F;
    }

    float safe_confidence = std::clamp(confidence, 0.0F, 1.0F);
    float target_current = rated_current[class_id];
    return 8.0F + (target_current - 8.0F) * safe_confidence;
}

}  // namespace bearing
