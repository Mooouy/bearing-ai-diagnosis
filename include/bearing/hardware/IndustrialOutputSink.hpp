#pragma once

#include "bearing/output/IOutputSink.hpp"

#include <array>
#include <atomic>
#include <cstdint>

namespace bearing {

// IndustrialOutputSink —— 树莓派真实工业输出端。
// 它把分类结果映射为 4-20mA PWM、轴承告警继电器、74HC595 状态灯和
// RS485 状态帧；默认仿真构建不会编译这个类。
class IndustrialOutputSink final : public IOutputSink {
public:
    IndustrialOutputSink() = default;
    ~IndustrialOutputSink() override;

    ErrorCode initialize() override;
    ErrorCode publish(const InferenceResult& result,
                      float& mapped_current_ma) override;
    void standby() override;
    void shutdown() override;

private:
    ErrorCode initialize_rs485();
    ErrorCode initialize_relay();
    ErrorCode initialize_leds();
    ErrorCode initialize_power_off_input();

    void set_rs485_send_mode();
    void set_rs485_receive_mode();
    ErrorCode send_rs485_frame(std::uint8_t frame_type,
                               std::uint8_t class_id,
                               float confidence,
                               float current_ma);

    ErrorCode write_pwm_current(float current_ma);
    float map_current_ma(int class_id, float confidence) const;

    void update_relay(int class_id);
    void reset_relay();
    void update_leds(int class_id);
    void clear_leds();
    void flush_leds();
    void shift_led_bit(bool value);

    static void handle_power_off_alert(int gpio,
                                       int level,
                                       std::uint32_t tick,
                                       void* user_data);
    void on_power_off();

    int serial_fd_ = -1;
    bool pigpio_initialized_ = false;
    bool rs485_initialized_ = false;
    bool relay_initialized_ = false;
    bool leds_initialized_ = false;
    std::atomic<bool> power_off_triggered_{false};
    std::array<bool, 8> led_shadow_{};
};

}  // namespace bearing
