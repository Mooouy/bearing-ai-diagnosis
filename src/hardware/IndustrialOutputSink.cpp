#include "bearing/hardware/IndustrialOutputSink.hpp"

#include <pigpio.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <termios.h>
#include <unistd.h>

namespace bearing {

namespace {

constexpr unsigned int pwm_pin = 18;
constexpr unsigned int pwm_frequency_hz = 1000;
constexpr float linear_k = 50.580F;
constexpr float linear_b = -0.0153F;
constexpr float min_current_ma = 4.0F;
constexpr float max_current_ma = 20.0F;
constexpr float standby_current_ma = 4.0F;
constexpr float infer_failed_current_ma = 5.0F;
constexpr float class_base_current_ma = 8.0F;

constexpr float bearing_normal_current_ma = 11.0F;
constexpr float bearing_ball_fault_current_ma = 14.0F;
constexpr float bearing_cage_fault_current_ma = 17.0F;
constexpr float bearing_outer_fault_current_ma = 20.0F;

constexpr unsigned int power_off_pin = 4;
constexpr unsigned int rs485_de_pin = 24;
constexpr const char* rs485_port = "/dev/ttyAMA0";
constexpr int rs485_baud = B9600;
constexpr std::uint8_t frame_header_1 = 0x55;
constexpr std::uint8_t frame_header_2 = 0xAA;
constexpr std::uint8_t frame_status = 0x01;
constexpr std::uint8_t frame_result = 0x02;
constexpr std::size_t frame_length = 8;

constexpr unsigned int relay_bearing_pin = 17;

constexpr unsigned int hc595_data_pin = 23;
constexpr unsigned int hc595_latch_pin = 26;
constexpr unsigned int hc595_clock_pin = 22;
constexpr std::uint8_t led_work = 0;
constexpr std::uint8_t led_outer_fault = 1;
constexpr std::uint8_t led_inner_fault = 3;
constexpr std::uint8_t led_ball_fault = 4;
constexpr std::uint8_t led_device_error = 6;
constexpr std::uint8_t led_beeper = 7;

unsigned int current_to_pwm_duty(float current_ma) {
    float safe_current =
        std::clamp(current_ma, min_current_ma, max_current_ma);
    float duty_percent =
        100.0F -
        (150.0F / 120.0F) * 2.0F *
            (safe_current - linear_b) / linear_k * 100.0F;
    duty_percent = std::clamp(duty_percent, 0.0F, 100.0F);

    unsigned int duty =
        static_cast<unsigned int>(duty_percent * 10000.0F + 0.5F);
    return std::min(duty, 1000000U);
}

void ensure_log_directory_exists() {
    mkdir("../log", 0755);
}

}  // namespace

IndustrialOutputSink::~IndustrialOutputSink() {
    shutdown();
}

ErrorCode IndustrialOutputSink::initialize() {
    if (gpioInitialise() < 0) {
        std::cerr << "[工业输出] pigpio 初始化失败" << std::endl;
        return ErrorCode::hardware_error;
    }
    pigpio_initialized_ = true;

    ErrorCode status = write_pwm_current(standby_current_ma);
    if (status != ErrorCode::success) {
        return status;
    }

    status = initialize_rs485();
    if (status != ErrorCode::success) {
        return status;
    }

    status = initialize_relay();
    if (status != ErrorCode::success) {
        return status;
    }

    status = initialize_leds();
    if (status != ErrorCode::success) {
        return status;
    }

    status = initialize_power_off_input();
    if (status != ErrorCode::success) {
        return status;
    }

    return ErrorCode::success;
}

ErrorCode IndustrialOutputSink::publish(const InferenceResult& result,
                                        float& mapped_current_ma) {
    mapped_current_ma = map_current_ma(result.class_id, result.confidence);
    ErrorCode status = write_pwm_current(mapped_current_ma);
    if (status != ErrorCode::success) {
        return status;
    }

    update_relay(result.class_id);
    update_leds(result.class_id);

    return send_rs485_frame(
        frame_result,
        static_cast<std::uint8_t>(result.class_id),
        result.confidence,
        mapped_current_ma);
}

void IndustrialOutputSink::standby() {
    write_pwm_current(infer_failed_current_ma);
    reset_relay();
    clear_leds();
    send_rs485_frame(frame_status, 0x05, 0.0F, infer_failed_current_ma);
}

void IndustrialOutputSink::shutdown() {
    if (pigpio_initialized_) {
        write_pwm_current(standby_current_ma);
        clear_leds();
        reset_relay();
    }

    if (serial_fd_ >= 0) {
        set_rs485_receive_mode();
        close(serial_fd_);
        serial_fd_ = -1;
    }

    rs485_initialized_ = false;
    relay_initialized_ = false;
    leds_initialized_ = false;

    if (pigpio_initialized_) {
        gpioTerminate();
        pigpio_initialized_ = false;
    }
}

ErrorCode IndustrialOutputSink::initialize_rs485() {
    gpioSetMode(rs485_de_pin, PI_OUTPUT);
    gpioWrite(rs485_de_pin, 0);

    serial_fd_ = open(rs485_port, O_RDWR | O_NOCTTY | O_NDELAY);
    if (serial_fd_ < 0) {
        std::cerr << "[RS485] 打开串口失败: " << rs485_port << std::endl;
        return ErrorCode::hardware_error;
    }

    termios options;
    tcgetattr(serial_fd_, &options);
    cfsetispeed(&options, rs485_baud);
    cfsetospeed(&options, rs485_baud);
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;
    options.c_cflag |= PARENB;
    options.c_cflag &= ~PARODD;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag |= CREAD | CLOCAL;
    options.c_iflag &= ~(IXON | IXOFF | IXANY);
    options.c_iflag |= INPCK;
    options.c_oflag = 0;
    options.c_lflag = 0;
    options.c_cc[VMIN] = 0;
    options.c_cc[VTIME] = 10;

    tcflush(serial_fd_, TCIFLUSH);
    if (tcsetattr(serial_fd_, TCSANOW, &options) != 0) {
        close(serial_fd_);
        serial_fd_ = -1;
        return ErrorCode::hardware_error;
    }

    rs485_initialized_ = true;
    return ErrorCode::success;
}

ErrorCode IndustrialOutputSink::initialize_relay() {
    gpioSetMode(relay_bearing_pin, PI_OUTPUT);
    gpioWrite(relay_bearing_pin, 0);
    relay_initialized_ = true;
    return ErrorCode::success;
}

ErrorCode IndustrialOutputSink::initialize_leds() {
    gpioSetMode(hc595_data_pin, PI_OUTPUT);
    gpioSetMode(hc595_latch_pin, PI_OUTPUT);
    gpioSetMode(hc595_clock_pin, PI_OUTPUT);
    gpioWrite(hc595_data_pin, 0);
    gpioWrite(hc595_latch_pin, 0);
    gpioWrite(hc595_clock_pin, 0);

    led_shadow_.fill(false);
    leds_initialized_ = true;
    flush_leds();
    return ErrorCode::success;
}

ErrorCode IndustrialOutputSink::initialize_power_off_input() {
    gpioSetMode(power_off_pin, PI_INPUT);
    if (gpioSetAlertFuncEx(power_off_pin,
                           handle_power_off_alert,
                           this) != 0) {
        return ErrorCode::hardware_error;
    }
    return ErrorCode::success;
}

void IndustrialOutputSink::set_rs485_send_mode() {
    gpioWrite(rs485_de_pin, 1);
    if (serial_fd_ >= 0) {
        tcdrain(serial_fd_);
    }
}

void IndustrialOutputSink::set_rs485_receive_mode() {
    gpioWrite(rs485_de_pin, 0);
}

ErrorCode IndustrialOutputSink::send_rs485_frame(std::uint8_t frame_type,
                                                 std::uint8_t class_id,
                                                 float confidence,
                                                 float current_ma) {
    if (!rs485_initialized_ || serial_fd_ < 0) {
        return ErrorCode::not_initialized;
    }

    float safe_confidence = std::clamp(confidence, 0.0F, 1.0F);
    std::uint16_t confidence_fixed =
        static_cast<std::uint16_t>(safe_confidence * 10000.0F + 0.5F);

    float integer_part = std::floor(current_ma);
    std::uint8_t frame[frame_length] = {
        frame_header_1,
        frame_header_2,
        frame_type,
        class_id,
        static_cast<std::uint8_t>(confidence_fixed >> 8U),
        static_cast<std::uint8_t>(confidence_fixed & 0xFFU),
        static_cast<std::uint8_t>(integer_part),
        static_cast<std::uint8_t>(
            std::round((current_ma - integer_part) * 100.0F))
    };

    set_rs485_send_mode();
    ssize_t bytes_written = write(serial_fd_, frame, frame_length);
    tcdrain(serial_fd_);
    set_rs485_receive_mode();

    if (bytes_written != static_cast<ssize_t>(frame_length)) {
        return ErrorCode::hardware_error;
    }
    return ErrorCode::success;
}

ErrorCode IndustrialOutputSink::write_pwm_current(float current_ma) {
    unsigned int duty = current_to_pwm_duty(current_ma);
    int status = gpioHardwarePWM(pwm_pin, pwm_frequency_hz, duty);
    if (status != 0) {
        std::cerr << "[PWM] 输出失败 current_ma=" << current_ma
                  << " status=" << status << std::endl;
        return ErrorCode::hardware_error;
    }
    return ErrorCode::success;
}

float IndustrialOutputSink::map_current_ma(int class_id,
                                           float confidence) const {
    static const float rated_current[] = {
        bearing_normal_current_ma,
        bearing_ball_fault_current_ma,
        bearing_cage_fault_current_ma,
        bearing_outer_fault_current_ma
    };

    if (class_id < 0 || class_id > 3) {
        return infer_failed_current_ma;
    }

    float safe_confidence = std::clamp(confidence, 0.0F, 1.0F);
    float target_current = rated_current[class_id];
    return class_base_current_ma +
           (target_current - class_base_current_ma) * safe_confidence;
}

void IndustrialOutputSink::update_relay(int class_id) {
    if (relay_initialized_) {
        gpioWrite(relay_bearing_pin, class_id != 0 ? 1 : 0);
    }
}

void IndustrialOutputSink::reset_relay() {
    if (relay_initialized_) {
        gpioWrite(relay_bearing_pin, 0);
    }
}

void IndustrialOutputSink::update_leds(int class_id) {
    if (!leds_initialized_) {
        return;
    }

    led_shadow_.fill(false);
    switch (class_id) {
        case 0:
            led_shadow_[led_work] = true;
            break;
        case 1:
            led_shadow_[led_device_error] = true;
            led_shadow_[led_ball_fault] = true;
            led_shadow_[led_beeper] = true;
            break;
        case 2:
            led_shadow_[led_device_error] = true;
            led_shadow_[led_inner_fault] = true;
            led_shadow_[led_beeper] = true;
            break;
        case 3:
            led_shadow_[led_device_error] = true;
            led_shadow_[led_outer_fault] = true;
            led_shadow_[led_beeper] = true;
            break;
        default:
            led_shadow_[led_device_error] = true;
            break;
    }
    flush_leds();
}

void IndustrialOutputSink::clear_leds() {
    if (!leds_initialized_) {
        return;
    }

    led_shadow_.fill(false);
    flush_leds();
}

void IndustrialOutputSink::flush_leds() {
    if (!leds_initialized_) {
        return;
    }

    // 74HC595 输出板是低电平点亮，所以写入影子状态时取反。
    for (int index = 0; index < 8; ++index) {
        shift_led_bit(!led_shadow_[index]);
    }
    gpioWrite(hc595_latch_pin, 1);
    gpioWrite(hc595_latch_pin, 0);
}

void IndustrialOutputSink::shift_led_bit(bool value) {
    gpioWrite(hc595_data_pin, value ? 1 : 0);
    gpioWrite(hc595_clock_pin, 0);
    gpioWrite(hc595_clock_pin, 1);
}

void IndustrialOutputSink::handle_power_off_alert(int /*gpio*/,
                                                  int level,
                                                  std::uint32_t /*tick*/,
                                                  void* user_data) {
    if (level == 1 && user_data != nullptr) {
        auto* sink = static_cast<IndustrialOutputSink*>(user_data);
        sink->on_power_off();
    }
}

void IndustrialOutputSink::on_power_off() {
    bool expected = false;
    if (!power_off_triggered_.compare_exchange_strong(expected, true)) {
        return;
    }

    std::cerr << "[PWR_OFF] 检测到断电输入，进入安全输出并请求关机"
              << std::endl;
    write_pwm_current(standby_current_ma);
    clear_leds();
    reset_relay();
    send_rs485_frame(frame_status, 0xFF, 0.0F, standby_current_ma);

    ensure_log_directory_exists();
    std::time_t now = std::time(nullptr);
    char time_buffer[32];
    std::strftime(time_buffer,
                  sizeof(time_buffer),
                  "%Y-%m-%d %H:%M:%S",
                  std::localtime(&now));
    FILE* file = std::fopen("../log/system.log", "a");
    if (file != nullptr) {
        std::fprintf(file, "%s Power-off protection triggered!\n", time_buffer);
        std::fclose(file);
    }

    std::system("sudo shutdown now");
}

}  // namespace bearing
