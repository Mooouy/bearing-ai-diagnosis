#include "bearing/hardware/Ad4134AcquisitionSource.hpp"

#include <wiringPi.h>
#include <linux/spi/spidev.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/ioctl.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace bearing {

namespace {

constexpr int default_spi_channel = 0;
constexpr int default_spi_speed_hz = 1000000;
constexpr int ad4134_reset_pin = 5;
constexpr int ad4134_power_down_pin = 12;
constexpr int ad4134_data_out_x_pin = 6;
constexpr int ad4134_data_out_y_pin = 13;
constexpr int ad4134_data_out_z_pin = 19;
constexpr int ad4134_data_ready_pin = 1;
constexpr int ad4134_data_clock_pin = 0;
constexpr int ad4134_data_clock_half_us = 0;
constexpr float ad4134_reference_voltage = 4.096F;
constexpr float ad4134_full_scale_24bit = 8388608.0F;

void delay_data_clock_half_period() {
    if (ad4134_data_clock_half_us > 0) {
        delayMicroseconds(ad4134_data_clock_half_us);
    }
}

std::int32_t sign_extend_24bit(std::uint32_t value) {
    if ((value & 0x800000U) != 0U) {
        return static_cast<std::int32_t>(value | 0xFF000000U);
    }
    return static_cast<std::int32_t>(value);
}

struct Ad4134Frame {
    std::int32_t x = 0;
    std::int32_t y = 0;
    std::int32_t z = 0;
};

Ad4134Frame read_parallel_frame() {
    while (digitalRead(ad4134_data_ready_pin) == HIGH) {
    }
    while (digitalRead(ad4134_data_ready_pin) == LOW) {
    }

    std::uint32_t x_bits = 0;
    std::uint32_t y_bits = 0;
    std::uint32_t z_bits = 0;

    for (int bit = 0; bit < 24; ++bit) {
        digitalWrite(ad4134_data_clock_pin, HIGH);
        delay_data_clock_half_period();

        x_bits = (x_bits << 1U) |
                 static_cast<std::uint32_t>(digitalRead(ad4134_data_out_x_pin) & 1);
        y_bits = (y_bits << 1U) |
                 static_cast<std::uint32_t>(digitalRead(ad4134_data_out_y_pin) & 1);
        z_bits = (z_bits << 1U) |
                 static_cast<std::uint32_t>(digitalRead(ad4134_data_out_z_pin) & 1);

        digitalWrite(ad4134_data_clock_pin, LOW);
        delay_data_clock_half_period();
    }

    Ad4134Frame frame;
    frame.x = sign_extend_24bit(x_bits);
    frame.y = sign_extend_24bit(y_bits);
    frame.z = sign_extend_24bit(z_bits);
    return frame;
}

float raw_count_to_voltage(std::int32_t raw_value) {
    return static_cast<float>(raw_value) *
           ad4134_reference_voltage /
           ad4134_full_scale_24bit;
}

}  // namespace

class Ad4134AcquisitionSource::Ad4134Device {
public:
    ~Ad4134Device() {
        shutdown();
    }

    ErrorCode initialize(int spi_channel, int spi_speed_hz) {
        ErrorCode status = initialize_gpio_pins();
        if (status != ErrorCode::success) {
            return status;
        }

        std::string spi_device = "/dev/spidev0." + std::to_string(spi_channel);
        spi_fd_ = open(spi_device.c_str(), O_RDWR);
        if (spi_fd_ < 0) {
            std::cerr << "打开 AD4134 SPI 设备失败: " << spi_device
                      << " errno=" << errno << std::endl;
            return ErrorCode::hardware_error;
        }

        if (!configure_spi_device(spi_speed_hz)) {
            close_spi();
            return ErrorCode::hardware_error;
        }

        spi_channel_ = spi_channel;
        spi_speed_hz_ = spi_speed_hz;

        status = configure_registers();
        if (status != ErrorCode::success) {
            close_spi();
            return status;
        }

        initialized_ = true;
        return ErrorCode::success;
    }

    void shutdown() {
        close_spi();
        initialized_ = false;
    }

    ErrorCode read_channels(std::vector<float>& x_axis,
                            std::vector<float>& y_axis,
                            std::vector<float>& z_axis,
                            std::size_t samples) {
        if (!initialized_) {
            return ErrorCode::not_initialized;
        }

        x_axis.clear();
        y_axis.clear();
        z_axis.clear();
        x_axis.reserve(samples);
        y_axis.reserve(samples);
        z_axis.reserve(samples);

        for (std::size_t index = 0; index < samples; ++index) {
            Ad4134Frame frame = read_parallel_frame();
            x_axis.push_back(raw_count_to_voltage(frame.x));
            y_axis.push_back(raw_count_to_voltage(frame.y));
            z_axis.push_back(raw_count_to_voltage(frame.z));
        }

        return ErrorCode::success;
    }

private:
    ErrorCode initialize_gpio_pins() {
        if (wiringPiSetupGpio() == -1) {
            std::cerr << "wiringPi GPIO 初始化失败" << std::endl;
            return ErrorCode::hardware_error;
        }

        pinMode(ad4134_data_out_x_pin, INPUT);
        pinMode(ad4134_data_out_y_pin, INPUT);
        pinMode(ad4134_data_out_z_pin, INPUT);
        pinMode(ad4134_data_ready_pin, INPUT);

        pinMode(ad4134_data_clock_pin, OUTPUT);
        digitalWrite(ad4134_data_clock_pin, LOW);

        // AD_CS 使用 /dev/spidev0.0 的内核 CE0，不在用户态手动拉片选。
        pinMode(ad4134_power_down_pin, OUTPUT);
        digitalWrite(ad4134_power_down_pin, HIGH);
        delay(5);

        pinMode(ad4134_reset_pin, OUTPUT);
        digitalWrite(ad4134_reset_pin, LOW);
        delay(10);
        digitalWrite(ad4134_reset_pin, HIGH);
        delay(20);

        return ErrorCode::success;
    }

    bool configure_spi_device(int spi_speed_hz) {
        std::uint8_t mode = SPI_MODE_3;
        if (ioctl(spi_fd_, SPI_IOC_WR_MODE, &mode) < 0) {
            return false;
        }

        std::uint8_t bits_per_word = 8;
        if (ioctl(spi_fd_, SPI_IOC_WR_BITS_PER_WORD, &bits_per_word) < 0) {
            return false;
        }

        std::uint32_t speed = static_cast<std::uint32_t>(spi_speed_hz);
        return ioctl(spi_fd_, SPI_IOC_WR_MAX_SPEED_HZ, &speed) >= 0;
    }

    ErrorCode configure_registers() {
        const std::pair<std::uint8_t, std::uint8_t> register_values[] = {
            {0x00, 0x80},
            {0x02, 0x01},
            {0x11, 0x20},
            {0x12, 0x02},
            {0x1E, 0x2A},
            {0x1D, 0x54},
            {0x13, 0x08}
        };

        for (const auto& register_value : register_values) {
            ErrorCode status =
                write_register(register_value.first, register_value.second);
            if (status != ErrorCode::success) {
                return status;
            }

            if (register_value.first == 0x00) {
                usleep(10000);
                continue;
            }

            std::uint8_t read_back = 0;
            status = read_register(register_value.first, read_back);
            if (status != ErrorCode::success) {
                return status;
            }

            const bool lower_two_bits_register = register_value.first == 0x12;
            bool matches = read_back == register_value.second;
            if (lower_two_bits_register) {
                matches = (read_back & 0x03U) == register_value.second;
            }
            if (!matches) {
                return ErrorCode::hardware_error;
            }
        }

        usleep(20000);
        return ErrorCode::success;
    }

    ErrorCode write_register(std::uint8_t reg, std::uint8_t value) {
        std::uint8_t tx[2] = {static_cast<std::uint8_t>(reg & 0x3FU), value};
        std::uint8_t rx[2] = {0, 0};
        spi_ioc_transfer transfer;
        std::memset(&transfer, 0, sizeof(transfer));
        transfer.tx_buf = reinterpret_cast<unsigned long>(tx);
        transfer.rx_buf = reinterpret_cast<unsigned long>(rx);
        transfer.len = 2;
        transfer.speed_hz = static_cast<std::uint32_t>(spi_speed_hz_);
        transfer.bits_per_word = 8;

        if (ioctl(spi_fd_, SPI_IOC_MESSAGE(1), &transfer) < 0) {
            return ErrorCode::hardware_error;
        }
        return ErrorCode::success;
    }

    ErrorCode read_register(std::uint8_t reg, std::uint8_t& value) {
        std::uint8_t tx[2] = {
            static_cast<std::uint8_t>(0x40U | (reg & 0x3FU)),
            0x00
        };
        std::uint8_t rx[2] = {0, 0};
        spi_ioc_transfer transfer;
        std::memset(&transfer, 0, sizeof(transfer));
        transfer.tx_buf = reinterpret_cast<unsigned long>(tx);
        transfer.rx_buf = reinterpret_cast<unsigned long>(rx);
        transfer.len = 2;
        transfer.speed_hz = static_cast<std::uint32_t>(spi_speed_hz_);
        transfer.bits_per_word = 8;

        if (ioctl(spi_fd_, SPI_IOC_MESSAGE(1), &transfer) < 0) {
            return ErrorCode::hardware_error;
        }

        value = rx[1];
        return ErrorCode::success;
    }

    void close_spi() {
        if (spi_fd_ >= 0) {
            close(spi_fd_);
            spi_fd_ = -1;
        }
    }

    int spi_fd_ = -1;
    int spi_channel_ = 0;
    int spi_speed_hz_ = default_spi_speed_hz;
    bool initialized_ = false;
};

Ad4134AcquisitionSource::Ad4134AcquisitionSource(std::size_t batch_size)
    : batch_size_(batch_size) {}

Ad4134AcquisitionSource::~Ad4134AcquisitionSource() {
    shutdown();
}

ErrorCode Ad4134AcquisitionSource::initialize() {
    device_ = std::make_unique<Ad4134Device>();
    ErrorCode status =
        device_->initialize(default_spi_channel, default_spi_speed_hz);
    if (status != ErrorCode::success) {
        device_.reset();
        return status;
    }

    initialized_ = true;
    next_sequence_ = 0;
    return ErrorCode::success;
}

ErrorCode Ad4134AcquisitionSource::read(RawVibrationBatch& batch) {
    if (!initialized_ || device_ == nullptr) {
        return ErrorCode::not_initialized;
    }

    ErrorCode status = device_->read_channels(
        batch.x_axis, batch.y_axis, batch.z_axis, batch_size_);
    if (status != ErrorCode::success) {
        return status;
    }

    const auto now = std::chrono::system_clock::now();
    batch.timestamp_ms = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch())
            .count());
    batch.sequence = next_sequence_++;
    batch.sample_rate_hz = 50000.0F;
    return ErrorCode::success;
}

void Ad4134AcquisitionSource::shutdown() {
    if (device_ != nullptr) {
        device_->shutdown();
        device_.reset();
    }
    initialized_ = false;
}

std::string Ad4134AcquisitionSource::name() const {
    return "Ad4134AcquisitionSource";
}

}  // namespace bearing
