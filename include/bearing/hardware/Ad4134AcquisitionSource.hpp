#pragma once

#include "bearing/acquisition/IAcquisitionSource.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace bearing {

// Ad4134AcquisitionSource —— 树莓派真实三轴振动采集源。
// 它把 AD4134 的 X/Y/Z 三路电压样本包装成 RawVibrationBatch，供后续
// CWT 和 ONNX 推理复用同一条处理链路。
class Ad4134AcquisitionSource final : public IAcquisitionSource {
public:
    explicit Ad4134AcquisitionSource(std::size_t batch_size = 5000);
    ~Ad4134AcquisitionSource() override;

    ErrorCode initialize() override;
    ErrorCode read(RawVibrationBatch& batch) override;
    void shutdown() override;
    std::string name() const override;

private:
    class Ad4134Device;

    std::unique_ptr<Ad4134Device> device_;
    std::size_t batch_size_;
    std::uint64_t next_sequence_ = 0;
    bool initialized_ = false;
};

}  // namespace bearing
