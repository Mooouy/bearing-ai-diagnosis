#pragma once

#include "bearing/core/ErrorCode.hpp"
#include "bearing/core/Types.hpp"

namespace bearing {

class IOutputSink {
public:
    virtual ~IOutputSink() = default;

    virtual ErrorCode initialize() = 0;
    virtual ErrorCode publish(const InferenceResult& result,
                              float& mapped_current_ma) = 0;
    virtual void standby() = 0;
    virtual void shutdown() = 0;
};

}  // namespace bearing
