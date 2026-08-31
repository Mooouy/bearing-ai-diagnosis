#pragma once

#include "bearing/core/ErrorCode.hpp"
#include "bearing/core/Types.hpp"

#include <string>

namespace bearing {

class IAcquisitionSource {
public:
    virtual ~IAcquisitionSource() = default;

    virtual ErrorCode initialize() = 0;
    virtual ErrorCode read(RawVibrationBatch& batch) = 0;
    virtual void shutdown() = 0;
    virtual std::string name() const = 0;
};

}  // namespace bearing
