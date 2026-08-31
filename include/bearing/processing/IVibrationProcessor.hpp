#pragma once

#include "bearing/core/ErrorCode.hpp"
#include "bearing/core/Types.hpp"

namespace bearing {

class IVibrationProcessor {
public:
    virtual ~IVibrationProcessor() = default;
    virtual ErrorCode process(const RawVibrationBatch& input,
                              ProcessedVibration& output) const = 0;
};

}  // namespace bearing
