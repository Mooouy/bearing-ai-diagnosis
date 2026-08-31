#pragma once

#include "bearing/core/ErrorCode.hpp"
#include "bearing/core/Types.hpp"

namespace bearing {

class IClassifier {
public:
    virtual ~IClassifier() = default;
    virtual ErrorCode infer(const cv::Mat& image,
                            InferenceResult& result) = 0;
};

}  // namespace bearing
