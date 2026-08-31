#pragma once

namespace bearing {

enum class ErrorCode {
    success = 0,
    not_initialized,
    already_running,
    invalid_argument,
    file_not_found,
    file_read_error,
    csv_format_error,
    processing_error,
    model_load_error,
    inference_error,
    output_error,
    hardware_error
};

}  // namespace bearing
