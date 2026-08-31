#pragma once

#include "bearing/acquisition/IAcquisitionSource.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace bearing {

class CsvAcquisitionSource final : public IAcquisitionSource {
public:
    CsvAcquisitionSource(std::string csv_path,
                         std::size_t batch_size,
                         bool loop_at_end);

    ErrorCode initialize() override;
    ErrorCode read(RawVibrationBatch& batch) override;
    void shutdown() override;
    std::string name() const override;

private:
    ErrorCode load_rows();
    ErrorCode parse_header(const std::string& line,
                           char& delimiter,
                           std::array<std::size_t, 3>& column_indices) const;
    ErrorCode parse_data_row(const std::string& line,
                             char delimiter,
                             const std::array<std::size_t, 3>& column_indices,
                             std::array<float, 3>& row) const;

    std::string csv_path_;
    std::size_t batch_size_;
    bool loop_at_end_;
    std::vector<std::array<float, 3>> rows_;
    std::size_t next_row_ = 0;
    std::uint64_t next_sequence_ = 0;
    bool initialized_ = false;
};

}  // namespace bearing
