#include "bearing/acquisition/CsvAcquisitionSource.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace bearing {

namespace {

// 按指定分隔符把一行文本拆成若干个字段
std::vector<std::string> split_csv_line(const std::string& line, char delimiter) {
    std::vector<std::string> cells;
    std::stringstream stream(line);
    std::string cell;
    while (std::getline(stream, cell, delimiter)) {
        cells.push_back(cell);
    }
    return cells;
}

// 去掉行尾可能存在的 \r（Windows 换行符遗留的回车符）
std::string strip_trailing_cr(std::string line) {
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    return line;
}

// 去掉行首可能存在的 UTF-8 BOM（部分 Excel 导出的 CSV 会带这个前缀）
std::string strip_utf8_bom(std::string line) {
    const std::string bom = "\xEF\xBB\xBF";
    if (line.compare(0, bom.size(), bom) == 0) {
        return line.substr(bom.size());
    }
    return line;
}

}  // namespace

CsvAcquisitionSource::CsvAcquisitionSource(std::string csv_path,
                                           std::size_t batch_size,
                                           bool loop_at_end)
    : csv_path_(std::move(csv_path)),
      batch_size_(batch_size),
      loop_at_end_(loop_at_end) {}

ErrorCode CsvAcquisitionSource::initialize() {
    ErrorCode load_result = load_rows();
    if (load_result != ErrorCode::success) {
        return load_result;
    }

    // 至少要有一个完整 batch 的数据，否则 read() 永远凑不满一批
    if (rows_.size() < batch_size_) {
        std::cerr << "振动数据行数不足: " << csv_path_
                  << " 需要至少 " << batch_size_
                  << " 行，实际有效行数为 " << rows_.size() << std::endl;
        return ErrorCode::file_read_error;
    }

    next_row_ = 0;
    next_sequence_ = 0;
    initialized_ = true;
    return ErrorCode::success;
}

ErrorCode CsvAcquisitionSource::load_rows() {
    std::ifstream file(csv_path_);
    if (!file.is_open()) {
        std::cerr << "无法打开振动数据文件: " << csv_path_ << std::endl;
        return ErrorCode::file_not_found;
    }

    std::string header_line;
    if (!std::getline(file, header_line)) {
        std::cerr << "振动数据文件为空: " << csv_path_ << std::endl;
        return ErrorCode::file_read_error;
    }
    header_line = strip_utf8_bom(header_line);
    header_line = strip_trailing_cr(header_line);

    char delimiter = ',';
    std::array<std::size_t, 3> column_indices{};
    ErrorCode header_result = parse_header(header_line, delimiter, column_indices);
    if (header_result != ErrorCode::success) {
        return header_result;
    }

    rows_.clear();
    bool reported_invalid_line = false;
    std::size_t line_number = 1;  // 第 1 行是表头，数据行从第 2 行开始计数
    std::string line;
    while (std::getline(file, line)) {
        ++line_number;
        line = strip_trailing_cr(line);

        std::array<float, 3> row{};
        ErrorCode row_result = ErrorCode::success;
        try {
            row_result = parse_data_row(line, delimiter, column_indices, row);
        } catch (const std::invalid_argument&) {
            // std::stof 遇到无法解析成数字的字段时抛出，只报告第一处错误行
            if (!reported_invalid_line) {
                std::cerr << "振动数据无法解析为数字: " << csv_path_
                          << " 第 " << line_number << " 行" << std::endl;
                reported_invalid_line = true;
            }
            continue;
        } catch (const std::out_of_range&) {
            // std::stof 遇到数值超出 float 表示范围时抛出
            if (!reported_invalid_line) {
                std::cerr << "振动数据数值超出范围: " << csv_path_
                          << " 第 " << line_number << " 行" << std::endl;
                reported_invalid_line = true;
            }
            continue;
        }

        if (row_result != ErrorCode::success) {
            // 字段数量不足的残缺行同样属于"无效行"，为了和上面数字解析失败的
            // 报告方式保持一致，这里也只报告第一处错误行，然后继续跳过该行
            if (!reported_invalid_line) {
                std::cerr << "振动数据行格式错误(字段数量不足): " << csv_path_
                          << " 第 " << line_number << " 行" << std::endl;
                reported_invalid_line = true;
            }
            continue;  // 字段数量不足的残缺行，跳过不计入有效数据
        }
        rows_.push_back(row);
    }

    return ErrorCode::success;
}

ErrorCode CsvAcquisitionSource::parse_header(
        const std::string& line,
        char& delimiter,
        std::array<std::size_t, 3>& column_indices) const {
    // 振动 CSV 既可能用制表符分隔，也可能用逗号分隔：先按制表符切分，
    // 如果只切出一列，说明真正的分隔符是逗号，再重新按逗号切分
    delimiter = '\t';
    std::vector<std::string> headers = split_csv_line(line, delimiter);
    if (headers.size() <= 1) {
        delimiter = ',';
        headers = split_csv_line(line, delimiter);
    }

    const std::array<std::string, 3> target_columns = {
        "de_normal_X", "de_normal_Y", "de_normal_Z"};

    // 三轴数据在文件中的列顺序并不固定，需要按列名查找，
    // 而不能假设 X/Y/Z 永远在固定的列号上
    for (std::size_t axis = 0; axis < target_columns.size(); ++axis) {
        bool found_column = false;
        for (std::size_t col = 0; col < headers.size(); ++col) {
            if (headers[col] == target_columns[axis]) {
                column_indices[axis] = col;
                found_column = true;
                break;
            }
        }
        if (!found_column) {
            std::cerr << "CSV 表头中未找到列: " << target_columns[axis]
                      << " (文件: " << csv_path_ << ")" << std::endl;
            return ErrorCode::csv_format_error;
        }
    }

    return ErrorCode::success;
}

ErrorCode CsvAcquisitionSource::parse_data_row(
        const std::string& line,
        char delimiter,
        const std::array<std::size_t, 3>& column_indices,
        std::array<float, 3>& row) const {
    std::vector<std::string> values_text = split_csv_line(line, delimiter);

    // 这一行至少要包含到最靠后那个目标列，否则就是残缺行，直接跳过
    std::size_t max_required_index = column_indices[0];
    for (std::size_t axis = 1; axis < column_indices.size(); ++axis) {
        if (column_indices[axis] > max_required_index) {
            max_required_index = column_indices[axis];
        }
    }
    if (values_text.size() <= max_required_index) {
        return ErrorCode::csv_format_error;
    }

    for (std::size_t axis = 0; axis < column_indices.size(); ++axis) {
        // std::stof 可能抛出 std::invalid_argument / std::out_of_range，
        // 由调用方 load_rows() 统一捕获并报告文件名和行号
        row[axis] = std::stof(values_text[column_indices[axis]]);
    }

    return ErrorCode::success;
}

ErrorCode CsvAcquisitionSource::read(RawVibrationBatch& batch) {
    if (!initialized_) {
        return ErrorCode::not_initialized;
    }

    const bool batch_fits_remaining_rows = next_row_ + batch_size_ <= rows_.size();
    if (!batch_fits_remaining_rows) {
        if (!loop_at_end_) {
            // 不循环的数据源读到末尾就停止，模拟一次性采集完成
            return ErrorCode::file_read_error;
        }
        // 循环模式下，剩余行不够一个 batch 时从头重新开始，模拟连续采集
        next_row_ = 0;
    }

    batch.x_axis.clear();
    batch.y_axis.clear();
    batch.z_axis.clear();
    batch.x_axis.reserve(batch_size_);
    batch.y_axis.reserve(batch_size_);
    batch.z_axis.reserve(batch_size_);

    // 按 X/Y/Z 三轴分别拼装成独立的向量，对应真实三轴振动传感器的输出格式
    for (std::size_t i = 0; i < batch_size_; ++i) {
        const std::array<float, 3>& row = rows_[next_row_];
        batch.x_axis.push_back(row[0]);
        batch.y_axis.push_back(row[1]);
        batch.z_axis.push_back(row[2]);
        ++next_row_;
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

void CsvAcquisitionSource::shutdown() {
    rows_.clear();
    next_row_ = 0;
    initialized_ = false;
}

std::string CsvAcquisitionSource::name() const {
    return "CsvAcquisitionSource";
}

}  // namespace bearing
