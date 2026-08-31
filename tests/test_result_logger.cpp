#include "bearing/logging/ResultLogger.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main() {
    // 用系统临时目录下的一个独立子目录做测试，避免污染项目目录，
    // 也避免和其他测试用例的输出文件互相冲突。
    std::filesystem::path log_dir =
        std::filesystem::temp_directory_path() / "bearing_result_logger_test";
    std::filesystem::remove_all(log_dir);

    bearing::ResultLogger logger(log_dir.string());

    bearing::ErrorCode status = logger.initialize();
    if (status != bearing::ErrorCode::success) {
        std::cerr << "initialize failed" << std::endl;
        return EXIT_FAILURE;
    }

    bearing::ProcessingRecord record;
    record.timestamp_ms = 1000;
    record.sequence = 7;
    record.success = true;
    record.inference.class_id = 2;
    record.inference.class_name = "保持架故障";
    record.inference.confidence = 0.87F;
    record.mapped_current_ma = 15.83F;
    record.inference.inference_time_us = 1234;
    record.error_message = "";

    status = logger.write(record);
    if (status != bearing::ErrorCode::success) {
        std::cerr << "write failed" << std::endl;
        return EXIT_FAILURE;
    }

    // 第二条记录专门验证 Fix 1：error_message 里同时混入逗号和换行，
    // write() 必须先把它们替换掉（逗号换成分号，换行换成空格），
    // 否则这一行会被多切出几列，或者被换行拆成两行，破坏 9 列契约。
    // 这里先把两条记录都写完，再统一打开文件读回来比对，更贴近真实
    // 使用场景（持续 write() 之后才去查看日志文件），也避免一边读
    // 一边写同一个文件。
    bearing::ProcessingRecord dirty_record;
    dirty_record.timestamp_ms = 2000;
    dirty_record.sequence = 8;
    dirty_record.success = false;
    dirty_record.inference.class_id = 0;
    dirty_record.inference.class_name = "正常";
    dirty_record.inference.confidence = 0.42F;
    dirty_record.mapped_current_ma = 4.0F;
    dirty_record.inference.inference_time_us = 99;
    dirty_record.error_message = "file not found, retry\nagain";

    status = logger.write(dirty_record);
    if (status != bearing::ErrorCode::success) {
        std::cerr << "write (dirty record) failed" << std::endl;
        return EXIT_FAILURE;
    }

    // 把日志文件原样读回来，逐行比对，确认表头和数据行都和约定的
    // CSV 格式完全一致——这是真实的文件 I/O，不使用任何 mock。
    std::filesystem::path expected_file = log_dir / "results.csv";
    std::ifstream input_file(expected_file);
    if (!input_file.is_open()) {
        std::cerr << "expected log file does not exist: "
                  << expected_file << std::endl;
        return EXIT_FAILURE;
    }

    std::string header_line;
    std::getline(input_file, header_line);
    const std::string expected_header =
        "timestamp_ms,sequence,success,class_id,class_name,confidence,"
        "current_ma,inference_time_us,error_message";
    if (header_line != expected_header) {
        std::cerr << "header mismatch: " << header_line << std::endl;
        return EXIT_FAILURE;
    }

    std::string data_line;
    std::getline(input_file, data_line);
    const std::string expected_data_line =
        "1000,7,1,2,保持架故障,0.87,15.83,1234,";
    if (data_line != expected_data_line) {
        std::cerr << "data row mismatch: got [" << data_line
                  << "] expected [" << expected_data_line << "]" << std::endl;
        return EXIT_FAILURE;
    }

    std::string dirty_data_line;
    std::getline(input_file, dirty_data_line);

    // 写出来的这一行本身必须是单行：getline() 已经按 '\n' 切过一次，
    // 如果字段里残留未替换的 '\r'，行尾会带上它，所以这里直接检查
    // 行内不再包含任何 '\n' 或 '\r'。
    if (dirty_data_line.find('\n') != std::string::npos ||
        dirty_data_line.find('\r') != std::string::npos) {
        std::cerr << "dirty data row contains embedded newline: ["
                  << dirty_data_line << "]" << std::endl;
        return EXIT_FAILURE;
    }

    // 按逗号切分整行，必须恰好切出 9 个字段（也就是恰好 8 个逗号），
    // 这是 CSV 9 列契约最直接的验证方式。
    std::istringstream dirty_line_stream(dirty_data_line);
    std::string field;
    std::vector<std::string> fields;
    while (std::getline(dirty_line_stream, field, ',')) {
        fields.push_back(field);
    }
    if (fields.size() != 9) {
        std::cerr << "dirty data row does not have exactly 9 fields, got "
                  << fields.size() << ": [" << dirty_data_line << "]"
                  << std::endl;
        return EXIT_FAILURE;
    }

    // error_message 是第 9 列（下标 8），期望值是把原始文本里的逗号
    // 换成分号、换行换成空格之后的结果。
    const std::string expected_error_message = "file not found; retry again";
    if (fields[8] != expected_error_message) {
        std::cerr << "sanitized error_message mismatch: got [" << fields[8]
                  << "] expected [" << expected_error_message << "]"
                  << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
