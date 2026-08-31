#include "bearing/logging/ResultLogger.hpp"

#include <filesystem>
#include <fstream>
#include <locale>
#include <string>
#include <utility>

namespace bearing {

namespace {

// CSV 表头：列的顺序是这个类对外的“契约”，下游脚本按位置解析每一列，
// 不能随意调整顺序。error_message 放在最后一列是因为它是唯一可能包含
// 任意长度文本（甚至理论上包含逗号）的字段，放在末尾不会影响前面
// 固定格式的数值列对齐，方便简单的按位置解析。
const char* const csv_header =
    "timestamp_ms,sequence,success,class_id,class_name,confidence,"
    "current_ma,inference_time_us,error_message";

// make_csv_safe —— CSV 用逗号分列、换行分行，自由文本里的逗号/换行必须
// 先替换，否则会破坏列对齐（一个逗号会多切出一列，一个换行会把一行
// 数据拆成两行）。这里只做最朴素的字符替换：逗号换成分号，换行/回车
// 换成空格，不引入正则表达式，保持逻辑一目了然。
std::string make_csv_safe(const std::string& field) {
    std::string safe_field = field;
    for (char& current_char : safe_field) {
        if (current_char == ',') {
            current_char = ';';
        } else if (current_char == '\n' || current_char == '\r') {
            current_char = ' ';
        }
    }
    return safe_field;
}

}  // namespace

// 构造函数只记录目录路径，不在这里创建目录或打开文件——
// 真正的副作用（建目录、开文件）放到 initialize() 里，这样构造一个
// ResultLogger 对象本身不会失败，失败只会在 initialize() 这一个
// 明确的时间点发生，方便调用方统一检查返回值。
ResultLogger::ResultLogger(std::string log_directory)
    : log_directory_(std::move(log_directory)) {}

// initialize —— 创建日志目录，并在固定文件名 results.csv 里写入表头。
// 文件名固定成 "<log_directory>/results.csv"，不按时间戳生成新文件，
// 这样每次诊断都追加到同一份文件里，方便用一个固定路径持续监控。
ErrorCode ResultLogger::initialize() {
    std::error_code fs_error;
    std::filesystem::create_directories(log_directory_, fs_error);
    if (fs_error) {
        return ErrorCode::output_error;
    }

    std::filesystem::path file_path =
        std::filesystem::path(log_directory_) / "results.csv";
    log_file_path_ = file_path.string();

    // 用截断模式打开一次，确保文件存在并写入表头；这一步只在 initialize()
    // 执行一次，后续每条记录由 write() 用追加模式单独打开/写入/关闭，
    // 避免长期持有一个文件句柄跨多次调用，类的状态更简单。
    std::ofstream output_file(log_file_path_, std::ios::out | std::ios::trunc);
    if (!output_file.is_open()) {
        return ErrorCode::output_error;
    }

    output_file << csv_header << "\n";
    if (!output_file.good()) {
        return ErrorCode::output_error;
    }

    return ErrorCode::success;
}

// write —— 把一条 ProcessingRecord 追加成 CSV 的一行。
// 列的顺序必须和 initialize() 写的表头一一对应：
// timestamp_ms, sequence, success, class_id, class_name, confidence,
// current_ma, inference_time_us, error_message（error_message 放最后，
// 原因见上面 csv_header 的注释）。
ErrorCode ResultLogger::write(const ProcessingRecord& record) {
    std::ofstream output_file(log_file_path_, std::ios::out | std::ios::app);
    if (!output_file.is_open()) {
        return ErrorCode::output_error;
    }

    // 用 classic locale 固定以 '.' 作小数点，避免不同系统 locale 把小数点
    // 写成逗号而破坏 CSV：confidence/current_ma 这两个浮点列一旦用了
    // 逗号当小数点，就会被误判成多出一列，把后面的列全部错位。
    output_file.imbue(std::locale::classic());

    // success 在 CSV 里用 1/0 表示而不是 true/false，方便后续用
    // pandas/Excel 直接当数值列做统计（求和、求平均成功率）。
    const int success_flag = record.success ? 1 : 0;

    output_file << record.timestamp_ms << ','
                << record.sequence << ','
                << success_flag << ','
                << record.inference.class_id << ','
                << make_csv_safe(record.inference.class_name) << ','
                << record.inference.confidence << ','
                << record.mapped_current_ma << ','
                << record.inference.inference_time_us << ','
                << make_csv_safe(record.error_message) << '\n';

    if (!output_file.good()) {
        return ErrorCode::output_error;
    }

    return ErrorCode::success;
}

}  // namespace bearing
