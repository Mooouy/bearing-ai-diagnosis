#pragma once

#include "bearing/core/ErrorCode.hpp"
#include "bearing/core/Types.hpp"

#include <string>

namespace bearing {

// ResultLogger —— 把每一次诊断结果追加写入一个 CSV 文件，方便事后用
// Excel/pandas 等工具回看历史诊断记录。这里只做“写文件”这一件事，
// 不负责打印到控制台（那是 ConsoleOutputSink 的职责），职责分开。
class ResultLogger {
public:
    explicit ResultLogger(std::string log_directory);

    // initialize —— 创建日志目录、打开日志文件并写入表头。
    // 必须在“后台线程开始写数据之前”调用成功，否则后续每一次 write()
    // 都会失败，所以失败要尽早暴露（返回 output_error），而不是拖到
    // 第一次写数据时才发现目录或文件有问题。
    ErrorCode initialize();

    // write —— 把一条诊断记录追加成 CSV 的一行，列的顺序固定，
    // 必须和 initialize() 里写的表头完全对应。
    ErrorCode write(const ProcessingRecord& record);

private:
    std::string log_directory_;
    std::string log_file_path_;
};

}  // namespace bearing
