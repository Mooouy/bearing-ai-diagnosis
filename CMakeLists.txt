cmake_minimum_required(VERSION 3.16)
project(BearingVibrationSystem VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_HARDWARE "Build Raspberry Pi hardware adapters" OFF)
option(BUILD_TESTING "Build automated tests" ON)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED IMPORTED_TARGET fftw3)

add_library(bearing_core
    src/acquisition/CsvAcquisitionSource.cpp
)
target_include_directories(bearing_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(bearing_core PUBLIC ${OpenCV_LIBS} Threads::Threads)
target_compile_options(bearing_core PRIVATE -Wall -Wextra -Wpedantic)

target_sources(bearing_core PRIVATE
    src/processing/VibrationProcessor.cpp
)
target_link_libraries(bearing_core PUBLIC PkgConfig::FFTW)

if(NOT DEFINED onnxruntime_DIR OR onnxruntime_DIR STREQUAL "")
    set(onnxruntime_DIR "/usr/local/lib/cmake/onnxruntime"
        CACHE PATH "Path to onnxruntime CMake config directory")
endif()

find_package(onnxruntime REQUIRED CONFIG)

target_sources(bearing_core PRIVATE
    src/inference/OnnxClassifier.cpp
)
target_link_libraries(bearing_core PUBLIC onnxruntime::onnxruntime)

# Task 6：控制台模拟输出 + CSV 结果日志。两者都是纯软件实现，
# 不引入任何新的第三方依赖（不含 pigpio/wiringPi/u8g2 等硬件库），
# 因此即使 ENABLE_HARDWARE=OFF 也能正常构建。
target_sources(bearing_core PRIVATE
    src/output/ConsoleOutputSink.cpp
    src/logging/ResultLogger.cpp
)

# Task 7：把已有的各个组件组装成一个三线程应用 + 命令行入口。
# main.cpp/ApplicationOptions.cpp/BearingApplication.cpp 是可执行文件
# 专属的源码，不放进 bearing_core 库：bearing_core 是“可复用的组件库”，
# 而应用编排（线程怎么连、命令行怎么解析）只属于这一个可执行文件。
add_executable(bearing_vibration_system
    src/app/main.cpp
    src/app/ApplicationOptions.cpp
    src/app/BearingApplication.cpp
)
target_link_libraries(bearing_vibration_system PRIVATE bearing_core)
target_compile_options(bearing_vibration_system PRIVATE
    -Wall -Wextra -Wpedantic)

if(ENABLE_HARDWARE)
    include(cmake/FindHardwareDependencies.cmake)

    target_sources(bearing_core PRIVATE
        src/hardware/Ad4134AcquisitionSource.cpp
        src/hardware/IndustrialOutputSink.cpp
        src/hardware/OledStatusDisplay.cpp
    )
    target_compile_definitions(bearing_core PUBLIC BEARING_ENABLE_HARDWARE=1)
    target_link_libraries(bearing_core PUBLIC
        ${PIGPIO_LIBRARY}
        ${WIRINGPI_LIBRARY}
        ${U8G2_LIBRARY}
        ${I2C_LIBRARY}
    )
endif()

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
