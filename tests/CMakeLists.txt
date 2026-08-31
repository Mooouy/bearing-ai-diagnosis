add_executable(test_smoke test_smoke.cpp)
add_test(NAME smoke COMMAND test_smoke)

add_executable(test_concurrent_queue test_concurrent_queue.cpp)
target_include_directories(test_concurrent_queue PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_concurrent_queue PRIVATE Threads::Threads)
add_test(NAME concurrent_queue COMMAND test_concurrent_queue)

add_executable(test_csv_acquisition test_csv_acquisition.cpp)
target_link_libraries(test_csv_acquisition PRIVATE bearing_core)
target_compile_definitions(test_csv_acquisition PRIVATE
    TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
add_test(NAME csv_acquisition COMMAND test_csv_acquisition)

add_executable(test_vibration_processor test_vibration_processor.cpp)
target_link_libraries(test_vibration_processor PRIVATE bearing_core)
add_test(NAME vibration_processor COMMAND test_vibration_processor)

add_executable(test_onnx_classifier test_onnx_classifier.cpp)
target_link_libraries(test_onnx_classifier PRIVATE bearing_core)
target_compile_definitions(test_onnx_classifier PRIVATE
    TEST_MODEL_PATH="${PROJECT_SOURCE_DIR}/models/bearing_vibration_model.onnx")
add_test(NAME onnx_classifier COMMAND test_onnx_classifier)

add_executable(test_console_output test_console_output.cpp)
target_link_libraries(test_console_output PRIVATE bearing_core)
add_test(NAME console_output COMMAND test_console_output)

add_executable(test_result_logger test_result_logger.cpp)
target_link_libraries(test_result_logger PRIVATE bearing_core)
add_test(NAME result_logger COMMAND test_result_logger)

# test_application_shutdown 需要 BearingApplication 的符号，但
# BearingApplication.cpp 是可执行文件 bearing_vibration_system 的源码，
# 没有放进 bearing_core 库里（放进去会和 Task 7 brief 里 Step 5 给定的
# add_executable 源码列表冲突）。所以这里把它作为测试目标的“额外源码”
# 直接编译进测试可执行文件，而不是把它移动到 bearing_core。
add_executable(test_application_shutdown
    test_application_shutdown.cpp
    ${PROJECT_SOURCE_DIR}/src/app/BearingApplication.cpp)
target_link_libraries(test_application_shutdown PRIVATE bearing_core)
add_test(NAME application_shutdown COMMAND test_application_shutdown)

# test_application_drain 覆盖 test_application_shutdown 没有覆盖到的场景：
# 外部 stop() 和一个仍在主动循环的采集线程之间的竞态（详见该文件开头的
# 注释）。同样需要把 BearingApplication.cpp 作为额外源码编译进来，原因
# 与上面 test_application_shutdown 完全一致。
add_executable(test_application_drain
    test_application_drain.cpp
    ${PROJECT_SOURCE_DIR}/src/app/BearingApplication.cpp)
target_link_libraries(test_application_drain PRIVATE bearing_core)
add_test(NAME application_drain COMMAND test_application_drain)
set_tests_properties(application_drain PROPERTIES TIMEOUT 4)
