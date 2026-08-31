set(BEARING_MISSING_HARDWARE_DEPS)

find_path(PIGPIO_INCLUDE_DIR pigpio.h)
find_library(PIGPIO_LIBRARY pigpio)
if(NOT PIGPIO_INCLUDE_DIR OR NOT PIGPIO_LIBRARY)
    list(APPEND BEARING_MISSING_HARDWARE_DEPS
        "pigpio.h + libpigpio (install pigpio development package)")
endif()

find_path(WIRINGPI_INCLUDE_DIR wiringPi.h)
find_library(WIRINGPI_LIBRARY wiringPi)
if(NOT WIRINGPI_INCLUDE_DIR OR NOT WIRINGPI_LIBRARY)
    list(APPEND BEARING_MISSING_HARDWARE_DEPS
        "wiringPi.h + libwiringPi (install wiringPi development package)")
endif()

find_path(U8G2_INCLUDE_DIR u8g2.h)
find_library(U8G2_LIBRARY u8g2)
if(NOT U8G2_INCLUDE_DIR OR NOT U8G2_LIBRARY)
    list(APPEND BEARING_MISSING_HARDWARE_DEPS
        "u8g2.h + libu8g2 (install u8g2 development package)")
endif()

find_path(LINUX_I2C_INCLUDE_DIR linux/i2c-dev.h)
find_library(I2C_LIBRARY i2c)
if(NOT LINUX_I2C_INCLUDE_DIR OR NOT I2C_LIBRARY)
    list(APPEND BEARING_MISSING_HARDWARE_DEPS
        "linux/i2c-dev.h + libi2c (install i2c-tools/libi2c-dev)")
endif()

if(BEARING_MISSING_HARDWARE_DEPS)
    string(REPLACE ";" "\n  - " BEARING_MISSING_TEXT
        "${BEARING_MISSING_HARDWARE_DEPS}")
    message(FATAL_ERROR
        "ENABLE_HARDWARE=ON requires Raspberry Pi hardware dependencies:\n"
        "  - ${BEARING_MISSING_TEXT}")
endif()

target_include_directories(bearing_core PUBLIC
    ${PIGPIO_INCLUDE_DIR}
    ${WIRINGPI_INCLUDE_DIR}
    ${U8G2_INCLUDE_DIR}
    ${LINUX_I2C_INCLUDE_DIR}
)
