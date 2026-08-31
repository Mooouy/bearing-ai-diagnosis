#include "bearing/output/ConsoleOutputSink.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

int main() {
    bearing::ConsoleOutputSink sink;

    struct Case {
        int class_id;
        float confidence;
        float expected_ma;
    };

    const Case cases[] = {
        {0, 1.0F, 11.0F},
        {1, 1.0F, 14.0F},
        {2, 1.0F, 17.0F},
        {3, 1.0F, 20.0F},
        {3, 0.5F, 14.0F}
    };

    for (const Case& test_case : cases) {
        float actual = sink.map_current_ma(
            test_case.class_id,
            test_case.confidence);
        if (std::fabs(actual - test_case.expected_ma) > 0.001F) {
            std::cerr << "current mapping failed for class "
                      << test_case.class_id << std::endl;
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}
