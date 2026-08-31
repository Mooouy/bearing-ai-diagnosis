#pragma once

#include <atomic>
#include <cstdint>

namespace bearing {

struct DiagnosticCounters {
    std::atomic<std::uint64_t> acquisition_loops{0};
    std::atomic<std::uint64_t> batches_acquired{0};
    std::atomic<std::uint64_t> raw_batches_pushed{0};
    std::atomic<std::uint64_t> processing_loops{0};
    std::atomic<std::uint64_t> raw_batches_received{0};
    std::atomic<std::uint64_t> cwt_images_created{0};
    std::atomic<std::uint64_t> inferences_started{0};
    std::atomic<std::uint64_t> inferences_finished{0};
    std::atomic<std::uint64_t> outputs_published{0};
    std::atomic<std::uint64_t> results_pushed{0};
    std::atomic<std::uint64_t> log_records_written{0};
};

}  // namespace bearing
