#pragma once

#include <cstdint>
#include <optional>

namespace dlio {

// Tracks one strictly increasing timestamp sequence. Rejected observations do
// not advance the sequence, so the next valid dt is measured from the last
// accepted sample.
class ImuTimestampTracker {
public:
  struct Observation {
    bool accepted = false;
    std::optional<double> dt_seconds;
    std::int64_t previous_stamp_ns = 0;
    std::uint64_t rejected_count = 0;
  };

  Observation observe(const std::int64_t stamp_ns) {
    Observation result;
    result.previous_stamp_ns = last_stamp_ns_;

    if (has_stamp_ && stamp_ns <= last_stamp_ns_) {
      ++rejected_count_;
      result.rejected_count = rejected_count_;
      return result;
    }

    result.accepted = true;
    if (has_stamp_) {
      // Preserve DLIO's existing clean-stream arithmetic: rclcpp::Time::seconds()
      // converts each absolute timestamp to double before subtraction. Ordering
      // still uses exact integer nanoseconds.
      const double stamp_seconds = static_cast<double>(stamp_ns) / 1e9;
      const double previous_seconds = static_cast<double>(last_stamp_ns_) / 1e9;
      result.dt_seconds = stamp_seconds - previous_seconds;
    }

    last_stamp_ns_ = stamp_ns;
    has_stamp_ = true;
    result.rejected_count = rejected_count_;
    return result;
  }

  // Starts a new timestamp sequence while preserving the lifetime rejection
  // counter for diagnostics.
  void resetSequence() {
    has_stamp_ = false;
    last_stamp_ns_ = 0;
  }

  std::uint64_t rejectedCount() const { return rejected_count_; }

private:
  bool has_stamp_ = false;
  std::int64_t last_stamp_ns_ = 0;
  std::uint64_t rejected_count_ = 0;
};

}  // namespace dlio
