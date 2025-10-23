#pragma once

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>

#include "../util/util_string.h"

namespace dxvk::log {

  inline std::string currentTimeString() {
    using namespace std::chrono;

    const auto now = system_clock::now();
    const auto millis = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    const std::time_t timeT = system_clock::to_time_t(now);

    std::tm timeInfo;
  #if defined(_WIN32)
    localtime_s(&timeInfo, &timeT);
  #else
    localtime_r(&timeT, &timeInfo);
  #endif

    std::ostringstream stream;
    stream << std::put_time(&timeInfo, "%H:%M:%S")
           << '.' << std::setfill('0') << std::setw(3) << millis.count();

    return stream.str();
  }

  template<typename... Args>
  inline std::string ehang(const Args&... args) {
    return str::format("[ehang][", currentTimeString(), "] DXVK: ", args...);
  }

}

