// src/dxvk/dxvk_precompile_stats.h
#pragma once
#include <atomic>
#include <cstdint>

namespace dxvk {
  extern std::atomic<std::uint64_t> g_precompile_runs; // 只有 extern
}
