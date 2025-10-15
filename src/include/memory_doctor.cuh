#pragma once
// memory_doctor.cuh - DEPRECATED WRAPPER
//
// ⚠️  DEPRECATION NOTICE ⚠️
// This file is deprecated and exists only for backward compatibility.
// Please update your includes to use the new modular structure:
//
// OLD:
//   #include "memory_doctor.cuh"
//
// NEW:
//   #include "../debug/memory_doctor.cuh"
//
// This wrapper will be removed in a future release.
//
// Migration Guide:
// - MemoryDoctor class moved to src/debug/
// - Global instance g_memory_doctor remains the same
// - No namespace changes (global class)
//
// Date: 2025-10-15
// Agent: Agent 4 (Refactoring)

#ifndef MEMORY_DOCTOR_CUH
#define MEMORY_DOCTOR_CUH

// Print deprecation warning at compile time
#warning "memory_doctor.cuh is deprecated. Use debug/memory_doctor.cuh instead."

// Include new modular header
#include "../debug/memory_doctor.cuh"

#endif // MEMORY_DOCTOR_CUH
