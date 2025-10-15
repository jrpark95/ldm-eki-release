#pragma once
// ldm_struct.cuh - DEPRECATED WRAPPER
//
// ⚠️  DEPRECATION NOTICE ⚠️
// This file is deprecated and exists only for backward compatibility.
// Please update your includes to use the new modular structure:
//
// OLD:
//   #include "ldm_struct.cuh"
//
// NEW:
//   #include "../data/config/ldm_struct.cuh"
//
// This wrapper will be removed in a future release.
//
// Date: 2025-10-15
// Agent: Agent 5 (Configuration Refactoring)

#ifndef LDM_STRUCT_CUH
#define LDM_STRUCT_CUH

// Print deprecation warning at compile time
#warning "ldm_struct.cuh is deprecated. Use data/config/ldm_struct.cuh instead."

// Include new location
#include "../data/config/ldm_struct.cuh"

#endif // LDM_STRUCT_CUH
