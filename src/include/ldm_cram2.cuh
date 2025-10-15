#pragma once
// ldm_cram2.cuh - DEPRECATED WRAPPER
//
// ⚠️  DEPRECATION NOTICE ⚠️
// This file is deprecated and exists only for backward compatibility.
// Please update your includes to use the new modular structure:
//
// OLD:
//   #include "ldm_cram2.cuh"
//
// NEW:
//   #include "../physics/ldm_cram2.cuh"
//
// This wrapper will be removed in a future release.
//
// Migration Guide:
// - CRAM48 functions moved to src/physics/
// - All functions remain as LDM class member functions
// - Device function cram_decay_calculation() unchanged
// - No namespace changes
//
// Date: 2025-10-15
// Agent: Agent 4 (Refactoring)

#ifndef LDM_CRAM2_CUH
#define LDM_CRAM2_CUH

// Print deprecation warning at compile time
#warning "ldm_cram2.cuh is deprecated. Use physics/ldm_cram2.cuh instead."

// Include new modular header
#include "../physics/ldm_cram2.cuh"

#endif // LDM_CRAM2_CUH
