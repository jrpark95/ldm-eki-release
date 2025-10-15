#pragma once
// ldm_nuclides.cuh - DEPRECATED WRAPPER
//
// ⚠️  DEPRECATION NOTICE ⚠️
// This file is deprecated and exists only for backward compatibility.
// Please update your includes to use the new modular structure:
//
// OLD:
//   #include "ldm_nuclides.cuh"
//
// NEW:
//   #include "../physics/ldm_nuclides.cuh"
//
// This wrapper will be removed in a future release.
//
// Migration Guide:
// - NuclideConfig class moved to src/physics/
// - Singleton pattern unchanged: NuclideConfig::getInstance()
// - NuclideInfo struct unchanged
// - No namespace changes (global class)
//
// Date: 2025-10-15
// Agent: Agent 4 (Refactoring)

#ifndef LDM_NUCLIDES_CUH
#define LDM_NUCLIDES_CUH

// Print deprecation warning at compile time
#warning "ldm_nuclides.cuh is deprecated. Use physics/ldm_nuclides.cuh instead."

// Include new modular header
#include "../physics/ldm_nuclides.cuh"

#endif // LDM_NUCLIDES_CUH
