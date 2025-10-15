#pragma once
// ldm_config.cuh - DEPRECATED WRAPPER
//
// ⚠️  DEPRECATION NOTICE ⚠️
// This file is deprecated and exists only for backward compatibility.
// Please update your includes to use the new modular structure:
//
// OLD:
//   #include "ldm_config.cuh"
//
// NEW:
//   #include "../data/config/ldm_config.cuh"
//
// This wrapper will be removed in a future release.
//
// Date: 2025-10-15
// Agent: Agent 5 (Configuration Refactoring)

#ifndef LDM_CONFIG_CUH
#define LDM_CONFIG_CUH

// Print deprecation warning at compile time
#warning "ldm_config.cuh is deprecated. Use data/config/ldm_config.cuh instead."

// Include new location
#include "../data/config/ldm_config.cuh"

#endif // LDM_CONFIG_CUH
