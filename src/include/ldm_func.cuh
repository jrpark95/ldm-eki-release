#pragma once

/**
 * @file ldm_func.cuh
 * @brief DEPRECATED: Simulation functions (legacy compatibility wrapper)
 *
 * ⚠️ WARNING: This file is deprecated and maintained only for backward compatibility.
 *
 * This header has been refactored into modular components:
 *   - ldm_func_simulation.cuh : Main simulation loops and execution control
 *   - ldm_func_particle.cuh   : Particle management and GPU memory operations
 *   - ldm_func_output.cuh     : Output handling and observation system
 *
 * New code should include the specific module headers directly:
 *   #include "simulation/ldm_func_simulation.cuh"
 *   #include "simulation/ldm_func_particle.cuh"
 *   #include "simulation/ldm_func_output.cuh"
 *
 * Migration guide: See AGENT2_REFACTORING_REPORT.md
 *
 * @deprecated Use modular headers instead (src/simulation/*.cuh)
 * @see src/simulation/ldm_func_simulation.cuh
 * @see src/simulation/ldm_func_particle.cuh
 * @see src/simulation/ldm_func_output.cuh
 */

// Issue deprecation warning at compile time (DISABLED for release)
// #ifdef __NVCC__
// #pragma message("WARNING: ldm_func.cuh is deprecated. Use modular headers from src/simulation/ instead.")
// #endif

// For backward compatibility, include all new modular headers
#include "../simulation/ldm_func_simulation.cuh"
#include "../simulation/ldm_func_particle.cuh"
#include "../simulation/ldm_func_output.cuh"

// Legacy compatibility: All functions are now available through the included headers above
// No additional code needed - this is purely a forwarding wrapper
