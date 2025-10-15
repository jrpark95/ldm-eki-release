#pragma once

/**
 * @file ldm_mdata.cuh
 * @brief DEPRECATED: Meteorological data management (legacy compatibility wrapper)
 *
 * ⚠️ WARNING: This file is deprecated and maintained only for backward compatibility.
 *
 * This header has been refactored into modular components:
 *   - ldm_mdata_loading.cuh   : GFS file loading and I/O operations
 *   - ldm_mdata_processing.cuh: Data processing and transformations
 *   - ldm_mdata_cache.cuh     : EKI meteorological data caching
 *
 * New code should include the specific module headers directly:
 *   #include "data/meteo/ldm_mdata_loading.cuh"
 *   #include "data/meteo/ldm_mdata_cache.cuh"
 *
 * Migration guide: See AGENT2_REFACTORING_REPORT.md
 *
 * @deprecated Use modular headers instead (src/data/meteo/*.cuh)
 * @see src/data/meteo/ldm_mdata_loading.cuh
 * @see src/data/meteo/ldm_mdata_processing.cuh
 * @see src/data/meteo/ldm_mdata_cache.cuh
 */

// Issue deprecation warning at compile time (DISABLED for release)
// #ifdef __NVCC__
// #pragma message("WARNING: ldm_mdata.cuh is deprecated. Use modular headers from src/data/meteo/ instead.")
// #endif

// For backward compatibility, include all new modular headers
#include "../data/meteo/ldm_mdata_loading.cuh"
#include "../data/meteo/ldm_mdata_processing.cuh"
#include "../data/meteo/ldm_mdata_cache.cuh"

// Legacy compatibility: All functions are now available through the included headers above
// No additional code needed - this is purely a forwarding wrapper
