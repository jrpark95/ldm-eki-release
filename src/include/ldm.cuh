/**
 * @file ldm.cuh (LEGACY WRAPPER)
 * @brief Backward compatibility wrapper for legacy include paths
 *
 * @deprecated This file is a compatibility shim only.
 *             All new code should include "../core/ldm.cuh" directly.
 *             This wrapper will be removed in a future release.
 *
 * @details The actual LDM class definition has been moved to:
 *          - src/core/ldm.cuh (class declaration)
 *          - src/core/ldm.cu  (method implementations)
 *
 *          This file exists solely to support legacy include statements
 *          like #include "ldm.cuh" from files in src/include/ directory.
 *
 * @note Files outside src/include/ should use relative paths:
 *       #include "../core/ldm.cuh"
 */

#pragma once

#ifndef LDM_INCLUDE_WRAPPER_GUARD
#define LDM_INCLUDE_WRAPPER_GUARD

// Forward to the actual implementation in src/core/
#include "../core/ldm.cuh"

#endif // LDM_INCLUDE_WRAPPER_GUARD
