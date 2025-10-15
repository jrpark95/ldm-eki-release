#pragma once
// ldm_eki_ipc.cuh - DEPRECATED WRAPPER
//
// ⚠️  DEPRECATION NOTICE ⚠️
// This file is deprecated and exists only for backward compatibility.
// Please update your includes to use the new modular structure:
//
// OLD:
//   #include "ldm_eki_ipc.cuh"
//
// NEW:
//   #include "../ipc/ldm_eki_writer.cuh"  // For EKIWriter
//   #include "../ipc/ldm_eki_reader.cuh"  // For EKIReader
//
// This wrapper will be removed in a future release.
//
// Migration Guide:
// - EKIWriter and EKIReader are now in LDM_EKI_IPC namespace
// - Use: using LDM_EKI_IPC::EKIWriter;
//   Or:  LDM_EKI_IPC::EKIWriter writer;
//
// Date: 2025-10-15
// Agent: Agent 4 (Refactoring)

#ifndef LDM_EKI_IPC_CUH
#define LDM_EKI_IPC_CUH

// Print deprecation warning at compile time
#warning "ldm_eki_ipc.cuh is deprecated. Use ipc/ldm_eki_writer.cuh and ipc/ldm_eki_reader.cuh instead."

// Include new modular headers
#include "../ipc/ldm_eki_writer.cuh"
#include "../ipc/ldm_eki_reader.cuh"

// Import namespace for backward compatibility
using namespace LDM_EKI_IPC;

#endif // LDM_EKI_IPC_CUH
