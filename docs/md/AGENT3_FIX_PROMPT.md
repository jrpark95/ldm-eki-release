# Agent 3 긴급 수정 프롬프트

```
Task: Fix src/include/ldm.cuh - Create simple wrapper (CRITICAL)

Issue Found by Agent 6:
- src/include/ldm.cuh still contains full LDM class definition (772 lines)
- src/core/ldm.cuh also contains LDM class definition (610 lines)
- Result: Duplicate definitions causing 69 compilation errors

Your Previous Work Status:
✅ COMPLETED: Created src/core/ldm.cuh and src/core/ldm.cu
✅ COMPLETED: Updated main files to use core/ldm.cuh
❌ INCOMPLETE: Did NOT update src/include/ldm.cuh to be a wrapper

Required Actions:

1. Backup the original file:
   mv src/include/ldm.cuh src/include/ldm.cuh.ORIGINAL_BACKUP

2. Create NEW src/include/ldm.cuh with ONLY this content:

   #pragma once
   // ldm.cuh - DEPRECATED WRAPPER
   //
   // ⚠️  DEPRECATION NOTICE ⚠️
   // This file is deprecated and exists only for backward compatibility.
   // Please update your includes to use the new location:
   //
   // OLD:
   //   #include "include/ldm.cuh"
   //
   // NEW:
   //   #include "core/ldm.cuh"
   //
   // This wrapper will be removed in a future release.
   //
   // Date: 2025-10-15
   // Agent: Agent 3 (Core Refactoring)

   #ifndef LDM_CUH_WRAPPER
   #define LDM_CUH_WRAPPER

   // Print deprecation warning at compile time
   #warning "include/ldm.cuh is deprecated. Use core/ldm.cuh instead."

   // Include new location
   #include "../core/ldm.cuh"

   #endif // LDM_CUH_WRAPPER

3. Verify the fix:
   wc -l src/include/ldm.cuh
   # Should be ~30 lines or less

   grep -c "class LDM" src/include/ldm.cuh
   # Should be 0 (no class definition, just include)

Expected Result:
- src/include/ldm.cuh: ~30 lines (wrapper only)
- src/include/ldm.cuh.ORIGINAL_BACKUP: 772 lines (backup)
- NO duplicate class definitions
- Compilation errors should drop from 69 to 0

DO NOT:
- Do NOT keep any class definitions in src/include/ldm.cuh
- Do NOT keep any global variables in src/include/ldm.cuh
- Do NOT keep any __constant__ declarations in src/include/ldm.cuh
- Do NOT modify src/core/ldm.cuh or src/core/ldm.cu

Report when complete with:
- Line count of new wrapper file
- Confirmation that backup was created
```
