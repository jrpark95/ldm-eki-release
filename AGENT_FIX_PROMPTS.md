# Agent 수정 작업 프롬프트

## 🔴 Agent 3 수정 프롬프트 (최우선)

```
Task: Complete ldm.cuh/cu separation (INCOMPLETE from original task)

Issue Found:
- ldm.cuh is still in src/include/ (771 lines, should be in src/core/)
- No ldm.cu file created (all implementations still in header)
- src/core/ directory exists but is empty

Required Actions:

1. Move and Split ldm.cuh
   - Move src/include/ldm.cuh → src/core/ldm.cuh
   - Extract all LDM class method implementations to src/core/ldm.cu
   - Keep only class declaration, inline methods in src/core/ldm.cuh

2. Create src/core/ldm.cuh (header)
   Content:
   - #pragma once
   - All #include statements
   - LDM class declaration
   - Inline getter/setter methods only
   - Expected size: ~200 lines

3. Create src/core/ldm.cu (implementation)
   Content:
   - #include "../core/ldm.cuh"
   - All LDM constructor/destructor implementations
   - All LDM method implementations (run, simulate, etc.)
   - All non-inline functions
   - Expected size: ~571 lines

4. Add documentation blocks
   - Follow FUNCTION_DOCUMENTATION_STYLE.md
   - Document all public methods in header
   - Add @brief, @details, @param, @return

5. Update src/include/ldm.cuh
   - Replace with simple forward include:
     #pragma once
     #include "../core/ldm.cuh"
   - Or mark as DEPRECATED

Expected Output:
- src/core/ldm.cuh (class declaration)
- src/core/ldm.cu (method implementations)
- Updated includes referencing new location

DO NOT proceed with other tasks until this is complete.
Report when finished with line counts.
```

---

## 🟡 Agent 2 수정 프롬프트 (대기: Agent 3 완료 후)

```
Task: Clean up original source files (INCOMPLETE)

Issue Found:
- New files created successfully in src/data/meteo/ and src/simulation/
- Original files still exist in src/include/:
  * ldm_mdata.cuh (87,819 bytes)
  * ldm_func.cuh (68,701 bytes)
- This creates duplicate definitions and confusion

Required Actions:

1. Backup original files
   cd src/include/
   mv ldm_mdata.cuh ldm_mdata.cuh.ORIGINAL_BACKUP
   mv ldm_func.cuh ldm_func.cuh.ORIGINAL_BACKUP

2. Create deprecation notice files (optional)
   # src/include/ldm_mdata.cuh
   #pragma once
   #warning "ldm_mdata.cuh is deprecated. Use src/data/meteo/ldm_mdata_*.cuh"
   #include "../data/meteo/ldm_mdata_loading.cuh"
   #include "../data/meteo/ldm_mdata_processing.cuh"
   #include "../data/meteo/ldm_mdata_cache.cuh"

   # src/include/ldm_func.cuh
   #pragma once
   #warning "ldm_func.cuh is deprecated. Use src/simulation/ldm_func_*.cuh"
   #include "../simulation/ldm_func_simulation.cuh"
   #include "../simulation/ldm_func_particle.cuh"
   #include "../simulation/ldm_func_output.cuh"

3. Verify new files are complete
   - Check all functions from originals exist in new files
   - Confirm line count matches (allowing for documentation overhead)

Expected Output:
- Original files renamed to .ORIGINAL_BACKUP
- Optional: Deprecation wrapper files created
- Verification report confirming completeness
```

---

## 🟡 Agent 4 수정 프롬프트 (대기: Agent 3 완료 후)

```
Task: Clean up original source files (INCOMPLETE)

Issue Found:
- New files created successfully in src/ipc/, src/physics/, src/debug/
- Original files still exist in src/include/:
  * ldm_eki_ipc.cuh (26,978 bytes)
  * ldm_cram2.cuh (9,318 bytes)
  * ldm_nuclides.cuh (7,182 bytes)
  * memory_doctor.cuh (10,068 bytes)
- This creates duplicate definitions

Required Actions:

1. Backup original files
   cd src/include/
   mv ldm_eki_ipc.cuh ldm_eki_ipc.cuh.ORIGINAL_BACKUP
   mv ldm_cram2.cuh ldm_cram2.cuh.ORIGINAL_BACKUP
   mv ldm_nuclides.cuh ldm_nuclides.cuh.ORIGINAL_BACKUP
   mv memory_doctor.cuh memory_doctor.cuh.ORIGINAL_BACKUP

2. Create deprecation wrapper files
   # src/include/ldm_eki_ipc.cuh
   #pragma once
   #include "../ipc/ldm_eki_writer.cuh"
   #include "../ipc/ldm_eki_reader.cuh"

   # src/include/ldm_cram2.cuh
   #pragma once
   #include "../physics/ldm_cram2.cuh"

   # src/include/ldm_nuclides.cuh
   #pragma once
   #include "../physics/ldm_nuclides.cuh"

   # src/include/memory_doctor.cuh
   #pragma once
   #include "../debug/memory_doctor.cuh"

3. Verify Python-C++ IPC compatibility
   - Check that shared memory layouts unchanged
   - Confirm EKIWriter/Reader interfaces preserved

Expected Output:
- 4 original files renamed to .ORIGINAL_BACKUP
- 4 deprecation wrapper files created
- IPC compatibility verification report
```

---

## 🟠 Agent 5 수정 프롬프트 (대기: Agent 3 완료 후)

```
Task: Verify kernels_cram and update remaining includes

Issue Found:
- ldm_kernels_cram.cuh moved to src/kernels/cram/ (confirmed)
- Need to verify if .cu file should exist
- Need to create wrapper in original location

Required Actions:

1. Check ldm_kernels_cram implementation
   - Read src/kernels/cram/ldm_kernels_cram.cuh
   - If it contains implementations, create .cu file
   - If header-only (templates/inline), mark as complete

2. Create wrapper in old location
   # src/kernels/ldm_kernels_cram.cuh
   #pragma once
   #include "cram/ldm_kernels_cram.cuh"

3. Verify struct/config files
   - Confirm src/data/config/ldm_struct.cuh is header-only
   - Confirm src/data/config/ldm_config.cuh is header-only
   - No .cu files needed for these

4. Update Makefile include paths (if not done)
   - Verify -I./src/data/config
   - Verify -I./src/kernels/cram
   - Add -I./src/core

Expected Output:
- Confirmation of kernels_cram status (header-only or split)
- Wrapper file created
- Makefile verification report
```

---

## 🔵 Agent 6 작업 프롬프트 (모든 수정 완료 후)

```
Task: Final integration and include path updates

Prerequisites: Agents 2, 3, 4, 5 must complete fixes above

Required Actions:

1. Update src/core/ldm.cuh includes (after Agent 3 finishes)
   Current problematic lines:
   - Line 759: #include "ldm_cram2.cuh"
             → #include "../physics/ldm_cram2.cuh"
   - Line 760: #include "ldm_kernels.cuh"
             → #include "../kernels/ldm_kernels.cuh"
   - Line 766: #include "ldm_mdata.cuh"
             → #include "../data/meteo/ldm_mdata_loading.cuh"
             → #include "../data/meteo/ldm_mdata_processing.cuh"
             → #include "../data/meteo/ldm_mdata_cache.cuh"
   - Line 768: #include "ldm_func.cuh"
             → #include "../simulation/ldm_func_simulation.cuh"
             → #include "../simulation/ldm_func_particle.cuh"
             → #include "../simulation/ldm_func_output.cuh"

2. Update main files
   - src/main_eki.cu
   - src/main.cu
   - src/main_receptor_debug.cu

   Change:
   #include "include/ldm.cuh"
   → #include "core/ldm.cuh"

3. Verify all include chains
   - Check for circular dependencies
   - Ensure all forward declarations present
   - Validate include guards (#pragma once)

4. Test compilation
   make clean
   make all-targets
   - Capture all errors
   - Fix include path issues

5. Create file structure map
   find src -name "*.cuh" -o -name "*.cu" | sort > FINAL_FILE_LIST.txt

6. Generate dependency graph (optional)
   - List which files include which
   - Identify any circular dependencies

Expected Output:
- All files compile successfully
- FINAL_FILE_LIST.txt (complete file inventory)
- INTEGRATION_REPORT.md (what was changed and why)
- Compilation test results
```

---

## 📋 Execution Order

**Phase 1 - Critical Fix:**
1. ✅ Execute Agent 3 prompt
2. ⏸️ **WAIT for Agent 3 completion**

**Phase 2 - Cleanup (Parallel):**
3. ✅ Execute Agent 2 prompt
4. ✅ Execute Agent 4 prompt
5. ✅ Execute Agent 5 prompt
6. ⏸️ **WAIT for Agents 2, 4, 5 completion**

**Phase 3 - Integration:**
7. ✅ Execute Agent 6 prompt
8. ✅ Final verification

---

## 🎯 Success Criteria

- [ ] src/core/ldm.cuh exists with class declaration
- [ ] src/core/ldm.cu exists with implementations
- [ ] No duplicate .cuh files in src/include/ (only wrappers or .ORIGINAL_BACKUP)
- [ ] All include paths use relative paths (../)
- [ ] make clean && make completes without errors
- [ ] ./ldm-eki runs successfully
- [ ] All 54+ files accounted for in FINAL_FILE_LIST.txt

---

## 📞 Communication Protocol

Each Agent should report:
1. "Starting fix for [issue]"
2. "Created files: [list]"
3. "Modified files: [list]"
4. "Line counts: [before] → [after]"
5. "Fix complete: [timestamp]"

Agent 6 should not proceed until receiving "Fix complete" from all prerequisite Agents.