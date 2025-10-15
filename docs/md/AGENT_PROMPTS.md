# Agent Execution Prompts

## 🔴 Agent 1 Prompt
```
Task: Split and refactor ldm_kernels.cuh (3,864 lines)

Files to read: src/kernels/ldm_kernels.cuh
Reference docs: PARALLEL_REFACTORING_FINAL.md, FUNCTION_DOCUMENTATION_STYLE.md, FINAL_SOURCE_STRUCTURE.md

Instructions:
1. Use grep -n "^__global__\|^__device__" to find function boundaries
2. Read file in 500-line chunks using offset/limit
3. Split into 6 logical files under src/kernels/:
   - device/ldm_kernels_device.cuh/cu (device utilities, ~150 lines)
   - particle/ldm_kernels_particle.cuh/cu (move_part_by_wind, ~800 lines)
   - particle/ldm_kernels_particle_ens.cuh/cu (ensemble version, ~900 lines)
   - eki/ldm_kernels_eki.cuh/cu (receptor dose, ~300 lines)
   - dump/ldm_kernels_dump.cuh/cu (dump operations, ~800 lines)
   - dump/ldm_kernels_dump_ens.cuh/cu (ensemble dump, ~900 lines)
4. Separate declarations (.cuh) from implementations (.cu)
5. Add documentation blocks following FUNCTION_DOCUMENTATION_STYLE.md
6. Create master include file src/kernels/ldm_kernels.cuh
7. Do NOT modify calculation logic - only restructure
```

---

## 🔵 Agent 2 Prompt
```
Task: Split and refactor data/simulation modules (3,526 lines)

Files to read:
- src/include/ldm_mdata.cuh (1,978 lines)
- src/include/ldm_func.cuh (1,548 lines)
Reference docs: PARALLEL_REFACTORING_FINAL.md, FUNCTION_DOCUMENTATION_STYLE.md, FINAL_SOURCE_STRUCTURE.md

Instructions:
1. Split ldm_mdata.cuh into src/data/meteo/:
   - ldm_mdata_loading.cuh/cu (file I/O operations, ~700 lines)
   - ldm_mdata_processing.cuh/cu (interpolation/conversion, ~700 lines)
   - ldm_mdata_cache.cuh/cu (EKI caching, ~578 lines)
2. Split ldm_func.cuh into src/simulation/:
   - ldm_func_simulation.cuh/cu (main loop, ~600 lines)
   - ldm_func_particle.cuh/cu (particle management, ~500 lines)
   - ldm_func_output.cuh/cu (output/logging, ~448 lines)
3. Separate headers from implementations
4. Add documentation blocks for all functions
5. Preserve exact numerical computations
```

---

## 🟢 Agent 3 Prompt
```
Task: Split and refactor visualization/init modules (2,393 lines)

Files to read:
- src/include/ldm_plot.cuh (824 lines)
- src/include/ldm_init.cuh (805 lines)
- src/include/ldm.cuh (764 lines)
Reference docs: PARALLEL_REFACTORING_FINAL.md, FUNCTION_DOCUMENTATION_STYLE.md, FINAL_SOURCE_STRUCTURE.md

Instructions:
1. Split ldm_plot.cuh into src/visualization/:
   - ldm_plot_vtk.cuh/cu (VTK output, ~500 lines)
   - ldm_plot_utils.cuh/cu (utilities, ~324 lines)
2. Split ldm_init.cuh into src/init/:
   - ldm_init_particles.cuh/cu (particle init, ~450 lines)
   - ldm_init_config.cuh/cu (config parsing, ~355 lines)
3. Move ldm.cuh to src/core/:
   - Keep as single file but separate ldm.cuh/cu
   - Extract class methods to .cu file
4. Add complete documentation blocks
5. Maintain all class interfaces unchanged
```

---

## 🟡 Agent 4 Prompt
```
Task: Split and refactor IPC/utility modules (1,403 lines)

Files to read:
- src/include/ldm_eki_ipc.cuh (690 lines)
- src/include/memory_doctor.cuh (258 lines)
- src/include/ldm_cram2.cuh (240 lines)
- src/include/ldm_nuclides.cuh (215 lines)
Reference docs: PARALLEL_REFACTORING_FINAL.md, FUNCTION_DOCUMENTATION_STYLE.md, FINAL_SOURCE_STRUCTURE.md

Instructions:
1. Split ldm_eki_ipc.cuh into src/ipc/:
   - ldm_eki_writer.cuh/cu (EKIWriter class, ~400 lines)
   - ldm_eki_reader.cuh/cu (EKIReader class, ~290 lines)
2. Move memory_doctor.cuh to src/debug/:
   - Separate into memory_doctor.cuh/cu
3. Move physics models to src/physics/:
   - ldm_cram2.cuh/cu (CRAM operations)
   - ldm_nuclides.cuh/cu (decay chains)
4. Add documentation for all public interfaces
5. Ensure Python-C++ IPC compatibility maintained
```

---

## 🟣 Agent 5 Prompt
```
Task: Handle configuration files and build system

Files to read:
- src/include/ldm_struct.cuh (179 lines)
- src/include/ldm_config.cuh (128 lines)
- src/kernels/ldm_kernels_cram.cuh (32 lines)
- Makefile
- src/eki/*.py (check for needed updates)
Reference docs: PARALLEL_REFACTORING_FINAL.md, FUNCTION_DOCUMENTATION_STYLE.md, FINAL_SOURCE_STRUCTURE.md

Instructions:
1. Move to src/data/config/:
   - ldm_struct.cuh (keep header-only, no .cu needed)
   - ldm_config.cuh (keep header-only, no .cu needed)
2. Move ldm_kernels_cram.cuh to src/kernels/cram/:
   - Separate into ldm_kernels_cram.cuh/cu
3. Create comprehensive Makefile update:
   - List all 27 new .cu files with correct paths
   - Organize by module (KERNEL_SOURCES, DATA_SOURCES, etc.)
   - Add proper include paths for new directory structure
4. Update Python scripts if they reference old paths
5. Create build_files_list.txt with all new files
6. Add documentation headers to config files
```

---

## 🔷 Agent 6 Prompt (After 1-5 Complete)
```
Task: Final integration and validation

Prerequisites: Wait for Agents 1-5 to complete

Instructions:
1. Verify all 54 files created (27 .cuh + 27 .cu)
2. Create master include files:
   - Update src/kernels/ldm_kernels.cuh to include all kernel subdirs
   - Create similar master includes for other modules if needed
3. Update main files:
   - Fix all includes in main_eki.cu, main.cu, main_receptor_debug.cu
   - Update include paths to new structure
4. Test compilation:
   - Run: make clean && make
   - Fix any missing includes or linking errors
5. Verify functionality:
   - Run: ./ldm-eki
   - Check output matches original
6. Create final report:
   - List all changes made
   - Document any issues found and resolved
   - Confirm calculation results unchanged
```

---

## 🎯 Common Instructions for All Agents

1. **Documentation**: Every function must have a documentation block following FUNCTION_DOCUMENTATION_STYLE.md
2. **No Logic Changes**: Do NOT modify any calculations, only restructure
3. **File Naming**: Follow exact paths in FINAL_SOURCE_STRUCTURE.md
4. **Testing**: Each agent should verify their files compile independently
5. **Progress Tracking**: Report completion of each major file
6. **Error Handling**: If a file is too large, use grep and offset/limit approach