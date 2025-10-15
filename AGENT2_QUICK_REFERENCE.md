# Agent 2 Quick Reference Guide

## File Mapping

### Original → New Files

**ldm_mdata.cuh (1,978 lines)** split into:
- `src/data/meteo/ldm_mdata_loading.cuh/cu` - File I/O (1,433 lines)
- `src/data/meteo/ldm_mdata_processing.cuh/cu` - Processing (placeholder)
- `src/data/meteo/ldm_mdata_cache.cuh/cu` - EKI cache (539 lines)

**ldm_func.cuh (1,548 lines)** split into:
- `src/simulation/ldm_func_simulation.cuh/cu` - Main loops (732 lines)
- `src/simulation/ldm_func_particle.cuh/cu` - GPU memory (166 lines)
- `src/simulation/ldm_func_output.cuh/cu` - Observations (606 lines)

## Function Location Reference

### Meteorological Data Functions
| Function | Old File | New File | Lines |
|----------|----------|----------|-------|
| `initializeFlexGFSData()` | ldm_mdata.cuh:4-1089 | ldm_mdata_loading.cu | 1086 |
| `loadFlexGFSData()` | ldm_mdata.cuh:1091-1376 | ldm_mdata_loading.cu | 286 |
| `loadFlexHeightData()` | ldm_mdata.cuh:1378-1435 | ldm_mdata_loading.cu | 58 |
| `calculateRequiredMeteoFiles()` | ldm_mdata.cuh:1441-1454 | ldm_mdata_cache.cu | 14 |
| `loadSingleMeteoFile()` | ldm_mdata.cuh:1456-1721 | ldm_mdata_cache.cu | 266 |
| `preloadAllEKIMeteorologicalData()` | ldm_mdata.cuh:1723-1973 | ldm_mdata_cache.cu | 251 |
| `cleanupEKIMeteorologicalData()` | ldm_mdata.cuh:1975-1979 | ldm_mdata_cache.cu | 5 |

### Simulation Functions
| Function | Old File | New File | Lines |
|----------|----------|----------|-------|
| `runSimulation()` | ldm_func.cuh:112-261 | ldm_func_simulation.cu | 150 |
| `runSimulation_eki()` | ldm_func.cuh:263-552 | ldm_func_simulation.cu | 290 |
| `runSimulation_eki_dump()` | ldm_func.cuh:1260-1549 | ldm_func_simulation.cu | 290 |

### Particle Management Functions
| Function | Old File | New File | Lines |
|----------|----------|----------|-------|
| `allocateGPUMemory()` | ldm_func.cuh:54-110 | ldm_func_particle.cu | 57 |
| `checkParticleNaN()` | ldm_func.cuh:554-591 | ldm_func_particle.cu | 38 |
| `checkMeteoDataNaN()` | ldm_func.cuh:593-660 | ldm_func_particle.cu | 68 |

### Output/Observation Functions
| Function | Old File | New File | Lines |
|----------|----------|----------|-------|
| `startTimer()` | ldm_func.cuh:42-45 | ldm_func_output.cu | 4 |
| `stopTimer()` | ldm_func.cuh:47-52 | ldm_func_output.cu | 6 |
| `initializeEKIObservationSystem()` | ldm_func.cuh:664-750 | ldm_func_output.cu | 87 |
| `computeReceptorObservations()` | ldm_func.cuh:752-844 | ldm_func_output.cu | 93 |
| `saveEKIObservationResults()` | ldm_func.cuh:846-900 | ldm_func_output.cu | 55 |
| `writeEKIObservationsToSharedMemory()` | ldm_func.cuh:902-907 | ldm_func_output.cu | 6 |
| `computeReceptorObservations_AllEnsembles()` | ldm_func.cuh:909-1021 | ldm_func_output.cu | 113 |
| `cleanupEKIObservationSystem()` | ldm_func.cuh:1023-1056 | ldm_func_output.cu | 34 |
| `resetEKIObservationSystemForNewIteration()` | ldm_func.cuh:1058-1088 | ldm_func_output.cu | 31 |
| `computeGridReceptorObservations()` | ldm_func.cuh:1092-1167 | ldm_func_output.cu | 76 |
| `saveGridReceptorData()` | ldm_func.cuh:1169-1225 | ldm_func_output.cu | 57 |
| `cleanupGridReceptorSystem()` | ldm_func.cuh:1227-1258 | ldm_func_output.cu | 32 |

## Integration Checklist for Agent 6

### Step 1: Update Makefile
```makefile
# Add to CUDA_SOURCES
CUDA_SOURCES += \
    src/data/meteo/ldm_mdata_loading.cu \
    src/data/meteo/ldm_mdata_processing.cu \
    src/data/meteo/ldm_mdata_cache.cu \
    src/simulation/ldm_func_simulation.cu \
    src/simulation/ldm_func_particle.cu \
    src/simulation/ldm_func_output.cu
```

### Step 2: Update Include Directories
```makefile
INCLUDES = -Isrc/include -Isrc/data/meteo -Isrc/simulation
```

### Step 3: Update ldm.cuh
```cpp
// Remove old includes
//#include "ldm_mdata.cuh"
//#include "ldm_func.cuh"

// Add new includes
#include "ldm_mdata_loading.cuh"
#include "ldm_mdata_cache.cuh"
#include "ldm_func_simulation.cuh"
#include "ldm_func_particle.cuh"
#include "ldm_func_output.cuh"
```

### Step 4: Deprecate Old Files
```bash
mkdir -p src/include/deprecated
mv src/include/ldm_mdata.cuh src/include/deprecated/
mv src/include/ldm_func.cuh src/include/deprecated/
```

### Step 5: Test Build
```bash
make clean
make all-targets
./ldm-eki  # Runtime test
```

## Verification Commands

```bash
# Count lines in new files
wc -l src/data/meteo/*.cu src/simulation/*.cu

# List all headers
find src/ -name "ldm_mdata*.cuh" -o -name "ldm_func*.cuh"

# Check function count
grep -h "^void LDM::" src/data/meteo/*.cuh src/simulation/*.cuh | wc -l

# Test syntax
nvcc -c src/data/meteo/ldm_mdata_loading.cu -Isrc/include --std=c++14 -o /tmp/test.o
```

## Troubleshooting

### Issue: Compilation errors about missing includes
**Solution**: Add `-Isrc/data/meteo -Isrc/simulation` to NVCC_FLAGS

### Issue: Undefined reference errors
**Solution**: Ensure all .cu files added to Makefile CUDA_SOURCES

### Issue: Duplicate symbol errors
**Solution**: Check that old ldm_mdata.cuh and ldm_func.cuh are not included

### Issue: Missing function declarations
**Solution**: Include the appropriate header file (loading/cache/simulation/particle/output)

## Contact

For questions about this refactoring:
- See: AGENT2_REFACTORING_REPORT.md (full technical details)
- Scripts: util/split_mdata_file.py, util/split_func_file.py
- Agent: Agent 2 (Data/Simulation Module Refactoring)
