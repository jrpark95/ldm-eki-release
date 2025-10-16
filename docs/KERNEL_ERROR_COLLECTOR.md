# Kernel Error Collection System

## Overview

The kernel error collection system silently collects CUDA kernel errors during simulation execution and reports them all at once when the simulation completes. This prevents cluttering the terminal output with error messages during runtime while ensuring all errors are properly logged.

## Architecture

### Components

1. **Header File**: `src/debug/kernel_error_collector.cuh`
   - Defines the error collection interface
   - Provides `CHECK_KERNEL_ERROR()` macro for convenient error checking

2. **Implementation**: `src/debug/kernel_error_collector.cu`
   - Collects errors in memory during simulation
   - Deduplicates errors (same error at same location counted once)
   - Sorts errors by occurrence count
   - Reports all errors with red/bold terminal formatting
   - Saves timestamped log files to `logs/error/`

3. **Integration Points**:
   - `src/main_eki.cu`: Enable/disable collection around simulations
   - `src/simulation/ldm_func_simulation.cu`: Check errors after kernel launches
   - `src/simulation/ldm_func_output.cu`: Check errors after observation kernels

## Usage

### Automatic Operation

The system runs automatically during simulation:

```cpp
// Enable collection before simulation
KernelErrorCollector::enableCollection();

// Run simulation (errors collected silently)
ldm.runSimulation_eki();

// Report all errors at end
KernelErrorCollector::reportAllErrors();
KernelErrorCollector::disableCollection();
```

### Error Checking Pattern

After every `cudaDeviceSynchronize()` call, errors are checked:

```cpp
cudaDeviceSynchronize();
CHECK_KERNEL_ERROR();  // Collects any kernel errors
```

### Manual Control

You can also use the system manually:

```cpp
#include "debug/kernel_error_collector.cuh"

// Enable collection
KernelErrorCollector::enableCollection();

// Your CUDA operations...
myKernel<<<blocks, threads>>>();
cudaDeviceSynchronize();
CHECK_KERNEL_ERROR();

// Report and clear
KernelErrorCollector::reportAllErrors();
KernelErrorCollector::clearErrors();
```

## Error Types

### Collected by This System

- **Asynchronous kernel launch errors**: Errors from `cudaGetLastError()` after `cudaDeviceSynchronize()`
- Examples:
  - Invalid kernel parameters
  - Out of bounds memory access
  - Invalid device function calls
  - Illegal memory access

### NOT Collected (Handled Separately)

- **Synchronous CUDA API errors**: Immediate failures from CUDA API calls
- Examples:
  - `cudaMemcpyToSymbol` failures (handled with fprintf)
  - `cudaMalloc` failures (handled with fprintf)
  - Device initialization errors

The synchronous errors are caught and printed immediately by existing error handling code using `fprintf(stderr, ...)`.

## Output Format

When errors are detected, the system prints:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  KERNEL ERROR REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total unique errors: 3
Total error occurrences: 156

1. [43 occurrences]
   Error: invalid argument
   Location: ldm_func_simulation.cu:385

2. [78 occurrences]
   Error: illegal memory access
   Location: ldm_kernels_particle.cu:124

3. [35 occurrences]
   Error: out of bounds
   Location: ldm_kernels_eki.cu:256

Detailed error log saved to: logs/error/kernel_errors_2025-10-16_12-34-56.log
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Log Files

When errors are found, a timestamped log file is created:

```
logs/error/kernel_errors_YYYY-MM-DD_HH-MM-SS.log
```

The log file contains the same formatted report as the terminal output.

## Implementation Details

### Error Deduplication

Errors are deduplicated by:
- Error message
- File name
- Line number

Multiple occurrences of the same error are counted, not stored separately.

### Thread Safety

The current implementation is **not thread-safe**. All error collection should happen on the main thread after `cudaDeviceSynchronize()` calls.

### Performance Impact

- **Minimal overhead**: Only a single `cudaGetLastError()` call per synchronization point
- **No impact when disabled**: Zero overhead when collection is disabled
- **No impact when no errors**: Zero overhead when no errors occur

## Verification

To verify the system is working:

1. **Run simulation**: `./ldm-eki`
2. **Check for report**: If kernel errors occurred, report will be printed at simulation end
3. **Check logs**: If report was printed, check `logs/error/` for timestamped log file

If no errors occur (correct behavior):
- No report printed
- No log files created
- No `logs/error/` directory created

## Example: Current Simulation

In the most recent simulation:
- ✅ No asynchronous kernel errors detected
- ✅ System worked correctly (no false positives)
- ⚠️ Synchronous API errors were detected (handled separately):
  - `[ERROR] Failed to copy height data to GPU: invalid device symbol`
  - These are from `cudaMemcpyToSymbol` calls, handled by existing error code

## Future Enhancements

Possible improvements:
1. Add thread-safe collection for multi-threaded kernels
2. Add error categorization (by severity)
3. Add statistical analysis of error patterns
4. Integration with existing Memory Doctor system
5. Add error suppression patterns (ignore known/expected errors)
