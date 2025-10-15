# LDM-EKI Output Style Guide

**Purpose**: Professional, clean, and consistent console output for production-grade scientific software.

**Last Updated**: 2025-01-15

---

## 1. Color System (from `src/include/colors.h`)

```cpp
namespace Color {
    const char* RESET   = "\033[0m";     // Reset to default
    const char* RED     = "\033[31m";    // Errors, failures
    const char* GREEN   = "\033[32m";    // Success, completion
    const char* YELLOW  = "\033[33m";    // Warnings, cautions
    const char* BLUE    = "\033[34m";    // Information
    const char* MAGENTA = "\033[35m";    // Ensemble operations
    const char* CYAN    = "\033[36m";    // System operations
    const char* BOLD    = "\033[1m";     // Emphasis
}
```

**Usage Pattern**:
```cpp
// Basic: Color the tag only
std::cout << Color::CYAN << "[GPU] " << Color::RESET
          << "Allocated 619 MB for meteorological data\n";

// Advanced: Color important values/keywords within message
std::cout << Color::CYAN << "[GPU] " << Color::RESET
          << "Allocated " << Color::BOLD << "619 MB" << Color::RESET
          << " for meteorological data\n";

// Section headers: Color the entire line
std::cout << Color::BOLD << Color::CYAN
          << "========================================\n"
          << Color::RESET;
```

**Color Application Rules**:
1. **Always color**: Tags (`[ERROR]`, `[GPU]`, etc.)
2. **Color for emphasis**: Important numbers, file paths, status keywords
3. **Full line color**: Section separators (====, ────, ━━━━)
4. **Never color**: Regular descriptive text

---

## 2. Tag System - Complete Overhaul

### 2.1 Core Tags (Essential - Always Keep)

| Tag | Color | When to Use | Example |
|-----|-------|-------------|---------|
| `[ERROR]` | `RED` | Fatal errors, exceptions, failures | `[ERROR] Failed to allocate GPU memory (out of memory)` |
| `[WARNING]` | `YELLOW` | Non-critical issues, potential problems | `[WARNING] Particle count below threshold (5/10)` |
| `[SUCCESS]` | `GREEN` | Major milestone completions | `[SUCCESS] Ensemble iteration 3/3 completed` |

### 2.2 Module Tags (Context Identification)

| Tag | Color | Module | When to Use |
|-----|-------|--------|-------------|
| `[GPU]` | `CYAN` | GPU operations | Memory allocation, data transfer, kernel launch |
| `[IPC]` | `BLUE` | Inter-process communication | Shared memory read/write, Python coordination |
| `[VTK]` | `BLUE` | VTK file output | File generation, I/O operations |
| `[ENSEMBLE]` | `MAGENTA` | Ensemble system | Multi-ensemble operations, ensemble-specific info |
| `[SYSTEM]` | `CYAN` | System operations | File cleanup, directory management, initialization |

### 2.3 Tags to REMOVE Completely

❌ **Delete these tags** - they add no value:
- `[DEBUG_BLOCK4]` - Temporary debugging artifact
- `[DEBUG_HGT]` - Height data debug (use `#ifdef DEBUG`)
- `[DEBUG_ITER1]` - Iteration-specific debug (use `#ifdef DEBUG`)
- `[DEBUG_EMISSIONS]` - Emissions debug (use `#ifdef DEBUG`)
- `[DEBUG_FULL_MATRIX]` - Matrix debug (use `#ifdef DEBUG`)
- `[EKI]` - Redundant (entire program is EKI)
- `[EKI_OBS]` - Use no tag or `[ENSEMBLE]`
- `[EKI_SIM]` - Use no tag or section header
- `[EKI_ENSEMBLE]` - Replace with `[ENSEMBLE]`
- `[EKI_ENSEMBLE_OBS]` - Replace with `[ENSEMBLE]`
- `[GPU_ALLOC]` - Replace with `[GPU]`
- `[GPU_VERIFY]` - Use `#ifdef DEBUG` with `[DEBUG]` tag
- `[NaN_CHECK]` - Use `#ifdef DEBUG` with `[DEBUG]` tag

### 2.4 Debug Tags (Conditional Compilation Only)

| Tag | Color | Usage |
|-----|-------|-------|
| `[DEBUG]` | `YELLOW` | Wrap in `#ifdef DEBUG` - general debugging |
| `[DEBUG_MEM]` | `YELLOW` | Wrap in `#ifdef DEBUG_VERBOSE` - memory inspection |
| `[DEBUG_GPU]` | `YELLOW` | Wrap in `#ifdef DEBUG_VERBOSE` - GPU verification |

**Pattern**:
```cpp
#ifdef DEBUG
    std::cout << Color::YELLOW << "[DEBUG] " << Color::RESET
              << "Particle 0: ens=0, timeidx=1, conc=" << conc << "\n";
#endif
```

---

## 3. Output Formatting Standards

### 3.1 Section Headers

**Format**: Bold Cyan with clean separators (FULL LINE COLOR)

```cpp
// Standard header (40 chars wide for 80-column terminals)
std::cout << "\n" << Color::BOLD << Color::CYAN
          << "========================================\n"
          << "  SIMULATION INITIALIZATION\n"
          << "========================================\n"
          << Color::RESET << "\n";
```

**Improved Version** (Unicode box drawing):
```cpp
// Unicode header - visually distinct
std::cout << "\n" << Color::BOLD << Color::CYAN
          << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
          << "  ENSEMBLE ITERATION 1/3\n"
          << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
          << Color::RESET << "\n";
```

**Subsection separator** (lighter, not full color):
```cpp
// Lighter separator for subsections
std::cout << "\n" << Color::CYAN << "────────────────────────────────────────\n"
          << Color::RESET;
```

### 3.2 Progress Indicators

**Bad** (Current):
```
[EKI_ENSEMBLE] Created particles for 10/100 ensembles
[EKI_ENSEMBLE] Created particles for 20/100 ensembles
[EKI_ENSEMBLE] Created particles for 30/100 ensembles
```

**Good** (Improved):
```cpp
// Single line with carriage return
std::cout << "\rInitializing ensembles: " << current << "/100 ("
          << (current*100/100) << "%)" << std::flush;

// Final line
std::cout << "\r" << Color::GREEN << "✓ " << Color::RESET
          << "Initialized 100 ensembles (998,400 particles)\n";
```

**Output**:
```
✓ Initialized 100 ensembles (998,400 particles)
```

### 3.3 Data Summaries

**Bad** (Current):
```
EKI Settings Loaded:
  Time Interval: 15 minutes
  Number of Receptors: 3
```

**Good** (Aligned and clean):
```cpp
std::cout << Color::BOLD << "Configuration Summary\n" << Color::RESET;
std::cout << "  Time Interval      : 15 minutes\n";
std::cout << "  Receptors          : 3\n";
std::cout << "  Ensemble Size      : 100 members\n";
std::cout << "  Observation Steps  : 24\n";
```

**Key**:
- 20-character width for labels (left-aligned)
- Colon separator with single space
- Units included
- Consistent indentation (2 spaces)

### 3.4 Observation Data (Massive Reduction Needed)

**Current Problem**: 72+ lines of repetitive observation data
```
[EKI_ENSEMBLE_OBS] Ens0 obs0: R1=0.000000e+00(0p) R2=0.000000e+00(0p) R3=0.000000e+00(0p)
[EKI_ENSEMBLE_OBS] Ens1 obs0: R1=0.000000e+00(0p) R2=0.000000e+00(0p) R3=0.000000e+00(0p)
... (100 lines per timestep!)
```

**Solution 1**: Disable by default, save to file instead
```cpp
// In production mode (release)
std::cout << "Collecting observations: timestep 0/24\r" << std::flush;
// Save detailed data to logs/observations_ensemble_iter1.txt

// In debug mode only
#ifdef DEBUG_VERBOSE
    std::cout << Color::MAGENTA << "[ENSEMBLE] " << Color::RESET
              << "Obs " << obs_id << " at t=" << time
              << "s: R1=" << r1_avg << " R2=" << r2_avg << " R3=" << r3_avg << "\n";
#endif
```

**Solution 2**: Summary statistics only
```cpp
std::cout << Color::MAGENTA << "[ENSEMBLE] " << Color::RESET
          << "Recorded observation " << obs_id << " at t=" << time << "s "
          << "(R1: " << r1_range << ", R2: " << r2_range << ", R3: " << r3_range << ")\n";
```

### 3.5 Timing Information

**Consistent Format**:
```cpp
// Always use seconds with 2-3 decimal places for times > 1s
std::cout << "Preloading completed in " << Color::BOLD << "2.82s" << Color::RESET << "\n";

// Use milliseconds for times < 1s
std::cout << "GPU transfer completed in " << Color::BOLD << "127ms" << Color::RESET << "\n";
```

### 3.6 Memory Statistics

**Good Format**:
```cpp
std::cout << Color::CYAN << "[GPU] " << Color::RESET << "Memory allocated:\n";
std::cout << "  Meteorological  : 619.2 MB\n";
std::cout << "  Particles       : 47.6 MB\n";
std::cout << "  Observations    : 0.3 MB\n";
std::cout << "  Total           : " << Color::BOLD << "667.1 MB\n" << Color::RESET;
```

---

## 4. Specific Refactoring Rules

### 4.1 Thread IDs - REMOVE

**Bad**:
```
Thread 9957682679808 loading file 0.txt...
```

**Good**:
```
Loading meteorological data: 1/3
```

### 4.2 Scientific Notation - Format Consistently

**Current Inconsistency**:
```
R1=2.46742e+14(6p)           // Sometimes e+14
R2=0.000000e+00(0p)          // Sometimes e+00 (unnecessary)
Capture radius: 2.500000e-02 degrees  // Overly precise
```

**Improved**:
```cpp
// For large numbers > 1e6, use scientific notation with 3 sig figs
if (value > 1e6) {
    std::cout << std::scientific << std::setprecision(2) << value;
}
// For small numbers, use fixed notation
else {
    std::cout << std::fixed << std::setprecision(4) << value;
}
```

**Result**:
```
R1=2.47e+14 (6p)
R2=0 (0p)
Capture radius: 0.0250°
```

### 4.3 Particle Count Formatting - Use Commas

**Bad**:
```
Total particles: 998400
```

**Good**:
```cpp
// Add thousand separators
std::locale comma_locale(std::locale(), new comma_numpunct());
std::cout.imbue(comma_locale);
std::cout << "Total particles: " << num_particles << "\n";
```

**Result**:
```
Total particles: 998,400
```

### 4.4 Redundant Messages - Consolidate

**Bad** (8 lines):
```
Cleaning output directory...
Output directory cleaned.
[DEBUG] Attempting to copy T matrix to T_const symbol...
[DEBUG] Successfully copied T matrix to T_const constant memory
```

**Good** (2 lines):
```
std::cout << "Cleaning output directory..." << std::flush;
std::cout << Color::GREEN << " ✓\n" << Color::RESET;

#ifdef DEBUG
std::cout << Color::YELLOW << "[DEBUG] " << Color::RESET
          << "Copied transformation matrix to GPU (14.4 KB)\n";
#endif
```

### 4.5 Hard-coded ANSI Codes - Replace with Constants

**Bad**:
```
[32mCRAM system initialization completed[0m
```

**Good**:
```cpp
std::cout << Color::GREEN << "✓ CRAM system initialization completed\n"
          << Color::RESET;
```

---

## 5. Message Content Guidelines

### 5.1 Clarity Checklist

Every message should answer:
- **What** happened? (action/result)
- **Where** did it happen? (optional: file, GPU, module)
- **Status**? (success/failure/progress)

**Bad** (vague):
```
Data loaded successfully
```

**Good** (specific):
```
Meteorological data loaded: 3 files, 619 MB (2.8s)
```

### 5.2 Avoid Redundancy

**Bad**:
```
Preloading meteorological data for fast iterations...
Starting meteorological data preloading for EKI...
```

**Good** (one message):
```
Preloading meteorological data (3 files)...
```

### 5.3 Actionable Errors

**Bad**:
```
[ERROR] File error occurred
```

**Good**:
```cpp
std::cout << Color::RED << Color::BOLD << "[ERROR] " << Color::RESET
          << "Failed to open " << Color::BOLD << filename << Color::RESET << "\n"
          << "  → Check file exists and has read permissions\n"
          << "  → Expected path: " << expected_path << "\n";
```

---

## 6. Before/After Examples

### Example 1: Initialization Sequence

**Before** (12 lines):
```
Cleaning output directory...
Output directory cleaned.
[DEBUG] Attempting to copy T matrix to T_const symbol (3600 floats, 14400 bytes)
[DEBUG] Successfully copied T matrix to T_const constant memory
Physics models: TURB=0, DRYDEP=0, WETDEP=0, RADDECAY=0
Loading Ensemble Kalman Inversion settings...
Loading EKI settings from data/eki_settings.txt...
EKI Settings Loaded:
  Time Interval: 15 minutes
  Number of Receptors: 3
Initializing CRAM system...
[32mCRAM system initialization completed[0m
```

**After** (6 lines):
```
Cleaning output directory... ✓
Loading configuration from data/eki_settings.txt...

Configuration Summary
  Time Interval      : 15 minutes
  Receptors          : 3
  Ensemble Size      : 100 members
  Physics Models     : TURB=off, DRYDEP=off, WETDEP=off, RADDECAY=off

✓ CRAM decay system initialized
```

### Example 2: Ensemble Particle Creation

**Before** (13 lines):
```
[EKI_ENSEMBLE] Initializing particles for 100 ensembles with 24 timesteps each
[EKI_ENSEMBLE] Particles per ensemble: 10000
[EKI_ENSEMBLE] Particles per timestep: 416
[EKI_ENSEMBLE] Total particles: 1000000
[EKI_ENSEMBLE] Created particles for 10/100 ensembles
[EKI_ENSEMBLE] Created particles for 20/100 ensembles
...
[EKI_ENSEMBLE] Created particles for 100/100 ensembles
[EKI_ENSEMBLE] Created 998400 total particles
[EKI_ENSEMBLE] Sorting particles by ensemble_id...
[EKI_ENSEMBLE] Particle initialization complete!
[EKI_ENSEMBLE] Memory layout: time-sorted for parallel ensemble execution
```

**After** (3 lines):
```
Initializing ensemble particles: 100/100 (100%) ✓
  Particles: 998,400 total (10,000 per ensemble)
  Memory layout: time-sorted for parallel execution
```

### Example 3: GPU Memory Operations

**Before** (18 lines):
```
[GPU_ALLOC] Allocating GPU memory for 998400 particles
[GPU_VERIFY] Verifying GPU data immediately after cudaMemcpy...
[GPU_VERIFY] Comparing CPU vs GPU data for first 3 particles:
[GPU_VERIFY] Particle 0:
  CPU: ens=0, timeidx=1, flag=0, conc=1.176405e+12
  GPU: ens=0, timeidx=1, flag=0, conc=1.176405e+12
  CPU conc[0]=1.176405e+12, sum(first 3)=1.176405e+12
  GPU conc[0]=1.176405e+12, sum(first 3)=1.176405e+12
[GPU_VERIFY] Particle 1:
...
[DEBUG] Particle memory copy successful, no CUDA errors detected
```

**After** (1-2 lines):
```
[GPU] Allocated 47.6 MB for 998,400 particles ✓
```

With debug mode:
```
[GPU] Allocated 47.6 MB for 998,400 particles ✓
[DEBUG] GPU verification: CPU/GPU data match (checked 3 particles)
```

---

## 7. Implementation Priority

### Phase 1: Critical Cleanup (Do First)
1. ✅ Remove all `[DEBUG_BLOCK*]`, `[DEBUG_ITER*]`, `[DEBUG_HGT]` tags
2. ✅ Consolidate `[EKI_*]` tags to simpler versions
3. ✅ Replace hard-coded ANSI codes (`[32m`, `[0m`) with `Color::` constants
4. ✅ Wrap verbose GPU/NaN checks in `#ifdef DEBUG`
5. ✅ Reduce observation output (24 lines → 1 line per timestep)

### Phase 2: Visual Improvements
6. ⭕ Add color to all remaining tags
7. ⭕ Use checkmarks (✓) for success messages
8. ⭕ Improve progress indicators (use `\r` for single-line updates)
9. ⭕ Format large numbers with thousand separators
10. ⭕ Standardize scientific notation (2-3 sig figs)

### Phase 3: Content Refinement
11. ⭕ Consolidate redundant messages
12. ⭕ Improve section headers (use Unicode box characters)
13. ⭕ Add memory/timing summaries at key milestones
14. ⭕ Ensure all errors have actionable suggestions

---

## 8. File-by-File Checklist

When refactoring a source file:

- [ ] Search for all `std::cout` and `printf` statements
- [ ] Remove deprecated tags (`[DEBUG_*]`, `[EKI_*]` variants)
- [ ] Apply color coding to remaining tags
- [ ] Wrap debug output in `#ifdef DEBUG`
- [ ] Consolidate progress loops (10 lines → 1 line)
- [ ] Replace hard-coded ANSI codes
- [ ] Check alignment in multi-line summaries
- [ ] Verify timing format (s vs ms consistency)
- [ ] Add checkmarks to success messages
- [ ] Remove thread IDs and unnecessary technical details
- [ ] Test output in terminal to verify colors work

---

## 9. Testing

After refactoring, verify:

1. **No information loss**: All critical data still visible
2. **Color codes work**: Test in standard Linux terminal
3. **Alignment correct**: No jagged columns in summaries
4. **Debug mode**: `#ifdef DEBUG` properly hides verbose output
5. **Performance**: No excessive I/O in hot loops (use `\r` for progress)
6. **Readability**: Show log to a colleague - is it clear?

**Test command**:
```bash
make clean && make && ./ldm-eki 2>&1 | tee test_output.log
```

---

## 10. Unicode Symbols (Optional Enhancement)

Modern terminals support Unicode - use sparingly:

| Symbol | Unicode | Usage |
|--------|---------|-------|
| ✓ | U+2713 | Success checkmark |
| ✗ | U+2717 | Failure cross |
| → | U+2192 | Pointer/arrow |
| ━ | U+2501 | Horizontal line (headers) |
| … | U+2026 | Ellipsis (loading...) |

**Pattern**:
```cpp
std::cout << Color::GREEN << "✓ " << Color::RESET << "Operation completed\n";
std::cout << Color::RED << "✗ " << Color::RESET << "Operation failed\n";
```

---

## 11. Special Cases

### 11.1 System Operations (File I/O, Cleanup)

Use `[SYSTEM]` tag with cyan color:

```cpp
std::cout << Color::CYAN << "[SYSTEM] " << Color::RESET
          << "Cleaning shared memory: /dev/shm/ldm_eki_* ... ";
std::cout << Color::GREEN << "✓\n" << Color::RESET;
```

### 11.2 Python Integration Messages

```cpp
std::cout << Color::BLUE << "[IPC] " << Color::RESET
          << "Waiting for Python EKI process (timeout: 60s)...\n";
```

### 11.3 Iteration Headers

```cpp
std::cout << "\n" << Color::BOLD << Color::CYAN;
std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
std::cout << " ITERATION " << iter << "/" << max_iter << "\n";
std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
std::cout << Color::RESET;
```

---

## 12. Python Output (Future Work)

For consistency, apply similar rules to Python scripts:

```python
from colorama import Fore, Style, init
init(autoreset=True)

# Tag with color
print(f"{Fore.MAGENTA}[ENSEMBLE] {Style.RESET_ALL}Computing Kalman gain...")

# Success message
print(f"{Fore.GREEN}✓ {Style.RESET_ALL}Ensemble update complete")

# Error message
print(f"{Fore.RED}{Style.BRIGHT}[ERROR] {Style.RESET_ALL}Matrix inversion failed")
```

---

## 13. Function Documentation Headers

Every function that produces console output should have a brief documentation comment:

```cpp
/**
 * @brief Initialize ensemble particles for all members
 * @details Creates particle arrays for N ensembles, allocates GPU memory,
 *          and displays progress with colored output
 * @output Progress indicator, memory statistics, success confirmation
 */
void LDM::initializeParticlesEKI_AllEnsembles() {
    std::cout << "\nInitializing ensemble particles...\n";
    // ... implementation ...
    std::cout << Color::GREEN << "✓ " << Color::RESET
              << "Initialized " << Color::BOLD << num_ensemble << Color::RESET
              << " ensembles\n\n";
}
```

**Documentation Template**:
```cpp
/**
 * @brief [One-line description of function purpose]
 * @details [Optional: Additional context, algorithm notes]
 * @output [What user will see in terminal]
 */
```

---

## 14. Whitespace and Readability

**Use blank lines strategically** to group related output:

```cpp
// Bad: Everything crammed together
std::cout << "Loading configuration...\n";
std::cout << "Config loaded\n";
std::cout << "Initializing GPU...\n";
std::cout << "GPU initialized\n";
std::cout << "Starting simulation...\n";

// Good: Logical grouping with whitespace
std::cout << "Loading configuration...\n";
std::cout << Color::GREEN << "✓ " << Color::RESET << "Configuration loaded\n\n";

std::cout << Color::CYAN << "[GPU] " << Color::RESET << "Initializing...\n";
std::cout << Color::GREEN << "✓ " << Color::RESET << "GPU initialized\n\n";

std::cout << Color::BOLD << Color::CYAN
          << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
          << "  STARTING SIMULATION\n"
          << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
          << Color::RESET << "\n";
```

**Whitespace Rules**:
1. Blank line **before** major section headers
2. Blank line **after** completion messages
3. No blank line within tightly related operations
4. Double blank line between major phases (initialization → simulation → results)

---

## 15. Keywords to Emphasize with Color

Certain keywords should be **bolded** or **colored** for visibility:

| Keyword Type | Example | Formatting |
|--------------|---------|------------|
| File paths | `data/input/setting.txt` | `Color::BOLD` |
| Counts/quantities | `100 ensembles`, `998,400 particles` | `Color::BOLD` |
| Status words | `completed`, `failed`, `ready` | Color-coded (GREEN/RED/YELLOW) |
| Time values | `2.82s`, `127ms` | `Color::BOLD` |
| Memory sizes | `619 MB`, `47.6 MB` | `Color::BOLD` |

**Example**:
```cpp
std::cout << Color::CYAN << "[GPU] " << Color::RESET
          << "Allocated " << Color::BOLD << "619 MB" << Color::RESET
          << " for meteorological data from "
          << Color::BOLD << "data/gfs/" << Color::RESET << "\n";
```

---

## Summary: Key Principles

1. **Less is more**: Remove 70% of debug output, keep essentials
2. **Color strategically**: Tags always, separators fully, keywords for emphasis
3. **One progress line**: Use `\r` for dynamic updates, not 10 lines for 10/100
4. **Strategic whitespace**: Group related operations, separate major phases
5. **Consistent formatting**: Alignment, units, precision, thousand separators
6. **Actionable errors**: Tell user what to do next
7. **Professional tone**: Scientific software, not chatbot
8. **Function documentation**: Add @output comment describing terminal output
9. **Test in terminal**: Colors must render correctly

**Goal**: Log output should be clear enough that a researcher can diagnose issues without reading source code.

---

## Quick Reference Card

```cpp
// Section header (full color)
std::cout << "\n" << Color::BOLD << Color::CYAN
          << "========================================\n"
          << "  MAJOR SECTION NAME\n"
          << "========================================\n"
          << Color::RESET << "\n";

// Tagged message with emphasis
std::cout << Color::CYAN << "[GPU] " << Color::RESET
          << "Allocated " << Color::BOLD << "619 MB" << Color::RESET << "\n";

// Success with whitespace
std::cout << Color::GREEN << "✓ " << Color::RESET
          << "Operation completed\n\n";

// Error with details
std::cout << Color::RED << Color::BOLD << "[ERROR] " << Color::RESET
          << "Failed to open " << Color::BOLD << filename << Color::RESET << "\n"
          << "  → Check file exists and has read permissions\n\n";

// Progress (single line update)
std::cout << "\rProcessing: " << i << "/" << total
          << " (" << (i*100/total) << "%)" << std::flush;

// Debug (conditional)
#ifdef DEBUG
    std::cout << Color::YELLOW << "[DEBUG] " << Color::RESET
              << "Variable value: " << value << "\n";
#endif
```

---

## Work Distribution for Parallel Refactoring

To maximize efficiency, this refactoring can be done across **3 parallel Claude sessions**. Below is the recommended file distribution based on line count and logical grouping.

### **Session 1: C++ Core Files** (~4,673 lines)

Main entry point and core simulation logic

**Files**:
1. `src/main_eki.cu` (691 lines) - Main entry point
2. `src/include/ldm_func.cuh` (1,523 lines) - Core simulation functions
3. `src/include/ldm_mdata.cuh` (1,942 lines) - Meteorological data loading
4. `src/include/ldm_init.cuh` (860 lines) - Particle initialization
5. `src/include/ldm.cuh` (770 lines) - LDM class

**Priority Tasks**:
- Improve main section headers with `Color::BOLD + Color::CYAN`
- Remove thread IDs from meteorological loading messages
- Consolidate `[EKI_ENSEMBLE]` → `[ENSEMBLE]`, reduce 10-line progress to 1 line
- Wrap `[NaN_CHECK]` in `#ifdef DEBUG`
- Replace hardcoded `[32m` → `Color::GREEN`

**Function Documentation**: Add `@output` annotations to functions with console output

---

### **Session 2: C++ IPC & Support + Utilities** (~2,810 lines)

IPC communication, VTK output, debugging support, and utility scripts

**C++ Files**:
1. `src/include/ldm_eki_ipc.cuh` (682 lines) - IPC communication
2. `src/include/ldm_plot.cuh` (818 lines) - VTK output
3. `src/include/memory_doctor.cuh` (251 lines) - Memory debugging
4. `src/include/ldm_cram2.cuh` (232 lines) - CRAM system

**Python Utility Files**:
5. `util/cleanup.py` (308 lines) - Data cleanup
6. `util/compare_all_receptors.py` (507 lines) - Visualization
7. `util/compare_logs.py` (161 lines) - Log comparison
8. `util/diagnose_convergence_issue.py` (152 lines) - Convergence diagnostics

**Priority Tasks**:
- Consolidate `[EKI_OBS]`, `[EKI_ENSEMBLE_OBS]` → `[ENSEMBLE]`
- Simplify `[GPU_ALLOC]` → `[GPU]`, wrap `[GPU_VERIFY]` in `#ifdef DEBUG`
- Add `[IPC]` tag with `Color::BLUE` for shared memory operations
- Add `[SYSTEM]` tag with `Color::CYAN` for file operations (cleanup.py)
- Improve Python utility output with colorama

**Python Colors**: Use `from colorama import Fore, Style, init`

---

### **Session 3: Python EKI Framework** (~3,384 lines)

Python inverse modeling algorithms and IPC

**Files**:
1. `src/eki/Model_Connection_np_Ensemble.py` (880 lines) - Forward model interface
2. `src/eki/Optimizer_EKI_np.py` (594 lines) - Kalman inversion algorithms
3. `src/eki/eki_ipc_reader.py` (399 lines) - IPC reader
4. `src/eki/RunEstimator.py` (343 lines) - Main estimator
5. `src/eki/eki_ipc_writer.py` (225 lines) - IPC writer
6. `src/eki/memory_doctor.py` (213 lines) - Memory debugging
7. `src/eki/Model_Connection_np.py` (171 lines) - Single model interface
8. `src/eki/Model_Connection_GPU.py` (153 lines) - GPU model interface
9. `src/eki/server.py` (36 lines) - Server (if used)

**Priority Tasks** (CRITICAL):
- **Reduce massive observation output**: 100+ lines per timestep → save to file + 1-line summary
- Add colorama imports: `from colorama import Fore, Style, init; init(autoreset=True)`
- Apply `[ENSEMBLE]` tag with `Fore.MAGENTA`
- Apply `[ERROR]` tag with `Fore.RED + Style.BRIGHT`
- Add success checkmarks: `Fore.GREEN + "✓ "`
- Make errors actionable: add "→ Check if..." suggestions

**Top Priority**: Reduce observation data output (target: 70% log file size reduction)

---

### Common Instructions (All Sessions)

Read `OUTPUT_STYLE_GUIDE.md` and improve output messages in assigned files.

**Phase 1 Tasks (Required)**:
1. Remove deprecated tags: `[DEBUG_BLOCK*]`, `[DEBUG_ITER*]`, `[EKI_*]` variants
2. Replace hardcoded ANSI codes with `Color::` constants
3. Wrap verbose output in `#ifdef DEBUG`
4. Improve progress indicators (10 lines → 1 line, use `\r`)

**Phase 2 Tasks (Recommended)**:
5. Add colors to tags
6. Add checkmarks (✓)
7. Emphasize keywords (numbers, file paths)
8. Improve section headers

**Important**: Improve readability without losing information.

---

### Coordination Notes

- Each session works **independently** on its assigned files
- All sessions reference the same `OUTPUT_STYLE_GUIDE.md`
- Test locally before committing: `make clean && make && ./ldm-eki`
- Commit format: `git commit -m "Refactor output: [Session X] [file list]"`
- No merge conflicts expected (each session has distinct files)

---

### Progress Tracking

After completing your session's work, update this checklist:

**Session 1 (C++ Core)**:
- [ ] `src/main_eki.cu`
- [ ] `src/include/ldm_func.cuh`
- [ ] `src/include/ldm_mdata.cuh`
- [ ] `src/include/ldm_init.cuh`
- [ ] `src/include/ldm.cuh`

**Session 2 (C++ IPC & Utils)**:
- [ ] `src/include/ldm_eki_ipc.cuh`
- [ ] `src/include/ldm_plot.cuh`
- [ ] `src/include/memory_doctor.cuh`
- [ ] `src/include/ldm_cram2.cuh`
- [ ] `util/cleanup.py`
- [ ] `util/compare_all_receptors.py`
- [ ] `util/compare_logs.py`
- [ ] `util/diagnose_convergence_issue.py`

**Session 3 (Python EKI)**:
- [ ] `src/eki/Model_Connection_np_Ensemble.py`
- [ ] `src/eki/Optimizer_EKI_np.py`
- [ ] `src/eki/eki_ipc_reader.py`
- [ ] `src/eki/RunEstimator.py`
- [ ] `src/eki/eki_ipc_writer.py`
- [ ] `src/eki/memory_doctor.py`
- [ ] `src/eki/Model_Connection_np.py`
- [ ] `src/eki/Model_Connection_GPU.py`
- [ ] `src/eki/server.py`
