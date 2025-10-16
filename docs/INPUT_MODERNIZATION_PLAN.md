# LDM-EKI Input File Modernization Plan

**Version:** 1.0
**Date:** 2025-10-16
**Author:** Code Modernization Specialist

---

## Executive Summary

This document outlines a comprehensive plan to modernize, streamline, and optimize the input file structure for the LDM-EKI simulation system. The current input configuration suffers from:

- **Redundancy**: Multiple files containing overlapping settings
- **Obsolete variables**: Network settings, unused radionuclide arrays
- **Poor organization**: Physics constants mixed with user settings
- **Lack of clarity**: Minimal documentation, inconsistent formats
- **Language inconsistency**: Mix of Korean comments and English keys

**Goal**: Transform the input system into a user-friendly, well-documented, and maintainable configuration structure with zero code breakage.

---

## Current State Analysis

### Existing Input Files

| File | Size | Purpose | Issues |
|------|------|---------|--------|
| `input/setting.txt` | 113 lines | Main simulation parameters | Contains unused network settings, physics constants, redundant radionuclide arrays |
| `input/eki_settings.txt` | 152 lines | EKI algorithm configuration | Well-structured but could be clearer |
| `input/source.txt` | 32 lines | Source location and terms | Contains unused `SOURCE_TERM` and `RELEASE_CASES` sections |
| `input/nuclides_config_1.txt` | 1 line | Nuclide definition | Minimal format, works but undocumented |
| `input/nuclides_config_60.txt` | Many lines | 60-nuclide decay chain | Not used in typical runs |

### Code Analysis Results

#### ✅ **USED Variables**

**From `setting.txt`:**
- `Time_end(s)`, `dt(s)`, `Plot_output_freq`, `Total_number_of_particle`
- `Rural/Urban`, `Pasquill-Gifford/Briggs-McElroy-Pooler`, `Data`
- Physics model switches: `turbulence_model`, `dry_deposition_model`, `wet_deposition_model`, `radioactive_decay_model`
- File paths: `input_base_path`, `output_base_path`, `ldaps_data_path`, `gfs_data_path`
- Terminal settings: `fixed_scroll_output`
- Grid dimensions: `dimX`, `dimY`, `dimZ_pres`, etc.

**From `eki_settings.txt`:**
- All EKI parameters are actively used
- Receptor configuration
- Emission time series
- Algorithm options

**From `source.txt`:**
- `[SOURCE]` section: lon, lat, height (actively used)

**From `nuclides_config_*.txt`:**
- Nuclide name, decay constant, deposition velocity

#### ❌ **UNUSED Variables** (To be removed)

**From `setting.txt`:**
```ini
# NETWORK SETTINGS # ← COMPLETELY UNUSED
socket_port: 8080
tcp_port: 12345
server_ip: 127.0.0.1

# RADIONUCLIDE DATA # ← REPLACED BY nuclides_config_*.txt
species_names: Zr-95,I-131,Cs-137,I-132
decay_constants: 1.252e-7,9.99e-7,1.00e-6,1.00e-6
deposition_velocities: 0.1,0.01,0.01,0.02
particle_sizes: 1.86,0.6,0.6,0.40
particle_densities: 5680.0,2500.0,2500.0,2500.0
size_standard_deviations: 3.8,0.01,0.01,4.8

# SPECIFIC REGION BOUNDARIES # ← UNUSED
region_lon_min: 141.0174
region_lon_max: 141.0288
region_lat_min: 37.4144
region_lat_max: 37.4234

# COORDINATE CONVERSION # ← DUPLICATE OF GRID SETTINGS
start_lat: 36.0  (appears twice)
start_lon: 140.0 (appears twice)
end_lat: 37.0
end_lon: 141.0
lat_step: 0.5
lon_step: 0.5

# SOME PHYSICAL CONSTANTS # ← CAN BE MOVED TO CODE
pi: 3.141592
reference_temperature: 293.15
etc.
```

**From `source.txt`:**
```ini
[GRID_CONFIG]         # ← DUPLICATE OF setting.txt
[SOURCE_TERM]         # ← UNUSED (nuclides_config is used instead)
[RELEASE_CASES]       # ← UNUSED
```

#### ⚠️ **HARDCODED IN CODE** (Should be moved to config)

Currently hardcoded values that should be user-configurable:
- EKI receptor capture radius default
- Default particle properties
- Some meteorological parameters

---

## Proposed New Structure

### Philosophy

1. **Separation of Concerns**: Divide by functional area
2. **User vs. Developer**: Separate frequently-changed user settings from rarely-changed system parameters
3. **Self-Documenting**: Every setting has inline explanation
4. **English Only**: All keys, values, and comments in English
5. **Consistent Format**: Maintain `KEY: value` format for easy parsing
6. **Beautiful Layout**: Clean sections with visual separators

### New File Organization

```
input/
├── simulation.conf          # Main user-facing simulation parameters
├── physics.conf             # Physics model configuration
├── eki.conf                 # EKI algorithm settings (modernized from eki_settings.txt)
├── source.conf              # Source term definition
├── nuclides.conf            # Radionuclide properties (replaces nuclides_config_*.txt)
├── advanced.conf            # Advanced/rarely-changed parameters
└── gfsdata/                 # Meteorological data (unchanged)
```

### Detailed File Specifications

---

## 📄 **File 1: `simulation.conf`**

**Purpose**: Primary user interface for simulation setup

**Content**:
```ini
################################################################################
#                    LDM-EKI SIMULATION CONFIGURATION
################################################################################
# This file contains the main user-facing simulation parameters.
# Modify these settings to control simulation duration, particle count,
# and basic atmospheric conditions.
#
# Units are specified in parentheses for each parameter.
# Boolean flags: 0 = OFF, 1 = ON
################################################################################

# ==============================================================================
# TEMPORAL SETTINGS
# ==============================================================================
# Defines simulation time span and temporal resolution

# Total simulation duration (seconds)
# Example: 21600 = 6 hours
time_end: 21600.0

# Time step for particle advancement (seconds)
# Smaller values = more accurate but slower
# Typical range: 10-100 seconds
time_step: 100.0

# Output frequency for VTK visualization files
# How many time steps between each output file
# Example: 1 = output every time step, 10 = every 10th step
vtk_output_frequency: 1

# ==============================================================================
# PARTICLE SETTINGS
# ==============================================================================

# Total number of simulation particles
# More particles = better statistics but slower computation
# Typical range: 1,000 - 1,000,000
# Note: Will be distributed across ensemble members in EKI mode
total_particles: 10000

# ==============================================================================
# ATMOSPHERIC CONDITIONS
# ==============================================================================

# Turbulence calculation mode
# 1 = Rural conditions (higher turbulence)
# 0 = Urban conditions (lower turbulence due to buildings)
rural_conditions: 1

# Stability category method
# 1 = Pasquill-Gifford (traditional, simpler)
# 0 = Briggs-McElroy-Pooler (more detailed)
use_pasquill_gifford: 1

# ==============================================================================
# METEOROLOGICAL DATA SOURCE
# ==============================================================================

# Meteorological data format
# 1 = GFS (Global Forecast System, 0.5° resolution)
# 0 = LDAPS (Korean regional model, higher resolution)
use_gfs_data: 1

# ==============================================================================
# TERMINAL OUTPUT
# ==============================================================================

# Fixed-height scrolling output mode
# 1 = Keep output within terminal height (cleaner)
# 0 = Scroll output continuously (shows full history)
fixed_scroll_output: 1

################################################################################
# End of simulation.conf
################################################################################
```

---

## 📄 **File 2: `physics.conf`**

**Purpose**: Physics model activation and physical constants

**Content**:
```ini
################################################################################
#                    PHYSICS MODEL CONFIGURATION
################################################################################
# Control which physical processes are included in the simulation.
# Toggle individual models on/off to study their effects.
#
# All flags: 1 = ENABLED, 0 = DISABLED
################################################################################

# ==============================================================================
# PHYSICS MODEL SWITCHES
# ==============================================================================

# Turbulent diffusion model
# Simulates random particle motion due to atmospheric turbulence
turbulence_model: 0

# Dry deposition model
# Gravitational settling and surface deposition
dry_deposition_model: 0

# Wet deposition model
# Removal by precipitation (rain, snow)
wet_deposition_model: 0

# Radioactive decay model
# Time-dependent decay using CRAM (Chebyshev Rational Approximation Method)
# Recommended: Keep ON for realistic radionuclide transport
radioactive_decay_model: 1

# ==============================================================================
# PHYSICAL CONSTANTS
# ==============================================================================
# Advanced users only - these rarely need modification

# Gravitational acceleration (m/s²)
gravity: 9.81

# Specific gas constant for dry air (J/(kg·K))
air_gas_constant: 287.05

# Reference temperature for atmospheric calculations (K)
reference_temperature: 293.15

# Boltzmann constant (J/K)
boltzmann_constant: 1.38e-23

# Dynamic viscosity of air (Pa·s)
dynamic_viscosity: 1.81e-5

# Kinematic viscosity of air (m²/s)
kinematic_viscosity: 0.15e-4

# Earth's mean radius (m)
earth_radius: 6371000.0

# Reference height for wind profile (m)
reference_height: 15.0

# Turbulence scaling factors
turbulence_mesoscale: 0.16
troposphere_factor: 50.0
stratosphere_factor: 0.1

# ==============================================================================
# DEFAULT PARTICLE PROPERTIES
# ==============================================================================
# Used when not specified in nuclides.conf

# Default particle density (kg/m³)
# Typical: 2500 for aerosols, 5680 for heavy metals
default_particle_density: 2500.0

# Default particle radius (m)
# Example: 6.0e-7 = 0.6 micrometers
default_particle_radius: 6.0e-7

# Default size distribution standard deviation
default_sigma: 3.0e-1

################################################################################
# End of physics.conf
################################################################################
```

---

## 📄 **File 3: `eki.conf`**

**Purpose**: Ensemble Kalman Inversion configuration (cleaned up version of `eki_settings.txt`)

**Content**:
```ini
################################################################################
#                    EKI ALGORITHM CONFIGURATION
################################################################################
# Ensemble Kalman Inversion (EKI) settings for source term estimation.
# This file controls the inverse modeling algorithm that estimates unknown
# emission rates from receptor observations.
################################################################################

# ==============================================================================
# TIME DISCRETIZATION
# ==============================================================================

# Time interval for emission rate estimation
# Emission rates are assumed constant within each interval
time_interval: 15.0

# Time unit for the interval
# Options: seconds, minutes, hours
time_unit: minutes

# ==============================================================================
# RECEPTOR CONFIGURATION
# ==============================================================================

# Number of monitoring receptors (observation points)
num_receptors: 3

# Receptor locations (latitude, longitude)
# Format: One location per line after the "=" line
# Space-separated: LATITUDE LONGITUDE
receptor_locations:
35.6500 129.6000
35.6000 129.8000
35.5000 129.9000

# Receptor capture radius (degrees)
# Particles within this distance from receptor center are counted
# Typical: 0.025° ≈ 2.5 km at mid-latitudes
receptor_capture_radius: 0.025

# ==============================================================================
# TRUE EMISSION TIME SERIES (REFERENCE SIMULATION)
# ==============================================================================
# This is the "ground truth" emission rate used to generate synthetic
# observations for testing the inversion algorithm.
# Units: Becquerels (Bq)
#
# Number of values should match: (simulation_duration / time_interval)
# Example: 6 hours / 15 minutes = 24 time steps

true_emissions:
1.90387731e+13
1.90387731e+13
1.90387731e+12
1.90387731e+11
1.90387731e+4
1.90387731e+3
1.90387731e+2
1.90387731e+1
2.26641204e+13
2.26641204e+13
2.26641204e+12
2.26641204e+11
2.26641204e+4
2.26641204e+3
2.26641204e+2
2.26641204e+1
1.51170139e+13
1.51170139e+13
1.51170139e+12
1.51170139e+11
1.51170139e+4
1.51170139e+3
1.51170139e+2
1.51170139e+1

# ==============================================================================
# PRIOR EMISSION ESTIMATE (INITIAL GUESS)
# ==============================================================================
# The starting point for the inversion algorithm.
# Can be a constant value or a time series.

# Prior mode: "constant" or "series"
prior_mode: constant

# Constant prior value (Bq) - used if prior_mode=constant
prior_constant: 1.0e+12

# Time series prior - used if prior_mode=series
# Uncomment and fill if needed:
# prior_series:
# 1.5e+8
# 1.5e+8
# ...

# ==============================================================================
# EKI ALGORITHM PARAMETERS
# ==============================================================================

# Ensemble size (number of ensemble members)
# Larger ensembles = better uncertainty quantification but slower
# Typical range: 50-200
ensemble_size: 100

# Observation noise level (relative)
# Accounts for measurement uncertainty
# Example: 0.1 = 10% noise
noise_level: 0.1

# Number of EKI iterations
# How many times to update the ensemble
# Typical: 3-10 iterations
max_iterations: 3

# ==============================================================================
# ADVANCED EKI OPTIONS
# ==============================================================================

# Perturbation option (On/Off)
perturb_observations: Off

# Adaptive step size (On/Off)
# Automatically adjusts update magnitude
adaptive_eki: Off

# Covariance localization (On/Off)
# Reduces spurious correlations
localized_eki: Off

# Regularization (On/Off)
# Penalizes overly complex solutions
regularization: Off

# Regularization parameter (0.0-1.0)
# Only used if regularization=On
renkf_lambda: 0.9

# ==============================================================================
# GPU CONFIGURATION
# ==============================================================================

# Use GPU for forward model (On/Off)
gpu_forward: On

# Use GPU for inverse calculations (On/Off)
gpu_inverse: On

# Number of GPUs to use
num_gpus: 2

# ==============================================================================
# SOURCE PARAMETERS
# ==============================================================================

# Source location mode
# Options: "Fixed" (known location) or "Unknown" (estimate location too)
source_location: Fixed

# Number of sources
num_sources: 1

# ==============================================================================
# OBSERVATION ERROR PARAMETERS
# ==============================================================================

# Receptor measurement error (relative)
receptor_error: 0.0

# Minimum Detectable Activity (MDA) level
receptor_mda: 0.0

# ==============================================================================
# DEBUGGING
# ==============================================================================

# Memory Doctor Mode (On/Off)
# Logs all IPC data transfers for debugging
# Warning: Creates large debug files in /tmp/eki_debug/
memory_doctor_mode: Off

################################################################################
# End of eki.conf
################################################################################
```

---

## 📄 **File 4: `source.conf`**

**Purpose**: Source location definition (simplified from `source.txt`)

**Content**:
```ini
################################################################################
#                    SOURCE TERM DEFINITION
################################################################################
# Defines the location and height of emission sources.
# Multiple sources can be specified (one per line).
#
# Format: LONGITUDE LATITUDE HEIGHT
# Units: degrees, degrees, meters
################################################################################

# ==============================================================================
# SOURCE LOCATION(S)
# ==============================================================================
# Each line defines one source in the format:
#   LONGITUDE  LATITUDE  HEIGHT(m)
#
# Example:
#   129.48  35.71  100.0    ← Source at 129.48°E, 35.71°N, 100m elevation
#
# Multiple sources:
#   129.48  35.71  100.0
#   140.50  37.00  50.0

129.48 35.71 100.0

################################################################################
# End of source.conf
################################################################################
```

---

## 📄 **File 5: `nuclides.conf`**

**Purpose**: Radionuclide properties (replaces `nuclides_config_*.txt`)

**Content**:
```ini
################################################################################
#                    RADIONUCLIDE CONFIGURATION
################################################################################
# Defines the properties of radionuclides being simulated.
#
# Format: NAME  DECAY_CONSTANT  DEPOSITION_VELOCITY
#
# Units:
#   DECAY_CONSTANT: per second (1/s)
#   DEPOSITION_VELOCITY: meters per second (m/s)
#
# To calculate decay constant from half-life:
#   λ = ln(2) / T_half
#   Example: Co-60 has T_half = 5.27 years
#            λ = 0.693 / (5.27 × 365.25 × 24 × 3600) = 4.168e-9 /s
################################################################################

# ==============================================================================
# SINGLE NUCLIDE MODE
# ==============================================================================
# For single-nuclide simulations, specify one line:
#
# NUCLIDE_NAME  DECAY_CONSTANT(/s)  DEPOSITION_VELOCITY(m/s)

Co-60  4.168e-9  1.0e-3

# ==============================================================================
# MULTI-NUCLIDE / DECAY CHAIN MODE
# ==============================================================================
# For decay chain simulations (e.g., I-131 → Xe-131m → Xe-131):
#
# Uncomment and add lines for each nuclide in the chain:
# I-131   9.99e-7   1.0e-3
# Xe-131m 2.84e-6   0.0
# Xe-131  0.0       0.0

# ==============================================================================
# COMMON RADIONUCLIDES
# ==============================================================================
# Uncomment as needed:
#
# Cs-137  7.30e-10  1.0e-3   (half-life: 30.17 years)
# Sr-90   7.63e-10  1.0e-4   (half-life: 28.8 years)
# I-131   9.99e-7   1.0e-2   (half-life: 8.02 days)
# Xe-133  1.52e-6   0.0      (half-life: 5.24 days, noble gas)
# Kr-85   2.05e-9   0.0      (half-life: 10.76 years, noble gas)

################################################################################
# End of nuclides.conf
################################################################################
```

---

## 📄 **File 6: `advanced.conf`**

**Purpose**: System parameters rarely modified by users

**Content**:
```ini
################################################################################
#                    ADVANCED SYSTEM CONFIGURATION
################################################################################
# This file contains advanced parameters that rarely need modification.
# Only edit these if you understand the internals of the LDM-EKI system.
#
# WARNING: Incorrect values may cause simulation failures or incorrect results.
################################################################################

# ==============================================================================
# FILE PATHS
# ==============================================================================

# Base directory for input files (relative or absolute)
input_base_path: ./input/

# Base directory for output files
output_base_path: ./output/

# LDAPS meteorological data path (if using LDAPS)
ldaps_data_path: ./input/ldapsdata/

# GFS meteorological data path (if using GFS)
gfs_data_path: ./input/gfsdata/

# ==============================================================================
# GRID DIMENSIONS
# ==============================================================================
# Internal computational grid sizes (usually set by meteorological data)

# LDAPS grid dimensions
ldaps_dimX: 602
ldaps_dimY: 781
ldaps_dimZ_pres: 24
ldaps_dimZ_etas: 71

# GFS grid dimensions
gfs_dimX: 720
gfs_dimY: 361
gfs_dimZ: 26

# ==============================================================================
# COORDINATE SYSTEM
# ==============================================================================

# LDAPS domain boundaries (degrees)
ldaps_lon_min: 121.06
ldaps_lon_max: 132.36
ldaps_lat_min: 32.20
ldaps_lat_max: 43.13

# Coordinate transformation parameters
lon_offset: -179.0
lat_offset: -90.0
coord_scale: 0.5
altitude_scale: 3000.0

# Vertical level spacing (meters)
vertical_interval: 10.0

# Default altitude for surface releases (meters)
default_altitude: 20.0

################################################################################
# End of advanced.conf
################################################################################
```

---

## Implementation Roadmap

### Phase 1: Code Preparation (Week 1)
**Goal**: Update parsing code to handle new file structure

1. **Create new parser functions** (`src/init/ldm_init_config.cu`)
   - `loadSimulationConfig()` - for `simulation.conf`
   - `loadPhysicsConfig()` - for `physics.conf`
   - `loadSourceConfig()` - for `source.conf`
   - `loadNuclidesConfig()` - for `nuclides.conf`
   - `loadAdvancedConfig()` - for `advanced.conf`
   - `loadEKIConfig()` - updated for `eki.conf`

2. **Add config migration helper**
   - Function to detect old file format
   - Automatic conversion to new format with warnings

3. **Update struct definitions** (`src/data/config/ldm_struct.cuh`)
   - Clean up global config structures
   - Remove unused fields (network settings, etc.)

### Phase 2: File Creation (Week 1)
**Goal**: Generate new input files

1. **Create template generator script**
   - `util/generate_config_templates.py`
   - Generates all new `.conf` files with comments

2. **Write migration script**
   - `util/migrate_legacy_inputs.py`
   - Reads old `setting.txt`, `eki_settings.txt`, `source.txt`
   - Writes new `.conf` files
   - Validates conversion

### Phase 3: Testing (Week 2)
**Goal**: Verify zero-impact on simulation results

1. **Regression testing**
   - Run identical simulation with old vs. new inputs
   - Compare outputs bit-by-bit
   - Validate EKI convergence

2. **Edge case testing**
   - Missing optional parameters
   - Malformed config files
   - Multi-source configurations

### Phase 4: Documentation (Week 2)
**Goal**: User-facing documentation

1. **Update CLAUDE.md**
   - New "Input File Reference" section
   - Migration guide from legacy files

2. **Create user guide**
   - `docs/INPUT_FILE_GUIDE.md`
   - Parameter descriptions with examples
   - Common configuration recipes

### Phase 5: Cleanup (Week 3)
**Goal**: Remove legacy code

1. **Deprecate old files**
   - Move to `input/legacy/` folder
   - Add deprecation warnings if old files detected

2. **Remove dead code**
   - Delete parsing for network settings
   - Clean up unused radionuclide array code

---

## Benefits Summary

### For Users
✅ **Clarity**: Every parameter has clear documentation
✅ **Safety**: Dangerous parameters isolated in `advanced.conf`
✅ **Simplicity**: Main workflow uses only `simulation.conf` and `eki.conf`
✅ **Examples**: Inline examples show typical values

### For Developers
✅ **Maintainability**: Logical file organization
✅ **Debuggability**: No mysterious unused variables
✅ **Extensibility**: Easy to add new parameters
✅ **Performance**: No parsing of obsolete settings

### For the Codebase
✅ **Modernization**: Industry-standard config structure
✅ **Reliability**: Type-safe parsing with validation
✅ **Consistency**: Uniform naming conventions
✅ **Documentation**: Self-documenting configuration

---

## Risk Mitigation

### Backward Compatibility
- ⚠️ Old files will be deprecated but still supported for 1 release cycle
- ⚠️ Migration script automates conversion
- ⚠️ Warning messages guide users to upgrade

### Testing Strategy
- ✅ Automated regression tests
- ✅ Bit-exact output verification
- ✅ User acceptance testing with real scenarios

### Rollback Plan
- 🔄 Git tag before implementation: `v1.0-legacy-inputs`
- 🔄 Keep old parsing code in commented form
- 🔄 Easy revert path documented

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Approve file structure** and naming conventions
3. **Begin Phase 1** implementation
4. **Schedule** testing window

---

## Appendix: Variable Removal List

### Complete list of variables to be removed from code:

**From `setting.txt` → DELETED:**
```
socket_port
tcp_port
server_ip
species_names (use nuclides.conf instead)
decay_constants (use nuclides.conf instead)
deposition_velocities (use nuclides.conf instead)
particle_sizes (deprecated - use nuclides.conf)
particle_densities (deprecated - use nuclides.conf)
size_standard_deviations (deprecated - use nuclides.conf)
region_lon_min
region_lon_max
region_lat_min
region_lat_max
```

**From `source.txt` → DELETED:**
```
[GRID_CONFIG] section
[SOURCE_TERM] section
[RELEASE_CASES] section
```

**Total reduction**: ~40 lines of unnecessary configuration
**Code cleanup**: ~200 lines of parsing code can be removed

---

**END OF PLAN**
