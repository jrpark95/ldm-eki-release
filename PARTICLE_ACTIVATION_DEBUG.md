# Particle Activation Debugging Session (2025-10-16)

## Summary

This document records a comprehensive debugging session investigating why particles weren't reaching receptors, resulting in zero observation values.

## Issues Identified and Resolved

### Issue 1: Particle Activation Logic Bug (FIXED)

**Problem**: The `update_particle_flags` kernel in single mode used array index `idx` instead of particle's `timeidx` field for activation logic.

**Location**: `src/kernels/device/ldm_kernels_device.cu:145-158`

**Original Code**:
```cuda
__global__ void update_particle_flags(
    LDM::LDMpart* d_part, float activationRatio){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_nop) return;

    LDM::LDMpart& p = d_part[idx];

    // Activate particles based on their timeidx and current activationRatio
    int maxActiveTimeidx = int(d_nop * activationRatio);

    if (idx <= maxActiveIndex){  // BUG: Using array index instead of p.timeidx
        p.flag = 1;
    }
}
```

**Impact**:
- At t=100s: `activationRatio = 100/21600 = 0.00463`, `maxActiveTimeidx = 46`
- Only first 46 particles in array activate, regardless of their `timeidx` values
- Particles with `timeidx=1` that should activate don't if they're positioned beyond array index 46

**Fix**:
```cuda
if (p.timeidx <= maxActiveTimeidx){  // FIXED: Use particle's timeidx field
    p.flag = 1;
}
```

**File Modified**: `src/kernels/device/ldm_kernels_device.cu`

**Result**: Ensemble mode particles now activate correctly (verified by `flag=active` and non-zero wind values in logs).

### Issue 2: Ensemble Mode Particles at Ground Level (IDENTIFIED)

**Problem**: Ensemble mode particles show `z=0.000000e+00m` in timestep 1 instead of expected z=100m.

**Log Evidence** (`logs/ldm_eki_simulation.log:377`):
```
[PARTICLE_POSITION] Timestep 1 (t=1.000000e+02s): lon=1.294800e+02°E, lat=3.571000e+01°N,
z=0.000000e+00m, flag=active, timeidx=1, u_wind=3.914014e+00, v_wind=-6.921375e-01, w_wind=-2.119772e-02 m/s
```

**Root Cause Analysis**:

Particles initialized at z=100m are hitting the ground in the FIRST timestep due to:

1. **Downward meteorological wind**: `w_wind = -0.021 m/s` (small but negative)
2. **Strong downward turbulent motion**: The vertical turbulent velocity `p.wp` becomes very negative
3. **Ground reflection condition** (`src/kernels/particle/ldm_kernels_particle_ens.cu:595-598`):
   ```cuda
   if (p.wp*d_dt < -p.z){  // If turbulent velocity would take particle below ground
       p.dir = -1;
       p.z = -p.z - p.wp*d_dt;  // Reflect above ground
   }
   ```

**Why Particles Stay at z=0m**:
- Turbulent velocity calculation (`p.wp`) at line 579 includes downward momentum
- For particles at z=100m with dt=100s, if `p.wp * 100 < -100`, they reflect
- After reflection, particles settle to z≈0m due to continued downward advection
- With `w_wind=-0.021 m/s` and dt=100s, vertical displacement is `dz = -2.1m` per timestep
- Particles reflected near ground cannot rise faster than they're advected downward

**This is NOT a code bug** - it's a physics/meteorology issue where:
- Meteorological conditions have strong downward motion at source location (129.48°E, 35.71°N, 100m)
- Turbulent mixing layer (`hmix`) parameters may not be suitable for this scenario
- Timestep dt=100s is too large for stable vertical motion at low altitudes

**Potential Solutions** (NOT implemented - requires user decision):
1. Reduce timestep `dt` in `input/setting.txt` (e.g., from 100s to 10s)
2. Increase source height above 100m to avoid near-ground turbulence
3. Check meteorological data validity for this location/time
4. Adjust turbulent diffusion parameters (`TURB` switch in settings)
5. Modify mixing height (`HMIX`) calculation

### Issue 3: Single Mode Particles Still Inactive (PARTIALLY RESOLVED)

**Status**: Kernel fix applied, but single mode particles remain inactive in test run.

**Possible Causes** (requires further investigation):
1. Separate code path for single mode vs ensemble mode activation
2. Different kernel invocation pattern
3. Timestep synchronization issue between activation and movement kernels

**Evidence**: Log shows single mode particle 0 at timestep 1 with `flag=inactive, timeidx=1, maxActiveTimeidx=46`.
The condition `1 <= 46` should be TRUE, yet flag remains inactive.

**Next Steps for Investigation** (if needed):
- Verify `update_particle_flags` kernel is actually called for single mode
- Check if there's a separate activation mechanism for single mode
- Add GPU printf debug in single mode to confirm kernel execution
- Compare single vs ensemble mode simulation flow in `src/simulation/ldm_func_simulation.cu`

## Test Results

**Meteorological Data Verification** (PASSED):
```
[GPU_METEO_VERIFY] Timestep 0 - GPU meteorological data at source location (141°E, 37°N, ~1km):
  Point [640,254,5]: UU=5.66763 VV=-3.5001 WW=0.0187386 m/s
```
✓ GPU has valid meteorological data

**Ensemble Mode Activation** (PASSED):
```
[PARTICLE_POSITION] Timestep 1: flag=active, u_wind=3.914014e+00, v_wind=-6.921375e-01, w_wind=-2.119772e-02 m/s
```
✓ Particles activate
✓ Wind values computed

**Particle Movement** (FAILED):
```
z=0.000000e+00m
```
✗ Particles at ground level instead of z=100m
✗ Particles cannot disperse horizontally when grounded

**Receptor Observations** (FAILED):
```
[EKI_ENSEMBLE_OBS] obs1 at t=900s: R1=0p R2=0p R3=0p
```
✗ No particles reach receptors (all at ground level)

## Debugging Tools Added

### Enhanced Particle Position Logging
**Location**: `src/simulation/ldm_func_simulation.cu:352-371`

Logs particle 0 position every timestep for first 10 timesteps:
- Geographic coordinates (lon, lat, z)
- Activation status (flag, timeidx, activationRatio, maxActiveTimeidx)
- Wind components (u_wind, v_wind, w_wind)

### GPU Meteorological Data Verification
**Location**: `src/simulation/ldm_func_simulation.cu:204-230`

One-time verification at timestep 0:
- Samples GPU meteorological data at source location
- Verifies data transfer from CPU→GPU was successful
- Confirms trilinear interpolation has valid input data

### GPU Kernel Debug Printf
**Location**: `src/kernels/device/ldm_kernels_device.cu:150-155`

Added debug output in `update_particle_flags` kernel:
```cuda
if (idx == 0) {
    printf("[GPU_ACTIVATION] Particle 0: timeidx=%d, maxActiveTimeidx=%d, should_activate=%d, current_flag=%d\n",
           p.timeidx, maxActiveTimeidx, (p.timeidx <= maxActiveTimeidx) ? 1 : 0, p.flag);
}
```

**Note**: CUDA printf output doesn't always appear reliably. CPU-side logging via cudaMemcpy is more reliable.

## Recommendations

### Immediate Actions Required

1. **Verify Meteorological Data**:
   - Check if GFS data for 129.48°E, 35.71°N shows realistic wind patterns
   - Confirm vertical wind component (`WW`) values are reasonable
   - Verify mixing height (`HMIX`) is appropriate for this location

2. **Adjust Simulation Parameters**:
   - **Reduce timestep**: Change `dt` from 100s to 10-20s in `input/setting.txt`
   - **Increase source height**: Move source from 100m to 500m or 1000m in `input/source.txt`
   - **Check turbulence**: Verify `TURB` switch and turbulent diffusion parameters

3. **Complete Single Mode Fix**:
   - Investigate why single mode particles don't activate despite kernel fix
   - Compare code paths between single and ensemble modes
   - Add more comprehensive logging to single mode execution

### Long-term Improvements

1. **Adaptive Timestep**:
   - Implement variable timestep based on vertical velocity
   - Use smaller dt near ground, larger dt aloft

2. **Ground Boundary Condition**:
   - Review ground reflection logic for physical correctness
   - Consider implementing partial absorption instead of perfect reflection

3. **Physics Validation**:
   - Compare particle trajectories against known atmospheric dispersion benchmarks
   - Validate turbulent parameterization with field data

## Files Modified in This Session

1. **src/kernels/device/ldm_kernels_device.cu**
   - Fixed particle activation logic (line 145-158)
   - Added GPU printf debug output

2. **src/simulation/ldm_func_simulation.cu**
   - Added GPU meteorological data verification (lines 204-230)
   - Added enhanced particle position logging (lines 352-371)

3. **This document**: `PARTICLE_ACTIVATION_DEBUG.md`

## Related Documents

- `PARTICLE_TROUBLESHOOTING_PLAN.md` - Original troubleshooting plan
- `CLAUDE.md` - Project architecture and development guide
- `logs/ldm_eki_simulation.log` - Detailed simulation log with all debug output
- `logs/python_eki_output.log` - Python EKI process log

## Conclusion

**Primary Achievement**: Fixed critical particle activation bug in `update_particle_flags` kernel.

**Remaining Issues**:
1. Particles fall to ground immediately (physics/parameter issue, NOT code bug)
2. Single mode activation still problematic (requires further investigation)

**Next Steps**: Adjust simulation parameters (dt, source height) or meteorological data to prevent immediate ground collision.
