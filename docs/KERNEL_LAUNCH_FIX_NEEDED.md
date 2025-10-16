# Critical Fix Required: Kernel Launch Sites Missing KernelScalars Parameter

**Date:** 2025-10-16
**Branch:** `fix/nonrdc-kparam-dptr`
**Severity:** CRITICAL - Particles not moving

## Problem

Particles are NOT moving because kernel launch sites are calling kernels WITHOUT the required `KernelScalars` parameter.

**What was done:**
- ✅ Removed __constant__ declarations
- ✅ Created KernelScalars struct
- ✅ Updated kernel signatures to accept KernelScalars
- ✅ Updated kernel implementations to use `ks.delta_time`, `ks.num_particles`, etc.

**What's MISSING:**
- ❌ Kernel launch sites not passing KernelScalars parameter
- ❌ Result: Kernels receive uninitialized struct with garbage values
- ❌ Consequence: `dt` is garbage → particles don't move → all doses = 0

## Required Changes

### File: `src/simulation/ldm_func_simulation.cu`

**3 kernel launch sites need updating:**

1. **Line ~81** - `runSimulation()` single mode
2. **Line ~323** - `runSimulation_eki()` ensemble mode
3. **Line ~332** - `runSimulation_eki()` single mode
4. **Line ~682** - `runSimulation_eki_dump()` ensemble mode
5. **Line ~691** - `runSimulation_eki_dump()` single mode

### Pattern to Apply

Before EACH kernel launch, add:

```cpp
// Populate KernelScalars
KernelScalars ks{};
ks.turb_switch = g_turb_switch;
ks.drydep = g_drydep;
ks.wetdep = g_wetdep;
ks.raddecay = g_raddecay;
ks.num_particles = nop;
ks.is_rural = isRural ? 1 : 0;
ks.is_pg = isPG ? 1 : 0;
ks.is_gfs = isGFS ? 1 : 0;
ks.delta_time = dt;
Grid Config grid_config = loadGridConfig();
ks.grid_start_lat = grid_config.start_lat;
ks.grid_start_lon = grid_config.start_lon;
ks.grid_lat_step = grid_config.lat_step;
ks.grid_lon_step = grid_config.lon_step;
ks.settling_vel = vsetaver;
ks.cunningham_fac = cunningham;
```

Then add `, ks` as the LAST parameter to the kernel launch.

### Example Fix

```cpp
// BEFORE (BROKEN):
move_part_by_wind_mpi<<<blocks, threadsPerBlock>>>
(d_part, t0, PROCESS_INDEX, d_dryDep, d_wetDep, mesh.lon_count, mesh.lat_count,
    device_meteorological_flex_unis0,
    device_meteorological_flex_pres0,
    device_meteorological_flex_unis1,
    device_meteorological_flex_pres1);

// AFTER (FIXED):
KernelScalars ks{};
// ... populate ks fields ...

move_part_by_wind_mpi<<<blocks, threadsPerBlock>>>
(d_part, t0, PROCESS_INDEX, d_dryDep, d_wetDep, mesh.lon_count, mesh.lat_count,
    device_meteorological_flex_unis0,
    device_meteorological_flex_pres0,
    device_meteorological_flex_unis1,
    device_meteorological_flex_pres1,
    ks);  // ← ADD THIS
```

### Ensemble Kernel Launches

For `move_part_by_wind_mpi_ens`, add ks as the LAST parameter (after `total_particles`):

```cpp
move_part_by_wind_mpi_ens<<<blocks, threadsPerBlock>>>
(d_part, t0, PROCESS_INDEX, d_dryDep, d_wetDep, mesh.lon_count, mesh.lat_count,
    device_meteorological_flex_unis0,
    device_meteorological_flex_pres0,
    device_meteorological_flex_unis1,
    device_meteorological_flex_pres1,
    total_particles,
    ks);  // ← ADD THIS
```

## Cleanup Also Needed

Remove these failing cudaMemcpyToSymbol calls (symbols no longer exist):

```cpp
// REMOVE (lines 132-139, 247, 606):
err = cudaMemcpyToSymbol(d_start_lat, &start_lat, sizeof(float));
err = cudaMemcpyToSymbol(d_start_lon, &start_lon, sizeof(float));
err = cudaMemcpyToSymbol(d_lat_step, &lat_step, sizeof(float));
err = cudaMemcpyToSymbol(d_lon_step, &lon_step, sizeof(float));
cudaMemcpyToSymbol(d_flex_hgt, flex_hgt.data(), sizeof(float) * dimZ_GFS);
```

---
**This fix is CRITICAL for particle movement to work!**
