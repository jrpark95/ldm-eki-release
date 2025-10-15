# Function Documentation Style Guide

## Standard Function Documentation Template

### For Regular Functions (C++/CUDA Host)

```cpp
/**
 * @brief Calculate particle advection using wind field data
 *
 * @details This function computes the new position of particles based on
 *          wind velocity fields using bilinear interpolation. The function
 *          handles boundary conditions and terrain interactions.
 *
 * @param[in]     particles    Array of particle structures to update
 * @param[in]     wind_u       U-component of wind velocity field [m/s]
 * @param[in]     wind_v       V-component of wind velocity field [m/s]
 * @param[in]     dt           Time step for integration [seconds]
 * @param[in,out] positions    Particle positions to be updated
 * @param[out]    velocities   Calculated particle velocities
 *
 * @return int Status code (0 for success, negative for errors)
 *
 * @note Requires pre-allocated device memory for all arrays
 * @note Wind fields must be in geographic coordinates (lat/lon)
 *
 * @warning Modifies particle positions in-place
 * @warning Not thread-safe for overlapping particle arrays
 *
 * @see launch_advection_kernel() for GPU execution
 * @see interpolate_wind_field() for wind data interpolation
 */
int calculate_advection(Particle* particles, float* wind_u, float* wind_v,
                       float dt, float3* positions, float3* velocities);
```

### For CUDA Kernels

```cuda
/**
 * @kernel move_part_by_wind
 * @brief GPU kernel for particle advection with wind transport
 *
 * @details Processes particle movement in parallel on GPU. Each thread handles
 *          one particle, computing advection, diffusion, and deposition.
 *          Uses shared memory for wind field caching to optimize memory access.
 *
 * @param[in,out] d_part       Device array of particles
 * @param[in]     d_wind       Wind field data structure
 * @param[in]     d_terrain    Terrain elevation data
 * @param[in]     num_parts    Total number of particles
 * @param[in]     dt           Time step [seconds]
 * @param[in]     current_time Simulation time [seconds since start]
 *
 * @grid_config
 *   - Block size: 256 threads (optimal for SM 6.1+)
 *   - Grid size: (num_parts + 255) / 256
 *   - Shared memory: 4KB per block for wind field cache
 *
 * @performance
 *   - Memory throughput: ~80% of theoretical maximum
 *   - Occupancy: 75% on RTX 3090
 *   - Typical runtime: 0.5ms for 1M particles
 *
 * @invariants
 *   - Preserves total particle mass
 *   - Maintains particle ordering
 *   - Energy conservation within 0.01%
 */
__global__ void move_part_by_wind(LDMpart* d_part, WindField* d_wind,
                                  float* d_terrain, int num_parts,
                                  float dt, float current_time);
```

### For Inline Device Functions

```cuda
/**
 * @device bilinear_interpolation
 * @brief Perform bilinear interpolation on 2D field
 *
 * @details Fast inline interpolation for grid-based data.
 *          Handles boundary cases with clamping.
 *
 * @param[in] field  2D field data (row-major order)
 * @param[in] x      X-coordinate (grid units)
 * @param[in] y      Y-coordinate (grid units)
 * @param[in] width  Grid width
 * @param[in] height Grid height
 *
 * @return Interpolated value at (x,y)
 *
 * @complexity O(1) - constant time
 * @memory_access 4 global memory reads (coalesced when possible)
 */
__device__ __forceinline__ float bilinear_interpolation(float* field,
                                                        float x, float y,
                                                        int width, int height);
```

### For Class Methods

```cpp
/**
 * @method LDM::initializeParticlesEKI
 * @brief Initialize particle ensemble for EKI optimization
 *
 * @details Creates initial particle distribution based on prior emission
 *          estimates. Supports both uniform and Gaussian distributions
 *          with configurable parameters from settings file.
 *
 * @param[in] ensemble_id    Ensemble member index (0 to num_ensemble-1)
 * @param[in] emission_rates Array of emission rates [kg/s]
 * @param[in] use_prior      Use prior distribution (true) or truth (false)
 *
 * @throws std::runtime_error if GPU memory allocation fails
 * @throws std::invalid_argument if ensemble_id out of range
 *
 * @pre Settings must be loaded via loadEKISettings()
 * @pre CUDA device must be initialized
 *
 * @post Particles allocated on device memory
 * @post Internal particle counter updated
 *
 * @algorithm
 *   1. Parse emission time series
 *   2. Calculate total particles needed
 *   3. Allocate GPU memory
 *   4. Initialize positions and properties
 *   5. Setup random number generators
 */
void LDM::initializeParticlesEKI(int ensemble_id, float* emission_rates,
                                bool use_prior);
```

## Documentation Tags Reference

### Essential Tags
- `@brief` - One-line summary (required)
- `@details` - Extended description
- `@param[in]` - Input parameter
- `@param[out]` - Output parameter
- `@param[in,out]` - Modified parameter
- `@return` - Return value description

### CUDA-Specific Tags
- `@kernel` - CUDA kernel name
- `@device` - Device function
- `@grid_config` - Launch configuration
- `@performance` - Performance metrics
- `@memory_access` - Memory pattern

### Quality Tags
- `@note` - Important information
- `@warning` - Potential issues
- `@invariants` - Preserved properties
- `@complexity` - Algorithm complexity
- `@throws` - Exceptions (C++ only)
- `@pre` - Preconditions
- `@post` - Postconditions

### Cross-Reference Tags
- `@see` - Related functions
- `@algorithm` - Algorithm steps
- `@equation` - Mathematical formulas

## Examples for Common Patterns

### Launcher Function
```cpp
/**
 * @brief Launch advection kernel with optimal configuration
 *
 * @details Wrapper function that calculates grid/block dimensions
 *          and launches the CUDA kernel with proper parameters.
 *
 * @param[in,out] particles Device particle array
 * @param[in]     count     Number of particles
 * @param[in]     stream    CUDA stream for async execution
 *
 * @note Grid size auto-calculated as (count + 255) / 256
 * @see advection_kernel() for actual computation
 */
void launch_advection(Particle* particles, int count, cudaStream_t stream);
```

### Initialization Function
```cpp
/**
 * @brief Load meteorological data from GFS files
 *
 * @details Reads GRIB2 format weather data and converts to internal
 *          grid representation. Handles time interpolation between
 *          available forecast hours.
 *
 * @param[in]  filename   Path to GFS data file
 * @param[in]  time       Target simulation time
 * @param[out] wind_data  Populated wind field structure
 *
 * @return true if successful, false otherwise
 *
 * @note File must be in GRIB2 format
 * @warning Large files (>1GB) may cause performance issues
 */
bool loadMeteorologicalData(const char* filename, float time,
                           WindField* wind_data);
```

### Utility Function
```cpp
/**
 * @brief Convert geographic to grid coordinates
 *
 * @details Transform latitude/longitude to grid indices using
 *          map projection parameters from configuration.
 *
 * @param[in]  lat   Latitude in degrees (-90 to 90)
 * @param[in]  lon   Longitude in degrees (-180 to 180)
 * @param[out] i     Grid row index
 * @param[out] j     Grid column index
 *
 * @invariants Preserves coordinate system orientation
 * @complexity O(1)
 */
void geo_to_grid(float lat, float lon, int& i, int& j);
```

## Consistency Rules

1. **Always start with @brief** - One line, no period
2. **Use @details for complexity** - Multi-line explanations
3. **Document all parameters** - Specify [in], [out], or [in,out]
4. **Include units** - Add [seconds], [m/s], [kg], etc.
5. **Warn about side effects** - Use @warning for modifications
6. **Cross-reference related code** - Use @see liberally
7. **Document performance** - For kernels, include metrics
8. **Specify preconditions** - What must be true before calling
9. **State postconditions** - What is guaranteed after return
10. **Keep it maintainable** - Update docs with code changes