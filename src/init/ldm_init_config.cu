/******************************************************************************
 * @file ldm_init_config.cu
 * @brief Configuration loading and initialization for LDM simulation system
 *
 * @details Implements configuration parsers for all input files:
 *          - setting.txt: Core simulation parameters
 *          - source.txt: Emission source locations and release cases
 *          - eki_settings.txt: Ensemble Kalman Inversion parameters
 *          - Modernized config files: simulation.conf, physics.conf, etc.
 *
 * @note Legacy file support maintained for backward compatibility
 * @note New modular config system introduced in 2025-10-17 (Phase 1)
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/

#include "../core/ldm.cuh"
#include "../physics/ldm_nuclides.cuh"
#include "colors.h"

/******************************************************************************
 * @brief Load simulation configuration from legacy setting.txt file
 *
 * @details Parses input/setting.txt to load core simulation parameters and
 *          source.txt for emission source definitions. This is the legacy
 *          configuration loader maintained for backward compatibility.
 *
 *          Configuration includes:
 *          - Temporal parameters: time_end, dt, output frequency
 *          - Particle properties: count, size distribution, density
 *          - Physics model switches: turbulence, deposition, decay
 *          - Atmospheric conditions: rural/urban, stability parameterization
 *          - Meteorological data source: GFS/LDAPS selection
 *          - Source locations: coordinates (lon, lat, height)
 *          - Release cases: emission values per source/time
 *
 * @pre Input files must exist:
 *      - input/setting.txt (simulation parameters)
 *      - input/source.txt (source locations and release cases)
 *      - cram/A60.csv (CRAM decay matrix if radioactive decay enabled)
 *
 * @post Member variables populated:
 *       - time_end, dt, freq_output, nop
 *       - isRural, isPG, isGFS
 *       - sources vector, concentrations vector
 *       - decayConstants, drydepositionVelocity vectors
 * @post Physics model switches set: g_turb_switch, g_drydep, g_wetdep, g_raddecay
 * @post CRAM system initialized (if radioactive decay enabled)
 * @post Output directory cleaned
 *
 * @algorithm
 *   1. Load setting.txt using ConfigReader
 *   2. Parse simulation parameters (time, particle count, etc.)
 *   3. Load physics model switches
 *   4. Parse species properties (decay constants, sizes, densities)
 *   5. Initialize CRAM decay system with dt from config
 *   6. Open source.txt for emission source parsing
 *   7. Parse [SOURCE] section: lon, lat, height coordinates
 *   8. Parse [SOURCE_TERM] section: decay constants, deposition velocities
 *   9. Parse [RELEASE_CASES] section: location, source term, emission value
 *  10. Close file and clean output directory
 *
 * @note Configuration values passed to GPU kernels via KernelScalars struct
 * @note No longer uses __constant__ memory (removed in refactoring)
 *
 * @see loadSimulationConfig() for modernized config file loader
 * @see input/setting.txt for legacy file format specification
 * @see input/source.txt for source definition format
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::loadSimulationConfiguration(){

    if (!g_config.loadConfig("input/setting.txt")) {
        std::cerr << "Failed to load configuration file" << std::endl;
        exit(1);
    }

    FILE* sourceFile;

    // Parse temporal parameters
    time_end = g_config.getFloat("Time_end(s)", 64800.0f);      // Simulation duration (seconds)
    dt = g_config.getFloat("dt(s)", 10.0f);                     // Time step (seconds)
    freq_output = g_config.getInt("Plot_output_freq", 10);      // VTK output frequency

    // Parse particle parameters
    nop = g_config.getInt("Total_number_of_particle", 10000);   // Total particle count

    // Parse atmospheric conditions
    isRural = g_config.getInt("Rural/Urban", 1);                // 1=Rural, 0=Urban
    isPG = g_config.getInt("Pasquill-Gifford/Briggs-McElroy-Pooler", 1);  // Stability scheme
    isGFS = g_config.getInt("Data", 1);                         // Meteorological data source

    // Load terminal output settings
    g_sim.fixedScrollOutput = g_config.getInt("fixed_scroll_output", 1);

    // Load physics model switches
    g_turb_switch = g_config.getInt("turbulence_model", 0);
    g_drydep = g_config.getInt("dry_deposition_model", 0);
    g_wetdep = g_config.getInt("wet_deposition_model", 0);
    g_raddecay = g_config.getInt("radioactive_decay_model", 1);

    // Display physics model configuration
    std::cout << Color::BOLD << "Physics Models" << Color::RESET << std::endl;
    std::cout << "  TURB=" << g_turb_switch
              << ", DRYDEP=" << g_drydep
              << ", WETDEP=" << g_wetdep
              << ", RADDECAY=" << g_raddecay << std::endl;

    // Clean output directory before simulation
    cleanOutputDirectory();

    // Parse species properties (up to 4 species supported)
    std::vector<std::string> species_names = g_config.getStringArray("species_names");
    std::vector<float> decay_constants = g_config.getFloatArray("decay_constants");
    std::vector<float> deposition_velocities = g_config.getFloatArray("deposition_velocities");
    std::vector<float> particle_sizes = g_config.getFloatArray("particle_sizes");
    std::vector<float> particle_densities = g_config.getFloatArray("particle_densities");
    std::vector<float> size_standard_deviations = g_config.getFloatArray("size_standard_deviations");

    for (int i = 0; i < 4 && i < species_names.size(); i++) {
        g_mpi.species[i] = species_names[i];
        g_mpi.decayConstants[i] = (i < decay_constants.size()) ? decay_constants[i] : 1.00e-6f;
        g_mpi.depositionVelocities[i] = (i < deposition_velocities.size()) ? deposition_velocities[i] : 0.01f;
        g_mpi.particleSizes[i] = (i < particle_sizes.size()) ? particle_sizes[i] : 0.6f;
        g_mpi.particleDensities[i] = (i < particle_densities.size()) ? particle_densities[i] : 2500.0f;
        g_mpi.sizeStandardDeviations[i] = (i < size_standard_deviations.size()) ? size_standard_deviations[i] : 0.01f;
    }

    // Initialize CRAM decay system with dynamic dt from configuration
    if (initialize_cram_system("cram/A60.csv")) {
        // Successfully computed exp(-A*dt) matrix for CRAM decay
    } else {
        std::cerr << "Warning: CRAM system initialization failed, using traditional decay" << std::endl;
    }

    // Open source configuration file
    std::string source_file_path = g_config.getString("input_base_path", "./input/") + "source.txt";
    sourceFile = fopen(source_file_path.c_str(), "r");

    if (!sourceFile){
        std::cerr << "Failed to open source.txt" << std::endl;
        exit(1);
    }

    // Parse source.txt with three sections: [SOURCE], [SOURCE_TERM], [RELEASE_CASES]
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), sourceFile)){
        if (buffer[0] == '#') continue;  // Skip comments

        // Parse [SOURCE] section: source coordinates (lon, lat, height)
        if (strstr(buffer, "[SOURCE]")) {
            while (fgets(buffer, sizeof(buffer), sourceFile) && !strstr(buffer, "[SOURCE_TERM]")) {
                if (buffer[0] == '#') continue;

                Source src;
                sscanf(buffer, "%f %f %f", &src.lon, &src.lat, &src.height);
                sources.push_back(src);
            }
            sources.pop_back();  // Remove sentinel entry
        }

        // Parse [SOURCE_TERM] section: decay constants and deposition velocities
        if (strstr(buffer, "[SOURCE_TERM]")){
            while (fgets(buffer, sizeof(buffer), sourceFile) && !strstr(buffer, "[RELEASE_CASES]")) {
                if (buffer[0] == '#') continue;

                int srcnum;
                float decay, depvel;
                sscanf(buffer, "%d %f %f", &srcnum, &decay, &depvel);
                decayConstants.push_back(decay);
                drydepositionVelocity.push_back(depvel);
            }
            decayConstants.pop_back();      // Remove sentinel entry
            drydepositionVelocity.pop_back();
        }

        // Parse [RELEASE_CASES] section: emission scenarios
        if (strstr(buffer, "[RELEASE_CASES]")){
            while (fgets(buffer, sizeof(buffer), sourceFile)) {
                if (buffer[0] == '#') continue;
                Concentration conc;
                sscanf(buffer, "%d %d %lf", &conc.location, &conc.sourceterm, &conc.value);
                concentrations.push_back(conc);
            }
        }
    }

    fclose(sourceFile);

    // Note: Configuration values now passed via KernelScalars struct to kernels
    // __constant__ memory symbols removed during 2025 refactoring (non-RDC compatibility)

}
/******************************************************************************
 * @brief Clean output directory before simulation starts
 *
 * @details Removes previous run artifacts from output/ directory to prevent
 *          data contamination between simulation runs. Platform-specific
 *          implementation using system calls.
 *
 * @post output/ directory cleared of:
 *       - *.vtk files (VTK particle visualization)
 *       - *.csv files (validation data)
 *       - *.txt files (text output)
 *
 * @note Platform-specific behavior:
 *       - Windows: Uses 'del /Q output\*.*' command
 *       - Linux/macOS: Uses 'rm -f output/*.{vtk,csv,txt}' commands
 * @note Errors suppressed (2>nul on Windows, 2>/dev/null on Unix)
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::cleanOutputDirectory() {
    std::cout << "Cleaning output directory... " << std::flush;

    // Remove all output files using platform-specific commands
    #ifdef _WIN32
        system("del /Q output\\*.* 2>nul");
    #else
        system("rm -f output/*.vtk 2>/dev/null");
        system("rm -f output/*.csv 2>/dev/null");
        system("rm -f output/*.txt 2>/dev/null");
    #endif

    std::cout << Color::GREEN << "" << Color::RESET << std::endl;
}

/******************************************************************************
 * @brief Load Ensemble Kalman Inversion settings from eki_settings.txt
 *
 * @details Parses input/eki_settings.txt to configure the EKI optimization
 *          framework. Uses a state machine to parse multi-line sections
 *          (receptor locations, emission time series) and key-value pairs.
 *
 *          Configuration includes:
 *          - Receptor definitions: locations (lat/lon), capture radius
 *          - Emission time series: true values (for observations), prior guess
 *          - EKI algorithm parameters: ensemble size, iterations, noise level
 *          - Algorithm variants: adaptive step size, localization, regularization
 *          - GPU acceleration settings: forward/inverse model GPU usage
 *          - Debug options: Memory Doctor mode for IPC diagnostics
 *
 * @pre input/eki_settings.txt must exist
 * @pre EKI mode must be enabled (function called when g_eki.mode = true)
 *
 * @post g_eki struct fully populated with EKI parameters
 * @post g_eki.receptor_locations: vector of (lat, lon) pairs
 * @post g_eki.true_emissions: time series for generating observations
 * @post g_eki.prior_emissions: initial guess for optimization
 * @post Algorithm switches set: adaptive_eki, localized_eki, regularization
 *
 * @algorithm State machine parser:
 *   1. Initialize g_eki with default values
 *   2. Parse file line by line:
 *      - Section headers toggle state flags:
 *        * RECEPTOR_LOCATIONS_MATRIX= → read receptor coordinates
 *        * TRUE_EMISSION_SERIES= → read true emission values
 *        * PRIOR_EMISSION_SERIES= → read prior emission values
 *      - Key-value pairs (KEY=VALUE) reset state flags and parse parameters
 *      - Matrix data lines parsed according to current state
 *   3. Validate configuration (e.g., num_receptors matches location count)
 *   4. Print essential EKI configuration summary
 *
 * @note File format:
 *       - Comments: Lines starting with #
 *       - Key-value: KEY=VALUE (no spaces around =)
 *       - Matrix sections: Header line followed by data lines
 * @note State machine ensures correct parsing of multi-line sections
 *
 * @see input/eki_settings.txt for configuration file format
 * @see src/eki/RunEstimator.py for Python EKI executor
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::loadEKISettings() {
    std::cout << "\n" << Color::CYAN << "[SYSTEM] " << Color::RESET
              << "Loading EKI settings... " << std::flush;

    FILE* ekiFile = fopen("input/eki_settings.txt", "r");
    if (!ekiFile) {
        std::cerr << "Failed to open input/eki_settings.txt" << std::endl;
        exit(1);
    }

    char buffer[256];

    // Initialize global EKI configuration with default values
    g_eki.mode = true;                          // EKI mode enabled
    g_eki.time_interval = 15.0f;                // 15 minute intervals
    g_eki.time_unit = "minutes";
    g_eki.num_receptors = 0;
    g_eki.prior_mode = "constant";
    g_eki.prior_constant = 1.5e+8f;             // Default prior emission rate (Bq/s)
    g_eki.ensemble_size = 50;                   // Number of ensemble members
    g_eki.noise_level = 0.01f;                  // 1% observation noise

    // Clear emission and receptor vectors
    g_eki.receptor_locations.clear();
    g_eki.true_emissions.clear();
    g_eki.prior_emissions.clear();

    // State machine flags for multi-line section parsing
    bool reading_receptor_matrix = false;
    bool reading_true_emissions = false;
    bool reading_prior_emissions = false;
    
    while (fgets(buffer, sizeof(buffer), ekiFile)) {
        // Skip comments and empty lines
        if (buffer[0] == '#' || buffer[0] == '\n' || buffer[0] == '\r') {
            continue;
        }

        // State machine: Check for multi-line section headers
        if (strstr(buffer, "RECEPTOR_LOCATIONS_MATRIX=")) {
            reading_receptor_matrix = true;
            reading_true_emissions = false;
            reading_prior_emissions = false;
            continue;
        }

        if (strstr(buffer, "TRUE_EMISSION_SERIES=")) {
            reading_receptor_matrix = false;
            reading_true_emissions = true;
            reading_prior_emissions = false;
            continue;
        }

        if (strstr(buffer, "PRIOR_EMISSION_SERIES=")) {
            reading_receptor_matrix = false;
            reading_true_emissions = false;
            reading_prior_emissions = true;
            continue;
        }

        // Reset section flags when encountering key-value pairs
        if (strchr(buffer, '=') != nullptr) {
            reading_receptor_matrix = false;
            reading_true_emissions = false;
            reading_prior_emissions = false;
        }

        // Parse matrix data based on current state
        if (reading_receptor_matrix) {
            float lat, lon;
            if (sscanf(buffer, "%f %f", &lat, &lon) == 2) {
                g_eki.receptor_locations.push_back(std::make_pair(lat, lon));
            }
        }
        else if (reading_true_emissions) {
            float emission;
            if (sscanf(buffer, "%f", &emission) == 1) {
                g_eki.true_emissions.push_back(emission);
            }
        }
        else if (reading_prior_emissions) {
            float emission;
            if (sscanf(buffer, "%f", &emission) == 1) {
                g_eki.prior_emissions.push_back(emission);
            }
        }
        
        // Parse key-value pairs (section flags already reset above)
        if (strchr(buffer, '=') != nullptr) {

            // Temporal parameters
            if (strstr(buffer, "EKI_TIME_INTERVAL=")) {
                sscanf(buffer, "EKI_TIME_INTERVAL=%f", &g_eki.time_interval);
            }
            else if (strstr(buffer, "EKI_TIME_UNIT=")) {
                char unit[32];
                sscanf(buffer, "EKI_TIME_UNIT=%s", unit);
                g_eki.time_unit = std::string(unit);
            }

            // Receptor configuration
            else if (strstr(buffer, "NUM_RECEPTORS=")) {
                sscanf(buffer, "NUM_RECEPTORS=%d", &g_eki.num_receptors);
            }
            else if (strstr(buffer, "RECEPTOR_CAPTURE_RADIUS=")) {
                sscanf(buffer, "RECEPTOR_CAPTURE_RADIUS=%f", &g_eki.receptor_capture_radius);
            }

            // Prior emission settings
            else if (strstr(buffer, "PRIOR_EMISSION_MODE=")) {
                char mode[32];
                sscanf(buffer, "PRIOR_EMISSION_MODE=%s", mode);
                g_eki.prior_mode = std::string(mode);
            }
            else if (strstr(buffer, "PRIOR_EMISSION_CONSTANT=")) {
                sscanf(buffer, "PRIOR_EMISSION_CONSTANT=%f", &g_eki.prior_constant);
            }

            // EKI algorithm parameters
            else if (strstr(buffer, "EKI_ENSEMBLE_SIZE=")) {
                sscanf(buffer, "EKI_ENSEMBLE_SIZE=%d", &g_eki.ensemble_size);
            }
            else if (strstr(buffer, "EKI_NOISE_LEVEL=")) {
                sscanf(buffer, "EKI_NOISE_LEVEL=%f", &g_eki.noise_level);
            }
            else if (strstr(buffer, "EKI_ITERATION=")) {
                sscanf(buffer, "EKI_ITERATION=%d", &g_eki.iteration);
            }
            else if (strstr(buffer, "EKI_PERTURB_OPTION=")) {
                char opt[32];
                sscanf(buffer, "EKI_PERTURB_OPTION=%s", opt);
                g_eki.perturb_option = std::string(opt);
            }

            // EKI algorithm variants
            else if (strstr(buffer, "EKI_ADAPTIVE=")) {
                char opt[32];
                sscanf(buffer, "EKI_ADAPTIVE=%s", opt);
                g_eki.adaptive_eki = std::string(opt);
            }
            else if (strstr(buffer, "EKI_LOCALIZED=")) {
                char opt[32];
                sscanf(buffer, "EKI_LOCALIZED=%s", opt);
                g_eki.localized_eki = std::string(opt);
            }
            else if (strstr(buffer, "EKI_REGULARIZATION=")) {
                char opt[32];
                sscanf(buffer, "EKI_REGULARIZATION=%s", opt);
                g_eki.regularization = std::string(opt);
            }
            else if (strstr(buffer, "EKI_RENKF_LAMBDA=")) {
                sscanf(buffer, "EKI_RENKF_LAMBDA=%f", &g_eki.renkf_lambda);
            }

            // GPU acceleration settings
            else if (strstr(buffer, "EKI_GPU_FORWARD=")) {
                char opt[32];
                sscanf(buffer, "EKI_GPU_FORWARD=%s", opt);
                g_eki.gpu_forward = std::string(opt);
            }
            else if (strstr(buffer, "EKI_GPU_INVERSE=")) {
                char opt[32];
                sscanf(buffer, "EKI_GPU_INVERSE=%s", opt);
                g_eki.gpu_inverse = std::string(opt);
            }
            else if (strstr(buffer, "EKI_NUM_GPU=")) {
                sscanf(buffer, "EKI_NUM_GPU=%d", &g_eki.num_gpu);
            }

            // Additional EKI parameters
            else if (strstr(buffer, "EKI_TIME_DAYS=")) {
                sscanf(buffer, "EKI_TIME_DAYS=%f", &g_eki.time_days);
            }
            else if (strstr(buffer, "EKI_INVERSE_TIME_INTERVAL=")) {
                sscanf(buffer, "EKI_INVERSE_TIME_INTERVAL=%f", &g_eki.inverse_time_interval);
            }
            else if (strstr(buffer, "EKI_RECEPTOR_ERROR=")) {
                sscanf(buffer, "EKI_RECEPTOR_ERROR=%f", &g_eki.receptor_error);
            }
            else if (strstr(buffer, "EKI_RECEPTOR_MDA=")) {
                sscanf(buffer, "EKI_RECEPTOR_MDA=%f", &g_eki.receptor_mda);
            }

            // Source configuration
            else if (strstr(buffer, "EKI_SOURCE_LOCATION=")) {
                char loc[32];
                sscanf(buffer, "EKI_SOURCE_LOCATION=%s", loc);
                g_eki.source_location = std::string(loc);
            }
            else if (strstr(buffer, "EKI_NUM_SOURCE=")) {
                sscanf(buffer, "EKI_NUM_SOURCE=%d", &g_eki.num_source);
            }

            // Debug mode
            else if (strstr(buffer, "MEMORY_DOCTOR_MODE=")) {
                char mode[32];
                sscanf(buffer, "MEMORY_DOCTOR_MODE=%s", mode);
                g_eki.memory_doctor_mode = (strcmp(mode, "On") == 0 || strcmp(mode, "on") == 0 ||
                                           strcmp(mode, "ON") == 0 || strcmp(mode, "1") == 0);
            }
        }
    }
    
    fclose(ekiFile);

    std::cout << Color::GREEN << "" << Color::RESET << std::endl;

    // Print essential EKI settings (condensed)
    std::cout << Color::BOLD << "EKI Configuration" << Color::RESET << std::endl;
    std::cout << "  Receptors          : " << Color::BOLD << g_eki.num_receptors << Color::RESET
              << " (radius: " << g_eki.receptor_capture_radius << "°)" << std::endl;
    std::cout << "  Emission timesteps : " << Color::BOLD << g_eki.true_emissions.size() << Color::RESET
              << " (" << g_eki.time_interval << " " << g_eki.time_unit << ")" << std::endl;
    std::cout << "  Ensemble size      : " << Color::BOLD << g_eki.ensemble_size << Color::RESET << std::endl;

    if (g_eki.memory_doctor_mode) {
        std::cout << "  Memory Doctor      : " << Color::YELLOW << "ON" << Color::RESET << std::endl;
    }
}

// ===========================================================================
// GRID RECEPTOR DEBUG MODE FUNCTIONS
// ===========================================================================

/******************************************************************************
 * @brief Initialize uniform grid of receptors for debugging/validation
 *
 * @details Creates a (2N+1)×(2N+1) square grid of receptors centered at the
 *          emission source location. Used in receptor-debug mode for detailed
 *          spatial analysis of particle dispersion patterns and validation
 *          against analytical solutions.
 *
 *          Grid structure:
 *          - Center: Source location (37°N, 141°E by default)
 *          - Extent: ±N grid points in lat/lon directions
 *          - Spacing: Uniform grid spacing in degrees
 *          - Example: grid_count=5, spacing=0.1° → 11×11=121 receptors
 *
 * @param[in] grid_count_param Grid extent (N receptors in each direction)
 *                             - Typical range: 5-10
 *                             - Total receptors = (2N+1)²
 * @param[in] grid_spacing_param Spacing between receptors (degrees)
 *                               - Typical range: 0.05-0.2°
 *                               - Approximate: 0.1° ≈ 11 km at mid-latitudes
 *
 * @pre CUDA device must be initialized
 * @pre Sufficient GPU memory for receptor arrays
 *
 * @post GPU arrays allocated and initialized:
 *       - d_grid_receptor_lats: Receptor latitudes (degrees N)
 *       - d_grid_receptor_lons: Receptor longitudes (degrees E)
 *       - d_grid_receptor_dose: Dose accumulation (initialized to 0)
 *       - d_grid_receptor_particle_count: Particle counts (initialized to 0)
 * @post Host storage vectors resized:
 *       - grid_receptor_observations
 *       - grid_receptor_particle_counts
 * @post Member variables set:
 *       - grid_count, grid_spacing, grid_receptor_total
 *
 * @algorithm
 *   1. Calculate total receptors = (2*grid_count + 1)²
 *   2. Generate receptor grid centered at source:
 *      for i in [-N, N]:
 *        for j in [-N, N]:
 *          lat = source_lat + i * grid_spacing
 *          lon = source_lon + j * grid_spacing
 *   3. Allocate GPU memory for receptor arrays
 *   4. Copy locations to GPU (cudaMemcpy)
 *   5. Initialize dose/count arrays to zero (cudaMemset)
 *   6. Resize host storage vectors
 *
 * @note Grid is always square and centered at source location
 * @note Large grids (>20×20) may impact performance due to memory overhead
 * @note Used exclusively in receptor-debug mode (not EKI mode)
 *
 * @memory GPU: 4 * total_receptors * sizeof(float) + 1 * total_receptors * sizeof(int)
 *         Example: 121 receptors = 2.4 KB total
 *
 * @see main_receptor_debug.cu for usage
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::initializeGridReceptors(int grid_count_param, float grid_spacing_param) {
    // Store grid parameters
    grid_count = grid_count_param;
    grid_spacing = grid_spacing_param;
    grid_receptor_total = (2 * grid_count + 1) * (2 * grid_count + 1);

    std::cout << Color::CYAN << "[SYSTEM] " << Color::RESET
              << "Initializing " << Color::BOLD << grid_receptor_total << Color::RESET
              << " grid receptors (" << (2*grid_count+1) << "×" << (2*grid_count+1)
              << ", spacing=" << grid_spacing << "°)" << std::endl;

    // Default source location (Fukushima coordinates)
    float source_lat = 37.0f;
    float source_lon = 141.0f;

    // Prepare host arrays for receptor locations
    std::vector<float> host_lats(grid_receptor_total);
    std::vector<float> host_lons(grid_receptor_total);

    // Generate uniform grid centered at source
    int receptor_idx = 0;
    for (int i = -grid_count; i <= grid_count; i++) {
        for (int j = -grid_count; j <= grid_count; j++) {
            float lat = source_lat + i * grid_spacing;
            float lon = source_lon + j * grid_spacing;

            host_lats[receptor_idx] = lat;
            host_lons[receptor_idx] = lon;
            receptor_idx++;
        }
    }

    // Allocate GPU memory for receptor data
    cudaMalloc(&d_grid_receptor_lats, grid_receptor_total * sizeof(float));
    cudaMalloc(&d_grid_receptor_lons, grid_receptor_total * sizeof(float));
    cudaMalloc(&d_grid_receptor_dose, grid_receptor_total * sizeof(float));
    cudaMalloc(&d_grid_receptor_particle_count, grid_receptor_total * sizeof(int));

    // Copy receptor locations to GPU
    cudaMemcpy(d_grid_receptor_lats, host_lats.data(), grid_receptor_total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid_receptor_lons, host_lons.data(), grid_receptor_total * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize GPU dose and particle count arrays to zero
    cudaMemset(d_grid_receptor_dose, 0, grid_receptor_total * sizeof(float));
    cudaMemset(d_grid_receptor_particle_count, 0, grid_receptor_total * sizeof(int));

    // Initialize host storage for observations
    grid_receptor_observations.resize(grid_receptor_total);
    grid_receptor_particle_counts.resize(grid_receptor_total);

    std::cout << Color::GREEN << "  " << Color::RESET
              << "Grid receptors initialized" << std::endl;
}

// ===========================================================================
// MODERNIZED CONFIG LOADING FUNCTIONS (Phase 1: 2025-10-17)
// ===========================================================================
// These functions implement the new modular configuration file structure
// described in docs/INPUT_MODERNIZATION_PLAN.md. Provides improved usability
// with self-documenting config files, logical grouping, and backward compatibility.

/******************************************************************************
 * @brief Load simulation parameters from modernized simulation.conf file
 *
 * @details Parses input/simulation.conf to load core simulation settings:
 *          - Temporal: time_end, time_step, vtk_output_frequency
 *          - Particles: total_particles
 *          - Atmosphere: rural_conditions, use_pasquill_gifford
 *          - Meteorology: use_gfs_data
 *          - Terminal: fixed_scroll_output
 *
 *          Part of Phase 1 input file modernization (2025-10-17).
 *
 * @pre input/simulation.conf must exist
 * @post Member variables populated: time_end, dt, freq_output, nop, isRural, isPG, isGFS
 * @post g_sim.fixedScrollOutput set
 * @post Configuration summary printed to console
 *
 * @see docs/INPUT_MODERNIZATION_PLAN.md for config file format
 * @see input/simulation.conf for configuration template
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::loadSimulationConfig() {
    std::cout << Color::CYAN << "[CONFIG] " << Color::RESET
              << "Loading simulation.conf... " << std::flush;

    // Load configuration file
    if (!g_config.loadConfig("input/simulation.conf")) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET
                  << "Failed to load input/simulation.conf" << std::endl;
        std::cerr << "         Please ensure the file exists and is readable." << std::endl;
        exit(1);
    }

    // ========== TEMPORAL SETTINGS ==========
    time_end = g_config.getFloat("time_end", 21600.0f);  // Default: 6 hours
    dt = g_config.getFloat("time_step", 100.0f);         // Default: 100 seconds
    freq_output = g_config.getInt("vtk_output_frequency", 1);  // Default: every timestep

    // ========== PARTICLE SETTINGS ==========
    nop = g_config.getInt("total_particles", 10000);     // Default: 10,000 particles

    // ========== ATMOSPHERIC CONDITIONS ==========
    isRural = g_config.getInt("rural_conditions", 1);    // Default: Rural (1)
    isPG = g_config.getInt("use_pasquill_gifford", 1);   // Default: Pasquill-Gifford (1)

    // ========== METEOROLOGICAL DATA SOURCE ==========
    isGFS = g_config.getInt("use_gfs_data", 1);          // Default: GFS (1)

    // ========== TERMINAL OUTPUT ==========
    g_sim.fixedScrollOutput = g_config.getInt("fixed_scroll_output", 1);  // Default: Enabled (1)

    std::cout << Color::GREEN << "done" << Color::RESET << std::endl;

    // ========== PRINT CONFIGURATION SUMMARY ==========
    std::cout << Color::BOLD << "Simulation Configuration" << Color::RESET << std::endl;

    // Temporal settings
    std::cout << "  Time settings      : " << Color::BOLD << time_end << "s" << Color::RESET
              << " (dt=" << dt << "s, "
              << "output_freq=" << freq_output << ")" << std::endl;

    // Particle count
    std::cout << "  Particles          : " << Color::BOLD << nop << Color::RESET << std::endl;

    // Atmospheric conditions
    std::cout << "  Atmosphere         : "
              << (isRural ? "Rural" : "Urban") << ", "
              << (isPG ? "Pasquill-Gifford" : "Briggs-McElroy-Pooler") << std::endl;

    // Meteorological data
    std::cout << "  Meteorology        : " << (isGFS ? "GFS" : "LDAPS") << std::endl;

    // Terminal output
    std::cout << "  Terminal output    : "
              << (g_sim.fixedScrollOutput ? "Fixed-scroll" : "Continuous-scroll") << std::endl;
}

/******************************************************************************
 * @brief Load physics model configuration from physics.conf
 *
 * @details Parses input/physics.conf to configure physics model switches:
 *          - turbulence_model: Turbulent diffusion (0=off, 1=on)
 *          - dry_deposition_model: Dry deposition (0=off, 1=on)
 *          - wet_deposition_model: Wet deposition (0=off, 1=on)
 *          - radioactive_decay_model: Decay (0=off, 1=on)
 *
 * @pre input/physics.conf must exist
 * @post Physics switches set: g_turb_switch, g_drydep, g_wetdep, g_raddecay
 * @post Configuration summary printed to console
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::loadPhysicsConfig() {
    std::cout << Color::CYAN << "[SYSTEM] " << Color::RESET
              << "Loading physics configuration... " << std::flush;

    // Load physics.conf using ConfigReader
    ConfigReader physics_config;
    if (!physics_config.loadConfig("input/physics.conf")) {
        std::cerr << Color::RED << "[ERROR]" << Color::RESET
                  << " Failed to load input/physics.conf" << std::endl;
        exit(1);
    }

    // Parse physics model switches
    g_turb_switch = physics_config.getInt("turbulence_model", 0);
    g_drydep = physics_config.getInt("dry_deposition_model", 0);
    g_wetdep = physics_config.getInt("wet_deposition_model", 0);
    g_raddecay = physics_config.getInt("radioactive_decay_model", 1);

    std::cout << Color::GREEN << "done" << Color::RESET << std::endl;

    // Print physics model status summary
    std::cout << Color::BOLD << "Physics Models" << Color::RESET << std::endl;
    std::cout << "  Turbulence         : " << (g_turb_switch ? Color::GREEN : Color::YELLOW)
              << (g_turb_switch ? "ON" : "OFF") << Color::RESET << std::endl;
    std::cout << "  Dry Deposition     : " << (g_drydep ? Color::GREEN : Color::YELLOW)
              << (g_drydep ? "ON" : "OFF") << Color::RESET << std::endl;
    std::cout << "  Wet Deposition     : " << (g_wetdep ? Color::GREEN : Color::YELLOW)
              << (g_wetdep ? "ON" : "OFF") << Color::RESET << std::endl;
    std::cout << "  Radioactive Decay  : " << (g_raddecay ? Color::GREEN : Color::YELLOW)
              << (g_raddecay ? "ON" : "OFF") << Color::RESET << std::endl;
}

/******************************************************************************
 * @brief Load source locations from source.conf file
 *
 * @details Parses input/source.conf to load emission source coordinates.
 *          Format: LONGITUDE LATITUDE HEIGHT (space-separated, degrees/meters)
 *
 *          Example:
 *            # Source 1: Fukushima Daiichi
 *            141.0 37.0 20.0
 *
 * @pre input/source.conf must exist
 * @post sources vector populated with Source structs (lon, lat, height)
 * @post At least one source must be defined (validation check)
 *
 * @note Lines starting with # are comments
 * @note Empty lines ignored
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::loadSourceConfig() {
    std::cout << Color::CYAN << "[SYSTEM] " << Color::RESET
              << "Loading source locations... " << std::flush;

    // Construct file path
    std::string source_file_path = g_config.getString("input_base_path", "./input/") + "source.conf";

    FILE* sourceFile = fopen(source_file_path.c_str(), "r");

    if (!sourceFile) {
        std::cerr << Color::RED << "Failed to open " << source_file_path << Color::RESET << std::endl;
        exit(1);
    }

    char buffer[256];
    int line_number = 0;

    // Clear existing sources
    sources.clear();

    while (fgets(buffer, sizeof(buffer), sourceFile)) {
        line_number++;

        // Skip comment lines starting with #
        if (buffer[0] == '#') continue;

        // Skip empty lines
        bool is_empty = true;
        for (int i = 0; buffer[i] != '\0'; i++) {
            if (buffer[i] != ' ' && buffer[i] != '\t' &&
                buffer[i] != '\n' && buffer[i] != '\r') {
                is_empty = false;
                break;
            }
        }
        if (is_empty) continue;

        // Parse source location: LON LAT HEIGHT
        Source src;
        int parsed = sscanf(buffer, "%f %f %f", &src.lon, &src.lat, &src.height);

        if (parsed == 3) {
            sources.push_back(src);
        } else {
            std::cerr << Color::YELLOW << "\n[WARNING] " << Color::RESET
                      << "Skipping invalid line " << line_number
                      << " in source.conf: " << buffer;
        }
    }

    fclose(sourceFile);

    // Validation: at least one source must be defined
    if (sources.empty()) {
        std::cerr << Color::RED << "\n[ERROR] " << Color::RESET
                  << "No valid sources found in source.conf" << std::endl;
        exit(1);
    }

    std::cout << Color::GREEN << "done" << Color::RESET << std::endl;

    // Print loaded sources summary
    std::cout << Color::BOLD << "Source Locations" << Color::RESET << std::endl;
    for (size_t i = 0; i < sources.size(); i++) {
        std::cout << "  Source " << (i+1) << "            : "
                  << sources[i].lon << "°E, "
                  << sources[i].lat << "°N, "
                  << sources[i].height << "m" << std::endl;
    }
}

/******************************************************************************
 * @brief Load nuclide configuration from nuclides.conf
 *
 * @details Parses nuclide properties with backward compatibility for legacy formats.
 *          Tries files in order:
 *          1. input/nuclides.conf (new format)
 *          2. input/nuclides_config_1.txt (legacy single nuclide)
 *          3. input/nuclides_config_60.txt (legacy 60-nuclide chain)
 *
 *          New format (space-separated):
 *            NUCLIDE_NAME DECAY_CONSTANT(s^-1) DEPOSITION_VELOCITY(m/s)
 *
 *          Legacy format (comma-separated):
 *            NUCLIDE_NAME,DECAY_CONSTANT,RATIO
 *
 * @pre At least one nuclide configuration file must exist
 * @post decayConstants vector populated with decay constants (s^-1)
 * @post drydepositionVelocity vector populated with deposition velocities (m/s)
 * @post g_num_nuclides set to number of nuclides loaded
 *
 * @note Decay constants forced to positive values (fabs applied)
 * @note Legacy format uses default deposition velocity = 1.0 m/s
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::loadNuclidesConfig() {
    std::cout << Color::CYAN << "[SYSTEM] " << Color::RESET
              << "Loading nuclide configuration... " << std::flush;

    FILE* nuclideFile = nullptr;
    std::string filename;

    // Try new format first (nuclides.conf)
    filename = "input/nuclides.conf";
    nuclideFile = fopen(filename.c_str(), "r");

    // Fall back to legacy format (nuclides_config_1.txt)
    if (!nuclideFile) {
        filename = "input/nuclides_config_1.txt";
        nuclideFile = fopen(filename.c_str(), "r");
    }

    // Fall back to 60-nuclide chain if available
    if (!nuclideFile) {
        filename = "input/nuclides_config_60.txt";
        nuclideFile = fopen(filename.c_str(), "r");
    }

    if (!nuclideFile) {
        std::cerr << std::endl << Color::RED << "[ERROR] " << Color::RESET
                  << "Cannot open nuclide configuration file" << std::endl;
        std::cerr << "  Tried: input/nuclides.conf" << std::endl;
        std::cerr << "         input/nuclides_config_1.txt" << std::endl;
        std::cerr << "         input/nuclides_config_60.txt" << std::endl;
        exit(1);
    }

    // Clear existing data
    decayConstants.clear();
    drydepositionVelocity.clear();

    char buffer[256];
    int line_number = 0;
    int nuclide_count = 0;

    while (fgets(buffer, sizeof(buffer), nuclideFile)) {
        line_number++;

        // Skip comments and empty lines
        if (buffer[0] == '#' || buffer[0] == '\n' || buffer[0] == '\r') {
            continue;
        }

        // Remove trailing newline
        buffer[strcspn(buffer, "\n\r")] = '\0';

        // Skip empty lines (after trimming)
        if (strlen(buffer) == 0) {
            continue;
        }

        char nuclide_name[64];
        float decay_const, dep_vel;

        // Try new format first (space-separated)
        int parsed = sscanf(buffer, "%s %f %f", nuclide_name, &decay_const, &dep_vel);

        if (parsed == 3) {
            // Successfully parsed new format
            decayConstants.push_back(fabs(decay_const));  // Ensure positive
            drydepositionVelocity.push_back(dep_vel);
            nuclide_count++;
        }
        else {
            // Try legacy comma-separated format
            float legacy_ratio;
            parsed = sscanf(buffer, "%[^,],%f,%f", nuclide_name, &decay_const, &legacy_ratio);

            if (parsed == 3) {
                // Successfully parsed legacy format
                decayConstants.push_back(fabs(decay_const));  // Ensure positive
                drydepositionVelocity.push_back(1.0f);  // Default deposition velocity
                nuclide_count++;
            }
            else {
                std::cerr << std::endl << Color::YELLOW << "[WARNING] " << Color::RESET
                          << "Failed to parse line " << line_number << " in " << filename << std::endl;
                std::cerr << "  Line: " << buffer << std::endl;
                continue;
            }
        }
    }

    fclose(nuclideFile);

    // Set global nuclide count
    g_num_nuclides = nuclide_count;

    if (nuclide_count == 0) {
        std::cerr << std::endl << Color::RED << "[ERROR] " << Color::RESET
                  << "No nuclides loaded from " << filename << std::endl;
        exit(1);
    }

    std::cout << Color::GREEN << "done" << Color::RESET << std::endl;

    // Print loaded configuration
    std::cout << Color::BOLD << "Nuclide Configuration" << Color::RESET << std::endl;
    std::cout << "  File               : " << filename << std::endl;
    std::cout << "  Nuclides loaded    : " << Color::BOLD << nuclide_count << Color::RESET << std::endl;

    // Print first nuclide as example
    if (nuclide_count > 0) {
        std::cout << "  Decay constant     : " << decayConstants[0] << " s⁻¹" << std::endl;
        std::cout << "  Deposition velocity: " << drydepositionVelocity[0] << " m/s" << std::endl;
    }
}

/******************************************************************************
 * @brief Load advanced system configuration from advanced.conf
 *
 * @details Validates grid dimensions and coordinate system parameters against
 *          compile-time constants. Provides early warning if config file
 *          dimensions differ from code constants.
 *
 *          Checks:
 *          - gfs_dimX vs Constants::dimX_GFS
 *          - gfs_dimY vs Constants::dimY_GFS
 *          - gfs_dimZ vs Constants::dimZ_GFS
 *
 * @pre input/advanced.conf must exist
 * @post Grid dimensions validated and reported
 *
 * @note Code always uses Constants namespace values (compile-time)
 * @note Dimension mismatch generates warning, not error
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::loadAdvancedConfig() {
    ConfigReader adv_config;

    if (!adv_config.loadConfig("input/advanced.conf")) {
        std::cerr << Color::RED << "[ERROR]" << Color::RESET
                  << " Failed to load input/advanced.conf" << std::endl;
        std::cerr << "This file contains advanced system parameters." << std::endl;
        std::cerr << "If missing, create it using util/generate_config_templates.py" << std::endl;
        exit(1);
    }

    // Load grid dimensions for validation
    int cfg_gfs_dimX = adv_config.getInt("gfs_dimX", Constants::dimX_GFS);
    int cfg_gfs_dimY = adv_config.getInt("gfs_dimY", Constants::dimY_GFS);
    int cfg_gfs_dimZ = adv_config.getInt("gfs_dimZ", Constants::dimZ_GFS);

    // Validate grid dimensions
    bool dimensions_match = (cfg_gfs_dimX == Constants::dimX_GFS) &&
                           (cfg_gfs_dimY == Constants::dimY_GFS) &&
                           (cfg_gfs_dimZ == Constants::dimZ_GFS);

    // Output validation result
    std::cout << Color::BOLD << "Advanced Configuration" << Color::RESET << std::endl;
    std::cout << "  Data paths: " << (isGFS ? "GFS" : "LDAPS") << std::endl;

    if (dimensions_match) {
        std::cout << "  Grid dimensions: " << Color::GREEN << "validated" << Color::RESET << std::endl;
    } else {
        std::cout << "  Grid dimensions: " << Color::YELLOW << "MISMATCH" << Color::RESET << std::endl;
        std::cout << Color::YELLOW << "  Warning: " << Color::RESET
                  << "Config dimensions differ from code constants" << std::endl;
        std::cout << "  Code will use Constants namespace values (compile-time)" << std::endl;
    }
}