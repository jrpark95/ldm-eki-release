#pragma once
#ifndef LDM_CLASS_DECLARED
#error "Include the header that declares class LDM before including ldm_init.cuh"
#endif
#include <chrono>

void LDM::loadSimulationConfiguration(){

    if (!g_config.loadConfig("data/input/setting.txt")) {
        std::cerr << "Failed to load configuration file" << std::endl;
        exit(1);
    }

    FILE* sourceFile;

    time_end = g_config.getFloat("Time_end(s)", 64800.0f);
    dt = g_config.getFloat("dt(s)", 10.0f);
    freq_output = g_config.getInt("Plot_output_freq", 10);
    nop = g_config.getInt("Total_number_of_particle", 10000);
    isRural = g_config.getInt("Rural/Urban", 1);
    isPG = g_config.getInt("Pasquill-Gifford/Briggs-McElroy-Pooler", 1);
    isGFS = g_config.getInt("Data", 1);
    
    
    // Load physics model settings
    g_turb_switch = g_config.getInt("turbulence_model", 0);
    g_drydep = g_config.getInt("dry_deposition_model", 0);
    g_wetdep = g_config.getInt("wet_deposition_model", 0);
    g_raddecay = g_config.getInt("radioactive_decay_model", 1);
    
    // Print physics model status
    std::cout << Color::BOLD << "Physics Models" << Color::RESET << std::endl;
    std::cout << "  TURB=" << g_turb_switch
              << ", DRYDEP=" << g_drydep
              << ", WETDEP=" << g_wetdep
              << ", RADDECAY=" << g_raddecay << std::endl;
    
    // Clean output directory before simulation
    cleanOutputDirectory();
    
    //loadRadionuclideData();

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
    
    // Initialize CRAM system with dynamic dt
    if (initialize_cram_system("cram/A60.csv")) {
        // Compute exp(-A*dt) matrix using the dt from configuration
    } else {
        std::cerr << "Warning: CRAM system initialization failed, using traditional decay" << std::endl;
    }

    std::string source_file_path = g_config.getString("input_base_path", "./data/input/") + "source.txt";
    sourceFile = fopen(source_file_path.c_str(), "r");

    if (!sourceFile){
        std::cerr << "Failed to open source.txt" << std::endl;
        exit(1);
    }

    char buffer[256];
    while (fgets(buffer, sizeof(buffer), sourceFile)){
        if (buffer[0] == '#') continue;
    
        // SOURCE coordinates
        if (strstr(buffer, "[SOURCE]")) {
            while (fgets(buffer, sizeof(buffer), sourceFile) && !strstr(buffer, "[SOURCE_TERM]")) {
                if (buffer[0] == '#') continue;
    
                Source src;
                sscanf(buffer, "%f %f %f", &src.lon, &src.lat, &src.height);
                sources.push_back(src);
            }
            sources.pop_back();
        }
    
        // SOURCE_TERM values
        if (strstr(buffer, "[SOURCE_TERM]")){
            while (fgets(buffer, sizeof(buffer), sourceFile) && !strstr(buffer, "[RELEASE_CASES]")) {
                if (buffer[0] == '#') continue;
    
                int srcnum;
                float decay, depvel;
                sscanf(buffer, "%d %f %f", &srcnum, &decay, &depvel);
                decayConstants.push_back(decay);
                drydepositionVelocity.push_back(depvel);
            }
            decayConstants.pop_back();
            drydepositionVelocity.pop_back();
        }
    
        // RELEASE_CASES
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

    //nop = floor(nop/(sources.size()*decayConstants.size()))*sources.size()*decayConstants.size();

    cudaError_t err;

    err = cudaMemcpyToSymbol(d_time_end, &time_end, sizeof(float));
    if (err != cudaSuccess) printf("Error copying to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_dt, &dt, sizeof(float));
    if (err != cudaSuccess) printf("Error copying to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_freq_output, &freq_output, sizeof(int));
    if (err != cudaSuccess) printf("Error copying to symbol: %s\n", cudaGetErrorString(err));
    
    
    err = cudaMemcpyToSymbol(d_turb_switch, &g_turb_switch, sizeof(int));
    if (err != cudaSuccess) printf("Error copying turb_switch to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_drydep, &g_drydep, sizeof(int));
    if (err != cudaSuccess) printf("Error copying drydep to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_wetdep, &g_wetdep, sizeof(int));
    if (err != cudaSuccess) printf("Error copying wetdep to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_raddecay, &g_raddecay, sizeof(int));
    if (err != cudaSuccess) printf("Error copying raddecay to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_nop, &nop, sizeof(int));
    if (err != cudaSuccess) printf("Error copying to symbol: %s\n", cudaGetErrorString(err));

    err = cudaMemcpyToSymbol(d_isRural, &isRural, sizeof(bool));
    if (err != cudaSuccess) printf("Error copying to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_isPG, &isPG, sizeof(bool));
    if (err != cudaSuccess) printf("Error copying to symbol: %s\n", cudaGetErrorString(err));

}

void LDM::initializeParticles(){
    if (concentrations.empty()) {
        std::cerr << "[ERROR] No concentrations data loaded for particle initialization" << std::endl;
        return;
    }
    
    if (sources.empty()) {
        std::cerr << "[ERROR] No sources data loaded for particle initialization" << std::endl;
        return;
    }
    
    int partPerConc = nop / concentrations.size();

    std::random_device rd;
    auto now = std::chrono::high_resolution_clock::now();
    auto seed = now.time_since_epoch().count() + PROCESS_INDEX * 1000;
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(g_mpi.particleSizes[PROCESS_INDEX], g_mpi.sizeStandardDeviations[PROCESS_INDEX]);

    int particle_count = 0;
    for (const auto& conc : concentrations) {
        if (conc.location - 1 >= sources.size()) {
            std::cerr << "[ERROR] Invalid source location index: " << conc.location << " (max: " << sources.size() << ")" << std::endl;
            continue;
        }

        for (int i = 0; i < partPerConc; ++i) {
            float x = (sources[conc.location - 1].lon + 179.0) / 0.5;
            float y = (sources[conc.location - 1].lat + 90) / 0.5;
            float z = sources[conc.location - 1].height;

            float random_radius = dist(gen);

            part.push_back(LDMpart(x, y, z,
                                   g_mpi.decayConstants[PROCESS_INDEX],
                                   conc.value,
                                   g_mpi.depositionVelocities[PROCESS_INDEX],
                                   random_radius,
                                   g_mpi.particleDensities[PROCESS_INDEX],
                                   i + 1));
            
            // Initialize multi-nuclide concentrations for the newly created particle
            LDMpart& current_particle = part.back();
            
            // Get nuclide configuration
            NuclideConfig* nucConfig = NuclideConfig::getInstance();
            int num_nuclides = nucConfig->getNumNuclides();
            
            
            // Set up multi-nuclide concentrations - use individual initial ratios from config
            for(int nuc = 0; nuc < MAX_NUCLIDES; nuc++) {
                if (nuc < num_nuclides) {
                    // Use individual initial ratio for each nuclide from config file
                    float initial_ratio = nucConfig->getInitialRatio(nuc);
                    current_particle.concentrations[nuc] = conc.value * initial_ratio;
                    
                } else {
                    current_particle.concentrations[nuc] = 0.0f;
                }
            }
            
            particle_count++;
        }
    }

    std::sort(part.begin(), part.end(), [](const LDMpart& a, const LDMpart& b) {
        return a.timeidx < b.timeidx;
    });
}

void LDM::initializeParticlesEKI(){
    // EKI mode particle initialization using true_emissions from g_eki
    if (g_eki.true_emissions.empty()) {
        std::cerr << "[ERROR] No true_emissions data loaded for EKI particle initialization" << std::endl;
        return;
    }
    
    if (sources.empty()) {
        std::cerr << "[ERROR] No sources data loaded for EKI particle initialization" << std::endl;
        return;
    }
    
    std::cout << Color::MAGENTA << "[ENSEMBLE] " << Color::RESET
              << "Initializing " << nop << " particles ("
              << g_eki.true_emissions.size() << " time steps)" << std::endl;
    
    int particles_per_interval = nop / g_eki.true_emissions.size();
    
    std::random_device rd;
    auto now = std::chrono::high_resolution_clock::now();
    auto seed = now.time_since_epoch().count() + PROCESS_INDEX * 1000;
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(g_mpi.particleSizes[PROCESS_INDEX], g_mpi.sizeStandardDeviations[PROCESS_INDEX]);

    int interval_idx = 0;
    int count_in_interval = 0;
    int particle_count = 0;

    // Clear existing particles
    part.clear();

    for (int i = 0; i < nop; ++i) {
        // Use first source location (same as original)
        float x = (sources[0].lon + 179.0) / 0.5;
        float y = (sources[0].lat + 90) / 0.5;
        float z = sources[0].height;

        float random_radius = dist(gen);

        // Use true_emissions value for this time interval
        float emission_value = g_eki.true_emissions[interval_idx];

        // For single mode EKI: set all particles to timeidx=1 so they activate immediately
        // (different from ensemble mode where particles activate at different times)
        part.push_back(LDMpart(x, y, z,
                               g_mpi.decayConstants[PROCESS_INDEX],
                               emission_value,
                               g_mpi.depositionVelocities[PROCESS_INDEX],
                               random_radius,
                               g_mpi.particleDensities[PROCESS_INDEX],
                               i+1));  // Changed from (i+1) to (1) for immediate activation
        
        // Initialize multi-nuclide concentrations for the newly created particle
        LDMpart& current_particle = part.back();
        
        // Get nuclide configuration
        NuclideConfig* nucConfig = NuclideConfig::getInstance();
        int num_nuclides = nucConfig->getNumNuclides();
        
        // Initialize multi-nuclide concentrations array
        // IMPORTANT: This is required because move_part_by_wind_mpi kernel updates p.conc as sum of concentrations[]
        for(int nuc = 0; nuc < MAX_NUCLIDES; nuc++) {
            if (nuc < num_nuclides) {
                float initial_ratio = nucConfig->getInitialRatio(nuc);
                current_particle.concentrations[nuc] = emission_value * initial_ratio;
            } else {
                current_particle.concentrations[nuc] = 0.0f;
            }
        }
        
        particle_count++;
        count_in_interval++;

        // Move to next time interval when we've allocated enough particles
        if (count_in_interval == particles_per_interval) {
            interval_idx++;
            count_in_interval = 0;

            // Don't exceed the available time steps
            if (interval_idx >= g_eki.true_emissions.size()) {
                interval_idx = g_eki.true_emissions.size() - 1;
            }
        }
    }

    std::cout << Color::GREEN << "  " << Color::RESET
              << "Initialized " << Color::BOLD << part.size() << Color::RESET
              << " particles" << std::endl;

    std::sort(part.begin(), part.end(), [](const LDMpart& a, const LDMpart& b) {
        return a.timeidx < b.timeidx;
    });
}

void LDM::initializeParticlesEKI_AllEnsembles(float* ensemble_states, int num_ensembles, int num_timesteps) {
    // EKI ensemble mode: Initialize particles for all ensemble members in parallel
    // ensemble_states: [num_ensembles × num_timesteps] matrix (e.g., 100×24)

    if (sources.empty()) {
        std::cerr << "[ERROR] No sources data loaded for EKI ensemble initialization" << std::endl;
        return;
    }

    int particles_per_ensemble = nop;  // Particle count specified in setting.txt
    int particles_per_timestep = particles_per_ensemble / num_timesteps;
    int total_particles = particles_per_ensemble * num_ensembles;

    std::cout << Color::MAGENTA << "[ENSEMBLE] " << Color::RESET
              << "Initializing " << Color::BOLD << num_ensembles << Color::RESET
              << " ensembles × " << Color::BOLD << num_timesteps << Color::RESET
              << " timesteps (" << Color::BOLD << total_particles << Color::RESET << " particles)" << std::endl;


    std::random_device rd;
    auto now = std::chrono::high_resolution_clock::now();
    auto seed = now.time_since_epoch().count() + PROCESS_INDEX * 1000;
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(g_mpi.particleSizes[PROCESS_INDEX], g_mpi.sizeStandardDeviations[PROCESS_INDEX]);

    // Clear and reserve memory
    part.clear();
    part.reserve(total_particles);

    // Generate particles for each ensemble
    for (int ens = 0; ens < num_ensembles; ens++) {
        for (int t = 0; t < num_timesteps; t++) {
            // Current timestep's emission rate (ensemble_states is row-major: [ens * num_timesteps + t])
            float emission_value = ensemble_states[ens * num_timesteps + t];

            // Generate particles for this timestep
            for (int p = 0; p < particles_per_timestep; p++) {
                // Source location
                float x = (sources[0].lon + 179.0) / 0.5;
                float y = (sources[0].lat + 90) / 0.5;
                float z = sources[0].height;
                float random_radius = dist(gen);

                // timeidx: Local particle ID within each ensemble (1~particles_per_ensemble)
                // All ensembles have identical timeidx range for independent activation
                int timeidx = t * particles_per_timestep + p + 1;

                // Generate particle
                LDMpart particle(x, y, z,
                                g_mpi.decayConstants[PROCESS_INDEX],
                                emission_value,
                                g_mpi.depositionVelocities[PROCESS_INDEX],
                                random_radius,
                                g_mpi.particleDensities[PROCESS_INDEX],
                                timeidx);

                // Set ensemble ID
                particle.ensemble_id = ens;

                // Initialize multi-nuclide concentrations array
                // IMPORTANT: This is required because move_part_by_wind_mpi kernel updates p.conc as sum of concentrations[]
                NuclideConfig* nucConfig = NuclideConfig::getInstance();
                int num_nuclides = nucConfig->getNumNuclides();
                for(int nuc = 0; nuc < MAX_NUCLIDES; nuc++) {
                    if (nuc < num_nuclides) {
                        float initial_ratio = nucConfig->getInitialRatio(nuc);
                        particle.concentrations[nuc] = emission_value * initial_ratio;
                    } else {
                        particle.concentrations[nuc] = 0.0f;
                    }
                }

                part.push_back(particle);
            }
        }

        // Progress indicator (only show every 20%)
        if ((ens + 1) % (num_ensembles / 5) == 0 || (ens + 1) == num_ensembles) {
            std::cout << "\r  Progress: " << (ens + 1) << "/" << num_ensembles
                      << " (" << ((ens + 1) * 100 / num_ensembles) << "%)" << std::flush;
        }
    }

    std::cout << "\r" << Color::GREEN << "  " << Color::RESET
              << "Created " << Color::BOLD << part.size() << Color::RESET << " particles";

    // CRITICAL: Sort by ensemble_id (independent simulations)
    std::sort(part.begin(), part.end(), [](const LDMpart& a, const LDMpart& b) {
        if (a.ensemble_id != b.ensemble_id)
            return a.ensemble_id < b.ensemble_id;
        else
            return a.timeidx < b.timeidx;
    });

    std::cout << " (sorted by ensemble)" << std::endl;
}

void LDM::calculateAverageSettlingVelocity(){
            
            // settling rho, temp
            float prho = 2500.0;
            float radi = 6.0e-7*1.0e+6; // m to um
            float dsig = 3.0e-1;
    
            float xdummy = sqrt(2.0f)*logf(dsig);
            float delta = 6.0/static_cast<float>(NI);
            float d01 = radi*pow(dsig,-3.0);
            float d02, x01, x02, dmean, kn, alpha, cun, dc;
    
            float fract[NI] = {0, };
            float schmi[NI] = {0, };
            float vsh[NI] = {0, };
    
    
            for(int i=1; i<NI+1; i++){
                d02 = d01;
                d01 = radi*pow(dsig, -3.0 + delta*static_cast<float>(i));
                x01 = logf(d01/radi)/xdummy;
                x02 = logf(d02/radi)/xdummy;

    
                fract[i-1] = 0.5*(std::erf(x01)-std::erf(x02));
                dmean = 1.0e-6*exp(0.5*log(d01*d02));
                kn = 2.0f*_lam/dmean;


                if(-1.1/kn <= log10(_eps)*log(10.0))
                    alpha = 1.257;
                else
                    alpha = 1.257+0.4*exp(-1.1/kn);
    
                cun = 1.0 + alpha*kn;
                dc = _kb*_Tr*cun/(3.0*PI*_myl*dmean);
                //schmi = pow(_nyl/dc,-2.0/3.0);
                vsh[i-1] = _ga*prho*dmean*dmean*cun/(18.0*_myl);

            }

            for(int i=1; i<NI+1; i++){
                vsetaver -= fract[i-1]*vsh[i-1];
                cunningham += fract[i-1]*cun;
            }

            cudaError_t err = cudaMemcpyToSymbol(d_vsetaver, &vsetaver, sizeof(float));
            if (err != cudaSuccess) printf("Error copying vsetaver to symbol: %s\n", cudaGetErrorString(err));
            err = cudaMemcpyToSymbol(d_cunningham, &cunningham, sizeof(float));
            if (err != cudaSuccess) printf("Error copying cunningham to symbol: %s\n", cudaGetErrorString(err));


}

void LDM::calculateSettlingVelocity(){
    float prho = g_mpi.particleDensities[PROCESS_INDEX];
    float radi = g_mpi.particleSizes[PROCESS_INDEX];
    float dsig = g_mpi.sizeStandardDeviations[PROCESS_INDEX];

    if (radi == 0.0f) {
        vsetaver = 1.0f;
        cunningham = -1.0f;
        cudaError_t err = cudaMemcpyToSymbol(d_vsetaver, &vsetaver, sizeof(float));
        if (err != cudaSuccess) printf("Error copying vsetaver to symbol: %s\n", cudaGetErrorString(err));
        err = cudaMemcpyToSymbol(d_cunningham, &cunningham, sizeof(float));
        if (err != cudaSuccess) printf("Error copying cunningham to symbol: %s\n", cudaGetErrorString(err));
        return;
    }

    float xdummy = sqrt(2.0f)*logf(dsig);
    float delta = 6.0/static_cast<float>(NI);
    float d01 = radi*pow(dsig,-3.0);
    float d02, x01, x02, dmean, kn, alpha, cun, dc;

    float fract[NI] = {0, };
    float schmi[NI] = {0, };
    float vsh[NI] = {0, };

    vsetaver = 0.0f;
    cunningham = 0.0f;

    for(int i=1; i<NI+1; i++){
        d02 = d01;
        d01 = radi*pow(dsig, -3.0 + delta*static_cast<float>(i));
        x01 = logf(d01/radi)/xdummy;
        x02 = logf(d02/radi)/xdummy;


        fract[i-1] = 0.5*(std::erf(x01)-std::erf(x02));
        dmean = 1.0e-6*exp(0.5*log(d01*d02));
        kn = 2.0f*_lam/dmean;


        if(-1.1/kn <= log10(_eps)*log(10.0))
            alpha = 1.257;
        else
            alpha = 1.257+0.4*exp(-1.1/kn);

        cun = 1.0 + alpha*kn;
        dc = _kb*_Tr*cun/(3.0*PI*_myl*dmean);
        //schmi = pow(_nyl/dc,-2.0/3.0);
        vsh[i-1] = _ga*prho*dmean*dmean*cun/(18.0*_myl);

    }

    for(int i=1; i<NI+1; i++){
        vsetaver -= fract[i-1]*vsh[i-1];
        cunningham += fract[i-1]*cun;
    }

    cudaError_t err = cudaMemcpyToSymbol(d_vsetaver, &vsetaver, sizeof(float));
    if (err != cudaSuccess) printf("Error copying vsetaver to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_cunningham, &cunningham, sizeof(float));
    if (err != cudaSuccess) printf("Error copying cunningham to symbol: %s\n", cudaGetErrorString(err));
}

void LDM::cleanOutputDirectory() {
    std::cout << "Cleaning output directory... " << std::flush;

    // Remove all files in output directory
    #ifdef _WIN32
        system("del /Q output\\*.* 2>nul");
    #else
        system("rm -f output/*.vtk 2>/dev/null");
        system("rm -f output/*.csv 2>/dev/null");
        system("rm -f output/*.txt 2>/dev/null");
    #endif

    std::cout << Color::GREEN << "" << Color::RESET << std::endl;
}

void LDM::loadEKISettings() {
    std::cout << Color::CYAN << "[SYSTEM] " << Color::RESET
              << "Loading EKI settings... " << std::flush;
    
    FILE* ekiFile = fopen("data/eki_settings.txt", "r");
    if (!ekiFile) {
        std::cerr << "Failed to open data/eki_settings.txt" << std::endl;
        exit(1);
    }
    
    char buffer[256];
    
    // Initialize global EKI config with default values
    g_eki.mode = true;  // Set to true since this function is called when EKI mode is ON
    g_eki.time_interval = 15.0f;
    g_eki.time_unit = "minutes";
    g_eki.num_receptors = 0;
    g_eki.prior_mode = "constant";
    g_eki.prior_constant = 1.5e+8f;
    g_eki.ensemble_size = 50;
    g_eki.noise_level = 0.01f;
    
    // Clear vectors
    g_eki.receptor_locations.clear();
    g_eki.true_emissions.clear();
    g_eki.prior_emissions.clear();
    
    bool reading_receptor_matrix = false;
    bool reading_true_emissions = false;
    bool reading_prior_emissions = false;
    
    while (fgets(buffer, sizeof(buffer), ekiFile)) {
        // Skip comments and empty lines
        if (buffer[0] == '#' || buffer[0] == '\n' || buffer[0] == '\r') {
            continue;
        }
        
        // Parsing debug disabled - working correctly
        
        // Check for section headers
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
        
        // Check if this line is a key-value pair (contains '=')
        // If so, reset section flags and parse as key-value
        if (strchr(buffer, '=') != nullptr) {
            reading_receptor_matrix = false;
            reading_true_emissions = false;
            reading_prior_emissions = false;
            // Section flags reset for key-value parsing
        }
        
        // Parse based on current section
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
        
        // Parse key-value pairs (now that section flags are reset if needed)
        if (strchr(buffer, '=') != nullptr) {
            
            // Parse key-value pairs
            if (strstr(buffer, "EKI_TIME_INTERVAL=")) {
                sscanf(buffer, "EKI_TIME_INTERVAL=%f", &g_eki.time_interval);
            }
            else if (strstr(buffer, "EKI_TIME_UNIT=")) {
                char unit[32];
                sscanf(buffer, "EKI_TIME_UNIT=%s", unit);
                g_eki.time_unit = std::string(unit);
            }
            else if (strstr(buffer, "NUM_RECEPTORS=")) {
                sscanf(buffer, "NUM_RECEPTORS=%d", &g_eki.num_receptors);
            }
            else if (strstr(buffer, "RECEPTOR_CAPTURE_RADIUS=")) {
                sscanf(buffer, "RECEPTOR_CAPTURE_RADIUS=%f", &g_eki.receptor_capture_radius);
            }
            else if (strstr(buffer, "PRIOR_EMISSION_MODE=")) {
                char mode[32];
                sscanf(buffer, "PRIOR_EMISSION_MODE=%s", mode);
                g_eki.prior_mode = std::string(mode);
            }
            else if (strstr(buffer, "PRIOR_EMISSION_CONSTANT=")) {
                sscanf(buffer, "PRIOR_EMISSION_CONSTANT=%f", &g_eki.prior_constant);
            }
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
            else if (strstr(buffer, "EKI_SOURCE_LOCATION=")) {
                char loc[32];
                sscanf(buffer, "EKI_SOURCE_LOCATION=%s", loc);
                g_eki.source_location = std::string(loc);
            }
            else if (strstr(buffer, "EKI_NUM_SOURCE=")) {
                sscanf(buffer, "EKI_NUM_SOURCE=%d", &g_eki.num_source);
            }
            else if (strstr(buffer, "MEMORY_DOCTOR_MODE=")) {
                char mode[32];
                sscanf(buffer, "MEMORY_DOCTOR_MODE=%s", mode);
                g_eki.memory_doctor_mode = (strcmp(mode, "On") == 0 || strcmp(mode, "on") == 0 ||
                                           strcmp(mode, "ON") == 0 || strcmp(mode, "1") == 0);
            }
        }
        
        // Reset section flags only when encountering empty line (not newline characters at end)
        // Note: buffer[0] should not be '\r' or '\n' for normal config lines
        // This check should only trigger for truly empty lines, not lines with Windows line endings
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

// ================== GRID RECEPTOR DEBUG MODE FUNCTIONS ==================

void LDM::initializeGridReceptors(int grid_count_param, float grid_spacing_param) {
    // Store grid parameters
    grid_count = grid_count_param;
    grid_spacing = grid_spacing_param;
    grid_receptor_total = (2 * grid_count + 1) * (2 * grid_count + 1);

    std::cout << Color::CYAN << "[SYSTEM] " << Color::RESET
              << "Initializing " << Color::BOLD << grid_receptor_total << Color::RESET
              << " grid receptors (" << (2*grid_count+1) << "×" << (2*grid_count+1)
              << ", spacing=" << grid_spacing << "°)" << std::endl;

    // Source location (from setting.txt default)
    float source_lat = 37.0f;
    float source_lon = 141.0f;

    // Prepare host arrays for receptor locations
    std::vector<float> host_lats(grid_receptor_total);
    std::vector<float> host_lons(grid_receptor_total);

    // Generate grid receptor locations
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