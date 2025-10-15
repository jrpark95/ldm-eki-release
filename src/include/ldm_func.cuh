
#include "ldm.cuh"

// External CRAM matrix declarations
extern float* d_A_matrix_global;
extern float* d_exp_matrix_global;

LDM::LDM()
    : device_meteorological_data_pres(nullptr),
      device_meteorological_data_unis(nullptr),
      device_meteorological_data_etas(nullptr),
      is_ensemble_mode(false),
      ensemble_size(0),
      ensemble_num_states(0),
      d_ensemble_dose(nullptr),
      enable_vtk_output(false){
        // Initialize CRAM A matrix system
        std::string cram_path = "./cram/A60.csv";
        if (!initialize_cram_system(cram_path.c_str())) {
            std::cerr << "[WARNING] Failed to initialize CRAM A matrix from " << cram_path << std::endl;
            std::cerr << "[WARNING] Multi-nuclide decay calculations will be disabled" << std::endl;
        }
    }

LDM::~LDM(){

        if (device_meteorological_data_pres){
            cudaFree(device_meteorological_data_pres);
        }
        if (device_meteorological_data_unis){
            cudaFree(device_meteorological_data_unis);
        }
        if (device_meteorological_data_etas){
            cudaFree(device_meteorological_data_etas);
        }
        if (d_part){
            cudaFree(d_part);
        }
        
    }

void LDM::startTimer(){
        
        timerStart = std::chrono::high_resolution_clock::now();
    }

void LDM::stopTimer(){

        timerEnd = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(timerEnd - timerStart);
        std::cout << "Elapsed time: " << duration.count()/1.0e6 << " seconds" << std::endl;
    }

void LDM::allocateGPUMemory(){
        if (part.empty()) {
            std::cerr << "[ERROR] No particles to copy to device (part vector is empty)" << std::endl;
            return;
        }

        std::cout << "[GPU_ALLOC] Allocating GPU memory for " << part.size() << " particles" << std::endl;

        size_t total_size = part.size() * sizeof(LDMpart);

        cudaError_t err = cudaMalloc((void**)&d_part, total_size);
        if (err != cudaSuccess){
            std::cerr << "[ERROR] Failed to allocate device memory for particles: " << cudaGetErrorString(err) << std::endl;
            size_t free_mem, total_mem;
            cudaMemGetInfo(&free_mem, &total_mem);
            std::cerr << "[ERROR] GPU memory info - Free: " << free_mem/(1024*1024) << " MB, Total: " << total_mem/(1024*1024) << " MB" << std::endl;
            std::cerr << "[ERROR] Requested: " << total_size/(1024*1024) << " MB" << std::endl;
            exit(EXIT_FAILURE);
        }

        err = cudaMemcpy(d_part, part.data(), total_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess){
            std::cerr << "[ERROR] Failed to copy particle data from host to device: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_part);
            exit(EXIT_FAILURE);
        }

        // CRITICAL DEBUG: Verify GPU data immediately after copy
        // Check if concentrations are preserved during GPU transfer
        std::cout << "[GPU_VERIFY] Verifying GPU data immediately after cudaMemcpy..." << std::endl;

        // Sample first 3 particles for verification
        int verify_count = std::min(3, (int)part.size());
        std::vector<LDMpart> gpu_verify(verify_count);

        err = cudaMemcpy(gpu_verify.data(), d_part, verify_count * sizeof(LDMpart), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "[GPU_VERIFY] Failed to read back GPU data: " << cudaGetErrorString(err) << std::endl;
        } else {
            std::cout << "[GPU_VERIFY] Comparing CPU vs GPU data for first " << verify_count << " particles:" << std::endl;
            for (int i = 0; i < verify_count; i++) {
                const LDMpart& cpu_p = part[i];
                const LDMpart& gpu_p = gpu_verify[i];

                std::cout << "[GPU_VERIFY] Particle " << i << ":" << std::endl;
                std::cout << "  CPU: ens=" << cpu_p.ensemble_id << ", timeidx=" << cpu_p.timeidx
                         << ", flag=" << cpu_p.flag << ", conc=" << cpu_p.conc << std::endl;
                std::cout << "  GPU: ens=" << gpu_p.ensemble_id << ", timeidx=" << gpu_p.timeidx
                         << ", flag=" << gpu_p.flag << ", conc=" << gpu_p.conc << std::endl;

                // Check concentrations array
                float cpu_sum = 0.0f, gpu_sum = 0.0f;
                for (int nuc = 0; nuc < 3; nuc++) {  // Check first 3 nuclides
                    cpu_sum += cpu_p.concentrations[nuc];
                    gpu_sum += gpu_p.concentrations[nuc];
                }
                std::cout << "  CPU conc[0]=" << cpu_p.concentrations[0] << ", sum(first 3)=" << cpu_sum << std::endl;
                std::cout << "  GPU conc[0]=" << gpu_p.concentrations[0] << ", sum(first 3)=" << gpu_sum << std::endl;

                // Check if data matches
                if (std::abs(cpu_p.conc - gpu_p.conc) > 1e-3f) {
                    std::cerr << "[GPU_VERIFY] MISMATCH: conc field differs!" << std::endl;
                }
                if (std::abs(cpu_p.concentrations[0] - gpu_p.concentrations[0]) > 1e-3f) {
                    std::cerr << "[GPU_VERIFY] MISMATCH: concentrations[0] differs!" << std::endl;
                }
            }
        }

        // Clear any previous CUDA errors before checking
        cudaGetLastError(); // Clear previous errors
        err = cudaGetLastError(); // Should now be cudaSuccess
        if (err != cudaSuccess) {
            std::cerr << "[ERROR] CUDA error still present after particle memory copy: " << cudaGetErrorString(err) << std::endl;
            std::cerr << "[ERROR] This indicates a problem with earlier CUDA operations" << std::endl;
            std::cerr << "[ERROR] This may cause NaN values in meteorological data interpolation" << std::endl;
        } else {
            std::cout << "[DEBUG] Particle memory copy successful, no CUDA errors detected" << std::endl;
        }
    }

void LDM::runSimulation(){
    
    cudaError_t err = cudaGetLastError();

    std::future<void> gfs_future;
    bool gfs_ready = false;

    int ded;
    
    float currentTime = 0.0f;
    int threadsPerBlock = 256;
    int blocks = (part.size() + threadsPerBlock - 1) / threadsPerBlock;
    int timestep = 0;
    float activationRatio = 0.0;
    float t0 = 0.0;
    float totalElapsedTime = 0.0;

    GridConfig grid_config = loadGridConfig();
    float start_lat = grid_config.start_lat;
    float start_lon = grid_config.start_lon;
    float end_lat = grid_config.end_lat;
    float end_lon = grid_config.end_lon;
    float lat_step = grid_config.lat_step;
    float lon_step = grid_config.lon_step;
    
    int lat_num = static_cast<int>(round((end_lat - start_lat) / lat_step) + 1);
    int lon_num = static_cast<int>(round((end_lon - start_lon) / lon_step) + 1);

    err = cudaMemcpyToSymbol(d_start_lat, &start_lat, sizeof(float));
    if (err != cudaSuccess) printf("Error copying start_lat to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_start_lon, &start_lon, sizeof(float));
    if (err != cudaSuccess) printf("Error copying start_lon to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_lat_step, &lat_step, sizeof(float));
    if (err != cudaSuccess) printf("Error copying lat_step to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_lon_step, &lon_step, sizeof(float));
    if (err != cudaSuccess) printf("Error copying lon_step to symbol: %s\n", cudaGetErrorString(err));

    Mesh mesh(start_lat, start_lon, lat_step, lon_step, lat_num, lon_num);

    size_t meshSize = mesh.lat_count * mesh.lon_count * sizeof(float);

    std::cout << mesh.lon_count << mesh.lat_count << std::endl;

    float* d_dryDep = nullptr;
    float* d_wetDep = nullptr;

    cudaMalloc((void**)&d_dryDep, meshSize);
    cudaMalloc((void**)&d_wetDep, meshSize);

    cudaMemset(d_dryDep, 0, meshSize);
    cudaMemset(d_wetDep, 0, meshSize);

    FlexPres* flexpresdata = new FlexPres[(dimX_GFS + 1) * dimY_GFS * dimZ_GFS];
    FlexUnis* flexunisdata = new FlexUnis[(dimX_GFS + 1) * dimY_GFS];

    //log_first_particle_concentrations(0, 0.0f);

    while(currentTime < time_end){


        stepStart = std::chrono::high_resolution_clock::now();

        currentTime += dt;

        activationRatio = (currentTime) / time_end;
        t0 = (currentTime - static_cast<int>(currentTime/time_interval)*time_interval) / time_interval;

        update_particle_flags<<<blocks, threadsPerBlock>>>
            (d_part, activationRatio);
        cudaDeviceSynchronize();

        NuclideConfig* nucConfig = NuclideConfig::getInstance();
        move_part_by_wind_mpi<<<blocks, threadsPerBlock>>>
        (d_part, t0, PROCESS_INDEX, d_dryDep, d_wetDep, mesh.lon_count, mesh.lat_count,
            device_meteorological_flex_unis0,
            device_meteorological_flex_pres0,
            device_meteorological_flex_unis1,
            device_meteorological_flex_pres1);
        cudaDeviceSynchronize();

        timestep++;

        // Grid receptor debug mode: Record observations at regular intervals
        if (is_grid_receptor_mode) {
            computeGridReceptorObservations(timestep, currentTime);
        }

        // Debug: Copy and print first particle position every 5 timesteps
        if(timestep % 5 == 0) {  // Every 5 timesteps for tracking
            LDMpart first_particle;
            cudaError_t err = cudaMemcpy(&first_particle, d_part, sizeof(LDMpart), cudaMemcpyDeviceToHost);
            if (err == cudaSuccess) {
                // Convert from GFS grid coordinates to geographic coordinates
                float lon = first_particle.x * 0.5f - 179.0f;
                float lat = first_particle.y * 0.5f - 90.0f;
                float z = first_particle.z;
                
                // Debug output disabled for release
                // printf("[CRAM2] Timestep %d: Particle 0: lon=%.6f, lat=%.6f, z=%.2f | Wind: u=%.6f, v=%.6f, w=%.6f m/s\n", 
                //        timestep, lon, lat, z, first_particle.u_wind, first_particle.v_wind, first_particle.w_wind);
                // fflush(stdout); // Force output
            }
        }

        if(timestep % freq_output==0){
            printf("-------------------------------------------------\n");
            printf("Time : %f\tsec\n", currentTime);
            printf("Time steps : \t%d of \t%d\n", timestep, (int)(time_end/dt));
            
            // Debug GFS loading condition at key timepoints
            if(timestep == 1080) {  // Exactly at 10800 seconds
                printf("[DEBUG] GFS condition check: currentTime=%.1f, time_interval=%d, left=%d, gfs_idx=%d, condition=%s\n", 
                       currentTime, time_interval, static_cast<int>(currentTime/time_interval), gfs_idx,
                       (static_cast<int>(currentTime/time_interval) > gfs_idx) ? "TRUE" : "FALSE");
            }

            //particle_output_ASCII(timestep);
            outputParticlesBinaryMPI(timestep);
            
            // Log concentration data for analysis
            log_first_particle_concentrations(timestep, currentTime);
            log_all_particles_nuclide_ratios(timestep, currentTime);
            log_first_particle_cram_detail(timestep, currentTime, dt);
            log_first_particle_decay_analysis(timestep, currentTime);
            
            // Export validation reference data
            exportValidationData(timestep, currentTime);
        }
        stepEnd = std::chrono::high_resolution_clock::now();
        auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(stepEnd - stepStart);
        totalElapsedTime += static_cast<double>(duration0.count()/1.0e6);

        // Check GFS loading condition every timestep 
        int left_val = static_cast<int>(currentTime/time_interval);
        if(timestep >= 1079 && timestep <= 1081) {
            printf("[DEBUG] Step %d: currentTime=%.1f, left=%d, gfs_idx=%d\n", timestep, currentTime, left_val, gfs_idx);
        }
        
        if(left_val > gfs_idx) {
            printf("[INFO] Condition met: currentTime=%.1f, time_interval=%d, left=%d, gfs_idx=%d\n", 
                   currentTime, time_interval, left_val, gfs_idx);
            loadFlexGFSData();
        }

    }

    std::cout << "Elapsed data time: " << totalElapsedTime << " seconds" << std::endl;

    
}

void LDM::runSimulation_eki(){

    cudaError_t err = cudaGetLastError();

    int ded;

    float currentTime = 0.0f;
    int threadsPerBlock = 256;
    int blocks = (part.size() + threadsPerBlock - 1) / threadsPerBlock;
    int timestep = 0;
    float activationRatio = 0.0;
    float t0 = 0.0;
    float totalElapsedTime = 0.0;

    // Debug: Display simulation mode and particle configuration
    std::cout << "\n========================================" << std::endl;
    if (is_ensemble_mode) {
        std::cout << "[EKI_SIM] Running in ENSEMBLE mode" << std::endl;
        std::cout << "[EKI_SIM] Total particles: " << part.size()
                  << " (" << ensemble_size << " ensembles × " << ensemble_num_states
                  << " states)" << std::endl;
    } else {
        std::cout << "[EKI_SIM] Running in SINGLE mode" << std::endl;
        std::cout << "[EKI_SIM] Total particles: " << part.size() << std::endl;
    }
    std::cout << "[EKI_SIM] GPU kernel configuration: " << blocks << " blocks × "
              << threadsPerBlock << " threads" << std::endl;
    std::cout << "========================================\n" << std::endl;

    GridConfig grid_config = loadGridConfig();
    float start_lat = grid_config.start_lat;
    float start_lon = grid_config.start_lon;
    float end_lat = grid_config.end_lat;
    float end_lon = grid_config.end_lon;
    float lat_step = grid_config.lat_step;
    float lon_step = grid_config.lon_step;
    
    int lat_num = static_cast<int>(round((end_lat - start_lat) / lat_step) + 1);
    int lon_num = static_cast<int>(round((end_lon - start_lon) / lon_step) + 1);

    err = cudaMemcpyToSymbol(d_start_lat, &start_lat, sizeof(float));
    if (err != cudaSuccess) printf("Error copying start_lat to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_start_lon, &start_lon, sizeof(float));
    if (err != cudaSuccess) printf("Error copying start_lon to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_lat_step, &lat_step, sizeof(float));
    if (err != cudaSuccess) printf("Error copying lat_step to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_lon_step, &lon_step, sizeof(float));
    if (err != cudaSuccess) printf("Error copying lon_step to symbol: %s\n", cudaGetErrorString(err));

    Mesh mesh(start_lat, start_lon, lat_step, lon_step, lat_num, lon_num);

    size_t meshSize = mesh.lat_count * mesh.lon_count * sizeof(float);

    std::cout << mesh.lon_count << mesh.lat_count << std::endl;

    float* d_dryDep = nullptr;
    float* d_wetDep = nullptr;

    cudaMalloc((void**)&d_dryDep, meshSize);
    cudaMalloc((void**)&d_wetDep, meshSize);

    cudaMemset(d_dryDep, 0, meshSize);
    cudaMemset(d_wetDep, 0, meshSize);

    // EKI mode: Check meteorological data preloading
    if (!g_eki_meteo.is_initialized) {
        std::cerr << "[ERROR] EKI meteorological data not initialized. Call preloadAllEKIMeteorologicalData() first." << std::endl;
        return;
    }
    std::cout << "EKI simulation starting - using preloaded meteorological data" << std::endl;

    // === NaN check 4: Before simulation start ===
    checkParticleNaN("Before simulation start");

    //log_first_particle_concentrations(0, 0.0f);

    while(currentTime < time_end){

        stepStart = std::chrono::high_resolution_clock::now();

        currentTime += dt;

        activationRatio = (currentTime) / time_end;
        t0 = (currentTime - static_cast<int>(currentTime/time_interval)*time_interval) / time_interval;

        // EKI mode: Automatic meteorological data selection (past/future pair)
        int meteo_time_interval = Constants::time_interval;
        int past_meteo_index = static_cast<int>(currentTime / meteo_time_interval);
        int future_meteo_index = past_meteo_index + 1;
        
        // Range check and meteorological data update
        if (past_meteo_index >= 0 && past_meteo_index < g_eki_meteo.num_time_steps) {
            // Past meteorological data (device_meteorological_flex_pres0, unis0)
            FlexPres* past_pres_ptr;
            FlexUnis* past_unis_ptr;
            
            cudaMemcpy(&past_pres_ptr, &g_eki_meteo.device_flex_pres_data[past_meteo_index], 
                       sizeof(FlexPres*), cudaMemcpyDeviceToHost);
            cudaMemcpy(&past_unis_ptr, &g_eki_meteo.device_flex_unis_data[past_meteo_index], 
                       sizeof(FlexUnis*), cudaMemcpyDeviceToHost);
            
            cudaMemcpy(device_meteorological_flex_pres0, past_pres_ptr, 
                       g_eki_meteo.pres_data_size, cudaMemcpyDeviceToDevice);
            cudaMemcpy(device_meteorological_flex_unis0, past_unis_ptr, 
                       g_eki_meteo.unis_data_size, cudaMemcpyDeviceToDevice);
            
            // Update height data as well
            flex_hgt = g_eki_meteo.host_flex_hgt_data[past_meteo_index];
            
            // Verify height data at first timestep only
            if (timestep == 0) {
                printf("[DEBUG_HGT_UPDATE] Timestep %d: Index %d height data first 5 values: ", timestep, past_meteo_index);
                for (int i = 0; i < std::min(5, (int)flex_hgt.size()); i++) {
                    printf("%.1f ", flex_hgt[i]);
                }
                printf("... %.1f\n", flex_hgt[flex_hgt.size()-1]);
            }
            
            // CRITICAL FIX: Copy height data to GPU constant memory
            cudaError_t hgt_err = cudaMemcpyToSymbol(d_flex_hgt, flex_hgt.data(), sizeof(float) * dimZ_GFS);
            if (hgt_err != cudaSuccess) {
                printf("[ERROR] Failed to copy height data to GPU: %s\n", cudaGetErrorString(hgt_err));
            }
        }

        // Future meteorological data (device_meteorological_flex_pres1, unis1)
        if (future_meteo_index >= 0 && future_meteo_index < g_eki_meteo.num_time_steps) {
            FlexPres* future_pres_ptr;
            FlexUnis* future_unis_ptr;
            
            cudaMemcpy(&future_pres_ptr, &g_eki_meteo.device_flex_pres_data[future_meteo_index], 
                       sizeof(FlexPres*), cudaMemcpyDeviceToHost);
            cudaMemcpy(&future_unis_ptr, &g_eki_meteo.device_flex_unis_data[future_meteo_index], 
                       sizeof(FlexUnis*), cudaMemcpyDeviceToHost);
            
            cudaMemcpy(device_meteorological_flex_pres1, future_pres_ptr, 
                       g_eki_meteo.pres_data_size, cudaMemcpyDeviceToDevice);
            cudaMemcpy(device_meteorological_flex_unis1, future_unis_ptr, 
                       g_eki_meteo.unis_data_size, cudaMemcpyDeviceToDevice);
        } else {
            // Last time period: Set future data same as past data (prevent extrapolation)
            if (past_meteo_index >= 0 && past_meteo_index < g_eki_meteo.num_time_steps) {
                FlexPres* past_pres_ptr;
                FlexUnis* past_unis_ptr;
                
                cudaMemcpy(&past_pres_ptr, &g_eki_meteo.device_flex_pres_data[past_meteo_index], 
                           sizeof(FlexPres*), cudaMemcpyDeviceToHost);
                cudaMemcpy(&past_unis_ptr, &g_eki_meteo.device_flex_unis_data[past_meteo_index], 
                           sizeof(FlexUnis*), cudaMemcpyDeviceToHost);
                
                cudaMemcpy(device_meteorological_flex_pres1, past_pres_ptr, 
                           g_eki_meteo.pres_data_size, cudaMemcpyDeviceToDevice);
                cudaMemcpy(device_meteorological_flex_unis1, past_unis_ptr, 
                           g_eki_meteo.unis_data_size, cudaMemcpyDeviceToDevice);
            }
        }

        // === Detailed NaN check at first timestep only ===
        if (timestep == 0) {
            printf("[NaN_CHECK] First timestep - detailed check before kernel execution\n");
            printf("  Current time: %.1fs, t0: %.6f\n", currentTime, t0);
            printf("  Past meteo index: %d, Future meteo index: %d\n", past_meteo_index, future_meteo_index);
            checkParticleNaN("Before first kernel execution", 5);
            checkMeteoDataNaN("After meteorological data update");
        }

        // Use ensemble kernel for ensemble mode, regular kernel for single mode
        if (is_ensemble_mode) {
            // Ensemble mode: activate particles independently per ensemble
            int total_particles = part.size();
            int particles_per_ensemble = nop;  // Particles per ensemble (10000)
            update_particle_flags_ens<<<blocks, threadsPerBlock>>>
                (d_part, activationRatio, total_particles, particles_per_ensemble);
        } else {
            // Single mode: use standard activation
            update_particle_flags<<<blocks, threadsPerBlock>>>
                (d_part, activationRatio);
        }
        cudaDeviceSynchronize();

        // === NaN check after first kernel ===
        if (timestep == 0) {
            checkParticleNaN("After update_particle_flags", 5);
        }

        NuclideConfig* nucConfig = NuclideConfig::getInstance();

        // Use ensemble kernel for ensemble mode, regular kernel for single mode
        if (is_ensemble_mode) {
            // Ensemble mode: process all particles (d_nop × ensemble_size)
            int total_particles = part.size();
            move_part_by_wind_mpi_ens<<<blocks, threadsPerBlock>>>
            (d_part, t0, PROCESS_INDEX, d_dryDep, d_wetDep, mesh.lon_count, mesh.lat_count,
                device_meteorological_flex_unis0,
                device_meteorological_flex_pres0,
                device_meteorological_flex_unis1,
                device_meteorological_flex_pres1,
                total_particles);
        } else {
            // Single mode: process d_nop particles
            move_part_by_wind_mpi<<<blocks, threadsPerBlock>>>
            (d_part, t0, PROCESS_INDEX, d_dryDep, d_wetDep, mesh.lon_count, mesh.lat_count,
                device_meteorological_flex_unis0,
                device_meteorological_flex_pres0,
                device_meteorological_flex_unis1,
                device_meteorological_flex_pres1);
        }
        cudaDeviceSynchronize();

        // === Most important check: After move_part_by_wind_mpi ===
        if (timestep == 0) {
            checkParticleNaN("After move_part_by_wind_mpi (first)", 5);
        } else if (timestep <= 3) {
            checkParticleNaN("After move_part_by_wind_mpi (timestep " + std::to_string(timestep) + ")", 3);
        }

        timestep++; 

        // Debug: Copy and print first particle position every 5 timesteps  
        if(timestep % 5 == 0) {  // Every 5 timesteps for tracking
            LDMpart first_particle;
            cudaError_t err = cudaMemcpy(&first_particle, d_part, sizeof(LDMpart), cudaMemcpyDeviceToHost);
            if (err == cudaSuccess) {
                // Convert from GFS grid coordinates to geographic coordinates
                float lon = first_particle.x * 0.5f - 179.0f;
                float lat = first_particle.y * 0.5f - 90.0f;
                float z = first_particle.z;
            }
        }

        // EKI observation system - check if we should record observations (every timestep)
        if (is_ensemble_mode) {
            // Ensemble mode: compute observations for all ensemble members
            computeReceptorObservations_AllEnsembles(timestep, currentTime, ensemble_size, ensemble_num_states);
        } else {
            // Single mode: compute observations for single simulation
            computeReceptorObservations(timestep, currentTime);
        }

        if(timestep % freq_output==0){
            printf("-------------------------------------------------\n");
            printf("[EKI] Time : %f\tsec\n", currentTime);
            printf("[EKI] Time steps : \t%d of \t%d\n", timestep, (int)(time_end/dt));
            printf("[EKI] Meteo indices - Past: %d, Future: %d (interpolation ratio t0=%.3f)\n",
                   past_meteo_index,
                   (future_meteo_index < g_eki_meteo.num_time_steps) ? future_meteo_index : past_meteo_index,
                   t0);

            // Use ensemble output for ensemble mode, regular output for single mode
            // VTK output is DISABLED for performance - only enabled for final EKI iteration via enable_vtk_output flag
            if (enable_vtk_output) {
                if (is_ensemble_mode) {
                    outputParticlesBinaryMPI_ens(timestep);
                } else {
                    outputParticlesBinaryMPI(timestep);
                }
            }

            // // Log concentration data for analysis
            // log_first_particle_concentrations(timestep, currentTime);
            // log_all_particles_nuclide_ratios(timestep, currentTime);
            // log_first_particle_cram_detail(timestep, currentTime, dt);
            // log_first_particle_decay_analysis(timestep, currentTime);
            
            // // Export validation reference data
            // exportValidationData(timestep, currentTime);
        }
        stepEnd = std::chrono::high_resolution_clock::now();
        auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(stepEnd - stepStart);
        totalElapsedTime += static_cast<double>(duration0.count()/1.0e6);

        // EKI mode: No meteorological data loading needed (all data already in memory)
        // Previous loadFlexGFSData() call removed

    }

    std::cout << "EKI simulation completed" << std::endl;
    std::cout << "Elapsed data time: " << totalElapsedTime << " seconds" << std::endl;
}

void LDM::checkParticleNaN(const std::string& location, int max_check) {
    if (part.empty()) {
        std::cout << "[NaN_CHECK] " << location << ": Particle data is empty" << std::endl;
        return;
    }

    int nan_count = 0;
    int check_count = std::min(max_check, static_cast<int>(part.size()));

    for (int i = 0; i < check_count; i++) {
        const LDMpart& p = part[i];
        bool has_nan = std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z) ||
                      std::isnan(p.u_wind) || std::isnan(p.v_wind) || std::isnan(p.w_wind) ||
                      std::isnan(p.up) || std::isnan(p.vp) || std::isnan(p.wp) ||
                      std::isnan(p.u) || std::isnan(p.v) || std::isnan(p.w);

        if (has_nan) {
            printf("[NaN_CHECK] %s: NaN detected in particle %d!\n", location.c_str(), i);
            printf("  Position: x=%.6f, y=%.6f, z=%.6f\n", p.x, p.y, p.z);
            printf("  Wind: u=%.6f, v=%.6f, w=%.6f\n", p.u_wind, p.v_wind, p.w_wind);
            printf("  Turbulence: up=%.6f, vp=%.6f, wp=%.6f\n", p.up, p.vp, p.wp);
            printf("  Velocity: u=%.6f, v=%.6f, w=%.6f\n", p.u, p.v, p.w);
            nan_count++;
            if (nan_count >= 3) break; // Print max 3
        }
    }

    if (nan_count > 0) {
        printf("[NaN_CHECK] %s: NaN detected in %d particles! (Checked %d out of %d total)\n",
               location.c_str(), nan_count, check_count, static_cast<int>(part.size()));
    } else {
        std::cout << "[NaN_CHECK] " << location << ": No NaN (checked first " << check_count << " particles)" << std::endl;
    }
}

void LDM::checkMeteoDataNaN(const std::string& location) {
    // Check first meteorological data in host memory
    if (g_eki_meteo.is_initialized && !g_eki_meteo.host_flex_pres_data.empty() &&
        g_eki_meteo.host_flex_pres_data[0] != nullptr) {

        FlexPres* pres_data = g_eki_meteo.host_flex_pres_data[0];
        FlexUnis* unis_data = g_eki_meteo.host_flex_unis_data[0];

        int nan_count = 0;
        int check_count = 100; // Check first 100 grid points

        for (int i = 0; i < check_count && i < (dimX_GFS + 1) * dimY_GFS * dimZ_GFS; i++) {
            bool has_nan = std::isnan(pres_data[i].UU) || std::isnan(pres_data[i].VV) ||
                          std::isnan(pres_data[i].WW) || std::isnan(pres_data[i].RHO) ||
                          std::isnan(pres_data[i].DRHO);

            if (has_nan) {
                printf("[NaN_CHECK] %s: NaN detected in meteorological data index %d!\n", location.c_str(), i);
                printf("  UU=%.6f, VV=%.6f, WW=%.6f, RHO=%.6f, DRHO=%.6f\n",
                       pres_data[i].UU, pres_data[i].VV, pres_data[i].WW,
                       pres_data[i].RHO, pres_data[i].DRHO);
                nan_count++;
                if (nan_count >= 3) break;
            }
        }

        if (nan_count > 0) {
            printf("[NaN_CHECK] %s: NaN detected in %d meteorological data points!\n", location.c_str(), nan_count);
        } else {
            std::cout << "[NaN_CHECK] " << location << ": No NaN in meteorological data" << std::endl;
        }

        // Also check current meteorological data in GPU memory
        if (device_meteorological_flex_pres0 && device_meteorological_flex_pres1) {
            printf("[NaN_CHECK] %s: Checking GPU meteorological data slots...\n", location.c_str());

            // Get sample data from GPU
            FlexPres sample_pres0, sample_pres1;
            FlexUnis sample_unis0, sample_unis1;

            cudaMemcpy(&sample_pres0, device_meteorological_flex_pres0, sizeof(FlexPres), cudaMemcpyDeviceToHost);
            cudaMemcpy(&sample_pres1, device_meteorological_flex_pres1, sizeof(FlexPres), cudaMemcpyDeviceToHost);
            cudaMemcpy(&sample_unis0, device_meteorological_flex_unis0, sizeof(FlexUnis), cudaMemcpyDeviceToHost);
            cudaMemcpy(&sample_unis1, device_meteorological_flex_unis1, sizeof(FlexUnis), cudaMemcpyDeviceToHost);

            bool gpu_has_nan = std::isnan(sample_pres0.UU) || std::isnan(sample_pres0.VV) ||
                              std::isnan(sample_pres0.WW) || std::isnan(sample_pres0.RHO) ||
                              std::isnan(sample_pres1.UU) || std::isnan(sample_pres1.VV) ||
                              std::isnan(sample_pres1.WW) || std::isnan(sample_pres1.RHO);

            if (gpu_has_nan) {
                printf("[NaN_CHECK] %s: NaN detected in GPU meteorological data slots!\n", location.c_str());
                printf("  Slot0: UU=%.6f, VV=%.6f, WW=%.6f, RHO=%.6f\n",
                       sample_pres0.UU, sample_pres0.VV, sample_pres0.WW, sample_pres0.RHO);
                printf("  Slot1: UU=%.6f, VV=%.6f, WW=%.6f, RHO=%.6f\n",
                       sample_pres1.UU, sample_pres1.VV, sample_pres1.WW, sample_pres1.RHO);
            } else {
                printf("[NaN_CHECK] %s: No NaN in GPU meteorological data slots\n", location.c_str());
            }
        }
    } else {
        std::cout << "[NaN_CHECK] " << location << ": Meteorological data not initialized" << std::endl;
    }
}

// ================== EKI OBSERVATION SYSTEM IMPLEMENTATION ==================

void LDM::initializeEKIObservationSystem() {
    std::cout << "[EKI_OBS] Initializing EKI observation system..." << std::endl;
    
    if (g_eki.receptor_locations.empty()) {
        std::cerr << "[ERROR] No receptor locations loaded for EKI observation system" << std::endl;
        return;
    }
    
    int num_receptors = g_eki.num_receptors;
    
    // Allocate GPU memory for receptor data
    cudaMalloc(&d_receptor_lats, num_receptors * sizeof(float));
    cudaMalloc(&d_receptor_lons, num_receptors * sizeof(float));
    cudaMalloc(&d_receptor_dose, num_receptors * sizeof(float));
    cudaMalloc(&d_receptor_particle_count, num_receptors * sizeof(int));
    
    // Prepare host arrays
    std::vector<float> host_lats(num_receptors);
    std::vector<float> host_lons(num_receptors);
    
    for (int i = 0; i < num_receptors; i++) {
        host_lats[i] = g_eki.receptor_locations[i].first;   // latitude
        host_lons[i] = g_eki.receptor_locations[i].second;  // longitude
    }

    // Debug: Print receptor locations being sent to GPU
    std::cout << "[EKI_OBS] Receptor locations being sent to GPU:" << std::endl;
    for (int i = 0; i < num_receptors; i++) {
        std::cout << "  Receptor " << (i+1) << ": (" << host_lats[i] << ", " << host_lons[i] << ")" << std::endl;
    }

    // Copy receptor locations to GPU
    cudaMemcpy(d_receptor_lats, host_lats.data(), num_receptors * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_receptor_lons, host_lons.data(), num_receptors * sizeof(float), cudaMemcpyHostToDevice);
    
    // Initialize dose accumulation
    cudaMemset(d_receptor_dose, 0, num_receptors * sizeof(float));
    cudaMemset(d_receptor_particle_count, 0, num_receptors * sizeof(int));

    // Initialize observation storage
    eki_observations.clear();
    eki_particle_counts.clear();
    eki_observation_count = 0;

    // Ensemble mode: Allocate or reset ensemble dose memory
    if (is_ensemble_mode) {
        int ensemble_dose_size = ensemble_size * g_eki.num_receptors * ensemble_num_states;

        if (d_ensemble_dose == nullptr) {
            // First time allocation
            cudaMalloc(&d_ensemble_dose, ensemble_dose_size * sizeof(float));
            cudaMalloc(&d_ensemble_particle_count, ensemble_dose_size * sizeof(int));

            std::cout << "[EKI_OBS] Ensemble mode detected: allocating GPU memory for ensemble dose" << std::endl;
            std::cout << "[EKI_OBS] Ensemble dose size: " << ensemble_size << " × "
                      << g_eki.num_receptors << " × " << ensemble_num_states
                      << " = " << ensemble_dose_size << " floats" << std::endl;
        } else {
            std::cout << "[EKI_OBS] Resetting ensemble GPU memory for new iteration" << std::endl;
        }

        // ALWAYS reset to zero (both first allocation and subsequent iterations)
        cudaMemset(d_ensemble_dose, 0, ensemble_dose_size * sizeof(float));
        cudaMemset(d_ensemble_particle_count, 0, ensemble_dose_size * sizeof(int));

        // Initialize host storage for ensemble observations
        eki_ensemble_observations.clear();
        eki_ensemble_particle_counts.clear();
        eki_ensemble_observations.resize(ensemble_size);
        eki_ensemble_particle_counts.resize(ensemble_size);
        for (int ens = 0; ens < ensemble_size; ens++) {
            eki_ensemble_observations[ens].resize(ensemble_num_states);
            eki_ensemble_particle_counts[ens].resize(ensemble_num_states);
            for (int t = 0; t < ensemble_num_states; t++) {
                eki_ensemble_observations[ens][t].resize(g_eki.num_receptors, 0.0f);
                eki_ensemble_particle_counts[ens][t].resize(g_eki.num_receptors, 0);
            }
        }
        std::cout << "[EKI_OBS] Host storage initialized for " << ensemble_size << " ensembles" << std::endl;
    }

    std::cout << "[EKI_OBS] Initialized for " << num_receptors << " receptors" << std::endl;
    std::cout << "[EKI_OBS] Capture radius: " << g_eki.receptor_capture_radius << " degrees" << std::endl;
}

void LDM::computeReceptorObservations(int timestep, float currentTime) {
    // FIXED: Match reference code ACCUMULATION mode
    // Reference calls kernel every timestep, accumulating into different time_idx slots
    // Timesteps 1-9 accumulate into time_idx=0, timesteps 10-18 into time_idx=1, etc.

    if (timestep == 0) {
        return;  // Skip timestep 0
    }

    float eki_interval_seconds = g_eki.time_interval * 60.0f;
    int timesteps_per_observation = (int)(eki_interval_seconds / dt);

    // Calculate time_idx: CORRECT FORMULA FOR ACCUMULATION
    // timestep 1-9   → time_idx = 0 (ALL accumulate to first observation)
    // timestep 10-18 → time_idx = 1 (ALL accumulate to second observation)
    // timestep 19-27 → time_idx = 2 (ALL accumulate to third observation)
    int time_idx = (timestep - 1) / timesteps_per_observation;

    int max_observations = static_cast<int>(time_end / eki_interval_seconds);
    if (time_idx >= max_observations) {
        return;
    }

    int num_receptors = g_eki.num_receptors;
    int num_timesteps = max_observations;

    // First time initialization: allocate 2D GPU memory if needed
    static bool gpu_memory_allocated = false;
    static float* d_receptor_dose_2d = nullptr;
    static int* d_receptor_particle_count_2d = nullptr;

    if (!gpu_memory_allocated) {
        // Allocate 2D arrays: [num_timesteps × num_receptors] - MATCH REFERENCE CODE
        int total_size = num_timesteps * num_receptors;
        cudaMalloc(&d_receptor_dose_2d, total_size * sizeof(float));
        cudaMalloc(&d_receptor_particle_count_2d, total_size * sizeof(int));
        cudaMemset(d_receptor_dose_2d, 0, total_size * sizeof(float));
        cudaMemset(d_receptor_particle_count_2d, 0, total_size * sizeof(int));
        gpu_memory_allocated = true;

        std::cout << "[EKI_OBS] Allocated 2D GPU memory: " << num_timesteps << " timesteps × "
                  << num_receptors << " receptors = " << total_size << " floats (REFERENCE LAYOUT)" << std::endl;
    }

    int blockSize = 256;
    int numBlocks = (nop + blockSize - 1) / blockSize;

    // Call kernel EVERY timestep to accumulate into correct time_idx slot
    compute_eki_receptor_dose<<<numBlocks, blockSize>>>(
        d_part,
        d_receptor_lats, d_receptor_lons,
        g_eki.receptor_capture_radius,
        d_receptor_dose_2d,
        d_receptor_particle_count_2d,
        num_receptors,
        num_timesteps,
        time_idx  // Pass time_idx to kernel
    );

    cudaDeviceSynchronize();

    // Only copy results at observation boundaries (timesteps 9, 18, 27, ...)
    if (timestep % timesteps_per_observation == 0) {
        // Copy accumulated results for this time_idx
        std::vector<float> host_dose(num_receptors);
        std::vector<int> host_particle_count(num_receptors);

        for (int r = 0; r < num_receptors; r++) {
            // MATCH REFERENCE CODE: [timestep][receptor] layout
            int idx = time_idx * num_receptors + r;
            cudaMemcpy(&host_dose[r], &d_receptor_dose_2d[idx], sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&host_particle_count[r], &d_receptor_particle_count_2d[idx], sizeof(int), cudaMemcpyDeviceToHost);
        }

        // Store observations
        // Note: time_idx is already correct (0-indexed), just push directly
        eki_observations.push_back(host_dose);
        eki_particle_counts.push_back(host_particle_count);
        eki_observation_count = time_idx;
        // Store time for the PREVIOUS period (matching reference)
        eki_observation_times.push_back(currentTime - eki_interval_seconds);

        // Debug output
        std::cout << "[EKI_OBS] Observation " << time_idx << " at timestep " << timestep
                  << " (t=" << currentTime << "s):";
        for (int r = 0; r < num_receptors; r++) {
            std::cout << " R" << (r+1) << "=" << host_dose[r] << "(" << host_particle_count[r] << "p)";
        }
        std::cout << std::endl;
    }
}

void LDM::saveEKIObservationResults() {
    if (eki_observations.empty()) {
        std::cout << "[EKI_OBS] No observations to save" << std::endl;
        return;
    }
    
    std::cout << "[EKI_OBS] Saving " << eki_observations.size() 
              << " observations for " << g_eki.num_receptors << " receptors" << std::endl;
    
    std::ofstream file("logs/eki_receptor_observations.txt");
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open logs/eki_receptor_observations.txt for writing" << std::endl;
        return;
    }
    
    // Write header
    file << "# EKI Receptor Observation Results\n";
    file << "# Time interval: " << g_eki.time_interval << " minutes\n";
    file << "# Capture radius: " << g_eki.receptor_capture_radius << " degrees\n";
    file << "# Receptor locations:\n";
    
    for (int r = 0; r < g_eki.num_receptors; r++) {
        file << "#   Receptor " << (r+1) << ": (" 
             << g_eki.receptor_locations[r].first << ", " 
             << g_eki.receptor_locations[r].second << ")\n";
    }
    
    file << "#\n";
    file << "# Format: Time(min)";
    for (int r = 0; r < g_eki.num_receptors; r++) {
        file << "\tReceptor" << (r+1) << "_Dose(Sv)";
    }
    file << "\n";
    
    // Write data
    for (size_t t = 0; t < eki_observations.size(); t++) {
        // Use actual observation time if available, otherwise estimate
        int time_minutes;
        if (t < eki_observation_times.size()) {
            time_minutes = (int)round(eki_observation_times[t] / 60.0f);  // Convert seconds to integer minutes
        } else {
            time_minutes = (int)(t * g_eki.time_interval);  // Fallback to expected time
        }
        file << time_minutes;
        
        for (int r = 0; r < g_eki.num_receptors; r++) {
            file << "\t" << std::scientific << eki_observations[t][r];
        }
        file << "\n";
    }
    
    file.close();
    std::cout << "[EKI_OBS] Results saved to logs/eki_receptor_observations.txt" << std::endl;
}

bool LDM::writeEKIObservationsToSharedMemory(void* writer_ptr) {
    std::cout << "[EKI_IPC] writeEKIObservationsToSharedMemory called but implementation moved to main_eki.cu" << std::endl;
    // Implementation moved to main_eki.cu to avoid header dependencies
    return true;
}

void LDM::computeReceptorObservations_AllEnsembles(int timestep, float currentTime, int num_ensembles, int num_timesteps) {
    // FIXED: Match reference code ACCUMULATION mode for ensemble
    // Reference calls kernel every timestep, accumulating into different time_idx slots

    if (timestep == 0) {
        return;  // Skip timestep 0
    }

    float eki_interval_seconds = g_eki.time_interval * 60.0f;
    int timesteps_per_observation = (int)(eki_interval_seconds / dt);

    // Calculate time_idx: CORRECT FORMULA FOR ACCUMULATION
    // timestep 1-9   → time_idx = 0 (ALL accumulate to first observation)
    // timestep 10-18 → time_idx = 1 (ALL accumulate to second observation)
    // timestep 19-27 → time_idx = 2 (ALL accumulate to third observation)
    int time_idx = (timestep - 1) / timesteps_per_observation;

    if (time_idx >= num_timesteps) {
        return;
    }

    int num_receptors = g_eki.num_receptors;
    int total_particles = part.size();

    // Verify ensemble dose memory is initialized
    if (d_ensemble_dose == nullptr) {
        std::cerr << "[ERROR] Ensemble dose memory not initialized!" << std::endl;
        std::cerr << "       This should have been allocated in initializeEKIObservationSystem()" << std::endl;
        std::cerr << "       Attempting emergency allocation..." << std::endl;

        int ensemble_dose_size = num_ensembles * num_receptors * num_timesteps;
        cudaMalloc(&d_ensemble_dose, ensemble_dose_size * sizeof(float));
        cudaMalloc(&d_ensemble_particle_count, ensemble_dose_size * sizeof(int));
        cudaMemset(d_ensemble_dose, 0, ensemble_dose_size * sizeof(float));
        cudaMemset(d_ensemble_particle_count, 0, ensemble_dose_size * sizeof(int));

        std::cout << "[EKI_ENSEMBLE_OBS] Emergency GPU memory allocation: "
                  << num_ensembles << "×" << num_receptors << "×" << num_timesteps
                  << " = " << ensemble_dose_size << " floats" << std::endl;

        // Initialize host storage
        eki_ensemble_observations.resize(num_ensembles);
        eki_ensemble_particle_counts.resize(num_ensembles);
        for (int ens = 0; ens < num_ensembles; ens++) {
            eki_ensemble_observations[ens].resize(num_timesteps);
            eki_ensemble_particle_counts[ens].resize(num_timesteps);
            for (int t = 0; t < num_timesteps; t++) {
                eki_ensemble_observations[ens][t].resize(num_receptors, 0.0f);
                eki_ensemble_particle_counts[ens][t].resize(num_receptors, 0);
            }
        }
    }

    int blockSize = 256;
    int numBlocks = (total_particles + blockSize - 1) / blockSize;

    // Call kernel EVERY timestep to accumulate into correct time_idx slot
    compute_eki_receptor_dose_ensemble<<<numBlocks, blockSize>>>(
        d_part,
        d_receptor_lats, d_receptor_lons,
        g_eki.receptor_capture_radius,
        d_ensemble_dose,
        d_ensemble_particle_count,
        num_ensembles,
        num_receptors,
        num_timesteps,
        time_idx,  // Pass time_idx to kernel
        total_particles
    );

    cudaDeviceSynchronize();

    // Only copy results at observation boundaries (timesteps 9, 18, 27, ...)
    if (timestep % timesteps_per_observation == 0) {
        // Copy results back to host
        int ensemble_dose_size = num_ensembles * num_receptors * num_timesteps;
        std::vector<float> host_ensemble_dose(ensemble_dose_size);
        std::vector<int> host_ensemble_particle_count(ensemble_dose_size);
        cudaMemcpy(host_ensemble_dose.data(), d_ensemble_dose,
                   ensemble_dose_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_ensemble_particle_count.data(), d_ensemble_particle_count,
                   ensemble_dose_size * sizeof(int), cudaMemcpyDeviceToHost);

        // Store in structured format - MATCH REFERENCE CODE
        // GPU layout: [ensemble][timestep][receptor]
        for (int ens = 0; ens < num_ensembles; ens++) {
            for (int r = 0; r < num_receptors; r++) {
                // Reference index: ens * (TIME * RECEPT) + r + time_idx * RECEPT
                int idx = ens * (num_timesteps * num_receptors) +
                         time_idx * num_receptors +
                         r;
                eki_ensemble_observations[ens][time_idx][r] = host_ensemble_dose[idx];
                eki_ensemble_particle_counts[ens][time_idx][r] = host_ensemble_particle_count[idx];
            }
        }

        // Log observations for first 3 ensembles
        std::cout << "[EKI_ENSEMBLE_OBS] Recorded observation " << time_idx << " at timestep " << timestep
                  << " (t=" << currentTime << "s)" << std::endl;
        const int num_sample_ensembles = std::min(3, num_ensembles);
        for (int ens = 0; ens < num_sample_ensembles; ens++) {
            std::cout << "[EKI_ENSEMBLE_OBS] Ens" << ens << " obs" << time_idx << ":";
            for (int r = 0; r < num_receptors; r++) {
                std::cout << " R" << (r+1) << "=" << eki_ensemble_observations[ens][time_idx][r]
                         << "(" << eki_ensemble_particle_counts[ens][time_idx][r] << "p)";
            }
            std::cout << std::endl;
        }
    }
}

void LDM::cleanupEKIObservationSystem() {
    if (d_receptor_lats) {
        cudaFree(d_receptor_lats);
        d_receptor_lats = nullptr;
    }
    if (d_receptor_lons) {
        cudaFree(d_receptor_lons);
        d_receptor_lons = nullptr;
    }
    if (d_receptor_dose) {
        cudaFree(d_receptor_dose);
        d_receptor_dose = nullptr;
    }
    if (d_receptor_particle_count) {
        cudaFree(d_receptor_particle_count);
        d_receptor_particle_count = nullptr;
    }
    if (d_ensemble_dose) {
        cudaFree(d_ensemble_dose);
        d_ensemble_dose = nullptr;
    }
    if (d_ensemble_particle_count) {
        cudaFree(d_ensemble_particle_count);
        d_ensemble_particle_count = nullptr;
    }

    eki_observations.clear();
    eki_particle_counts.clear();
    eki_ensemble_observations.clear();
    eki_ensemble_particle_counts.clear();
    eki_observation_count = 0;

    std::cout << "[EKI_OBS] Cleanup completed" << std::endl;
}

void LDM::resetEKIObservationSystemForNewIteration() {
    std::cout << "[EKI_OBS] Resetting observation system for new iteration..." << std::endl;

    // Reset GPU memory without deallocating
    if (is_ensemble_mode && d_ensemble_dose != nullptr) {
        int ensemble_dose_size = ensemble_size * g_eki.num_receptors * ensemble_num_states;
        cudaMemset(d_ensemble_dose, 0, ensemble_dose_size * sizeof(float));
        cudaMemset(d_ensemble_particle_count, 0, ensemble_dose_size * sizeof(int));

        std::cout << "[EKI_OBS] Reset GPU memory: " << ensemble_dose_size << " floats" << std::endl;
    }

    // Clear and reinitialize host storage
    eki_ensemble_observations.clear();
    eki_ensemble_particle_counts.clear();
    eki_ensemble_observations.resize(ensemble_size);
    eki_ensemble_particle_counts.resize(ensemble_size);

    for (int ens = 0; ens < ensemble_size; ens++) {
        eki_ensemble_observations[ens].resize(ensemble_num_states);
        eki_ensemble_particle_counts[ens].resize(ensemble_num_states);
        for (int t = 0; t < ensemble_num_states; t++) {
            eki_ensemble_observations[ens][t].resize(g_eki.num_receptors, 0.0f);
            eki_ensemble_particle_counts[ens][t].resize(g_eki.num_receptors, 0);
        }
    }

    std::cout << "[EKI_OBS] Host storage reinitialized for " << ensemble_size
              << " ensembles, " << ensemble_num_states << " timesteps" << std::endl;
}

// ================== GRID RECEPTOR DEBUG MODE IMPLEMENTATION ==================

void LDM::computeGridReceptorObservations(int timestep, float currentTime) {
    if (!is_grid_receptor_mode) {
        return;  // Only run in grid receptor mode
    }

    // Record observations every 60 seconds (600 timesteps with dt=100s -> every 60s = 6 timesteps)
    const int observation_interval = 6;  // Every 6 timesteps = 600s

    if (timestep == 0 || timestep % observation_interval != 0) {
        return;
    }

    // Debug output
    static int obs_count = 0;
    obs_count++;
    if (obs_count <= 5 || obs_count % 10 == 0) {
        std::cout << "[GRID_OBS] Recording observation #" << obs_count
                  << " at timestep " << timestep << " (t=" << currentTime << "s)" << std::endl;
    }

    // Reset dose and particle count for this timestep
    cudaMemset(d_grid_receptor_dose, 0, grid_receptor_total * sizeof(float));
    cudaMemset(d_grid_receptor_particle_count, 0, grid_receptor_total * sizeof(int));

    // Launch GPU kernel to compute receptor doses
    // Reuse the same kernel as EKI observation system
    int blockSize = 256;
    int numBlocks = (nop + blockSize - 1) / blockSize;

    // Use receptor capture radius of 0.1 degrees (same as EKI default)
    float capture_radius = 0.1f;

    // Grid receptors don't need time_idx - call with num_timesteps=1, time_idx=0
    compute_eki_receptor_dose<<<numBlocks, blockSize>>>(
        d_part,
        d_grid_receptor_lats,
        d_grid_receptor_lons,
        capture_radius,
        d_grid_receptor_dose,
        d_grid_receptor_particle_count,
        grid_receptor_total,
        1,  // num_timesteps = 1 (no time dimension)
        0   // time_idx = 0 (always use slot 0)
    );

    cudaDeviceSynchronize();

    // Copy results back to host
    std::vector<float> host_dose(grid_receptor_total);
    std::vector<int> host_particle_count(grid_receptor_total);

    cudaMemcpy(host_dose.data(), d_grid_receptor_dose, grid_receptor_total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_particle_count.data(), d_grid_receptor_particle_count, grid_receptor_total * sizeof(int), cudaMemcpyDeviceToHost);

    // Store observations for each receptor
    for (int r = 0; r < grid_receptor_total; r++) {
        grid_receptor_observations[r].push_back(host_dose[r]);
        grid_receptor_particle_counts[r].push_back(host_particle_count[r]);
    }

    // Store observation time
    grid_observation_times.push_back(currentTime);

    // Occasional debug output showing max values
    if (obs_count <= 5 || obs_count % 10 == 0) {
        float max_dose = *std::max_element(host_dose.begin(), host_dose.end());
        int max_particles = *std::max_element(host_particle_count.begin(), host_particle_count.end());
        std::cout << "[GRID_OBS] Max dose: " << max_dose << " Sv, Max particles: " << max_particles << std::endl;
    }
}

void LDM::saveGridReceptorData() {
    if (!is_grid_receptor_mode || grid_observation_times.empty()) {
        std::cout << "[GRID] No grid receptor data to save" << std::endl;
        return;
    }

    std::cout << "[GRID] Saving data for " << grid_receptor_total << " receptors" << std::endl;
    std::cout << "[GRID] Number of time steps: " << grid_observation_times.size() << std::endl;

    // Save each receptor's data to a separate CSV file
    for (int r = 0; r < grid_receptor_total; r++) {
        // Create filename with zero-padded receptor number
        std::ostringstream filename;
        filename << "grid_receptors/receptor_" << std::setfill('0') << std::setw(4) << r << ".csv";

        std::ofstream outfile(filename.str());
        if (!outfile.is_open()) {
            std::cerr << "[ERROR] Could not open file: " << filename.str() << std::endl;
            continue;
        }

        // Write CSV header
        outfile << "Time(s),Dose(Sv),ParticleCount\n";

        // Write data for each timestep
        for (size_t t = 0; t < grid_observation_times.size(); t++) {
            outfile << grid_observation_times[t] << ","
                    << grid_receptor_observations[r][t] << ","
                    << grid_receptor_particle_counts[r][t] << "\n";
        }

        outfile.close();

        // Progress indicator
        if ((r+1) % 20 == 0 || r == grid_receptor_total - 1) {
            std::cout << "[GRID] Saved " << (r+1) << "/" << grid_receptor_total << " receptor files" << std::endl;
        }
    }

    // Save grid metadata
    std::ofstream metafile("grid_receptors/grid_metadata.txt");
    if (metafile.is_open()) {
        metafile << "Grid Configuration\n";
        metafile << "==================\n";
        metafile << "Grid count: " << grid_count << " (in each direction)\n";
        metafile << "Grid spacing: " << grid_spacing << " degrees\n";
        metafile << "Total receptors: " << grid_receptor_total << "\n";
        metafile << "Grid dimensions: " << (2*grid_count+1) << "×" << (2*grid_count+1) << "\n";
        metafile << "Number of time steps: " << grid_observation_times.size() << "\n";
        metafile << "Observation interval: 600 seconds (10 minutes)\n";
        metafile << "Receptor capture radius: 0.1 degrees\n";
        metafile.close();
    }

    std::cout << "[GRID] All receptor data saved successfully to grid_receptors/" << std::endl;
}

void LDM::cleanupGridReceptorSystem() {
    if (!is_grid_receptor_mode) {
        return;
    }

    std::cout << "[GRID] Cleaning up grid receptor system..." << std::endl;

    // Free GPU memory
    if (d_grid_receptor_lats) {
        cudaFree(d_grid_receptor_lats);
        d_grid_receptor_lats = nullptr;
    }
    if (d_grid_receptor_lons) {
        cudaFree(d_grid_receptor_lons);
        d_grid_receptor_lons = nullptr;
    }
    if (d_grid_receptor_dose) {
        cudaFree(d_grid_receptor_dose);
        d_grid_receptor_dose = nullptr;
    }
    if (d_grid_receptor_particle_count) {
        cudaFree(d_grid_receptor_particle_count);
        d_grid_receptor_particle_count = nullptr;
    }

    // Clear host storage
    grid_receptor_observations.clear();
    grid_receptor_particle_counts.clear();
    grid_observation_times.clear();

    std::cout << "[GRID] Cleanup completed" << std::endl;
}

void LDM::runSimulation_eki_dump(){

    cudaError_t err = cudaGetLastError();

    int ded;

    float currentTime = 0.0f;
    int threadsPerBlock = 256;
    int blocks = (part.size() + threadsPerBlock - 1) / threadsPerBlock;
    int timestep = 0;
    float activationRatio = 0.0;
    float t0 = 0.0;
    float totalElapsedTime = 0.0;

    // Debug: Display simulation mode and particle configuration
    std::cout << "\n========================================" << std::endl;
    if (is_ensemble_mode) {
        std::cout << "[EKI_SIM] Running in ENSEMBLE mode" << std::endl;
        std::cout << "[EKI_SIM] Total particles: " << part.size()
                  << " (" << ensemble_size << " ensembles × " << ensemble_num_states
                  << " states)" << std::endl;
    } else {
        std::cout << "[EKI_SIM] Running in SINGLE mode" << std::endl;
        std::cout << "[EKI_SIM] Total particles: " << part.size() << std::endl;
    }
    std::cout << "[EKI_SIM] GPU kernel configuration: " << blocks << " blocks × "
              << threadsPerBlock << " threads" << std::endl;
    std::cout << "========================================\n" << std::endl;

    GridConfig grid_config = loadGridConfig();
    float start_lat = grid_config.start_lat;
    float start_lon = grid_config.start_lon;
    float end_lat = grid_config.end_lat;
    float end_lon = grid_config.end_lon;
    float lat_step = grid_config.lat_step;
    float lon_step = grid_config.lon_step;
    
    int lat_num = static_cast<int>(round((end_lat - start_lat) / lat_step) + 1);
    int lon_num = static_cast<int>(round((end_lon - start_lon) / lon_step) + 1);

    err = cudaMemcpyToSymbol(d_start_lat, &start_lat, sizeof(float));
    if (err != cudaSuccess) printf("Error copying start_lat to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_start_lon, &start_lon, sizeof(float));
    if (err != cudaSuccess) printf("Error copying start_lon to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_lat_step, &lat_step, sizeof(float));
    if (err != cudaSuccess) printf("Error copying lat_step to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_lon_step, &lon_step, sizeof(float));
    if (err != cudaSuccess) printf("Error copying lon_step to symbol: %s\n", cudaGetErrorString(err));

    Mesh mesh(start_lat, start_lon, lat_step, lon_step, lat_num, lon_num);

    size_t meshSize = mesh.lat_count * mesh.lon_count * sizeof(float);

    std::cout << mesh.lon_count << mesh.lat_count << std::endl;

    float* d_dryDep = nullptr;
    float* d_wetDep = nullptr;

    cudaMalloc((void**)&d_dryDep, meshSize);
    cudaMalloc((void**)&d_wetDep, meshSize);

    cudaMemset(d_dryDep, 0, meshSize);
    cudaMemset(d_wetDep, 0, meshSize);

    // EKI mode: Check meteorological data preloading
    if (!g_eki_meteo.is_initialized) {
        std::cerr << "[ERROR] EKI meteorological data not initialized. Call preloadAllEKIMeteorologicalData() first." << std::endl;
        return;
    }
    std::cout << "EKI simulation starting - using preloaded meteorological data" << std::endl;

    // === NaN check 4: Before simulation start ===
    checkParticleNaN("Before simulation start");

    //log_first_particle_concentrations(0, 0.0f);

    while(currentTime < time_end){

        stepStart = std::chrono::high_resolution_clock::now();

        currentTime += dt;

        activationRatio = (currentTime) / time_end;
        t0 = (currentTime - static_cast<int>(currentTime/time_interval)*time_interval) / time_interval;

        // EKI mode: Automatic meteorological data selection (past/future pair)
        int meteo_time_interval = Constants::time_interval;
        int past_meteo_index = static_cast<int>(currentTime / meteo_time_interval);
        int future_meteo_index = past_meteo_index + 1;
        
        // Range check and meteorological data update
        if (past_meteo_index >= 0 && past_meteo_index < g_eki_meteo.num_time_steps) {
            // Past meteorological data (device_meteorological_flex_pres0, unis0)
            FlexPres* past_pres_ptr;
            FlexUnis* past_unis_ptr;
            
            cudaMemcpy(&past_pres_ptr, &g_eki_meteo.device_flex_pres_data[past_meteo_index], 
                       sizeof(FlexPres*), cudaMemcpyDeviceToHost);
            cudaMemcpy(&past_unis_ptr, &g_eki_meteo.device_flex_unis_data[past_meteo_index], 
                       sizeof(FlexUnis*), cudaMemcpyDeviceToHost);
            
            cudaMemcpy(device_meteorological_flex_pres0, past_pres_ptr, 
                       g_eki_meteo.pres_data_size, cudaMemcpyDeviceToDevice);
            cudaMemcpy(device_meteorological_flex_unis0, past_unis_ptr, 
                       g_eki_meteo.unis_data_size, cudaMemcpyDeviceToDevice);
            
            // Update height data as well
            flex_hgt = g_eki_meteo.host_flex_hgt_data[past_meteo_index];
            
            // Verify height data at first timestep only
            if (timestep == 0) {
                printf("[DEBUG_HGT_UPDATE] Timestep %d: Index %d height data first 5 values: ", timestep, past_meteo_index);
                for (int i = 0; i < std::min(5, (int)flex_hgt.size()); i++) {
                    printf("%.1f ", flex_hgt[i]);
                }
                printf("... %.1f\n", flex_hgt[flex_hgt.size()-1]);
            }
            
            // CRITICAL FIX: Copy height data to GPU constant memory
            cudaError_t hgt_err = cudaMemcpyToSymbol(d_flex_hgt, flex_hgt.data(), sizeof(float) * dimZ_GFS);
            if (hgt_err != cudaSuccess) {
                printf("[ERROR] Failed to copy height data to GPU: %s\n", cudaGetErrorString(hgt_err));
            }
        }

        // Future meteorological data (device_meteorological_flex_pres1, unis1)
        if (future_meteo_index >= 0 && future_meteo_index < g_eki_meteo.num_time_steps) {
            FlexPres* future_pres_ptr;
            FlexUnis* future_unis_ptr;
            
            cudaMemcpy(&future_pres_ptr, &g_eki_meteo.device_flex_pres_data[future_meteo_index], 
                       sizeof(FlexPres*), cudaMemcpyDeviceToHost);
            cudaMemcpy(&future_unis_ptr, &g_eki_meteo.device_flex_unis_data[future_meteo_index], 
                       sizeof(FlexUnis*), cudaMemcpyDeviceToHost);
            
            cudaMemcpy(device_meteorological_flex_pres1, future_pres_ptr, 
                       g_eki_meteo.pres_data_size, cudaMemcpyDeviceToDevice);
            cudaMemcpy(device_meteorological_flex_unis1, future_unis_ptr, 
                       g_eki_meteo.unis_data_size, cudaMemcpyDeviceToDevice);
        } else {
            // Last time period: Set future data same as past data (prevent extrapolation)
            if (past_meteo_index >= 0 && past_meteo_index < g_eki_meteo.num_time_steps) {
                FlexPres* past_pres_ptr;
                FlexUnis* past_unis_ptr;
                
                cudaMemcpy(&past_pres_ptr, &g_eki_meteo.device_flex_pres_data[past_meteo_index], 
                           sizeof(FlexPres*), cudaMemcpyDeviceToHost);
                cudaMemcpy(&past_unis_ptr, &g_eki_meteo.device_flex_unis_data[past_meteo_index], 
                           sizeof(FlexUnis*), cudaMemcpyDeviceToHost);
                
                cudaMemcpy(device_meteorological_flex_pres1, past_pres_ptr, 
                           g_eki_meteo.pres_data_size, cudaMemcpyDeviceToDevice);
                cudaMemcpy(device_meteorological_flex_unis1, past_unis_ptr, 
                           g_eki_meteo.unis_data_size, cudaMemcpyDeviceToDevice);
            }
        }

        // === Detailed NaN check at first timestep only ===
        if (timestep == 0) {
            printf("[NaN_CHECK] First timestep - detailed check before kernel execution\n");
            printf("  Current time: %.1fs, t0: %.6f\n", currentTime, t0);
            printf("  Past meteo index: %d, Future meteo index: %d\n", past_meteo_index, future_meteo_index);
            checkParticleNaN("Before first kernel execution", 5);
            checkMeteoDataNaN("After meteorological data update");
        }

        // Use ensemble kernel for ensemble mode, regular kernel for single mode
        if (is_ensemble_mode) {
            // Ensemble mode: activate particles independently per ensemble
            int total_particles = part.size();
            int particles_per_ensemble = nop;  // Particles per ensemble (10000)
            update_particle_flags_ens<<<blocks, threadsPerBlock>>>
                (d_part, activationRatio, total_particles, particles_per_ensemble);
        } else {
            // Single mode: use standard activation
            update_particle_flags<<<blocks, threadsPerBlock>>>
                (d_part, activationRatio);
        }
        cudaDeviceSynchronize();

        // === NaN check after first kernel ===
        if (timestep == 0) {
            checkParticleNaN("After update_particle_flags", 5);
        }

        NuclideConfig* nucConfig = NuclideConfig::getInstance();

        // Use ensemble kernel for ensemble mode, regular kernel for single mode
        if (is_ensemble_mode) {
            // Ensemble mode: process all particles (d_nop × ensemble_size)
            int total_particles = part.size();
            move_part_by_wind_mpi_ens_dump<<<blocks, threadsPerBlock>>>
            (d_part, t0, PROCESS_INDEX, d_dryDep, d_wetDep, mesh.lon_count, mesh.lat_count,
                device_meteorological_flex_unis0,
                device_meteorological_flex_pres0,
                device_meteorological_flex_unis1,
                device_meteorological_flex_pres1,
                total_particles);
        } else {
            // Single mode: process d_nop particles
            move_part_by_wind_mpi_dump<<<blocks, threadsPerBlock>>>
            (d_part, t0, PROCESS_INDEX, d_dryDep, d_wetDep, mesh.lon_count, mesh.lat_count,
                device_meteorological_flex_unis0,
                device_meteorological_flex_pres0,
                device_meteorological_flex_unis1,
                device_meteorological_flex_pres1);
        }
        cudaDeviceSynchronize();

        // === Most important check: After move_part_by_wind_mpi ===
        if (timestep == 0) {
            checkParticleNaN("After move_part_by_wind_mpi (first)", 5);
        } else if (timestep <= 3) {
            checkParticleNaN("After move_part_by_wind_mpi (timestep " + std::to_string(timestep) + ")", 3);
        }

        timestep++; 

        // Debug: Copy and print first particle position every 5 timesteps  
        if(timestep % 5 == 0) {  // Every 5 timesteps for tracking
            LDMpart first_particle;
            cudaError_t err = cudaMemcpy(&first_particle, d_part, sizeof(LDMpart), cudaMemcpyDeviceToHost);
            if (err == cudaSuccess) {
                // Convert from GFS grid coordinates to geographic coordinates
                float lon = first_particle.x * 0.5f - 179.0f;
                float lat = first_particle.y * 0.5f - 90.0f;
                float z = first_particle.z;
            }
        }

        // EKI observation system - check if we should record observations (every timestep)
        if (is_ensemble_mode) {
            // Ensemble mode: compute observations for all ensemble members
            computeReceptorObservations_AllEnsembles(timestep, currentTime, ensemble_size, ensemble_num_states);
        } else {
            // Single mode: compute observations for single simulation
            computeReceptorObservations(timestep, currentTime);
        }

        if(timestep % freq_output==0){
            printf("-------------------------------------------------\n");
            printf("[EKI] Time : %f\tsec\n", currentTime);
            printf("[EKI] Time steps : \t%d of \t%d\n", timestep, (int)(time_end/dt));
            printf("[EKI] Meteo indices - Past: %d, Future: %d (interpolation ratio t0=%.3f)\n",
                   past_meteo_index,
                   (future_meteo_index < g_eki_meteo.num_time_steps) ? future_meteo_index : past_meteo_index,
                   t0);

            // Use ensemble output for ensemble mode, regular output for single mode
            // VTK output is DISABLED for performance - only enabled for final EKI iteration via enable_vtk_output flag
            if (enable_vtk_output) {
                if (is_ensemble_mode) {
                    outputParticlesBinaryMPI_ens(timestep);
                } else {
                    outputParticlesBinaryMPI(timestep);
                }
            }

            // // Log concentration data for analysis
            // log_first_particle_concentrations(timestep, currentTime);
            // log_all_particles_nuclide_ratios(timestep, currentTime);
            // log_first_particle_cram_detail(timestep, currentTime, dt);
            // log_first_particle_decay_analysis(timestep, currentTime);
            
            // // Export validation reference data
            // exportValidationData(timestep, currentTime);
        }
        stepEnd = std::chrono::high_resolution_clock::now();
        auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(stepEnd - stepStart);
        totalElapsedTime += static_cast<double>(duration0.count()/1.0e6);

        // EKI mode: No meteorological data loading needed (all data already in memory)
        // Previous loadFlexGFSData() call removed

    }

    std::cout << "EKI simulation completed" << std::endl;
    std::cout << "Elapsed data time: " << totalElapsedTime << " seconds" << std::endl;
}