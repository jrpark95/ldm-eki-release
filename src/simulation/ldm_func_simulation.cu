/**
 * @file ldm_func_simulation.cu
 * @brief Simulation module implementation
 */

#include "../core/ldm.cuh"
#include "ldm_func_simulation.cuh"
#include "../debug/kernel_error_collector.cuh"
#include "../colors.h"
#include <unistd.h>  // for isatty()

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

    // Grid parameters now passed via KernelScalars struct, not constant memory
    Mesh mesh(start_lat, start_lon, lat_step, lon_step, lat_num, lon_num);

    size_t meshSize = mesh.lat_count * mesh.lon_count * sizeof(float);

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
            (d_part, activationRatio, nop);
        cudaDeviceSynchronize();
        CHECK_KERNEL_ERROR();

        NuclideConfig* nucConfig = NuclideConfig::getInstance();

        // Populate KernelScalars for particle movement kernel
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
        ks.grid_start_lat = grid_config.start_lat;
        ks.grid_start_lon = grid_config.start_lon;
        ks.grid_lat_step = grid_config.lat_step;
        ks.grid_lon_step = grid_config.lon_step;
        ks.settling_vel = vsetaver;
        ks.cunningham_fac = cunningham;
        ks.T_matrix = d_T_matrix;
        ks.flex_hgt = d_flex_hgt;

        move_part_by_wind_mpi<<<blocks, threadsPerBlock>>>
        (d_part, t0, PROCESS_INDEX, d_dryDep, d_wetDep, mesh.lon_count, mesh.lat_count,
            device_meteorological_flex_unis0,
            device_meteorological_flex_pres0,
            device_meteorological_flex_unis1,
            device_meteorological_flex_pres1,
            ks);
        cudaDeviceSynchronize();
        CHECK_KERNEL_ERROR();

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

    // Display simulation mode and particle configuration
    std::cout << "\n" << Color::BOLD << Color::CYAN
              << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
              << "  SIMULATION CONFIGURATION\n"
              << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
              << Color::RESET;
    if (is_ensemble_mode) {
        std::cout << "  Mode           : " << Color::MAGENTA << "ENSEMBLE" << Color::RESET << std::endl;
        std::cout << "  Particles      : " << Color::BOLD << part.size() << Color::RESET
                  << " (" << ensemble_size << " × " << ensemble_num_states << " states)" << std::endl;
    } else {
        std::cout << "  Mode           : SINGLE" << std::endl;
        std::cout << "  Particles      : " << Color::BOLD << part.size() << Color::RESET << std::endl;
    }
    std::cout << "  GPU config     : " << blocks << " blocks × " << threadsPerBlock << " threads\n" << std::endl;

    GridConfig grid_config = loadGridConfig();
    float start_lat = grid_config.start_lat;
    float start_lon = grid_config.start_lon;
    float end_lat = grid_config.end_lat;
    float end_lon = grid_config.end_lon;
    float lat_step = grid_config.lat_step;
    float lon_step = grid_config.lon_step;
    
    int lat_num = static_cast<int>(round((end_lat - start_lat) / lat_step) + 1);
    int lon_num = static_cast<int>(round((end_lon - start_lon) / lon_step) + 1);

    // Grid parameters now passed via KernelScalars struct, not constant memory
    Mesh mesh(start_lat, start_lon, lat_step, lon_step, lat_num, lon_num);

    size_t meshSize = mesh.lat_count * mesh.lon_count * sizeof(float);

    float* d_dryDep = nullptr;
    float* d_wetDep = nullptr;

    cudaMalloc((void**)&d_dryDep, meshSize);
    cudaMalloc((void**)&d_wetDep, meshSize);

    cudaMemset(d_dryDep, 0, meshSize);
    cudaMemset(d_wetDep, 0, meshSize);

    // EKI mode: Check meteorological data preloading
    if (!g_eki_meteo.is_initialized) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET
                  << "EKI meteorological data not initialized. Call preloadAllEKIMeteorologicalData() first." << std::endl;
        return;
    }
    std::cout << "\nEKI simulation starting - using preloaded meteorological data\n" << std::endl;

    std::cout << Color::BOLD << Color::CYAN
              << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
              << "  RUNNING SIMULATION\n"
              << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
              << Color::RESET << std::endl;

#ifdef DEBUG
    // === NaN check 4: Before simulation start ===
    checkParticleNaN("Before simulation start");
#endif

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

            // DEBUG: Verify GPU meteorological data at first timestep
            if (timestep == 0) {
                FlexPres sample_pres[3];
                // Sample at particle's expected location (source: 37.0°N, 141.0°E)
                // GFS coordinates: lon_idx = (141 + 179) / 0.5 = 640, lat_idx = (37 + 90) / 0.5 = 254
                int test_xidx = 640;  // 141°E
                int test_yidx = 254;  // 37°N
                int test_zidx = 5;    // ~1000m altitude

                int idx0 = test_xidx * dimY_GFS * dimZ_GFS + test_yidx * dimZ_GFS + test_zidx;
                int idx1 = (test_xidx+1) * dimY_GFS * dimZ_GFS + test_yidx * dimZ_GFS + test_zidx;
                int idx2 = test_xidx * dimY_GFS * dimZ_GFS + (test_yidx+1) * dimZ_GFS + test_zidx;

                cudaMemcpy(sample_pres, &device_meteorological_flex_pres0[idx0], sizeof(FlexPres), cudaMemcpyDeviceToHost);
                cudaMemcpy(sample_pres+1, &device_meteorological_flex_pres0[idx1], sizeof(FlexPres), cudaMemcpyDeviceToHost);
                cudaMemcpy(sample_pres+2, &device_meteorological_flex_pres0[idx2], sizeof(FlexPres), cudaMemcpyDeviceToHost);

                extern std::ofstream* g_log_file;
                if (g_log_file && g_log_file->is_open()) {
                    *g_log_file << "[GPU_METEO_VERIFY] Timestep 0 - GPU meteorological data at source location (141°E, 37°N, ~1km):\n";
                    *g_log_file << "  Grid indices: xidx=" << test_xidx << ", yidx=" << test_yidx << ", zidx=" << test_zidx << "\n";
                    *g_log_file << "  Point [" << test_xidx << "," << test_yidx << "," << test_zidx << "]: UU=" << sample_pres[0].UU << " VV=" << sample_pres[0].VV << " WW=" << sample_pres[0].WW << " m/s\n";
                    *g_log_file << "  Point [" << (test_xidx+1) << "," << test_yidx << "," << test_zidx << "]: UU=" << sample_pres[1].UU << " VV=" << sample_pres[1].VV << " WW=" << sample_pres[1].WW << " m/s\n";
                    *g_log_file << "  Point [" << test_xidx << "," << (test_yidx+1) << "," << test_zidx << "]: UU=" << sample_pres[2].UU << " VV=" << sample_pres[2].VV << " WW=" << sample_pres[2].WW << " m/s\n";
                    *g_log_file << std::flush;
                }
            }

            // Update height data as well
            flex_hgt = g_eki_meteo.host_flex_hgt_data[past_meteo_index];

#ifdef DEBUG
            // Verify height data at first timestep only
            if (timestep == 0) {
                printf("[DEBUG] Timestep %d: Index %d height data first 5 values: ", timestep, past_meteo_index);
                for (int i = 0; i < std::min(5, (int)flex_hgt.size()); i++) {
                    printf("%.1f ", flex_hgt[i]);
                }
                printf("... %.1f\n", flex_hgt[flex_hgt.size()-1]);
            }
#endif
            
            // Copy height data to GPU memory
            cudaError_t hgt_err = cudaMemcpy(d_flex_hgt, flex_hgt.data(), sizeof(float) * dimZ_GFS, cudaMemcpyHostToDevice);
            if (hgt_err != cudaSuccess) {
                // Log to file only (collected by Kernel Error Collector for batch reporting)
                extern std::ofstream* g_log_file;
                if (g_log_file && g_log_file->is_open()) {
                    *g_log_file << "[ERROR] Failed to copy height data to GPU: "
                                << cudaGetErrorString(hgt_err) << "\n" << std::flush;
                }
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

#ifdef DEBUG
        // === Detailed NaN check at first timestep only ===
        if (timestep == 0) {
            printf("[DEBUG] First timestep - detailed check before kernel execution\n");
            printf("  Current time: %.1fs, t0: %.6f\n", currentTime, t0);
            printf("  Past meteo index: %d, Future meteo index: %d\n", past_meteo_index, future_meteo_index);
            checkParticleNaN("Before first kernel execution", 5);
            checkMeteoDataNaN("After meteorological data update");
        }
#endif

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
                (d_part, activationRatio, nop);
        }
        cudaDeviceSynchronize();
        CHECK_KERNEL_ERROR();

#ifdef DEBUG
        // === NaN check after first kernel ===
        if (timestep == 0) {
            checkParticleNaN("After update_particle_flags", 5);
        }
#endif

        NuclideConfig* nucConfig = NuclideConfig::getInstance();

        // Populate KernelScalars for particle movement kernel
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
        ks.grid_start_lat = grid_config.start_lat;
        ks.grid_start_lon = grid_config.start_lon;
        ks.grid_lat_step = grid_config.lat_step;
        ks.grid_lon_step = grid_config.lon_step;
        ks.settling_vel = vsetaver;
        ks.cunningham_fac = cunningham;
        ks.T_matrix = d_T_matrix;
        ks.flex_hgt = d_flex_hgt;

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
                total_particles,
                ks);
        } else {
            // Single mode: process d_nop particles
            move_part_by_wind_mpi<<<blocks, threadsPerBlock>>>
            (d_part, t0, PROCESS_INDEX, d_dryDep, d_wetDep, mesh.lon_count, mesh.lat_count,
                device_meteorological_flex_unis0,
                device_meteorological_flex_pres0,
                device_meteorological_flex_unis1,
                device_meteorological_flex_pres1,
                ks);
        }
        cudaDeviceSynchronize();
        CHECK_KERNEL_ERROR();

#ifdef DEBUG
        // === Most important check: After move_part_by_wind_mpi ===
        if (timestep == 0) {
            checkParticleNaN("After move_part_by_wind_mpi (first)", 5);
        } else if (timestep <= 3) {
            checkParticleNaN("After move_part_by_wind_mpi (timestep " + std::to_string(timestep) + ")", 3);
        }
#endif

        timestep++; 

        // Debug: Copy and print first particle position every timestep (first 10 steps)
        if(timestep <= 10) {
            LDMpart first_particle;
            cudaError_t err = cudaMemcpy(&first_particle, d_part, sizeof(LDMpart), cudaMemcpyDeviceToHost);
            if (err == cudaSuccess) {
                // Convert from GFS grid coordinates to geographic coordinates
                float lon = first_particle.x * 0.5f - 179.0f;
                float lat = first_particle.y * 0.5f - 90.0f;
                float z = first_particle.z;

                extern std::ofstream* g_log_file;
                if (g_log_file && g_log_file->is_open()) {
                    *g_log_file << "[PARTICLE_POSITION] Timestep " << timestep << " (t=" << currentTime << "s): "
                                << "lon=" << lon << "°E, lat=" << lat << "°N, z=" << z << "m, "
                                << "flag=" << (first_particle.flag ? "active" : "inactive") << ", "
                                << "timeidx=" << first_particle.timeidx << ", "
                                << "activationRatio=" << activationRatio << ", "
                                << "maxActiveTimeidx=" << int(part.size() * activationRatio) << ", "
                                << "u_wind=" << first_particle.u_wind << ", v_wind=" << first_particle.v_wind << ", w_wind=" << first_particle.w_wind << " m/s\n"
                                << std::flush;
                }
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
            static bool first_time = true;

            int total_steps = (int)(time_end/dt);
            float progress = (float)timestep / total_steps * 100.0f;
            int bar_width = 40;
            int bar_filled = (int)(progress / 100.0f * bar_width);

            // Terminal: Use ANSI codes for in-place update (goes to both terminal and log via TeeStreambuf)
            if (!first_time && g_sim.fixedScrollOutput) {
                fprintf(stderr, "\033[3A");  // Move up 3 lines (only affects terminal, ignored in log file)
            }

            fprintf(stderr, "\r-------------------------------------------------\033[K\n");
            fprintf(stderr, "\rTime: %8.1f sec │ Step: %4d/%4d [",
                   currentTime, timestep, total_steps);
            for (int i = 0; i < bar_width; i++) {
                if (i < bar_filled) fprintf(stderr, "█");
                else fprintf(stderr, "░");
            }
            fprintf(stderr, "] %.1f%%\033[K\n", progress);
            fprintf(stderr, "\rMeteo: Past=%d Future=%d │ t0=%.3f\033[K\n",
                   past_meteo_index,
                   (future_meteo_index < g_eki_meteo.num_time_steps) ? future_meteo_index : past_meteo_index,
                   t0);
            fflush(stderr);

            first_time = false;

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

    // Close the progress bar with a final separator line
    fprintf(stderr, "\r-------------------------------------------------\033[K\n");
    fflush(stderr);

    std::cout << "EKI simulation completed" << std::endl;
    std::cout << "Elapsed data time: " << totalElapsedTime << " seconds" << std::endl;
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

    // Display simulation mode and particle configuration
    std::cout << "\n" << Color::BOLD << Color::CYAN
              << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
              << "  SIMULATION CONFIGURATION\n"
              << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
              << Color::RESET;
    if (is_ensemble_mode) {
        std::cout << "  Mode           : " << Color::MAGENTA << "ENSEMBLE" << Color::RESET << std::endl;
        std::cout << "  Particles      : " << Color::BOLD << part.size() << Color::RESET
                  << " (" << ensemble_size << " × " << ensemble_num_states << " states)" << std::endl;
    } else {
        std::cout << "  Mode           : SINGLE" << std::endl;
        std::cout << "  Particles      : " << Color::BOLD << part.size() << Color::RESET << std::endl;
    }
    std::cout << "  GPU config     : " << blocks << " blocks × " << threadsPerBlock << " threads\n" << std::endl;

    GridConfig grid_config = loadGridConfig();
    float start_lat = grid_config.start_lat;
    float start_lon = grid_config.start_lon;
    float end_lat = grid_config.end_lat;
    float end_lon = grid_config.end_lon;
    float lat_step = grid_config.lat_step;
    float lon_step = grid_config.lon_step;
    
    int lat_num = static_cast<int>(round((end_lat - start_lat) / lat_step) + 1);
    int lon_num = static_cast<int>(round((end_lon - start_lon) / lon_step) + 1);

    // Grid parameters now passed via KernelScalars struct, not constant memory
    Mesh mesh(start_lat, start_lon, lat_step, lon_step, lat_num, lon_num);

    size_t meshSize = mesh.lat_count * mesh.lon_count * sizeof(float);

    float* d_dryDep = nullptr;
    float* d_wetDep = nullptr;

    cudaMalloc((void**)&d_dryDep, meshSize);
    cudaMalloc((void**)&d_wetDep, meshSize);

    cudaMemset(d_dryDep, 0, meshSize);
    cudaMemset(d_wetDep, 0, meshSize);

    // EKI mode: Check meteorological data preloading
    if (!g_eki_meteo.is_initialized) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET
                  << "EKI meteorological data not initialized. Call preloadAllEKIMeteorologicalData() first." << std::endl;
        return;
    }
    std::cout << "\nEKI simulation starting - using preloaded meteorological data\n" << std::endl;

    std::cout << Color::BOLD << Color::CYAN
              << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
              << "  RUNNING SIMULATION\n"
              << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
              << Color::RESET << std::endl;

#ifdef DEBUG
    // === NaN check 4: Before simulation start ===
    checkParticleNaN("Before simulation start");
#endif

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

            // DEBUG: Verify GPU meteorological data at first timestep
            if (timestep == 0) {
                FlexPres sample_pres[3];
                // Sample at particle's expected location (source: 37.0°N, 141.0°E)
                // GFS coordinates: lon_idx = (141 + 179) / 0.5 = 640, lat_idx = (37 + 90) / 0.5 = 254
                int test_xidx = 640;  // 141°E
                int test_yidx = 254;  // 37°N
                int test_zidx = 5;    // ~1000m altitude

                int idx0 = test_xidx * dimY_GFS * dimZ_GFS + test_yidx * dimZ_GFS + test_zidx;
                int idx1 = (test_xidx+1) * dimY_GFS * dimZ_GFS + test_yidx * dimZ_GFS + test_zidx;
                int idx2 = test_xidx * dimY_GFS * dimZ_GFS + (test_yidx+1) * dimZ_GFS + test_zidx;

                cudaMemcpy(sample_pres, &device_meteorological_flex_pres0[idx0], sizeof(FlexPres), cudaMemcpyDeviceToHost);
                cudaMemcpy(sample_pres+1, &device_meteorological_flex_pres0[idx1], sizeof(FlexPres), cudaMemcpyDeviceToHost);
                cudaMemcpy(sample_pres+2, &device_meteorological_flex_pres0[idx2], sizeof(FlexPres), cudaMemcpyDeviceToHost);

                extern std::ofstream* g_log_file;
                if (g_log_file && g_log_file->is_open()) {
                    *g_log_file << "[GPU_METEO_VERIFY] Timestep 0 - GPU meteorological data at source location (141°E, 37°N, ~1km):\n";
                    *g_log_file << "  Grid indices: xidx=" << test_xidx << ", yidx=" << test_yidx << ", zidx=" << test_zidx << "\n";
                    *g_log_file << "  Point [" << test_xidx << "," << test_yidx << "," << test_zidx << "]: UU=" << sample_pres[0].UU << " VV=" << sample_pres[0].VV << " WW=" << sample_pres[0].WW << " m/s\n";
                    *g_log_file << "  Point [" << (test_xidx+1) << "," << test_yidx << "," << test_zidx << "]: UU=" << sample_pres[1].UU << " VV=" << sample_pres[1].VV << " WW=" << sample_pres[1].WW << " m/s\n";
                    *g_log_file << "  Point [" << test_xidx << "," << (test_yidx+1) << "," << test_zidx << "]: UU=" << sample_pres[2].UU << " VV=" << sample_pres[2].VV << " WW=" << sample_pres[2].WW << " m/s\n";
                    *g_log_file << std::flush;
                }
            }

            // Update height data as well
            flex_hgt = g_eki_meteo.host_flex_hgt_data[past_meteo_index];

#ifdef DEBUG
            // Verify height data at first timestep only
            if (timestep == 0) {
                printf("[DEBUG] Timestep %d: Index %d height data first 5 values: ", timestep, past_meteo_index);
                for (int i = 0; i < std::min(5, (int)flex_hgt.size()); i++) {
                    printf("%.1f ", flex_hgt[i]);
                }
                printf("... %.1f\n", flex_hgt[flex_hgt.size()-1]);
            }
#endif
            
            // Copy height data to GPU memory
            cudaError_t hgt_err = cudaMemcpy(d_flex_hgt, flex_hgt.data(), sizeof(float) * dimZ_GFS, cudaMemcpyHostToDevice);
            if (hgt_err != cudaSuccess) {
                // Log to file only (collected by Kernel Error Collector for batch reporting)
                extern std::ofstream* g_log_file;
                if (g_log_file && g_log_file->is_open()) {
                    *g_log_file << "[ERROR] Failed to copy height data to GPU: "
                                << cudaGetErrorString(hgt_err) << "\n" << std::flush;
                }
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

#ifdef DEBUG
        // === Detailed NaN check at first timestep only ===
        if (timestep == 0) {
            printf("[DEBUG] First timestep - detailed check before kernel execution\n");
            printf("  Current time: %.1fs, t0: %.6f\n", currentTime, t0);
            printf("  Past meteo index: %d, Future meteo index: %d\n", past_meteo_index, future_meteo_index);
            checkParticleNaN("Before first kernel execution", 5);
            checkMeteoDataNaN("After meteorological data update");
        }
#endif

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
                (d_part, activationRatio, nop);
        }
        cudaDeviceSynchronize();
        CHECK_KERNEL_ERROR();

#ifdef DEBUG
        // === NaN check after first kernel ===
        if (timestep == 0) {
            checkParticleNaN("After update_particle_flags", 5);
        }
#endif

        NuclideConfig* nucConfig = NuclideConfig::getInstance();

        // Populate KernelScalars for particle movement kernel
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
        ks.grid_start_lat = grid_config.start_lat;
        ks.grid_start_lon = grid_config.start_lon;
        ks.grid_lat_step = grid_config.lat_step;
        ks.grid_lon_step = grid_config.lon_step;
        ks.settling_vel = vsetaver;
        ks.cunningham_fac = cunningham;
        ks.T_matrix = d_T_matrix;
        ks.flex_hgt = d_flex_hgt;

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
                total_particles,
                ks);
        } else {
            // Single mode: process d_nop particles
            move_part_by_wind_mpi_dump<<<blocks, threadsPerBlock>>>
            (d_part, t0, PROCESS_INDEX, d_dryDep, d_wetDep, mesh.lon_count, mesh.lat_count,
                device_meteorological_flex_unis0,
                device_meteorological_flex_pres0,
                device_meteorological_flex_unis1,
                device_meteorological_flex_pres1,
                ks);
        }
        cudaDeviceSynchronize();
        CHECK_KERNEL_ERROR();

#ifdef DEBUG
        // === Most important check: After move_part_by_wind_mpi ===
        if (timestep == 0) {
            checkParticleNaN("After move_part_by_wind_mpi (first)", 5);
        } else if (timestep <= 3) {
            checkParticleNaN("After move_part_by_wind_mpi (timestep " + std::to_string(timestep) + ")", 3);
        }
#endif

        timestep++; 

        // Debug: Copy and print first particle position every timestep (first 10 steps)
        if(timestep <= 10) {
            LDMpart first_particle;
            cudaError_t err = cudaMemcpy(&first_particle, d_part, sizeof(LDMpart), cudaMemcpyDeviceToHost);
            if (err == cudaSuccess) {
                // Convert from GFS grid coordinates to geographic coordinates
                float lon = first_particle.x * 0.5f - 179.0f;
                float lat = first_particle.y * 0.5f - 90.0f;
                float z = first_particle.z;

                extern std::ofstream* g_log_file;
                if (g_log_file && g_log_file->is_open()) {
                    *g_log_file << "[PARTICLE_POSITION] Timestep " << timestep << " (t=" << currentTime << "s): "
                                << "lon=" << lon << "°E, lat=" << lat << "°N, z=" << z << "m, "
                                << "flag=" << (first_particle.flag ? "active" : "inactive") << ", "
                                << "timeidx=" << first_particle.timeidx << ", "
                                << "activationRatio=" << activationRatio << ", "
                                << "maxActiveTimeidx=" << int(part.size() * activationRatio) << ", "
                                << "u_wind=" << first_particle.u_wind << ", v_wind=" << first_particle.v_wind << ", w_wind=" << first_particle.w_wind << " m/s\n"
                                << std::flush;
                }
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
            static bool first_time = true;

            int total_steps = (int)(time_end/dt);
            float progress = (float)timestep / total_steps * 100.0f;
            int bar_width = 40;
            int bar_filled = (int)(progress / 100.0f * bar_width);

            // Terminal: Use ANSI codes for in-place update (goes to both terminal and log via TeeStreambuf)
            if (!first_time && g_sim.fixedScrollOutput) {
                fprintf(stderr, "\033[3A");  // Move up 3 lines (only affects terminal, ignored in log file)
            }

            fprintf(stderr, "\r-------------------------------------------------\033[K\n");
            fprintf(stderr, "\rTime: %8.1f sec │ Step: %4d/%4d [",
                   currentTime, timestep, total_steps);
            for (int i = 0; i < bar_width; i++) {
                if (i < bar_filled) fprintf(stderr, "█");
                else fprintf(stderr, "░");
            }
            fprintf(stderr, "] %.1f%%\033[K\n", progress);
            fprintf(stderr, "\rMeteo: Past=%d Future=%d │ t0=%.3f\033[K\n",
                   past_meteo_index,
                   (future_meteo_index < g_eki_meteo.num_time_steps) ? future_meteo_index : past_meteo_index,
                   t0);
            fflush(stderr);

            first_time = false;

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

    // Close the progress bar with a final separator line
    fprintf(stderr, "\r-------------------------------------------------\033[K\n");
    fflush(stderr);

    std::cout << "EKI simulation completed" << std::endl;
    std::cout << "Elapsed data time: " << totalElapsedTime << " seconds" << std::endl;
}
