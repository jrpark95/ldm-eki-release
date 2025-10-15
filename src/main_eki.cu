#include "ldm.cuh"
#include "ldm_nuclides.cuh"
#include "ldm_eki_ipc.cuh"
#include "colors.h"

// Standard library includes
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <random>

// Physics model global variables (will be loaded from setting.txt)
int g_num_nuclides;  // Default value, updated from nuclide config
int g_turb_switch;    // Default values, overwritten by setting.txt
int g_drydep;
int g_wetdep; 
int g_raddecay;

int main(int argc, char** argv) {

    // Single process mode (MPI removed)

    // Clean logs directory and redirect output
    std::cout << Color::BOLD << Color::CYAN
              << "=== LDM-EKI: Source Term Inversion with Ensemble Kalman Methods ==="
              << Color::RESET << std::endl;
    std::cout << "Cleaning previous simulation data..." << std::endl;

    // Create necessary directories
    system("mkdir -p logs");
    system("mkdir -p output/plot_vtk_prior");
    system("mkdir -p output/plot_vtk_ens");
    system("mkdir -p output/results");
    system("mkdir -p logs/eki_iterations");

    // Use centralized cleanup script for data cleanup
    std::cout << "Running cleanup script (util/cleanup.py)..." << std::endl;
    int cleanup_ret = system("python3 util/cleanup.py");

    if (cleanup_ret == 0) {
        std::cout << Color::GREEN << "Cleanup completed successfully" << Color::RESET << std::endl;
    } else {
        std::cout << Color::YELLOW << "Cleanup script returned code " << cleanup_ret << Color::RESET << std::endl;
        // If user declined cleanup (exit code 0 from aborted script), continue anyway
        if (cleanup_ret != 0) {
            std::cout << "Continuing without cleanup..." << std::endl;
        }
    }
    
    // Open log file for output redirection
    std::ofstream logFile("logs/ldm_eki_simulation.log");
    if (!logFile.is_open()) {
        std::cerr << Color::RED << "[ERROR] Could not open log file logs/ldm_eki_simulation.log"
                  << Color::RESET << std::endl;
        return 1;
    }
    
    // Redirect cout and cerr to log file while keeping console output
    std::streambuf* coutbuf = std::cout.rdbuf();
    std::streambuf* cerrbuf = std::cerr.rdbuf();
    
    // Use tee-like functionality: output to both console and file
    class TeeStreambuf : public std::streambuf {
        std::streambuf* sb1;
        std::streambuf* sb2;
    public:
        TeeStreambuf(std::streambuf* sb1, std::streambuf* sb2) : sb1(sb1), sb2(sb2) {}
        int overflow(int c) {
            if (c == EOF) return !EOF;
            int r1 = sb1->sputc(c);
            int r2 = sb2->sputc(c);
            return (r1 == EOF || r2 == EOF) ? EOF : c;
        }
        int sync() {
            int r1 = sb1->pubsync();
            int r2 = sb2->pubsync();
            return (r1 == 0 && r2 == 0) ? 0 : -1;
        }
    };
    
    static TeeStreambuf tee_cout(coutbuf, logFile.rdbuf());
    static TeeStreambuf tee_cerr(cerrbuf, logFile.rdbuf());
    
    std::cout.rdbuf(&tee_cout);
    std::cerr.rdbuf(&tee_cerr);

    std::cout << "Log redirection active - output goes to console and logs/ldm_eki_simulation.log" << std::endl;

    // Load nuclide configuration
    NuclideConfig* nucConfig = NuclideConfig::getInstance();
    std::string nuclide_config_file = "./data/input/nuclides_config_1.txt";

    if (!nucConfig->loadFromFile(nuclide_config_file)) {
        std::cerr << Color::RED << "[ERROR] Failed to load nuclide configuration"
                  << Color::RESET << std::endl;
        return 1;
    }

    // Update global nuclide count
    g_num_nuclides = nucConfig->getNumNuclides();

    LDM ldm;

    ldm.loadSimulationConfiguration();

    // Load EKI settings
    std::cout << "Loading Ensemble Kalman Inversion settings..." << std::endl;
    ldm.loadEKISettings();

    // Initialize Memory Doctor if enabled in settings
    extern MemoryDoctor g_memory_doctor;
    g_memory_doctor.setEnabled(ldm.getEKIConfig().memory_doctor_mode);

    // Initialize CRAM system with A60.csv matrix
    std::cout << "Initializing CRAM system..." << std::endl;
    if (!ldm.initialize_cram_system("./cram/A60.csv")) {
        std::cerr << Color::RED << "[ERROR] CRAM system initialization failed"
                  << Color::RESET << std::endl;
        return 1;
    }
    std::cout << Color::GREEN << "CRAM system initialization completed"
              << Color::RESET << std::endl;

    ldm.calculateAverageSettlingVelocity();
    ldm.initializeParticlesEKI();

    // Preload all meteorological data for EKI mode
    std::cout << "Preloading meteorological data for fast iterations..." << std::endl;
    if (!ldm.preloadAllEKIMeteorologicalData()) {
        std::cerr << Color::RED << "[ERROR] Failed to preload meteorological data"
                  << Color::RESET << std::endl;
        return 1;
    }
    std::cout << Color::GREEN << "All meteorological data preloaded successfully"
              << Color::RESET << std::endl;

    ldm.allocateGPUMemory();

    // NOTE: Don't initialize EKI observation system here for single mode
    // It will be initialized properly for ensemble mode in the iteration loop

    // Initialize EKI IPC Writer
    LDM_EKI_IPC::EKIWriter eki_writer;
    int ensemble_size = ldm.getEKIConfig().ensemble_size;
    int num_receptors = ldm.getEKIConfig().receptor_locations.size();
    
    // Calculate number of timesteps dynamically
    // time_end (seconds) / EKI_TIME_INTERVAL (minutes) = number of observations
    const EKIConfig& eki_config = ldm.getEKIConfig();
    float eki_interval_minutes = eki_config.time_interval;
    float eki_interval_seconds = eki_interval_minutes * 60.0f;
    int num_timesteps = (int)(time_end / eki_interval_seconds);

    // Initialize EKI writer with full configuration
    if (!eki_writer.initialize(eki_config, num_timesteps)) {
        std::cerr << Color::RED << "[ERROR] Failed to initialize EKI IPC Writer"
                  << Color::RESET << std::endl;
        return 1;
    }

    ldm.startTimer();

    // Initialize EKI observation system for single mode
    std::cout << "Initializing observation system for single mode..." << std::endl;
    ldm.initializeEKIObservationSystem();

    // Enable VTK output for single mode run
    ldm.enable_vtk_output = true;
    std::cout << "[VTK] Output enabled for single mode run" << std::endl;

    // Run EKI simulation with preloaded meteorological data
    std::cout << "Running forward simulation..." << std::endl;
    ldm.runSimulation_eki();

    ldm.stopTimer();

    std::cout << Color::GREEN << "Simulation completed successfully"
              << Color::RESET << std::endl;

    // Save EKI observation results
    ldm.saveEKIObservationResults();

    // Write observations to shared memory
    std::cout << "Writing observations to shared memory..." << std::endl;
    
    // Get actual observation data from LDM object
    const std::vector<std::vector<float>>& observations = ldm.getEKIObservations();
    
    if (observations.empty()) {
        std::cerr << Color::RED << "[ERROR] No observations collected during simulation"
                  << Color::RESET << std::endl;
        return 1;
    }
    
    // Convert observations matrix to flat array for shared memory
    // observations[timestep][receptor] -> flat array[receptor * timesteps + timestep]
    std::vector<float> flat_observations(num_receptors * num_timesteps, 0.0f);
    
    for (size_t t = 0; t < observations.size() && t < static_cast<size_t>(num_timesteps); t++) {
        for (size_t r = 0; r < static_cast<size_t>(num_receptors) && r < observations[t].size(); r++) {
            flat_observations[r * num_timesteps + t] = observations[t][r];
        }
    }
    
    std::cout << "Collected " << observations.size() << " observation timesteps with "
              << (observations.empty() ? 0 : observations[0].size()) << " receptors each" << std::endl;

    bool success = eki_writer.writeObservations(flat_observations.data(), num_receptors, num_timesteps);

    if (!success) {
        std::cerr << Color::RED << "[ERROR] Failed to write observations to shared memory"
                  << Color::RESET << std::endl;
        return 1;
    }

    std::cout << Color::GREEN << "Observations successfully written to shared memory"
              << Color::RESET << std::endl;

    // Launch Python EKI script in background
    std::cout << "Launching Python EKI script in background..." << std::endl;
    int ret = system("PYTHONPATH=src/eki:$PYTHONPATH python src/eki/RunEstimator.py input_config input_data > logs/python_eki_output.log 2>&1 &");
    if (ret != 0) {
        std::cout << Color::YELLOW << "Warning: Failed to launch Python script (code: " << ret << ")"
                  << Color::RESET << std::endl;
    } else {
        std::cout << Color::GREEN << "Python script launched successfully in background"
                  << Color::RESET << std::endl;
        std::cout << "Python output will be saved to logs/python_eki_output.log" << std::endl;
        // Give Python a moment to start up
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // ========================================================================
    // ITERATION LOOP: Process multiple rounds of ensemble states from Python
    // ========================================================================
    int max_iterations = ldm.getEKIConfig().iteration;
    int current_iteration = 0;

    std::cout << "Maximum iterations configured: " << max_iterations << std::endl;

    LDM_EKI_IPC::EKIReader eki_reader;
    bool continue_iterations = true;

    while (continue_iterations && current_iteration < max_iterations) {
        current_iteration++;

        // ========================================================================
        // Wait for ensemble states from Python
        // ========================================================================
        std::cout << "\n" << Color::BOLD << Color::CYAN
                  << "========================================"
                  << Color::RESET << std::endl;
        std::cout << Color::BOLD << "ITERATION " << current_iteration << " - Waiting for ensemble states from Python..."
                  << Color::RESET << std::endl;
        std::cout << Color::BOLD << Color::CYAN
                  << "========================================"
                  << Color::RESET << "\n" << std::endl;

        if (!eki_reader.waitForEnsembleData(60)) {  // 60 second timeout
            if (current_iteration == 1) {
                std::cerr << Color::RED << "[ERROR] Timeout waiting for initial ensemble data from Python"
                          << Color::RESET << std::endl;
                std::cerr << Color::RED << "[ERROR] Python may have crashed or failed to send data"
                          << Color::RESET << std::endl;
            } else {
                std::cout << Color::GREEN << "No more ensemble data received - Python has completed all iterations"
                          << Color::RESET << std::endl;
                continue_iterations = false;
            }
            break;
        }

        // Read ensemble states
        std::vector<float> ensemble_data;
        int num_states, num_ensemble;

        if (!eki_reader.readEnsembleStates(ensemble_data, num_states, num_ensemble)) {
            std::cerr << Color::RED << "[ERROR] Failed to read ensemble states from shared memory"
                      << Color::RESET << std::endl;
            continue_iterations = false;
            break;
        }

        // ========================================================================
        // Display received ensemble data
        // ========================================================================
        std::cout << "\nEnsemble states received from Python" << std::endl;
        std::cout << "Matrix dimensions: " << num_states << " states × "
                  << num_ensemble << " ensemble members" << std::endl;

        // Calculate statistics
        float min_val = *std::min_element(ensemble_data.begin(), ensemble_data.end());
        float max_val = *std::max_element(ensemble_data.begin(), ensemble_data.end());
        float sum = std::accumulate(ensemble_data.begin(), ensemble_data.end(), 0.0f);
        float mean_val = sum / ensemble_data.size();

        std::cout << "Data statistics:" << std::endl;
        std::cout << "  Min:  " << std::scientific << min_val << std::endl;
        std::cout << "  Max:  " << max_val << std::endl;
        std::cout << "  Mean: " << mean_val << std::endl;

        // DEBUG: Count zeros and negative values for ALL iterations
        int zero_count = 0;
        int negative_count = 0;
        int tiny_count = 0;  // Values < 1e6
        for (const auto& val : ensemble_data) {
            if (val == 0.0f) zero_count++;
            if (val < 0.0f) negative_count++;
            if (val > 0.0f && val < 1.0e6f) tiny_count++;
        }

        std::cout << "[DEBUG_ITER" << current_iteration << "] Value analysis:" << std::endl;
        std::cout << "  - Zero values: " << zero_count << " / " << ensemble_data.size()
                  << " (" << (100.0f * zero_count / ensemble_data.size()) << "%)" << std::endl;
        std::cout << "  - Negative values: " << negative_count << std::endl;
        std::cout << "  - Tiny values (<1e6): " << tiny_count << std::endl;

        // Highlight if negatives are received
        if (negative_count > 0) {
            std::cout << Color::RED << "\n  WARNING: Negative values detected in iteration "
                      << current_iteration << Color::RESET << std::endl;
        } else {
            std::cout << Color::GREEN << "\n  All values non-negative in iteration "
                      << current_iteration << Color::RESET << std::endl;
        }

        // DEBUG: Save what LDM receives for comparison
        {
            std::string debug_filename = "/tmp/eki_debug/ldm_receives_iter_" + std::to_string(current_iteration) + ".txt";
            std::ofstream debug_file(debug_filename);
            if (debug_file.is_open()) {
                debug_file << "Iteration " << current_iteration << " data received by LDM\n";
                debug_file << "Shape: " << num_states << " x " << num_ensemble << "\n";
                debug_file << "Min: " << min_val << ", Max: " << max_val << ", Mean: " << mean_val << "\n";
                debug_file << "Negative count: " << negative_count << "\n";
                debug_file << "First 10 values: ";
                for (int idx = 0; idx < std::min(10, (int)ensemble_data.size()); idx++) {
                    debug_file << ensemble_data[idx] << " ";
                }
                debug_file << "\n";
                debug_file.close();
                std::cout << "[DEBUG] Saved received data info to " << debug_filename << std::endl;
            }
        }

        // Compare with previous iteration if available
        static float prev_min = 0.0f, prev_max = 0.0f, prev_mean = 0.0f;
        if (current_iteration > 1) {
            std::cout << "[DEBUG_ITER" << current_iteration << "] Change from previous iteration:" << std::endl;
            std::cout << "  - Min change: " << ((min_val - prev_min) / prev_min * 100.0f) << "%" << std::endl;
            std::cout << "  - Max change: " << ((max_val - prev_max) / prev_max * 100.0f) << "%" << std::endl;
            std::cout << "  - Mean change: " << ((mean_val - prev_mean) / prev_mean * 100.0f) << "%" << std::endl;
        }
        prev_min = min_val;
        prev_max = max_val;
        prev_mean = mean_val;

        // Only display detailed data for first iteration
        if (current_iteration == 1) {
            std::cout << "\nSample data (first state, first 20 ensemble members):" << std::endl;
            int display_count = std::min(20, num_ensemble);
            for (int i = 0; i < display_count; i++) {
                std::cout << "  [state 0, ensemble " << i << "] = " << ensemble_data[i] << std::endl;
            }

            std::cout << "\nSample data (first ensemble member, first 10 states):" << std::endl;
            int state_display = std::min(10, num_states);
            for (int s = 0; s < state_display; s++) {
                std::cout << "  [state " << s << ", ensemble 0] = "
                          << ensemble_data[s * num_ensemble] << std::endl;
            }
        }

        // ========================================================================
        // Ensemble Mode: Initialize particles with ensemble states
        // ========================================================================
        std::cout << "\n[ENSEMBLE] Preparing ensemble simulation for iteration " << current_iteration << std::endl;

        // Data format conversion: Python sends [state0_ens0, state0_ens1, ..., state1_ens0, ...]
        // We need: [ens0_state0, ens0_state1, ..., ens1_state0, ...] for row-major ensemble matrix
        std::vector<float> ensemble_matrix(num_ensemble * num_states);

        for (int s = 0; s < num_states; s++) {
            for (int e = 0; e < num_ensemble; e++) {
                // Python: ensemble_data[s * num_ensemble + e]
                // LDM needs: ensemble_matrix[e * num_states + s]
                ensemble_matrix[e * num_states + s] = ensemble_data[s * num_ensemble + e];
            }
        }

        // DEBUG: Check for zero values in ensemble matrix - CHECK ALL ENSEMBLES (first iteration only)
        if (current_iteration == 1) {
            int zero_count = 0;
            int nonzero_count = 0;
            float min_val = 1e20f, max_val = -1e20f;
            std::vector<std::string> zero_locations;

            std::cout << "[DEBUG_FULL_MATRIX] Checking ALL " << num_ensemble << " ensembles × " << num_states << " states..." << std::endl;

            for (int e = 0; e < num_ensemble; e++) {
                for (int s = 0; s < num_states; s++) {
                    float val = ensemble_matrix[e * num_states + s];
                    if (val == 0.0f) {
                        zero_count++;
                        if (zero_locations.size() < 50) {
                            zero_locations.push_back("[E" + std::to_string(e) + ",S" + std::to_string(s) + "]");
                        }
                    } else {
                        nonzero_count++;
                        min_val = std::min(min_val, val);
                        max_val = std::max(max_val, val);
                    }
                }
            }

            std::cout << "[DEBUG_FULL_MATRIX] RESULTS FOR ENTIRE MATRIX:" << std::endl;
            std::cout << "[DEBUG_FULL_MATRIX]   Total values: " << (num_ensemble * num_states) << std::endl;
            std::cout << "[DEBUG_FULL_MATRIX]   Zero values: " << zero_count
                      << " (" << (100.0f * zero_count / (num_ensemble * num_states)) << "%)" << std::endl;
            std::cout << "[DEBUG_FULL_MATRIX]   Non-zero values: " << nonzero_count << std::endl;

            if (nonzero_count > 0) {
                std::cout << "[DEBUG_FULL_MATRIX]   Min (non-zero): " << min_val << std::endl;
                std::cout << "[DEBUG_FULL_MATRIX]   Max: " << max_val << std::endl;
            }

            if (zero_count > 0) {
                std::cout << Color::YELLOW << "[DEBUG_FULL_MATRIX]   WARNING: Zero values detected in ensemble matrix!"
                          << Color::RESET << std::endl;
                std::cout << "[DEBUG_FULL_MATRIX]   First 50 zero locations:" << std::endl;
                int count = 0;
                for (const auto& loc : zero_locations) {
                    std::cout << loc << " ";
                    if (++count % 10 == 0) std::cout << std::endl << "                                      ";
                }
                std::cout << std::endl;

                // Check which states have zeros across ensembles
                std::cout << "[DEBUG_FULL_MATRIX]   Checking which states have zeros:" << std::endl;
                for (int s = 0; s < num_states; s++) {
                    int state_zeros = 0;
                    for (int e = 0; e < num_ensemble; e++) {
                        if (ensemble_matrix[e * num_states + s] == 0.0f) {
                            state_zeros++;
                        }
                    }
                    if (state_zeros > 0) {
                        std::cout << "[DEBUG_FULL_MATRIX]     State " << s << ": " << state_zeros << " zeros out of " << num_ensemble << " ensembles" << std::endl;
                    }
                }
            } else {
                std::cout << "[DEBUG_FULL_MATRIX]   ✓ NO ZEROS FOUND - All values are non-zero!" << std::endl;
            }
        }

        // Set ensemble mode flags (only on first iteration)
        if (current_iteration == 1) {
            ldm.is_ensemble_mode = true;
            ldm.ensemble_size = num_ensemble;
            ldm.ensemble_num_states = num_timesteps;  // Use observation timesteps (48) not state timesteps (24)

            // Select ensemble 7 for VTK output (fixed, not random)
            ldm.selected_ensemble_ids.clear();
            ldm.selected_ensemble_ids.push_back(7);

            std::cout << "[ENSEMBLE] Mode configured: " << num_ensemble << " ensembles, "
                      << num_timesteps << " observation timesteps" << std::endl;
            std::cout << "[ENSEMBLE] Selected ensemble 7 for VTK output (fixed)" << std::endl;

            // First time: cleanup single mode and initialize ensemble mode observation system
            ldm.cleanupEKIObservationSystem();
            ldm.initializeEKIObservationSystem();
        }

        // Enable VTK output ONLY on the final iteration for performance
        if (current_iteration == max_iterations) {
            ldm.enable_vtk_output = true;
            std::cout << "[VTK] Output enabled for final iteration " << current_iteration << std::endl;
        } else {
            ldm.enable_vtk_output = false;
            std::cout << "[VTK] Output disabled for iteration " << current_iteration << " (performance optimization)" << std::endl;
        }

        // Clear previous particles for reinitialization
        ldm.part.clear();

        // Initialize particles for all ensembles with new states
        ldm.initializeParticlesEKI_AllEnsembles(ensemble_matrix.data(), num_ensemble, num_states);

        // DEBUG: Check concentrations array after initialization (ALL iterations)
        // if (current_iteration == 1) {  // Commented out to check ALL iterations
            std::cout << "[DEBUG_ITER" << current_iteration << "] Checking concentrations[] after CPU initialization..." << std::endl;
            int check_count = 0;
            int particle_zero_count = 0;
            int particle_nonzero_count = 0;
            for (size_t i = 0; i < std::min(size_t(1000), ldm.part.size()); i++) {
                float total_conc = 0.0f;
                for (int nuc = 0; nuc < 60; nuc++) {
                    total_conc += ldm.part[i].concentrations[nuc];
                }
                if (total_conc == 0.0f) {
                    particle_zero_count++;
                } else {
                    particle_nonzero_count++;
                }
                if (check_count < 5) {
                    std::cout << "[DEBUG] Particle " << i << " (ens=" << ldm.part[i].ensemble_id
                              << ", timeidx=" << ldm.part[i].timeidx
                              << "): conc=" << ldm.part[i].conc
                              << ", sum(concentrations)=" << total_conc
                              << ", concentrations[0]=" << ldm.part[i].concentrations[0] << std::endl;
                    check_count++;
                }
            }
            std::cout << "[DEBUG_ITER" << current_iteration << "] First 1000 particles: " << particle_nonzero_count << " non-zero, "
                      << particle_zero_count << " zero concentrations" << std::endl;

            // DEBUG: Check Block 4 particles (indices 832-1040) that appear as zero in VTK
            std::cout << "[DEBUG_BLOCK4] Checking particles 832-841 (should be timeidx 4):" << std::endl;
            for (int i = 832; i <= 841 && i < ldm.part.size(); i++) {
                float total_conc = 0.0f;
                for (int nuc = 0; nuc < 60; nuc++) {
                    total_conc += ldm.part[i].concentrations[nuc];
                }
                std::cout << "[DEBUG_BLOCK4] Particle " << i
                          << ": ens=" << ldm.part[i].ensemble_id
                          << ", timeidx=" << ldm.part[i].timeidx
                          << ", flag=" << ldm.part[i].flag
                          << ", conc=" << ldm.part[i].conc
                          << ", sum=" << total_conc << std::endl;
            }
        // }  // End of if (current_iteration == 1) - Commented out to check ALL iterations

        // Verify particle count after initialization
        if (current_iteration == 1) {
            size_t expected_particles = static_cast<size_t>(num_ensemble) * num_states * (10000 / 24);
            std::cout << "[ENSEMBLE] Total particles after initialization: " << ldm.part.size() << std::endl;
            std::cout << "[ENSEMBLE] Expected particles: ~" << expected_particles
                      << " (" << num_ensemble << " ensembles × " << num_states
                      << " states × " << (10000/24) << " particles/state)" << std::endl;
        } else {
            std::cout << "[ENSEMBLE] Particles reinitialized: " << ldm.part.size() << std::endl;
        }

        if (ldm.part.size() == 0) {
            std::cerr << Color::RED << "[ERROR] No particles initialized! Check initializeParticlesEKI_AllEnsembles()"
                      << Color::RESET << std::endl;
            continue_iterations = false;
            break;
        }

        // Reallocate GPU memory for new particles
        if (ldm.d_part) {
            // IMPORTANT: Ensure all GPU operations are complete before freeing memory
            cudaDeviceSynchronize();
            cudaFree(ldm.d_part);
            ldm.d_part = nullptr;
        }
        ldm.allocateGPUMemory();  // Will allocate for new part.size()

        // Reset EKI observation system for this iteration
        // NOTE: Don't call cleanupEKIObservationSystem() here as it deallocates GPU memory
        // Instead, just reset the memory and host storage
        ldm.resetEKIObservationSystemForNewIteration();

    // Run ensemble simulation
    std::cout << "[ENSEMBLE] Starting forward simulation..." << std::endl;
    ldm.startTimer();
    ldm.runSimulation_eki();
    ldm.stopTimer();

    std::cout << Color::GREEN << "[ENSEMBLE] Simulation completed" << Color::RESET << std::endl;

    // ========================================================================
    // Send ensemble observations back to Python
    // ========================================================================
    std::cout << "\n[ENSEMBLE] Preparing to send observations to Python..." << std::endl;

    // Format: [num_ensemble × num_receptors × num_timesteps]
    // Python expects: flat array that can be reshaped to (num_ensemble, num_receptors, num_timesteps)

    // Get references to EKI data (eki_config already declared at line 138)
    auto& ensemble_observations = ldm.getEKIEnsembleObservations();

    // Flatten the 3D vector to a 1D array
    int total_obs_elements = num_ensemble * eki_config.num_receptors * num_timesteps;
    std::vector<float> flat_ensemble_observations(total_obs_elements);

    int idx = 0;
    for (int ens = 0; ens < num_ensemble; ens++) {
        // IMPORTANT: Match reference implementation order - timestep-major flattening
        // For each ensemble, we want: [T0_R0...R2, T1_R0...R2, ..., T23_R0...R2]
        for (int t = 0; t < num_timesteps; t++) {                  // timestep first (outer loop)
            for (int r = 0; r < eki_config.num_receptors; r++) {   // receptor second (inner loop)
                // eki_ensemble_observations is [ensemble][timestep][receptor]
                // We access as [ens][t][r] and flatten in timestep-major order
                flat_ensemble_observations[idx++] = ensemble_observations[ens][t][r];
            }
        }
    }

        // Initialize ensemble observation shared memory
        if (!eki_writer.initializeEnsembleObservations(num_ensemble, eki_config.num_receptors, num_timesteps)) {
            std::cerr << Color::RED << "[ERROR] Failed to initialize ensemble observation shared memory"
                      << Color::RESET << std::endl;
            continue_iterations = false;
            break;
        }

        // Write ensemble observations to shared memory
        if (!eki_writer.writeEnsembleObservations(flat_ensemble_observations.data(), num_ensemble, eki_config.num_receptors, num_timesteps, current_iteration)) {
            std::cerr << Color::RED << "[ERROR] Failed to write ensemble observations to shared memory"
                      << Color::RESET << std::endl;
            continue_iterations = false;
            break;
        }

    std::cout << Color::GREEN << "[ENSEMBLE] Successfully sent " << total_obs_elements
              << " observation values to shared memory" << Color::RESET << std::endl;
    std::cout << "[ENSEMBLE] Shape: [" << num_ensemble << " × "
              << eki_config.num_receptors << " × " << num_timesteps << "]" << std::endl;

    std::cout << Color::BOLD << Color::GREEN << "Iteration " << current_iteration << " completed"
              << Color::RESET << "\n" << std::endl;

    } // End of iteration loop

    // ========================================================================
    // Cleanup (after all iterations)
    // ========================================================================
    std::cout << Color::BOLD << Color::CYAN
              << "\nAll iterations completed. Cleaning up resources..."
              << Color::RESET << std::endl;
    std::cout << Color::GREEN << "Total iterations processed: " << current_iteration
              << Color::RESET << std::endl;

    // Cleanup EKI observation system
    ldm.cleanupEKIObservationSystem();

    // Cleanup meteorological data memory
    std::cout << "Cleaning up meteorological data memory..." << std::endl;
    ldm.cleanupEKIMeteorologicalData();

    // Cleanup shared memory
    eki_writer.cleanup();
    LDM_EKI_IPC::EKIWriter::unlinkSharedMemory();
    LDM_EKI_IPC::EKIReader::unlinkEnsembleSharedMemory();

    // Restore original stream buffers
    std::cout.rdbuf(coutbuf);
    std::cerr.rdbuf(cerrbuf);
    logFile.close();

    std::cout << Color::GREEN << "Simulation completed. All output saved to logs/ldm_eki_simulation.log"
              << Color::RESET << std::endl;

    // ========================================================================
    // Automatic Post-Processing: Generate Visualization
    // ========================================================================
    std::cout << "\n[VISUALIZATION] Generating comparison graphs..." << std::endl;

    // Check if visualization script exists
    std::ifstream viz_script("util/compare_all_receptors.py");
    if (viz_script.good()) {
        viz_script.close();

        int viz_ret = system("python3 util/compare_all_receptors.py > /tmp/ldm_viz.log 2>&1");

        if (viz_ret == 0) {
            std::cout << Color::GREEN << "[VISUALIZATION] Successfully generated: output/results/all_receptors_comparison.png"
                      << Color::RESET << std::endl;
        } else {
            std::cout << Color::YELLOW << "[VISUALIZATION] Visualization script failed (exit code: " << viz_ret << ")"
                      << Color::RESET << std::endl;
            std::cout << "[VISUALIZATION] Check /tmp/ldm_viz.log for details" << std::endl;
            std::cout << "[VISUALIZATION] You can manually run: python3 util/compare_all_receptors.py" << std::endl;
        }
    } else {
        std::cout << Color::YELLOW << "[VISUALIZATION] Script not found: util/compare_all_receptors.py"
                  << Color::RESET << std::endl;
        std::cout << "[VISUALIZATION] Skipping automatic visualization" << std::endl;
    }

    return 0;
}