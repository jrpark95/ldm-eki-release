#include "ldm.cuh"
#include "ldm_nuclides.cuh"
#include <omp.h>

int LDM::countActiveParticles(){
    int count = 0;
    for(int i = 0; i < nop; ++i) if(part[i].flag == 1) count++;
    return count;
}

void LDM::swapByteOrder(float& value){
    char* valuePtr = reinterpret_cast<char*>(&value);
    std::swap(valuePtr[0], valuePtr[3]);
    std::swap(valuePtr[1], valuePtr[2]);
}

void LDM::swapByteOrder(int& value){
    char* valuePtr = reinterpret_cast<char*>(&value);
    std::swap(valuePtr[0], valuePtr[3]);
    std::swap(valuePtr[1], valuePtr[2]);
}

void LDM::outputParticlesBinaryMPI(int timestep){

    cudaMemcpy(part.data(), d_part, nop * sizeof(LDMpart), cudaMemcpyDeviceToHost);
    int numa = 0;
    int numb = 0;

    int part_num = countActiveParticles();

    // Debug output disabled for release

    std::ostringstream filenameStream;
    std::string path;

    // Create output directory for VTK files - NEW STRUCTURE
    path = "output/plot_vtk_prior";

    #ifdef _WIN32
        _mkdir(path.c_str());
        filenameStream << path << "\\" << "plot_" << std::setfill('0')
                       << std::setw(5) << timestep << ".vtk";
    #else
        mkdir(path.c_str(), 0777);
        filenameStream << path << "/" << "plot_" << std::setfill('0')
                       << std::setw(5) << timestep << ".vtk";
    #endif
    std::string filename = filenameStream.str();

    std::ofstream vtkFile(filename, std::ios::binary);

    if (!vtkFile.is_open()){
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return;
    }

    vtkFile << "# vtk DataFile Version 4.0\n";
    vtkFile << "particle data\n";
    vtkFile << "BINARY\n";
    vtkFile << "DATASET POLYDATA\n";

    vtkFile << "POINTS " << part_num << " float\n";
    float zsum = 0.0;
    for (int i = 0; i < nop; ++i){
        if(!part[i].flag) continue;
        // float x = part[i].x;
        // float y = part[i].y;
        // float z = part[i].z/3000.0;

        float x = -179.0 + part[i].x*0.5;
        float y = -90.0 + part[i].y*0.5;
        float z = part[i].z/3000.0;
        zsum += part[i].z;

        swapByteOrder(x);
        swapByteOrder(y);
        swapByteOrder(z);

        vtkFile.write(reinterpret_cast<char*>(&x), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&y), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&z), sizeof(float));
    }
    // __Z[timestep-1]=zsum/static_cast<float>(part_num);


    vtkFile << "POINT_DATA " << part_num << "\n";
    // vtkFile << "SCALARS u_wind float 1\n";
    // vtkFile << "LOOKUP_TABLE default\n";
    // for (int i = 0; i < nop; ++i){
    //     if(!part[i].flag) continue;
    //     float vval = part[i].u_wind;
    //     swapByteOrder(vval); 
    //     vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    // }

    // vtkFile << "SCALARS v_wind float 1\n"; 
    // vtkFile << "LOOKUP_TABLE default\n";
    // for (int i = 0; i < nop; ++i){
    //     if(!part[i].flag) continue;
    //     float vval = part[i].v_wind;
    //     swapByteOrder(vval); 
    //     vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    // }

    // vtkFile << "SCALARS w_wind float 1\n"; 
    // vtkFile << "LOOKUP_TABLE default\n";
    // for (int i = 0; i < nop; ++i){
    //     if(!part[i].flag) continue;
    //     float vval = part[i].w_wind;
    //     swapByteOrder(vval); 
    //     vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    // }

    // vtkFile << "SCALARS virtual_dist float 1\n";
    // vtkFile << "LOOKUP_TABLE default\n";
    // for (int i = 0; i < nop; ++i){
    //     if(!part[i].flag) continue;
    //     float vval = part[i].virtual_distance;
    //     swapByteOrder(vval); 
    //     vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    // }

    vtkFile << "SCALARS Q float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nop; ++i){
        if(!part[i].flag) continue;
        float vval = part[i].conc;
        swapByteOrder(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS time_idx int 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < nop; ++i){
        if(!part[i].flag) continue;
        int vval = part[i].timeidx;
        swapByteOrder(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(int));
    }

    // // Output representative nuclide concentration (I-131, index 32)
    // const int representative_nuclide = 32;  // I-131 is at index 32 in the 60-nuclide configuration
    // vtkFile << "SCALARS I131_concentration float 1\n";
    // vtkFile << "LOOKUP_TABLE default\n";
    // for (int i = 0; i < nop; ++i){
    //     if(!part[i].flag) continue;
    //     float vval = part[i].concentrations[representative_nuclide];
    //     swapByteOrder(vval);
    //     vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    // }


    vtkFile.close();
}

void LDM::outputParticlesBinaryMPI_ens(int timestep){

    // Set OpenMP threads for optimal performance (28 physical cores × 2 = 56 threads)
    // Use 48 threads to leave headroom for system and GPU tasks
    omp_set_num_threads(50);

    static bool first_call = true;
    if (first_call) {
        std::cout << "[OpenMP] Ensemble output parallelization enabled with "
                  << omp_get_max_threads() << " threads" << std::endl;
        std::cout << "[VTK_OUTPUT] Selected ensemble for output: ensemble "
                  << selected_ensemble_ids[0] << " (fixed)" << std::endl;
        first_call = false;
    }

    // Ensemble mode: use actual particle count from part vector
    // This was set during initializeParticlesEKI_AllEnsembles()
    size_t total_particles = part.size();

    if (total_particles == 0) {
        std::cerr << "[ERROR] part vector is empty in outputParticlesBinaryMPI_ens!" << std::endl;
        return;
    }

    // Copy all particles from GPU
    cudaMemcpy(part.data(), d_part, total_particles * sizeof(LDMpart), cudaMemcpyDeviceToHost);

    // // DEBUG: Check concentrations after GPU->Host copy (EVERY output at timestep 60)
    // if (timestep == 60) {
    //     std::cout << "[DEBUG_VTK] Checking concentrations after GPU->Host copy..." << std::endl;
    //     std::cout << "[DEBUG_VTK] Total particles: " << total_particles << std::endl;

    //     // Check ALL particles, not just first 10000
    //     int zero_count = 0;
    //     int nonzero_count = 0;
    //     int checked = 0;

    //     // Sample check: first 10000 AND particles around index 379392 (ensemble 38)
    //     std::cout << "[DEBUG_VTK] Checking first 10000 particles..." << std::endl;
    //     for (size_t i = 0; i < std::min(size_t(10000), total_particles); i++) {
    //         if (part[i].flag) {
    //             checked++;
    //             if (part[i].conc == 0.0f) {
    //                 zero_count++;
    //             } else {
    //                 nonzero_count++;
    //             }
    //         }
    //     }
    //     std::cout << "[DEBUG_VTK] First 10000: checked=" << checked
    //               << ", Zero=" << zero_count << ", Non-zero=" << nonzero_count << std::endl;

    //     // Check around ensemble 38 region (particles 379392 ~ 389375)
    //     std::cout << "[DEBUG_VTK] Checking ensemble 38 region (379392~389375)..." << std::endl;
    //     zero_count = nonzero_count = checked = 0;
    //     for (size_t i = 379392; i < std::min(size_t(389376), total_particles); i++) {
    //         if (part[i].flag) {
    //             checked++;
    //             if (part[i].conc == 0.0f) {
    //                 zero_count++;
    //                 if (zero_count <= 5) {
    //                     std::cout << "[DEBUG_VTK] ZERO in ens38: particle " << i
    //                               << " (ens=" << part[i].ensemble_id
    //                               << ", conc=" << part[i].conc
    //                               << ", conc[0]=" << part[i].concentrations[0] << ")" << std::endl;
    //                 }
    //             } else {
    //                 nonzero_count++;
    //             }
    //         }
    //     }
    //     std::cout << "[DEBUG_VTK] Ensemble 38 region: checked=" << checked
    //               << ", Zero=" << zero_count << ", Non-zero=" << nonzero_count << std::endl;
    // }

    // Create output_ens directory - NEW STRUCTURE
    std::string path = "output/plot_vtk_ens";

    #ifdef _WIN32
        _mkdir(path.c_str());
    #else
        mkdir(path.c_str(), 0777);
    #endif

    // Group particles by ensemble_id and write separate files
    if (!is_ensemble_mode) {
        std::cerr << "[WARNING] outputParticlesBinaryMPI_ens called but not in ensemble mode" << std::endl;
        return;
    }

    // Check if selected ensembles are initialized
    if (selected_ensemble_ids.empty()) {
        std::cerr << "[WARNING] No ensembles selected for output. Skipping VTK output." << std::endl;
        return;
    }

    // Pre-filter particles only for selected ensembles (3 ensembles instead of all 100)
    std::vector<std::vector<int>> ensemble_particle_indices(ensemble_size);
    for (int i = 0; i < total_particles; ++i) {
        if (part[i].flag && part[i].ensemble_id >= 0 && part[i].ensemble_id < ensemble_size) {
            // Only filter if this ensemble is selected for output
            bool is_selected = false;
            for (int selected_id : selected_ensemble_ids) {
                if (part[i].ensemble_id == selected_id) {
                    is_selected = true;
                    break;
                }
            }
            if (is_selected) {
                ensemble_particle_indices[part[i].ensemble_id].push_back(i);
            }
        }
    }

    // Process only selected ensembles (3 ensembles) with OpenMP parallelization
    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < selected_ensemble_ids.size(); idx++) {
        int ens = selected_ensemble_ids[idx];
        // Use pre-filtered particle indices
        const auto& particle_indices = ensemble_particle_indices[ens];
        int part_num = particle_indices.size();

        if (part_num == 0) continue; // Skip empty ensembles

        // Create filename: output_ens/ens_XXX_timestep_XXXXX.vtk
        std::ostringstream filenameStream;
        #ifdef _WIN32
            filenameStream << path << "\\ens_" << std::setfill('0') << std::setw(3) << ens
                          << "_timestep_" << std::setw(5) << timestep << ".vtk";
        #else
            filenameStream << path << "/ens_" << std::setfill('0') << std::setw(3) << ens
                          << "_timestep_" << std::setw(5) << timestep << ".vtk";
        #endif
        std::string filename = filenameStream.str();

        std::ofstream vtkFile(filename, std::ios::binary);

        if (!vtkFile.is_open()){
            std::cerr << "Cannot open file for writing: " << filename << std::endl;
            continue;
        }

        vtkFile << "# vtk DataFile Version 4.0\n";
        vtkFile << "Ensemble " << ens << " particle data\n";
        vtkFile << "BINARY\n";
        vtkFile << "DATASET POLYDATA\n";

        vtkFile << "POINTS " << part_num << " float\n";
        float zsum = 0.0;
        for (int idx : particle_indices){
            float x = -179.0 + part[idx].x*0.5;
            float y = -90.0 + part[idx].y*0.5;
            float z = part[idx].z/3000.0;
            zsum += part[idx].z;

            swapByteOrder(x);
            swapByteOrder(y);
            swapByteOrder(z);

            vtkFile.write(reinterpret_cast<char*>(&x), sizeof(float));
            vtkFile.write(reinterpret_cast<char*>(&y), sizeof(float));
            vtkFile.write(reinterpret_cast<char*>(&z), sizeof(float));
        }

        vtkFile << "POINT_DATA " << part_num << "\n";
        // vtkFile << "SCALARS u_wind float 1\n";
        // vtkFile << "LOOKUP_TABLE default\n";
        // for (int idx : particle_indices){
        //     float vval = part[idx].u_wind;
        //     swapByteOrder(vval);
        //     vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
        // }

        // vtkFile << "SCALARS v_wind float 1\n";
        // vtkFile << "LOOKUP_TABLE default\n";
        // for (int idx : particle_indices){
        //     float vval = part[idx].v_wind;
        //     swapByteOrder(vval);
        //     vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
        // }

        // vtkFile << "SCALARS w_wind float 1\n";
        // vtkFile << "LOOKUP_TABLE default\n";
        // for (int idx : particle_indices){
        //     float vval = part[idx].w_wind;
        //     swapByteOrder(vval);
        //     vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
        // }

        // vtkFile << "SCALARS virtual_dist float 1\n";
        // vtkFile << "LOOKUP_TABLE default\n";
        // for (int idx : particle_indices){
        //     float vval = part[idx].virtual_distance;
        //     swapByteOrder(vval);
        //     vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
        // }

        vtkFile << "SCALARS time_idx int 1\n";
        vtkFile << "LOOKUP_TABLE default\n";
        for (int i = 0; i < nop; ++i){
            if(!part[i].flag) continue;
            int vval = part[i].timeidx;
            swapByteOrder(vval);
            vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(int));
        }

        vtkFile << "SCALARS Q float 1\n";
        vtkFile << "LOOKUP_TABLE default\n";

        // DEBUG: Check Q values being written to VTK (EVERY output at timestep 60)
        if (timestep == 60) {
            int zero_count = 0;
            int nonzero_count = 0;
            std::cout << "[DEBUG_VTK_WRITE] Ensemble " << ens << ": Checking Q values before writing..." << std::endl;

            // Check first 10 particles
            int check_count = 0;
            for (int idx : particle_indices) {
                float vval = part[idx].conc;
                if (check_count < 10) {
                    std::cout << "[DEBUG_VTK_WRITE] P" << idx
                              << " (ens=" << part[idx].ensemble_id
                              << ", timeidx=" << part[idx].timeidx
                              << ", flag=" << part[idx].flag
                              << "): conc=" << vval;
                    // Also check concentrations[0]
                    std::cout << ", concentrations[0]=" << part[idx].concentrations[0] << std::endl;
                    check_count++;
                }

                if (vval == 0.0f) {
                    zero_count++;
                } else {
                    nonzero_count++;
                }
            }
            std::cout << "[DEBUG_VTK_WRITE] Ensemble " << ens << ": Total=" << particle_indices.size()
                      << ", Zero=" << zero_count
                      << ", Non-zero=" << nonzero_count << std::endl;
        }

        // DEBUG: Check actual write values (ALWAYS, to catch last iteration)
        int write_count = 0;

        for (int idx : particle_indices){
            float vval = part[idx].conc;

            if (write_count < 5) {
                float orig_vval = vval;
                swapByteOrder(vval);
                std::cout << "[DEBUG_WRITE] P" << idx << ": orig=" << orig_vval
                          << ", swapped=" << vval
                          << ", bytes=";
                unsigned char* bytes = reinterpret_cast<unsigned char*>(&vval);
                for (int b = 0; b < 4; b++) {
                    printf("%02x ", bytes[b]);
                }
                std::cout << std::endl;
                write_count++;
                // Continue to write
                vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
            } else {
                swapByteOrder(vval);
                vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
            }
        }

        vtkFile << "SCALARS time_idx int 1\n";
        vtkFile << "LOOKUP_TABLE default\n";
        for (int idx : particle_indices){
            int vval = part[idx].timeidx;
            swapByteOrder(vval);
            vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(int));
        }

        vtkFile.close();
    } // end of ensemble loop
}

// Log first particle's concentrations over time
void LDM::log_first_particle_concentrations(int timestep, float currentTime) {
    // Copy particles from device to host
    cudaMemcpy(part.data(), d_part, nop * sizeof(LDMpart), cudaMemcpyDeviceToHost);
    
    // Ensure validation directory exists
    #ifdef _WIN32
        _mkdir("validation");
    #else
        mkdir("validation", 0777);
    #endif
    
    // Create or append to CSV file
    static bool first_write = true;
    std::string filename = "validation/first_particle_concentrations.csv";
    std::ofstream csvFile;
    
    if (first_write) {
        csvFile.open(filename, std::ios::out);
        // Write header with dynamic nuclide names
        csvFile << "timestep,time(s),total_conc";
        NuclideConfig* nucConfig = NuclideConfig::getInstance();
        int num_nuclides = nucConfig->getNumNuclides();
        for (int i = 0; i < num_nuclides; i++) {
            csvFile << "," << nucConfig->getNuclideName(i);
        }
        csvFile << std::endl;
        first_write = false;
    } else {
        csvFile.open(filename, std::ios::app);
    }
    
    // Find first active particle
    bool found_active = false;
    for (size_t idx = 0; idx < part.size(); idx++) {
        const auto& p = part[idx];
        if (p.flag) {
            found_active = true;
            NuclideConfig* nucConfig = NuclideConfig::getInstance();
            int num_nuclides = nucConfig->getNumNuclides();
            csvFile << timestep << "," << currentTime << "," << p.conc;
            for (int i = 0; i < num_nuclides; i++) {
                csvFile << "," << p.concentrations[i];
            }
            csvFile << std::endl;
            break; // Only log the first active particle
        }
    }
    if (!found_active) {
        if (!part.empty()) {
            const auto& p = part[0];
            NuclideConfig* nucConfig = NuclideConfig::getInstance();
            int num_nuclides = nucConfig->getNumNuclides();
            csvFile << timestep << "," << currentTime << "," << p.conc;
            for (int i = 0; i < num_nuclides; i++) {
                csvFile << "," << p.concentrations[i];
            }
            csvFile << std::endl;
        }
    }
    
    csvFile.close();
}

// Log all particles' nuclide ratios over time
void LDM::log_all_particles_nuclide_ratios(int timestep, float currentTime) {
    // Copy particles from device to host
    cudaMemcpy(part.data(), d_part, nop * sizeof(LDMpart), cudaMemcpyDeviceToHost);
    
    // Ensure validation directory exists
    #ifdef _WIN32
        _mkdir("validation");
    #else
        mkdir("validation", 0777);
    #endif
    
    // Create or append to CSV file
    static bool first_write = true;
    std::string filename = "validation/all_particles_nuclide_ratios.csv";
    std::ofstream csvFile;
    
    if (first_write) {
        csvFile.open(filename, std::ios::out);
        // Write header
        csvFile << "timestep,time(s),active_particles,total_conc";
        for (int i = 0; i < MAX_NUCLIDES; i++) {
            csvFile << ",total_Q_" << i << ",ratio_Q_" << i;
        }
        csvFile << std::endl;
        first_write = false;
    } else {
        csvFile.open(filename, std::ios::app);
    }
    
    // Calculate totals for all active particles
    float total_concentrations[MAX_NUCLIDES] = {0.0f};
    float total_conc = 0.0f;
    int active_particles = 0;
    
    for (const auto& p : part) {
        if (p.flag) {
            active_particles++;
            total_conc += p.conc;
            for (int i = 0; i < MAX_NUCLIDES; i++) {
                total_concentrations[i] += p.concentrations[i];
            }
        }
    }
    
    // Write data
    csvFile << timestep << "," << currentTime << "," << active_particles << "," << total_conc;
    for (int i = 0; i < MAX_NUCLIDES; i++) {
        float ratio = (total_conc > 0) ? (total_concentrations[i] / total_conc) : 0.0f;
        csvFile << "," << total_concentrations[i] << "," << ratio;
    }
    csvFile << std::endl;
    
    csvFile.close();
}

// CRAM 계산 전후 농도 변화를 상세히 로깅하는 함수
void LDM::log_first_particle_cram_detail(int timestep, float currentTime, float dt_used) {
    // Copy particles from device to host
    cudaMemcpy(part.data(), d_part, nop * sizeof(LDMpart), cudaMemcpyDeviceToHost);
    
    // Ensure validation directory exists
    #ifdef _WIN32
        _mkdir("validation");
    #else
        mkdir("validation", 0777);
    #endif
    
    // Create or append to detailed CSV file
    static bool first_write = true;
    std::string filename = "validation/first_particle_cram_detail.csv";
    std::ofstream csvFile;
    
    if (first_write) {
        csvFile.open(filename, std::ios::out);
        // Write detailed header with decay information
        csvFile << "timestep,time(s),dt(s),particle_age(s)";
        
        NuclideConfig* nucConfig = NuclideConfig::getInstance();
        int num_nuclides = nucConfig->getNumNuclides();
        
        for (int i = 0; i < num_nuclides; i++) {
            std::string name = nucConfig->getNuclideName(i);
            float half_life = nucConfig->getHalfLife(i);
            csvFile << "," << name << "_conc," << name << "_half_life," << name << "_decay_factor";
        }
        csvFile << ",total_mass,mass_conservation_check" << std::endl;
        first_write = false;
    } else {
        csvFile.open(filename, std::ios::app);
    }
    
    // Find first active particle and log detailed information
    for (const auto& p : part) {
        if (p.flag) {
            NuclideConfig* nucConfig = NuclideConfig::getInstance();
            int num_nuclides = nucConfig->getNumNuclides();
            
            csvFile << timestep << "," << currentTime << "," << dt_used << "," << p.age;
            
            float total_mass = 0.0f;
            for (int i = 0; i < num_nuclides; i++) {
                float half_life = nucConfig->getHalfLife(i);
                float decay_constant = log(2.0f) / (half_life * 3600.0f); // Convert hours to seconds
                float decay_factor = exp(-decay_constant * dt_used);
                
                csvFile << "," << p.concentrations[i] << "," << half_life << "," << decay_factor;
                total_mass += p.concentrations[i];
            }
            
            // Mass conservation check (should decrease due to decay)
            static float initial_mass = -1.0f;
            if (initial_mass < 0) initial_mass = total_mass;
            float conservation_ratio = total_mass / initial_mass;
            
            csvFile << "," << total_mass << "," << conservation_ratio << std::endl;
            break; // Only log the first active particle
        }
    }
    
    csvFile.close();
}

// CRAM4 검증용 데이터 출력 함수 - 간단한 농도분포 저장
void LDM::exportValidationData(int timestep, float currentTime) {
    // 검증용 폴더 생성
    static bool validation_dir_created = false;
    if (!validation_dir_created) {
        #ifdef _WIN32
            _mkdir("validation");
        #else
            mkdir("validation", 0777);
        #endif
        validation_dir_created = true;
    }
    
    // Copy particles from device to host
    cudaMemcpy(part.data(), d_part, nop * sizeof(LDMpart), cudaMemcpyDeviceToHost);
    
    // 주요 타임스텝에서만 격자 데이터 출력 (용량 절약)
    if (timestep % 50 == 0 || timestep <= 10 || timestep >= 710) {
        exportConcentrationGrid(timestep, currentTime);
    }
    
    // 모든 타임스텝에서 핵종별 총 농도 출력
    exportNuclideTotal(timestep, currentTime);
    
    if (timestep % 100 == 0) {
        std::cout << "[VALIDATION] Exported reference data for timestep " << timestep << std::endl;
    }
}

// 농도 격자 데이터 출력 (100x100x20 격자)
void LDM::exportConcentrationGrid(int timestep, float currentTime) {
    // 후쿠시마 주변 영역 설정 (139-143°E, 36-39°N, 0-2000m)
    const float min_lon = 139.0f, max_lon = 143.0f;
    const float min_lat = 36.0f, max_lat = 39.0f; 
    const float min_alt = 0.0f, max_alt = 2000.0f;
    const int grid_x = 100, grid_y = 100, grid_z = 20;
    
    const float dx = (max_lon - min_lon) / grid_x;
    const float dy = (max_lat - min_lat) / grid_y;
    const float dz = (max_alt - min_alt) / grid_z;
    
    // 격자 초기화
    std::vector<std::vector<std::vector<float>>> concentration_grid(
        grid_x, std::vector<std::vector<float>>(grid_y, std::vector<float>(grid_z, 0.0f)));
    std::vector<std::vector<std::vector<int>>> count_grid(
        grid_x, std::vector<std::vector<int>>(grid_y, std::vector<int>(grid_z, 0)));
    
    // 활성 입자들을 격자에 매핑
    for (const auto& p : part) {
        if (!p.flag) continue;
        
        // GFS 좌표를 지리 좌표로 변환
        float lon = -179.0f + p.x * 0.5f;
        float lat = -90.0f + p.y * 0.5f;
        float alt = p.z;
        
        // 격자 범위 확인
        if (lon < min_lon || lon >= max_lon || lat < min_lat || lat >= max_lat || 
            alt < min_alt || alt >= max_alt) continue;
            
        // 격자 인덱스 계산
        int ix = static_cast<int>((lon - min_lon) / dx);
        int iy = static_cast<int>((lat - min_lat) / dy);
        int iz = static_cast<int>((alt - min_alt) / dz);
        
        // 경계 확인
        if (ix >= 0 && ix < grid_x && iy >= 0 && iy < grid_y && iz >= 0 && iz < grid_z) {
            concentration_grid[ix][iy][iz] += p.conc;
            count_grid[ix][iy][iz]++;
        }
    }
    
    // 격자 데이터를 CSV 파일로 저장
    std::ostringstream filename;
    filename << "validation/concentration_grid_" << std::setfill('0') << std::setw(5) << timestep << ".csv";
    
    std::ofstream csvFile(filename.str());
    csvFile << "x_index,y_index,z_index,lon,lat,alt,concentration,particle_count" << std::endl;
    
    for (int ix = 0; ix < grid_x; ix++) {
        for (int iy = 0; iy < grid_y; iy++) {
            for (int iz = 0; iz < grid_z; iz++) {
                if (concentration_grid[ix][iy][iz] > 0 || count_grid[ix][iy][iz] > 0) {
                    float lon = min_lon + (ix + 0.5f) * dx;
                    float lat = min_lat + (iy + 0.5f) * dy;
                    float alt = min_alt + (iz + 0.5f) * dz;
                    
                    csvFile << ix << "," << iy << "," << iz << "," 
                           << lon << "," << lat << "," << alt << ","
                           << concentration_grid[ix][iy][iz] << "," 
                           << count_grid[ix][iy][iz] << std::endl;
                }
            }
        }
    }
    csvFile.close();
}

// 핵종별 총 농도 출력
void LDM::exportNuclideTotal(int timestep, float currentTime) {
    static bool first_write = true;
    std::string filename = "validation/nuclide_totals.csv";
    std::ofstream csvFile;
    
    if (first_write) {
        csvFile.open(filename, std::ios::out);
        csvFile << "timestep,time(s),active_particles,total_conc";
        NuclideConfig* nucConfig = NuclideConfig::getInstance();
        int num_nuclides = nucConfig->getNumNuclides();
        for (int i = 0; i < num_nuclides; i++) {
            csvFile << ",total_" << nucConfig->getNuclideName(i);
        }
        csvFile << std::endl;
        first_write = false;
    } else {
        csvFile.open(filename, std::ios::app);
    }
    
    // 핵종별 총 농도 계산
    std::vector<float> total_concentrations(MAX_NUCLIDES, 0.0f);
    float total_conc = 0.0f;
    int active_particles = 0;
    
    for (const auto& p : part) {
        if (p.flag) {
            active_particles++;
            total_conc += p.conc;
            NuclideConfig* nucConfig = NuclideConfig::getInstance();
            int num_nuclides = nucConfig->getNumNuclides();
            for (int i = 0; i < num_nuclides; i++) {
                total_concentrations[i] += p.concentrations[i];
            }
        }
    }
    
    // 데이터 출력
    csvFile << timestep << "," << currentTime << "," << active_particles << "," << total_conc;
    NuclideConfig* nucConfig = NuclideConfig::getInstance();
    int num_nuclides = nucConfig->getNumNuclides();
    for (int i = 0; i < num_nuclides; i++) {
        csvFile << "," << total_concentrations[i];
    }
    csvFile << std::endl;
    
    csvFile.close();
}

// 핵종별 반감기 정보와 함께 농도 변화를 로깅
void LDM::log_first_particle_decay_analysis(int timestep, float currentTime) {
    // Copy particles from device to host
    cudaMemcpy(part.data(), d_part, nop * sizeof(LDMpart), cudaMemcpyDeviceToHost);
    
    // Ensure validation directory exists
    #ifdef _WIN32
        _mkdir("validation");
    #else
        mkdir("validation", 0777);
    #endif
    
    static bool first_write = true;
    std::string filename = "validation/first_particle_decay_analysis.csv";
    std::ofstream csvFile;
    
    if (first_write) {
        csvFile.open(filename, std::ios::out);
        csvFile << "timestep,time(s),nuclide_name,concentration,half_life_hours,decay_constant_per_sec,theoretical_concentration,relative_error" << std::endl;
        first_write = false;
    } else {
        csvFile.open(filename, std::ios::app);
    }
    
    // Find first active particle
    for (const auto& p : part) {
        if (p.flag) {
            NuclideConfig* nucConfig = NuclideConfig::getInstance();
            int num_nuclides = nucConfig->getNumNuclides();
            
            // 각 핵종별로 상세 분석
            for (int i = 0; i < num_nuclides; i++) {
                std::string name = nucConfig->getNuclideName(i);
                float half_life = nucConfig->getHalfLife(i);
                float decay_constant = log(2.0f) / (half_life * 3600.0f);
                
                // 이론적 농도 계산 (초기 농도 0.1에서 시작)
                float theoretical_conc = 0.1f * exp(-decay_constant * p.age);
                float relative_error = (p.concentrations[i] - theoretical_conc) / theoretical_conc * 100.0f;
                
                csvFile << timestep << "," << currentTime << "," << name << "," 
                       << p.concentrations[i] << "," << half_life << "," << decay_constant << "," 
                       << theoretical_conc << "," << relative_error << std::endl;
            }
            break; // Only analyze the first active particle
        }
    }
    
    csvFile.close();
}
