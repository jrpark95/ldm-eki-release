// ldm_eki_ipc.cuh - POSIX Shared Memory IPC for LDM-EKI communication
#ifndef LDM_EKI_IPC_CUH
#define LDM_EKI_IPC_CUH

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <numeric>
#include <algorithm>
#include "memory_doctor.cuh"

// Global memory doctor instance
MemoryDoctor g_memory_doctor;

namespace LDM_EKI_IPC {

// Shared memory names
constexpr const char* SHM_CONFIG_NAME = "/ldm_eki_config";
constexpr const char* SHM_DATA_NAME = "/ldm_eki_data";
constexpr const char* SHM_ENSEMBLE_OBS_CONFIG_NAME = "/ldm_eki_ensemble_obs_config";
constexpr const char* SHM_ENSEMBLE_OBS_DATA_NAME = "/ldm_eki_ensemble_obs_data";

// Legacy configuration structure (12 bytes) - kept for backward compatibility
struct EKIConfigBasic {
    int32_t ensemble_size;  // Number of ensemble members
    int32_t num_receptors;  // Number of receptors
    int32_t num_timesteps;  // Number of time steps
};

// Full configuration structure (128 bytes exactly)
struct EKIConfigFull {
    // Basic info (12 bytes)
    int32_t ensemble_size;
    int32_t num_receptors;
    int32_t num_timesteps;

    // Algorithm parameters (44 bytes)
    int32_t iteration;
    float renkf_lambda;
    float noise_level;
    float time_days;
    float time_interval;          // EKI time interval (e.g., 15.0 minutes)
    float inverse_time_interval;
    float receptor_error;
    float receptor_mda;
    float prior_constant;         // Prior emission constant value (e.g., 1.5e+8)
    int32_t num_source;
    int32_t num_gpu;

    // Option strings (64 bytes = 8 strings × 8 bytes)
    char perturb_option[8];    // "Off"
    char adaptive_eki[8];      // "Off"
    char localized_eki[8];     // "Off"
    char regularization[8];    // "On"
    char gpu_forward[8];       // "On"
    char gpu_inverse[8];       // "On"
    char source_location[8];   // "Fixed"
    char time_unit[8];         // "minutes"

    // Memory Doctor Mode (8 bytes)
    char memory_doctor[8];     // "On" or "Off"

    // Total: 12 + 44 + 64 + 8 = 128 bytes (no padding needed)
};

// Data header structure (12 bytes + data)
struct EKIDataHeader {
    int32_t status;      // 0=writing, 1=ready
    int32_t rows;        // Number of receptors
    int32_t cols;        // Number of timesteps
    // float data[] follows immediately after header
};

class EKIWriter {
private:
    int config_fd;
    int data_fd;
    void* config_map;
    void* data_map;
    size_t data_size;
    bool initialized;

public:
    EKIWriter() : config_fd(-1), data_fd(-1), config_map(nullptr), 
                  data_map(nullptr), data_size(0), initialized(false) {}
    
    ~EKIWriter() {
        cleanup();
    }

    // Initialize shared memory segments with full configuration
    bool initialize(const ::EKIConfig& eki_config, int num_timesteps) {
        if (initialized) {
            std::cerr << "EKIWriter already initialized" << std::endl;
            return false;
        }

        // Calculate data size: header + receptor data
        data_size = sizeof(EKIDataHeader) + eki_config.num_receptors * num_timesteps * sizeof(float);

        // Create config shared memory (now using EKIConfigFull size)
        config_fd = shm_open(SHM_CONFIG_NAME, O_CREAT | O_RDWR, 0660);
        if (config_fd < 0) {
            perror("shm_open config");
            return false;
        }

        if (ftruncate(config_fd, sizeof(EKIConfigFull)) != 0) {
            perror("ftruncate config");
            close(config_fd);
            return false;
        }

        config_map = mmap(nullptr, sizeof(EKIConfigFull), PROT_READ | PROT_WRITE,
                         MAP_SHARED, config_fd, 0);
        if (config_map == MAP_FAILED) {
            perror("mmap config");
            close(config_fd);
            return false;
        }

        // Write full configuration
        auto* config = reinterpret_cast<EKIConfigFull*>(config_map);
        memset(config, 0, sizeof(EKIConfigFull));

        // Basic info
        config->ensemble_size = eki_config.ensemble_size;
        config->num_receptors = eki_config.num_receptors;
        config->num_timesteps = num_timesteps;

        // Algorithm parameters
        config->iteration = eki_config.iteration;
        config->renkf_lambda = eki_config.renkf_lambda;
        config->noise_level = eki_config.noise_level;
        config->time_days = eki_config.time_days;
        config->time_interval = eki_config.time_interval;
        config->inverse_time_interval = eki_config.inverse_time_interval;
        config->receptor_error = eki_config.receptor_error;
        config->receptor_mda = eki_config.receptor_mda;
        config->prior_constant = eki_config.prior_constant;
        config->num_source = eki_config.num_source;
        config->num_gpu = eki_config.num_gpu;

        // Option strings (safe copy with null termination)
        memset(config->perturb_option, 0, 8);
        strncpy(config->perturb_option, eki_config.perturb_option.c_str(), 7);

        memset(config->adaptive_eki, 0, 8);
        strncpy(config->adaptive_eki, eki_config.adaptive_eki.c_str(), 7);

        memset(config->localized_eki, 0, 8);
        strncpy(config->localized_eki, eki_config.localized_eki.c_str(), 7);

        memset(config->regularization, 0, 8);
        strncpy(config->regularization, eki_config.regularization.c_str(), 7);

        memset(config->gpu_forward, 0, 8);
        strncpy(config->gpu_forward, eki_config.gpu_forward.c_str(), 7);

        memset(config->gpu_inverse, 0, 8);
        strncpy(config->gpu_inverse, eki_config.gpu_inverse.c_str(), 7);

        memset(config->source_location, 0, 8);
        strncpy(config->source_location, eki_config.source_location.c_str(), 7);

        memset(config->time_unit, 0, 8);
        strncpy(config->time_unit, eki_config.time_unit.c_str(), 7);

        memset(config->memory_doctor, 0, 8);
        strncpy(config->memory_doctor, eki_config.memory_doctor_mode ? "On" : "Off", 7);

        // Create data shared memory
        data_fd = shm_open(SHM_DATA_NAME, O_CREAT | O_RDWR, 0660);
        if (data_fd < 0) {
            perror("shm_open data");
            munmap(config_map, sizeof(EKIConfig));
            close(config_fd);
            return false;
        }

        if (ftruncate(data_fd, data_size) != 0) {
            perror("ftruncate data");
            munmap(config_map, sizeof(EKIConfig));
            close(config_fd);
            close(data_fd);
            return false;
        }

        data_map = mmap(nullptr, data_size, PROT_READ | PROT_WRITE, 
                       MAP_SHARED, data_fd, 0);
        if (data_map == MAP_FAILED) {
            perror("mmap data");
            munmap(config_map, sizeof(EKIConfig));
            close(config_fd);
            close(data_fd);
            return false;
        }

        // Initialize data header
        auto* header = reinterpret_cast<EKIDataHeader*>(data_map);
        header->status = 0;  // Writing status
        header->rows = eki_config.num_receptors;
        header->cols = num_timesteps;

        initialized = true;
        std::cout << "EKI IPC Writer initialized (Full Config):" << std::endl;
        std::cout << "  - " << eki_config.ensemble_size << " ensembles, "
                  << eki_config.num_receptors << " receptors, "
                  << num_timesteps << " timesteps" << std::endl;
        std::cout << "  - Iteration: " << eki_config.iteration
                  << ", Regularization: " << eki_config.regularization << std::endl;
        std::cout << "  - GPU: " << eki_config.num_gpu << " devices"
                  << ", Forward: " << eki_config.gpu_forward
                  << ", Inverse: " << eki_config.gpu_inverse << std::endl;
        return true;
    }

    // Write observation data
    bool writeObservations(const float* observations, int rows, int cols) {
        if (!initialized) {
            std::cerr << "EKIWriter not initialized" << std::endl;
            return false;
        }

        auto* header = reinterpret_cast<EKIDataHeader*>(data_map);

        // Verify dimensions
        if (rows != header->rows || cols != header->cols) {
            std::cerr << "Dimension mismatch: expected " << header->rows
                      << "x" << header->cols << ", got " << rows << "x" << cols << std::endl;
            return false;
        }

        // Set writing status
        header->status = 0;

        // Copy data
        float* data_ptr = reinterpret_cast<float*>(
            reinterpret_cast<uint8_t*>(data_map) + sizeof(EKIDataHeader)
        );

        memcpy(data_ptr, observations, rows * cols * sizeof(float));

        // Memory Doctor: Log sent data (iteration 0 for initial observations)
        if (g_memory_doctor.isEnabled()) {
            g_memory_doctor.logSentData("initial_observations", observations, rows, cols, 0,
                                       "LDM->Python initial EKI observations");
        }

        // Set ready status
        header->status = 1;

        std::cout << "EKI observations written: " << rows << "x" << cols
                  << " matrix (" << rows * cols * sizeof(float) << " bytes)" << std::endl;
        return true;
    }

    // Get current basic configuration
    bool getConfig(int& ensemble_size, int& num_receptors, int& num_timesteps) {
        if (!initialized) {
            return false;
        }

        auto* config = reinterpret_cast<EKIConfigFull*>(config_map);
        ensemble_size = config->ensemble_size;
        num_receptors = config->num_receptors;
        num_timesteps = config->num_timesteps;
        return true;
    }

    // Cleanup resources
    void cleanup() {
        if (data_map) {
            munmap(data_map, data_size);
            data_map = nullptr;
        }
        if (config_map) {
            munmap(config_map, sizeof(EKIConfigFull));
            config_map = nullptr;
        }
        if (data_fd >= 0) {
            close(data_fd);
            data_fd = -1;
        }
        if (config_fd >= 0) {
            close(config_fd);
            config_fd = -1;
        }
        initialized = false;
    }

    // Initialize ensemble observation shared memory segments
    bool initializeEnsembleObservations(int ensemble_size, int num_receptors, int num_timesteps) {
        if (!initialized) {
            std::cerr << "EKIWriter not initialized. Call initialize() first." << std::endl;
            return false;
        }

        // Create ensemble observation config shared memory
        int ens_obs_config_fd = shm_open(SHM_ENSEMBLE_OBS_CONFIG_NAME, O_CREAT | O_RDWR | O_TRUNC, 0666);
        if (ens_obs_config_fd < 0) {
            std::cerr << "Failed to create ensemble obs config shared memory: " << strerror(errno) << std::endl;
            return false;
        }

        // Set size for config (12 bytes)
        if (ftruncate(ens_obs_config_fd, sizeof(EKIConfigBasic)) != 0) {
            std::cerr << "Failed to set ensemble obs config size: " << strerror(errno) << std::endl;
            close(ens_obs_config_fd);
            return false;
        }

        // Map config memory
        void* ens_obs_config_map = mmap(nullptr, sizeof(EKIConfigBasic), PROT_READ | PROT_WRITE, MAP_SHARED, ens_obs_config_fd, 0);
        if (ens_obs_config_map == MAP_FAILED) {
            std::cerr << "Failed to map ensemble obs config memory: " << strerror(errno) << std::endl;
            close(ens_obs_config_fd);
            return false;
        }

        // Write config
        EKIConfigBasic* ens_obs_config = static_cast<EKIConfigBasic*>(ens_obs_config_map);
        ens_obs_config->ensemble_size = ensemble_size;
        ens_obs_config->num_receptors = num_receptors;
        ens_obs_config->num_timesteps = num_timesteps;

        // Unmap config (data will be written separately)
        munmap(ens_obs_config_map, sizeof(EKIConfigBasic));
        close(ens_obs_config_fd);

        std::cout << "[EKI_IPC] Ensemble observation config initialized: "
                  << ensemble_size << " ensembles, "
                  << num_receptors << " receptors, "
                  << num_timesteps << " timesteps" << std::endl;

        return true;
    }

    // Write ensemble observations to shared memory (with iteration tracking)
    bool writeEnsembleObservations(const float* observations, int ensemble_size, int num_receptors, int num_timesteps, int iteration = -1) {
        if (!initialized) {
            std::cerr << "EKIWriter not initialized" << std::endl;
            return false;
        }

        // Calculate data size
        size_t ens_obs_data_size = ensemble_size * num_receptors * num_timesteps * sizeof(float);

        // Create/open data shared memory
        int ens_obs_data_fd = shm_open(SHM_ENSEMBLE_OBS_DATA_NAME, O_CREAT | O_RDWR | O_TRUNC, 0666);
        if (ens_obs_data_fd < 0) {
            std::cerr << "Failed to create ensemble obs data shared memory: " << strerror(errno) << std::endl;
            return false;
        }

        // Set data size
        if (ftruncate(ens_obs_data_fd, ens_obs_data_size) != 0) {
            std::cerr << "Failed to set ensemble obs data size: " << strerror(errno) << std::endl;
            close(ens_obs_data_fd);
            return false;
        }

        // Map data memory
        void* ens_obs_data_map = mmap(nullptr, ens_obs_data_size, PROT_READ | PROT_WRITE, MAP_SHARED, ens_obs_data_fd, 0);
        if (ens_obs_data_map == MAP_FAILED) {
            std::cerr << "Failed to map ensemble obs data memory: " << strerror(errno) << std::endl;
            close(ens_obs_data_fd);
            return false;
        }

        // Write data
        memcpy(ens_obs_data_map, observations, ens_obs_data_size);

        // Calculate statistics for validation
        float min_val = observations[0];
        float max_val = observations[0];
        float sum_val = 0.0f;
        int total_elements = ensemble_size * num_receptors * num_timesteps;

        for (int i = 0; i < total_elements; i++) {
            float val = observations[i];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
            sum_val += val;
        }

        std::cout << "[EKI_IPC] Ensemble observations written to shared memory:" << std::endl;
        std::cout << "  - Size: " << ens_obs_data_size << " bytes" << std::endl;
        std::cout << "  - Shape: [" << ensemble_size << " × " << num_receptors << " × " << num_timesteps << "]" << std::endl;
        std::cout << "  - Min: " << min_val << ", Max: " << max_val << ", Mean: " << (sum_val / total_elements) << std::endl;

        // Memory Doctor: Log sent ensemble observations with iteration
        if (g_memory_doctor.isEnabled()) {
            std::string info = "EKI iteration " + std::to_string(iteration) + ": " + std::to_string(ensemble_size) + " ensembles";
            g_memory_doctor.logSentData("ensemble_observations", observations,
                                       ensemble_size * num_receptors, num_timesteps, iteration, info);
        }

        // Unmap data
        munmap(ens_obs_data_map, ens_obs_data_size);
        close(ens_obs_data_fd);

        return true;
    }

    // Unlink shared memory (call at program exit)
    // NOTE: We don't unlink ensemble observation files since Python needs to read them
    static void unlinkSharedMemory() {
        shm_unlink(SHM_CONFIG_NAME);
        shm_unlink(SHM_DATA_NAME);
        // Don't unlink ensemble observation files - Python needs them!
        // shm_unlink(SHM_ENSEMBLE_OBS_CONFIG_NAME);
        // shm_unlink(SHM_ENSEMBLE_OBS_DATA_NAME);
        std::cout << "EKI shared memory unlinked (kept ensemble obs for Python)" << std::endl;
    }
};

// ============================================================================
// EKIReader - Read ensemble state data from Python via shared memory
// ============================================================================

// Shared memory names for ensemble data (Python → C++)
constexpr const char* SHM_ENSEMBLE_CONFIG_NAME = "/ldm_eki_ensemble_config";
constexpr const char* SHM_ENSEMBLE_DATA_NAME = "/ldm_eki_ensemble_data";

// Ensemble configuration structure (12 bytes)
struct EnsembleConfig {
    int32_t num_states;      // Number of state variables (e.g., 24 timesteps)
    int32_t num_ensemble;    // Number of ensemble members (e.g., 100)
    int32_t timestep_id;     // Current timestep identifier
};

// Ensemble data header structure (12 bytes + data)
struct EnsembleDataHeader {
    int32_t status;          // 0=writing, 1=ready
    int32_t rows;            // Number of states
    int32_t cols;            // Number of ensemble members
    // float data[] follows immediately after header
};

class EKIReader {
private:
    int config_fd;
    int data_fd;
    void* config_map;
    void* data_map;
    size_t data_size;
    bool initialized;

public:
    EKIReader() : config_fd(-1), data_fd(-1), config_map(nullptr),
                  data_map(nullptr), data_size(0), initialized(false) {}

    ~EKIReader() {
        cleanup();
    }

    // Wait for ensemble data to be ready
    bool waitForEnsembleData(int timeout_seconds = 60, int expected_iteration = -1) {
        std::cout << "[EKI_READER] Waiting for ensemble data from Python (timeout: "
                  << timeout_seconds << "s)..." << std::endl;

        const char* config_path = "/dev/shm/ldm_eki_ensemble_config";
        const char* data_path = "/dev/shm/ldm_eki_ensemble_data";

        // Store last iteration ID to detect fresh data
        static int last_iteration_id = -1;

        // Wait for config file to appear with fresh data
        for (int i = 0; i < timeout_seconds; i++) {
            if (access(config_path, F_OK) == 0 && access(data_path, F_OK) == 0) {
                // Read config to check iteration ID
                int config_fd = open(config_path, O_RDONLY);
                if (config_fd >= 0) {
                    EnsembleConfig config;
                    ssize_t bytes_read = read(config_fd, &config, sizeof(config));
                    close(config_fd);

                    if (bytes_read == sizeof(config)) {
                        // Check if this is new data (different iteration ID)
                        if (config.timestep_id > last_iteration_id) {
                            // New iteration detected, now check if data is ready
                            int test_fd = open(data_path, O_RDONLY);
                            if (test_fd >= 0) {
                                EnsembleDataHeader header;
                                bytes_read = read(test_fd, &header, sizeof(header));
                                close(test_fd);

                                if (bytes_read == sizeof(header) && header.status == 1) {
                                    std::cout << "[EKI_READER] Fresh ensemble data detected! Iteration ID: "
                                              << config.timestep_id << " (previous: " << last_iteration_id << ")" << std::endl;
                                    last_iteration_id = config.timestep_id;
                                    return true;
                                }
                            }
                        } else if (config.timestep_id == last_iteration_id && i > 5) {
                            // Same iteration ID after 5 seconds - probably stale data
                            if (i % 5 == 0) {
                                std::cout << "[EKI_READER] Waiting for new data... (current iteration ID: "
                                          << config.timestep_id << ")" << std::endl;
                            }
                        }
                    }
                }
            }
            sleep(1);
        }

        std::cerr << "[EKI_READER] Timeout waiting for ensemble data" << std::endl;
        return false;
    }

    // Read ensemble configuration
    bool readEnsembleConfig(int& num_states, int& num_ensemble, int& timestep_id) {
        const char* shm_path = "/dev/shm/ldm_eki_ensemble_config";

        config_fd = open(shm_path, O_RDONLY);
        if (config_fd < 0) {
            perror("[EKI_READER] open config failed");
            return false;
        }

        EnsembleConfig config;
        ssize_t bytes_read = read(config_fd, &config, sizeof(config));
        close(config_fd);
        config_fd = -1;

        if (bytes_read != sizeof(config)) {
            std::cerr << "[EKI_READER] Failed to read config (got " << bytes_read << " bytes)" << std::endl;
            return false;
        }

        num_states = config.num_states;
        num_ensemble = config.num_ensemble;
        timestep_id = config.timestep_id;

        std::cout << "[EKI_READER] Config loaded: " << num_states << " states × "
                  << num_ensemble << " ensemble (timestep " << timestep_id << ")" << std::endl;
        return true;
    }

    // Read ensemble state data
    bool readEnsembleStates(std::vector<float>& output, int& num_states, int& num_ensemble) {
        // First read config
        int timestep_id;
        if (!readEnsembleConfig(num_states, num_ensemble, timestep_id)) {
            return false;
        }

        const char* shm_path = "/dev/shm/ldm_eki_ensemble_data";

        data_fd = open(shm_path, O_RDONLY);
        if (data_fd < 0) {
            perror("[EKI_READER] open data failed");
            return false;
        }

        // Get file size
        struct stat st;
        if (fstat(data_fd, &st) != 0) {
            perror("[EKI_READER] fstat failed");
            close(data_fd);
            data_fd = -1;
            return false;
        }

        size_t file_size = st.st_size;
        size_t expected_size = sizeof(EnsembleDataHeader) + num_states * num_ensemble * sizeof(float);

        if (file_size != expected_size) {
            std::cerr << "[EKI_READER] Size mismatch: file=" << file_size
                      << " bytes, expected=" << expected_size << " bytes" << std::endl;
            close(data_fd);
            data_fd = -1;
            return false;
        }

        // Map entire file
        data_map = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, data_fd, 0);
        if (data_map == MAP_FAILED) {
            perror("[EKI_READER] mmap failed");
            close(data_fd);
            data_fd = -1;
            return false;
        }

        // Read header
        auto* header = reinterpret_cast<EnsembleDataHeader*>(data_map);

        if (header->status != 1) {
            std::cerr << "[EKI_READER] Data not ready (status=" << header->status << ")" << std::endl;
            munmap(data_map, file_size);
            close(data_fd);
            data_map = nullptr;
            data_fd = -1;
            return false;
        }

        if (header->rows != num_states || header->cols != num_ensemble) {
            std::cerr << "[EKI_READER] Dimension mismatch: header says " << header->rows
                      << "×" << header->cols << ", config says " << num_states << "×" << num_ensemble << std::endl;
            munmap(data_map, file_size);
            close(data_fd);
            data_map = nullptr;
            data_fd = -1;
            return false;
        }

        // Read data
        float* data_ptr = reinterpret_cast<float*>(
            reinterpret_cast<uint8_t*>(data_map) + sizeof(EnsembleDataHeader)
        );

        size_t data_count = num_states * num_ensemble;
        output.resize(data_count);
        std::memcpy(output.data(), data_ptr, data_count * sizeof(float));

        std::cout << "[EKI_READER] Ensemble states loaded: " << num_states << "×" << num_ensemble
                  << " matrix (" << data_count * sizeof(float) << " bytes)" << std::endl;

        // Calculate statistics
        float min_val = *std::min_element(output.begin(), output.end());
        float max_val = *std::max_element(output.begin(), output.end());
        float sum = std::accumulate(output.begin(), output.end(), 0.0f);
        float mean_val = sum / data_count;

        std::cout << "[EKI_READER] Data range: [" << min_val << ", " << max_val
                  << "], mean: " << mean_val << std::endl;

        // Memory Doctor: Log received ensemble states with iteration from timestep_id
        if (g_memory_doctor.isEnabled()) {
            std::string info = "EKI iteration " + std::to_string(timestep_id) + " from Python";
            g_memory_doctor.logReceivedData("ensemble_states", output.data(),
                                          num_states, num_ensemble, timestep_id, info);
        }

        // Cleanup mapping
        munmap(data_map, file_size);
        close(data_fd);
        data_map = nullptr;
        data_fd = -1;

        return true;
    }

    // Cleanup resources
    void cleanup() {
        if (data_map) {
            munmap(data_map, data_size);
            data_map = nullptr;
        }
        if (config_map) {
            munmap(config_map, sizeof(EnsembleConfig));
            config_map = nullptr;
        }
        if (data_fd >= 0) {
            close(data_fd);
            data_fd = -1;
        }
        if (config_fd >= 0) {
            close(config_fd);
            config_fd = -1;
        }
        initialized = false;
    }

    // Unlink ensemble shared memory
    static void unlinkEnsembleSharedMemory() {
        shm_unlink(SHM_ENSEMBLE_CONFIG_NAME);
        shm_unlink(SHM_ENSEMBLE_DATA_NAME);
        std::cout << "[EKI_READER] Ensemble shared memory unlinked" << std::endl;
    }
};

} // namespace LDM_EKI_IPC

#endif // LDM_EKI_IPC_CUH