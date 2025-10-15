#include "ldm.cuh"


void LDM::initializeFlexGFSData(){

    std::cout << "[DEBUG] Starting read_meteorological_flex_gfs_init3 function..." << std::endl;

    flex_hgt.resize(dimZ_GFS);
    std::cout << "[DEBUG] flex_hgt vector resized to " << dimZ_GFS << " elements" << std::endl;

    const char* filename = "../gfsdata/0p5/2.txt";
    int recordMarker;

    size_t pres_data_size = (dimX_GFS + 1) * dimY_GFS * dimZ_GFS;
    size_t unis_data_size = (dimX_GFS + 1) * dimY_GFS;
    std::cout << "[DEBUG] Allocating memory for FlexPres data: " << pres_data_size << " elements" << std::endl;
    std::cout << "[DEBUG] Allocating memory for FlexUnis data: " << unis_data_size << " elements" << std::endl;

    FlexPres* flexpresdata = new FlexPres[pres_data_size];
    FlexUnis* flexunisdata = new FlexUnis[unis_data_size];

    if (!flexpresdata || !flexunisdata) {
        std::cerr << "[ERROR] Failed to allocate memory for meteorological data" << std::endl;
        if (flexpresdata) delete[] flexpresdata;
        if (flexunisdata) delete[] flexunisdata;
        return;
    }

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "[ERROR] Cannot open file: " << filename << std::endl;
        delete[] flexpresdata;
        delete[] flexunisdata;
        return;
    }
    std::cout << "[DEBUG] Successfully opened file: " << filename << std::endl;

    // Read HMIX data
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].HMIX), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            // Sample debug output for verification
            if ((i == 100 && j == 50) || (i == 200 && j == 150) || (i == 360 && j == 180)) {
                std::cout << "[VERIFY] HMIX[" << i << "," << j << "] = " << flexunisdata[index].HMIX << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].TROP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].USTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            // if(i==50) std::cout << "flexunisdata.USTR[" << j << "] = " << flexunisdata.USTR[index] << std::endl;
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].WSTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].OBKL), sizeof(float));
            //std::cout << "OBKL = " << flexunisdata[index].OBKL << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].LPREC), sizeof(float));
            //if(flexunisdata[index].LPREC>0.0f)std::cout << "LPREC = " << flexunisdata[index].LPREC << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].CPREC), sizeof(float));
            //std::cout << "CPREC = " << flexunisdata[index].CPREC << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].TCC), sizeof(float));
            //std::cout << "TCC = " << flexunisdata[index].TCC << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            int intBuffer;
            file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
            flexunisdata[index].CLDH = static_cast<float>(intBuffer);
            //file.read(reinterpret_cast<char*>(&flexunisdata[index].CLDH), sizeof(int));
            //std::cout << "CLDH = " << flexunisdata[index].CLDH << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].RHO), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                //if(i==50&&j==100) std::cout << "flexpresdata.RHO[" << k << "] = " << flexpresdata[index].RHO << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].TT), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                //if(i==50&&j==100) std::cout << "flexpresdata.TT[" << k << "] = " << flexpresdata[index].TT << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].UU), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // Debug disabled
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].VV), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // Debug disabled
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].WW), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // Debug disabled
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].VDEP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            //if(i==50&&j==100) 
            //if(i==50&&j==100) std::cout << "VDEP[" << i << "," << j << "] = " << flexunisdata[index].VDEP << std::endl;
            //if(i<10&&j<10) std::cout << "VDEP[" << i << "," << j << "] = " << flexunisdata[index].VDEP << std::endl;
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                int intBuffer;
                file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
                flexpresdata[index].CLDS = static_cast<float>(intBuffer);
                // file.read(reinterpret_cast<char*>(&flexpresdata[index].CLDS), sizeof(int));
                //std::cout << "CLDS = " << static_cast<int>(flexpresdata[index].CLDS) << std::endl;
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // Debug disabled
            }
        }
    }



    // for (int k = 0; k < 1; ++k) {
    //     for (int i = 0; i < dimX_GFS+1; ++i) {
    //         for (int j = 0; j < dimY_GFS; ++j) {
    //             int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
    //             file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
    //             file.read(reinterpret_cast<char*>(&flexpresdata[index].VDEP), sizeof(float));
    //             file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
    //             if(i==50&&j==100) std::cout << "flexpresdata.VDEP[" << k << "] = " << flexpresdata[index].VDEP << std::endl;
    //         }
    //     }
    // }

    // Debug height data before DRHO calculation
    std::cout << "[DRHO_DEBUG] Height check: flex_hgt[0]=" << flex_hgt[0] << ", flex_hgt[1]=" << flex_hgt[1] 
              << ", diff=" << (flex_hgt[1] - flex_hgt[0]) << std::endl;
    
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            float rho0 = flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].RHO;
            float rho1 = flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + 1].RHO;
            float hgt_diff = flex_hgt[1] - flex_hgt[0];
            
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO = (rho1 - rho0) / hgt_diff;
            
            // Debug first few points
            if (i < 2 && j < 2) {
                std::cout << "[DRHO_CALC] [" << i << "," << j << "]: rho0=" << rho0 << ", rho1=" << rho1 
                          << ", hgt_diff=" << hgt_diff << ", DRHO=" << flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO << std::endl;
            }

            //if(i==100 && j==50) printf("drho(%d) = %f\n", 0, flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO);
            
            for (int k = 1; k < dimZ_GFS-2; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                flexpresdata[index].DRHO = 
                (flexpresdata[index+1].RHO - flexpresdata[index-1].RHO) / (flex_hgt[k+1]-flex_hgt[k-1]);
                //if(i==100 && j==50) printf("drho(%d) = %f\n", k, flexpresdata[index].DRHO);
            }

            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO = 
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-3].DRHO;

            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-1].DRHO = 
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO;

            //if(i==100 && j==50) printf("drho(%d) = %f\n", dimZ_GFS-2, flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO);

        }
    }
    
    // Debug simplified - just check if data loaded
    std::cout << "[INFO] Wind data loaded - UU[0,0,0]=" << flexpresdata[0].UU 
              << ", VV[0,0,0]=" << flexpresdata[0].VV 
              << ", WW[0,0,0]=" << flexpresdata[0].WW << std::endl;
    
    // Check file read status
    if (file.fail() || file.bad()) {
        std::cerr << "[ERROR] File read error detected. fail=" << file.fail() << ", bad=" << file.bad() << std::endl;
    } else {
        std::cout << "[DEBUG] File read completed successfully" << std::endl;
    }



    file.close();

    cudaMalloc((void**)&device_meteorological_flex_pres0, 
    (dimX_GFS+1) * dimY_GFS * dimZ_GFS * sizeof(FlexPres));
    cudaMalloc((void**)&device_meteorological_flex_unis0, 
    (dimX_GFS+1) * dimY_GFS * sizeof(FlexUnis));
    
    cudaError_t copy_err = cudaMemcpy(device_meteorological_flex_pres0, 
        flexpresdata, (dimX_GFS+1) * dimY_GFS * dimZ_GFS * sizeof(FlexPres), 
        cudaMemcpyHostToDevice);
    if (copy_err != cudaSuccess) {
        std::cerr << "[ERROR] Failed to copy meteorological pres data to GPU: " << cudaGetErrorString(copy_err) << std::endl;
    } else {
        std::cout << "[DEBUG] Meteorological pres data copied successfully" << std::endl;
    }
    
    copy_err = cudaMemcpy(device_meteorological_flex_unis0, 
        flexunisdata, (dimX_GFS+1) * dimY_GFS * sizeof(FlexUnis), 
        cudaMemcpyHostToDevice);
    if (copy_err != cudaSuccess) {
        std::cerr << "[ERROR] Failed to copy meteorological unis data to GPU: " << cudaGetErrorString(copy_err) << std::endl;
    } else {
        std::cout << "[DEBUG] Meteorological unis data copied successfully" << std::endl;
    }


    // int size2D = (dimX_GFS + 1) * dimY_GFS;
    // int size3D = (dimX_GFS + 1) * dimY_GFS * dimZ_GFS;

    // this->host_unisA0 = new float4[size2D]; // HMIX, USTR, WSTR, OBKL
    // this->host_unisB0 = new float4[size2D]; // VDEP, LPREC, CPREC, TCC
    // this->host_presA0 = new float4[size3D]; // DRHO, RHO, TT, QV
    // this->host_presB0 = new float4[size3D]; // UU, VV, WW, 0.0 

    // for(int i = 0; i < size2D; i++){
    //     host_unisA0[i] = make_float4(
    //         flexunisdata[i].HMIX,
    //         flexunisdata[i].USTR,
    //         flexunisdata[i].WSTR,
    //         flexunisdata[i].OBKL
    //     );
    //     host_unisB0[i] = make_float4(
    //         flexunisdata[i].VDEP,
    //         flexunisdata[i].LPREC,
    //         flexunisdata[i].CPREC,
    //         flexunisdata[i].TCC
    //     );
    // }

    // for(int i = 0; i < size3D; i++){
    //     host_presA0[i] = make_float4(
    //         flexpresdata[i].DRHO,
    //         flexpresdata[i].RHO,
    //         flexpresdata[i].TT,
    //         flexpresdata[i].QV
    //     );
    //     host_presB0[i] = make_float4(
    //         flexpresdata[i].UU,
    //         flexpresdata[i].VV,
    //         flexpresdata[i].WW,
    //         0.0f        
    //     );
    // }

    // size_t width  = dimX_GFS + 1; 
    // size_t height = dimY_GFS;

    // cudaMallocArray(&d_unisArrayA0, &channelDesc2D, width, dimY_GFS);
    // cudaMallocArray(&d_unisArrayB0, &channelDesc2D, dimX_GFS + 1, dimY_GFS);

    // cudaMemcpy2DToArray(
    //     d_unisArrayA0,
    //     0, 0,
    //     host_unisA0,
    //     width*sizeof(float4),
    //     width*sizeof(float4),
    //     height,
    //     cudaMemcpyHostToDevice
    // );
    
    // cudaMemcpy2DToArray(
    //     d_unisArrayB0,
    //     0, 0,
    //     host_unisB0,
    //     width*sizeof(float4),
    //     width*sizeof(float4),
    //     height,
    //     cudaMemcpyHostToDevice
    // );

    // cudaExtent extent = make_cudaExtent(dimX_GFS+1, dimY_GFS, dimZ_GFS);
    // cudaMalloc3DArray(&d_presArrayA0, &channelDesc3D, extent);
    // cudaMalloc3DArray(&d_presArrayB0, &channelDesc3D, extent);

    // cudaMemcpy3DParms copyParamsA0 = {0};
    // copyParamsA0.srcPtr = make_cudaPitchedPtr(
    //     (void*)host_presA0, 
    //     (dimX_GFS+1)*sizeof(float4), // pitch in bytes
    //     (dimX_GFS+1),                // width in elements
    //     dimY_GFS                     // height
    // );
    // copyParamsA0.dstArray = d_presArrayA0;
    // copyParamsA0.extent   = extent;
    // copyParamsA0.kind     = cudaMemcpyHostToDevice;
    // cudaMemcpy3D(&copyParamsA0);

    // cudaMemcpy3DParms copyParamsB0 = {0};
    // copyParamsB0.srcPtr = make_cudaPitchedPtr(
    //     (void*)host_presB0, 
    //     (dimX_GFS+1)*sizeof(float4), // pitch in bytes
    //     (dimX_GFS+1),                // width in elements
    //     dimY_GFS                     // height
    // );
    // copyParamsB0.dstArray = d_presArrayB0;
    // copyParamsB0.extent   = extent;
    // copyParamsB0.kind     = cudaMemcpyHostToDevice;
    // cudaMemcpy3D(&copyParamsB0);

    
    filename = "../gfsdata/0p5/3.txt";

    file.open(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].HMIX), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].TROP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].USTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            // if(i==50) std::cout << "flexunisdata.USTR[" << j << "] = " << flexunisdata.USTR[index] << std::endl;
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].WSTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].OBKL), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].LPREC), sizeof(float));
            //if(flexunisdata[index].LPREC>0.0)printf("lsp = %e\n", flexunisdata[index].LPREC);
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].CPREC), sizeof(float));
            //if(flexunisdata[index].CPREC>0.0)printf("convprec = %e\n", flexunisdata[index].CPREC);
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].TCC), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            int intBuffer;
            file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
            flexunisdata[index].CLDH = static_cast<float>(intBuffer);
            //file.read(reinterpret_cast<char*>(&flexunisdata[index].CLDH), sizeof(int));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].RHO), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.RHO[" << k << "] = " << flexpresdata[index].RHO << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].TT), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.TT[" << k << "] = " << flexpresdata[index].TT << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].UU), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.UU[" << k << "] = " << flexpresdata[index].UU << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].VV), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.VV[" << k << "] = " << flexpresdata[index].VV << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].WW), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.WW[" << k << "] = " << flexpresdata[index].WW << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].VDEP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            //if(i==50&&j==100) std::cout << "VDEP[" << i << "," << j << "] = " << flexunisdata[index].VDEP << std::endl;
        }
    }


    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                int intBuffer;
                file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
                flexpresdata[index].CLDS = static_cast<float>(intBuffer);
                // file.read(reinterpret_cast<char*>(&flexpresdata[index].CLDS), sizeof(int));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.WW[" << k << "] = " << flexpresdata[index].WW << std::endl;
            }
        }
    }

    // for (int j = 0; j < dimY_GFS; ++j) {
    //     int index = 647 * dimY_GFS + j; 
    //     std::cout << "CLDH(" << 647 << ", " << j << ") = " << flexunisdata[index].CLDH << std::endl;
    //     index = 647 * dimY_GFS * dimZ_GFS + j * dimZ_GFS + 5; 
    //     std::cout << "CLDS(" << 647 << ", " << j << ") = " << flexpresdata[index].CLDS << std::endl;
    // }


    // for (int i = 0; i < dimX_GFS+1; ++i) {
    //     for (int j = 0; j < dimY_GFS; ++j) {
    //         for (int k = 0; k < dimZ_GFS; ++k) {
    //             int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
    //             file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
    //             file.read(reinterpret_cast<char*>(&flexpresdata[index].VDEP), sizeof(float));
    //             file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
    //             if(i==50&&j==100) std::cout << "flexpresdata.VDEP[" << k << "] = " << flexpresdata[index].VDEP << std::endl;
    //         }
    //     }
    // }


    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO = 
            (flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + 1].RHO - flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].RHO) /
            (flex_hgt[1]-flex_hgt[0]);

            //if(i==100 && j==50) printf("drho(%d) = %f\n", 0, flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO);
            
            for (int k = 1; k < dimZ_GFS-2; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                flexpresdata[index].DRHO = 
                (flexpresdata[index+1].RHO - flexpresdata[index-1].RHO) / (flex_hgt[k+1]-flex_hgt[k-1]);
                //if(i==100 && j==50) printf("drho(%d) = %f\n", k, flexpresdata[index].DRHO);
            }

            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO = 
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-3].DRHO;

            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-1].DRHO = 
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO;

            //if(i==100 && j==50) printf("drho(%d) = %f\n", dimZ_GFS-2, flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO);

        }
    }

    file.close();

    cudaMalloc((void**)&device_meteorological_flex_pres1, 
    (dimX_GFS+1) * dimY_GFS * dimZ_GFS * sizeof(FlexPres));
    cudaMalloc((void**)&device_meteorological_flex_unis1, 
    (dimX_GFS+1) * dimY_GFS * sizeof(FlexUnis));
    
    cudaMemcpy(device_meteorological_flex_pres1, 
        flexpresdata, (dimX_GFS+1) * dimY_GFS * dimZ_GFS * sizeof(FlexPres), 
        cudaMemcpyHostToDevice);
    cudaMemcpy(device_meteorological_flex_unis1, 
        flexunisdata, (dimX_GFS+1) * dimY_GFS * sizeof(FlexUnis), 
        cudaMemcpyHostToDevice);

    
    // this->host_unisA1 = new float4[size2D]; // HMIX, USTR, WSTR, OBKL
    // this->host_unisB1 = new float4[size2D]; // VDEP, LPREC, CPREC, TCC
    // this->host_presA1 = new float4[size3D]; // DRHO, RHO, TT, QV
    // this->host_presB1 = new float4[size3D]; // UU, VV, WW, 0.0 

    // for(int i = 0; i < size2D; i++){
    //     host_unisA1[i] = make_float4(
    //         flexunisdata[i].HMIX,
    //         flexunisdata[i].USTR,
    //         flexunisdata[i].WSTR,
    //         flexunisdata[i].OBKL
    //     );
    //     host_unisB1[i] = make_float4(
    //         flexunisdata[i].VDEP,
    //         flexunisdata[i].LPREC,
    //         flexunisdata[i].CPREC,
    //         flexunisdata[i].TCC
    //     );
    // }

    // for(int i = 0; i < size3D; i++){
    //     host_presA1[i] = make_float4(
    //         flexpresdata[i].DRHO,
    //         flexpresdata[i].RHO,
    //         flexpresdata[i].TT,
    //         flexpresdata[i].QV
    //     );
    //     host_presB1[i] = make_float4(
    //         flexpresdata[i].UU,
    //         flexpresdata[i].VV,
    //         flexpresdata[i].WW,
    //         0.0f        
    //     );
    // }

    // cudaMallocArray(&d_unisArrayA1, &channelDesc2D, width, dimY_GFS);
    // cudaMallocArray(&d_unisArrayB1, &channelDesc2D, dimX_GFS + 1, dimY_GFS);

    // cudaMemcpy2DToArray(
    //     d_unisArrayA1,
    //     0, 0,
    //     host_unisA1,
    //     width*sizeof(float4),
    //     width*sizeof(float4),
    //     height,
    //     cudaMemcpyHostToDevice
    // );
    
    // cudaMemcpy2DToArray(
    //     d_unisArrayB1,
    //     0, 0,
    //     host_unisB1,
    //     width*sizeof(float4),
    //     width*sizeof(float4),
    //     height,
    //     cudaMemcpyHostToDevice
    // );

    // cudaMalloc3DArray(&d_presArrayA1, &channelDesc3D, extent);
    // cudaMalloc3DArray(&d_presArrayB1, &channelDesc3D, extent);

    // cudaMemcpy3DParms copyParamsA1 = {0};
    // copyParamsA1.srcPtr = make_cudaPitchedPtr(
    //     (void*)host_presA1, 
    //     (dimX_GFS+1)*sizeof(float4), // pitch in bytes
    //     (dimX_GFS+1),                // width in elements
    //     dimY_GFS                     // height
    // );
    // copyParamsA1.dstArray = d_presArrayA1;
    // copyParamsA1.extent   = extent;
    // copyParamsA1.kind     = cudaMemcpyHostToDevice;
    // cudaMemcpy3D(&copyParamsA1);

    // cudaMemcpy3DParms copyParamsB1 = {0};
    // copyParamsB1.srcPtr = make_cudaPitchedPtr(
    //     (void*)host_presB1, 
    //     (dimX_GFS+1)*sizeof(float4), // pitch in bytes
    //     (dimX_GFS+1),                // width in elements
    //     dimY_GFS                     // height
    // );
    // copyParamsB1.dstArray = d_presArrayB1;
    // copyParamsB1.extent   = extent;
    // copyParamsB1.kind     = cudaMemcpyHostToDevice;
    // cudaMemcpy3D(&copyParamsB1);


    
    
    filename = "../gfsdata/0p5/4.txt";

    file.open(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].HMIX), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].TROP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].USTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            // if(i==50) std::cout << "flexunisdata.USTR[" << j << "] = " << flexunisdata.USTR[index] << std::endl;
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].WSTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].OBKL), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].LPREC), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].CPREC), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].TCC), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            int intBuffer;
            file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
            flexunisdata[index].CLDH = static_cast<float>(intBuffer);
            //file.read(reinterpret_cast<char*>(&flexunisdata[index].CLDH), sizeof(int));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].RHO), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.RHO[" << k << "] = " << flexpresdata[index].RHO << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].TT), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.TT[" << k << "] = " << flexpresdata[index].TT << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].UU), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.UU[" << k << "] = " << flexpresdata[index].UU << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].VV), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.VV[" << k << "] = " << flexpresdata[index].VV << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].WW), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.WW[" << k << "] = " << flexpresdata[index].WW << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].VDEP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            //if(i==50&&j==100) std::cout << "VDEP[" << i << "," << j << "] = " << flexunisdata[index].VDEP << std::endl;
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                int intBuffer;
                file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
                flexpresdata[index].CLDS = static_cast<float>(intBuffer);
                //file.read(reinterpret_cast<char*>(&flexpresdata[index].CLDS), sizeof(int));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.WW[" << k << "] = " << flexpresdata[index].WW << std::endl;
            }
        }
    }



    // for (int i = 0; i < dimX_GFS+1; ++i) {
    //     for (int j = 0; j < dimY_GFS; ++j) {
    //         for (int k = 0; k < dimZ_GFS; ++k) {
    //             int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
    //             file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
    //             file.read(reinterpret_cast<char*>(&flexpresdata[index].VDEP), sizeof(float));
    //             file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
    //             if(i==50&&j==100) std::cout << "flexpresdata.VDEP[" << k << "] = " << flexpresdata[index].VDEP << std::endl;
    //         }
    //     }
    // }


    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO = 
            (flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + 1].RHO - flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].RHO) /
            (flex_hgt[1]-flex_hgt[0]);

            //if(i==100 && j==50) printf("drho(%d) = %f\n", 0, flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO);
            
            for (int k = 1; k < dimZ_GFS-2; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                flexpresdata[index].DRHO = 
                (flexpresdata[index+1].RHO - flexpresdata[index-1].RHO) / (flex_hgt[k+1]-flex_hgt[k-1]);
                //if(i==100 && j==50) printf("drho(%d) = %f\n", k, flexpresdata[index].DRHO);
            }

            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO = 
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-3].DRHO;

            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-1].DRHO = 
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO;

            //if(i==100 && j==50) printf("drho(%d) = %f\n", dimZ_GFS-2, flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO);

        }
    }

    file.close();

    cudaMalloc((void**)&device_meteorological_flex_pres2, 
    (dimX_GFS+1) * dimY_GFS * dimZ_GFS * sizeof(FlexPres));
    cudaMalloc((void**)&device_meteorological_flex_unis2, 
    (dimX_GFS+1) * dimY_GFS * sizeof(FlexUnis));
    
    cudaMemcpy(device_meteorological_flex_pres2, 
        flexpresdata, (dimX_GFS+1) * dimY_GFS * dimZ_GFS * sizeof(FlexPres), 
        cudaMemcpyHostToDevice);
    cudaMemcpy(device_meteorological_flex_unis2, 
        flexunisdata, (dimX_GFS+1) * dimY_GFS * sizeof(FlexUnis), 
        cudaMemcpyHostToDevice);


    // this->host_unisA2 = new float4[size2D]; // HMIX, USTR, WSTR, OBKL
    // this->host_unisB2 = new float4[size2D]; // VDEP, LPREC, CPREC, TCC
    // this->host_presA2 = new float4[size3D]; // DRHO, RHO, TT, QV
    // this->host_presB2 = new float4[size3D]; // UU, VV, WW, 0.0 

    // for(int i = 0; i < size2D; i++){
    //     host_unisA2[i] = make_float4(
    //         flexunisdata[i].HMIX,
    //         flexunisdata[i].USTR,
    //         flexunisdata[i].WSTR,
    //         flexunisdata[i].OBKL
    //     );
    //     host_unisB2[i] = make_float4(
    //         flexunisdata[i].VDEP,
    //         flexunisdata[i].LPREC,
    //         flexunisdata[i].CPREC,
    //         flexunisdata[i].TCC
    //     );
    // }

    // for(int i = 0; i < size3D; i++){
    //     host_presA2[i] = make_float4(
    //         flexpresdata[i].DRHO,
    //         flexpresdata[i].RHO,
    //         flexpresdata[i].TT,
    //         flexpresdata[i].QV
    //     );
    //     host_presB2[i] = make_float4(
    //         flexpresdata[i].UU,
    //         flexpresdata[i].VV,
    //         flexpresdata[i].WW,
    //         0.0f        
    //     );
    // }

    // cudaMallocArray(&d_unisArrayA2, &channelDesc2D, width, dimY_GFS);
    // cudaMallocArray(&d_unisArrayB2, &channelDesc2D, dimX_GFS + 1, dimY_GFS);

    // cudaMemcpy2DToArray(
    //     d_unisArrayA2,
    //     0, 0,
    //     host_unisA2,
    //     width*sizeof(float4),
    //     width*sizeof(float4),
    //     height,
    //     cudaMemcpyHostToDevice
    // );
    
    // cudaMemcpy2DToArray(
    //     d_unisArrayB2,
    //     0, 0,
    //     host_unisB2,
    //     width*sizeof(float4),
    //     width*sizeof(float4),
    //     height,
    //     cudaMemcpyHostToDevice
    // );

    // cudaMalloc3DArray(&d_presArrayA2, &channelDesc3D, extent);
    // cudaMalloc3DArray(&d_presArrayB2, &channelDesc3D, extent);

    // cudaMemcpy3DParms copyParamsA2 = {0};
    // copyParamsA2.srcPtr = make_cudaPitchedPtr(
    //     (void*)host_presA2, 
    //     (dimX_GFS+1)*sizeof(float4), // pitch in bytes
    //     (dimX_GFS+1),                // width in elements
    //     dimY_GFS                     // height
    // );
    // copyParamsA2.dstArray = d_presArrayA2;
    // copyParamsA2.extent   = extent;
    // copyParamsA2.kind     = cudaMemcpyHostToDevice;
    // cudaMemcpy3D(&copyParamsA2);

    // cudaMemcpy3DParms copyParamsB2 = {0};
    // copyParamsB2.srcPtr = make_cudaPitchedPtr(
    //     (void*)host_presB2, 
    //     (dimX_GFS+1)*sizeof(float4), // pitch in bytes
    //     (dimX_GFS+1),                // width in elements
    //     dimY_GFS                     // height
    // );
    // copyParamsB2.dstArray = d_presArrayB2;
    // copyParamsB2.extent   = extent;
    // copyParamsB2.kind     = cudaMemcpyHostToDevice;
    // cudaMemcpy3D(&copyParamsB2);

        
    delete[] flexunisdata;
    delete[] flexpresdata;

}

void LDM::loadFlexGFSData(){

    gfs_idx ++;

    char filename[256];
    sprintf(filename, "../gfsdata/0p5/%d.txt", gfs_idx+3);
    std::cout << "[DEBUG] Loading next gfs file: " << filename << std::endl;
    int recordMarker;

    // FlexUnis flexunisdata;
    // FlexPres flexpresdata;

    FlexPres* flexpresdata = new FlexPres[(dimX_GFS + 1) * dimY_GFS * dimZ_GFS];
    FlexUnis* flexunisdata = new FlexUnis[(dimX_GFS + 1) * dimY_GFS];

    cudaMemcpy(device_meteorological_flex_pres0, 
        device_meteorological_flex_pres1, (dimX_GFS+1) * dimY_GFS * dimZ_GFS * sizeof(FlexPres), 
        cudaMemcpyDeviceToDevice);
    cudaMemcpy(device_meteorological_flex_unis0, 
        device_meteorological_flex_unis1, (dimX_GFS+1) * dimY_GFS * sizeof(FlexUnis), 
        cudaMemcpyDeviceToDevice);

    size_t width  = dimX_GFS + 1; 
    size_t height = dimY_GFS;

    // flexunisdata.HMIX.resize((dimX_GFS + 1) * dimY_GFS);
    // flexunisdata.TROP.resize((dimX_GFS + 1) * dimY_GFS);
    // flexunisdata.USTR.resize((dimX_GFS + 1) * dimY_GFS);
    // flexunisdata.WSTR.resize((dimX_GFS + 1) * dimY_GFS);
    // flexunisdata.OBKL.resize((dimX_GFS + 1) * dimY_GFS);

    // flexpresdata.RHO.resize((dimX_GFS + 1) * dimY_GFS * dimZ_GFS);
    // flexpresdata.TT.resize((dimX_GFS + 1) * dimY_GFS * dimZ_GFS);
    // flexpresdata.UU.resize((dimX_GFS + 1) * dimY_GFS * dimZ_GFS);
    // flexpresdata.VV.resize((dimX_GFS + 1) * dimY_GFS * dimZ_GFS);
    // flexpresdata.WW.resize((dimX_GFS + 1) * dimY_GFS * dimZ_GFS);

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }
    else{
        //std::cout << "open file: " << filename << std::endl;
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].HMIX), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].TROP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].USTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            // if(i==50) std::cout << "flexunisdata.USTR[" << j << "] = " << flexunisdata.USTR[index] << std::endl;
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].WSTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].OBKL), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].LPREC), sizeof(float));
            //std::cout << "LPREC = " << flexunisdata[index].LPREC << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].CPREC), sizeof(float));
            //std::cout << "cprec = " << flexunisdata[index].CPREC << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].TCC), sizeof(float));
            //std::cout << "TCC = " << flexunisdata[index].TCC << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            int intBuffer;
            file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
            flexunisdata[index].CLDH = static_cast<float>(intBuffer);
            //file.read(reinterpret_cast<char*>(&flexunisdata[index].CLDH), sizeof(int));
            //std::cout << "CLDH = " << flexunisdata[index].CLDH << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].RHO), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                //if(i==50&&j==100) std::cout << "flexpresdata.RHO[" << k << "] = " << flexpresdata[index].RHO << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].TT), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                //if(i==50&&j==100) std::cout << "flexpresdata.TT[" << k << "] = " << flexpresdata[index].TT << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].UU), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // Debug disabled
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].VV), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // Debug disabled
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].WW), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // Debug disabled
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].VDEP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            //if(i==50&&j==100) std::cout << "VDEP[" << i << "," << j << "] = " << flexunisdata[index].VDEP << std::endl;
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                int intBuffer;
                file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
                flexpresdata[index].CLDS = static_cast<float>(intBuffer);
                //file.read(reinterpret_cast<char*>(&flexpresdata[index].CLDS), sizeof(int));
                //std::cout << "CLDS = " << static_cast<int>(flexpresdata[index].CLDS) << std::endl;
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.WW[" << k << "] = " << flexpresdata[index].WW << std::endl;
            }
        }
    }



    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO = 
            (flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + 1].RHO - flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].RHO) /
            (flex_hgt[1]-flex_hgt[0]);

            //if(i==100 && j==50) printf("drho(%d) = %f\n", 0, flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO);
            
            for (int k = 1; k < dimZ_GFS-2; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                flexpresdata[index].DRHO = 
                (flexpresdata[index+1].RHO - flexpresdata[index-1].RHO) / (flex_hgt[k+1]-flex_hgt[k-1]);
                //if(i==100 && j==50) printf("drho(%d) = %f\n", k, flexpresdata[index].DRHO);
            }

            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO = 
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-3].DRHO;

            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-1].DRHO = 
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO;

            //if(i==100 && j==50) printf("drho(%d) = %f\n", dimZ_GFS-2, flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO);

        }
    }

    file.close();
    
    cudaMemcpy(device_meteorological_flex_pres1, 
        flexpresdata, (dimX_GFS+1) * dimY_GFS * dimZ_GFS * sizeof(FlexPres), 
        cudaMemcpyHostToDevice);
    cudaMemcpy(device_meteorological_flex_unis1, 
        flexunisdata, (dimX_GFS+1) * dimY_GFS * sizeof(FlexUnis), 
        cudaMemcpyHostToDevice);


    char filename_hgt[256];
    sprintf(filename_hgt, "../gfsdata/0p5/hgt_%d.txt", gfs_idx+2);
    //std::cout << filename_hgt << std::endl;

    int recordMarker_hgt;

    std::ifstream file_hgt(filename_hgt, std::ios::binary);
    if (!file_hgt) {
        std::cerr << "Cannot open file: " << filename_hgt << std::endl;
        return;
    }

    for (int index = 0; index < dimZ_GFS; ++index) {
        file_hgt.read(reinterpret_cast<char*>(&recordMarker_hgt), sizeof(int));
        file_hgt.read(reinterpret_cast<char*>(&flex_hgt[index]), sizeof(float));
        file_hgt.read(reinterpret_cast<char*>(&recordMarker_hgt), sizeof(int));
        // std::cout << "flex_hgt[" << index << "] = " << flex_hgt[index] << std::endl;
    }

    file_hgt.close();
    
    cudaError_t err = cudaMemcpyToSymbol(d_flex_hgt, flex_hgt.data(), sizeof(float) * (dimZ_GFS));
    if (err != cudaSuccess) std::cerr << "Failed to copy data to constant memory: " << cudaGetErrorString(err) << std::endl;
        

}

void LDM::loadFlexHeightData(){ 

    std::cout << "[DEBUG] Starting read_meteorological_flex_hgt function..." << std::endl;
    
    flex_hgt.resize(dimZ_GFS);
    std::cout << "[DEBUG] flex_hgt vector resized to " << dimZ_GFS << " elements" << std::endl;

    const char* filename = "../gfsdata/0p5/hgt_2.txt";
    int recordMarker;

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "[ERROR] Cannot open file: " << filename << std::endl;
        return;
    }
    std::cout << "[DEBUG] Successfully opened file: " << filename << std::endl;

    for (int index = 0; index < dimZ_GFS; ++index) {
        file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        if (file.fail()) {
            std::cerr << "[ERROR] Failed to read record marker at index " << index << std::endl;
            return;
        }
        
        file.read(reinterpret_cast<char*>(&flex_hgt[index]), sizeof(float));
        if (file.fail()) {
            std::cerr << "[ERROR] Failed to read flex_hgt data at index " << index << std::endl;
            return;
        }
        
        // Check for NaN in height data
        if (std::isnan(flex_hgt[index])) {
            std::cout << "[HGT_NAN] flex_hgt[" << index << "] is NaN!" << std::endl;
        }
        
        file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        if (file.fail()) {
            std::cerr << "[ERROR] Failed to read closing record marker at index " << index << std::endl;
            return;
        }
        
        // std::cout << "flex_hgt[" << index << "] = " << flex_hgt[index] << std::endl;
    }

    file.close();
    std::cout << "[DEBUG] Successfully read " << dimZ_GFS << " height levels from file" << std::endl;
    
    std::cout << "[DEBUG] Attempting to copy " << dimZ_GFS << " float values to d_flex_hgt constant memory" << std::endl;
    cudaError_t err = cudaMemcpyToSymbol(d_flex_hgt, flex_hgt.data(), sizeof(float) * (dimZ_GFS));
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] Failed to copy flex_hgt to GPU constant memory: " << cudaGetErrorString(err) << std::endl;
        std::cerr << "[ERROR] Size attempted: " << sizeof(float) * (dimZ_GFS) << " bytes (" << dimZ_GFS << " floats)" << std::endl;
    } else {
        std::cout << "[DEBUG] Successfully copied flex_hgt to GPU constant memory" << std::endl;
    }
        

}

// =====================================
// Meteorological data preloading functions for EKI
// =====================================

int LDM::calculateRequiredMeteoFiles() {
    // Calculate required number of files from total simulation time and meteorological data interval in settings.txt
    float total_simulation_time = time_end;  // Total simulation time (seconds)
    int meteo_time_interval = Constants::time_interval;  // Meteorological data time interval (seconds)

    // Number of files needed = (total simulation time / meteorological interval) + 1 (including file 0)
    int num_files = static_cast<int>(std::ceil(total_simulation_time / meteo_time_interval)) + 1;

    std::cout << "Total simulation time: " << total_simulation_time << " seconds" << std::endl;
    std::cout << "Meteorological data time interval: " << meteo_time_interval << " seconds" << std::endl;
    std::cout << "Required meteorological data files: " << num_files << " files (0~" << (num_files-1) << ")" << std::endl;
    
    return num_files;
}

bool LDM::loadSingleMeteoFile(int file_index, FlexPres*& pres_data, FlexUnis*& unis_data, std::vector<float>& hgt_data) {
    // Generate file paths
    char filename[256];
    sprintf(filename, "../gfsdata/0p5/%d.txt", file_index);
    
    char hgt_filename[256];
    sprintf(hgt_filename, "../gfsdata/0p5/hgt_%d.txt", file_index);
    
    
    // Memory allocation
    size_t pres_size = (dimX_GFS + 1) * dimY_GFS * dimZ_GFS;
    size_t unis_size = (dimX_GFS + 1) * dimY_GFS;
    
    pres_data = new FlexPres[pres_size];
    unis_data = new FlexUnis[unis_size];
    hgt_data.resize(dimZ_GFS);
    
    if (!pres_data || !unis_data) {
        std::cerr << "[ERROR] Memory allocation failed: " << filename << std::endl;
        if (pres_data) delete[] pres_data;
        if (unis_data) delete[] unis_data;
        return false;
    }
    
    // Read meteorological data file
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "[ERROR] Failed to open file: " << filename << std::endl;
        delete[] pres_data;
        delete[] unis_data;
        return false;
    }
    
    int recordMarker;
    
    // Read HMIX data
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&unis_data[index].HMIX), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    // Read TROP data
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&unis_data[index].TROP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    // Read USTR data
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&unis_data[index].USTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    // Read WSTR data
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&unis_data[index].WSTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    // Read OBKL data
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&unis_data[index].OBKL), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    // Read LPREC data
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&unis_data[index].LPREC), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    // Read CPREC data
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&unis_data[index].CPREC), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    // Read TCC data
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&unis_data[index].TCC), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    // Read CLDH data
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            int intBuffer;
            file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
            unis_data[index].CLDH = static_cast<float>(intBuffer);
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    // Read 3D pressure data (RHO, TT, UU, VV, WW)
    // RHO
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&pres_data[index].RHO), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            }
        }
    }
    
    // TT
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&pres_data[index].TT), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            }
        }
    }
    
    // UU
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&pres_data[index].UU), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            }
        }
    }
    
    // VV
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&pres_data[index].VV), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            }
        }
    }
    
    // WW
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&pres_data[index].WW), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            }
        }
    }
    
    // Read VDEP data
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&unis_data[index].VDEP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    // Read CLDS data
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                int intBuffer;
                file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
                pres_data[index].CLDS = static_cast<float>(intBuffer);
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            }
        }
    }
    
    file.close();
    
    // Read height data
    std::ifstream hgt_file(hgt_filename, std::ios::binary);
    if (!hgt_file) {
        std::cerr << "[ERROR] Failed to open height file: " << hgt_filename << std::endl;
        delete[] pres_data;
        delete[] unis_data;
        return false;
    }
    
    std::cout << "[DEBUG_HGT] Loading: " << hgt_filename << std::endl;
    
    for (int index = 0; index < dimZ_GFS; ++index) {
        hgt_file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        hgt_file.read(reinterpret_cast<char*>(&hgt_data[index]), sizeof(float));
        hgt_file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
    }
    hgt_file.close();
    
    // Verify height data values for first and last files
    if (file_index == 0 || file_index == 4) {
        std::cout << "[DEBUG_HGT] File " << file_index << " height data sample: ";
        for (int i = 0; i < std::min(5, (int)dimZ_GFS); i++) {
            std::cout << hgt_data[i] << " ";
        }
        std::cout << "... " << hgt_data[dimZ_GFS-1] << std::endl;
    }
    
    // DRHO 계산 (기존 코드와 동일)
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            float rho0 = pres_data[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].RHO;
            float rho1 = pres_data[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + 1].RHO;
            float hgt_diff = hgt_data[1] - hgt_data[0];
            
            pres_data[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO = (rho1 - rho0) / hgt_diff;
            
            for (int k = 1; k < dimZ_GFS-2; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                pres_data[index].DRHO = 
                (pres_data[index+1].RHO - pres_data[index-1].RHO) / (hgt_data[k+1]-hgt_data[k-1]);
            }

            pres_data[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO = 
            pres_data[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-3].DRHO;

            pres_data[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-1].DRHO = 
            pres_data[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO;
        }
    }
    
    return true;
}

bool LDM::preloadAllEKIMeteorologicalData() {
    std::cout << "Starting meteorological data preloading for EKI..." << std::endl;
    
    // Clean up existing data
    g_eki_meteo.cleanup();
    
    // Calculate required number of files
    int num_files = calculateRequiredMeteoFiles();
    if (num_files <= 0) {
        std::cerr << "[ERROR] Invalid file count: " << num_files << std::endl;
        return false;
    }
    
    // Set metadata
    g_eki_meteo.num_time_steps = num_files;
    g_eki_meteo.pres_data_size = (dimX_GFS + 1) * dimY_GFS * dimZ_GFS * sizeof(FlexPres);
    g_eki_meteo.unis_data_size = (dimX_GFS + 1) * dimY_GFS * sizeof(FlexUnis);
    g_eki_meteo.hgt_data_size = dimZ_GFS * sizeof(float);
    
    // Initialize host memory vectors
    g_eki_meteo.host_flex_pres_data.resize(num_files);
    g_eki_meteo.host_flex_unis_data.resize(num_files);
    g_eki_meteo.host_flex_hgt_data.resize(num_files);
    
    std::cout << "Starting parallel CPU loading of " << num_files << " files..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Use thread pool for true parallel execution
    std::vector<std::thread> threads;
    std::vector<bool> results(num_files, false);
    std::mutex completion_mutex;
    std::atomic<int> completed_files(0);
    
    for (int i = 0; i < num_files; i++) {
        threads.emplace_back([this, i, &results, &completion_mutex, &completed_files]() {
            FlexPres* pres_data = nullptr;
            FlexUnis* unis_data = nullptr;
            std::vector<float> hgt_data;

            std::cout << "Thread " << std::this_thread::get_id()
                      << " loading file " << i << ".txt..." << std::endl;

            bool success = this->loadSingleMeteoFile(i, pres_data, unis_data, hgt_data);
            if (success) {
                std::lock_guard<std::mutex> lock(completion_mutex);
                g_eki_meteo.host_flex_pres_data[i] = pres_data;
                g_eki_meteo.host_flex_unis_data[i] = unis_data;
                g_eki_meteo.host_flex_hgt_data[i] = hgt_data;
                results[i] = true;
                completed_files++;
                std::cout << "Meteorological data file " << i << ".txt loaded successfully ("
                          << completed_files.load() << "/" << g_eki_meteo.num_time_steps << ")" << std::endl;
            } else {
                std::lock_guard<std::mutex> lock(completion_mutex);
                std::cerr << "[ERROR] Failed to load file " << i << ".txt!" << std::endl;
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // 결과 확인
    bool all_success = true;
    for (int i = 0; i < num_files; i++) {
        if (!results[i]) {
            all_success = false;
        }
    }
    
    std::cout << "Parallel loading result: " << completed_files.load() << "/" << num_files << " completed" << std::endl;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (!all_success) {
        std::cerr << "[ERROR] Some meteorological data files failed to load" << std::endl;
        g_eki_meteo.cleanup();
        return false;
    }
    
    std::cout << "CPU parallel loading completed (" << duration.count() << "ms)" << std::endl;
    
    // GPU memory allocation and copying
    std::cout << "Starting GPU memory allocation and data transfer..." << std::endl;
    auto gpu_start_time = std::chrono::high_resolution_clock::now();
    
    // Allocate GPU memory pointer arrays
    cudaError_t err1 = cudaMalloc((void**)&g_eki_meteo.device_flex_pres_data, num_files * sizeof(FlexPres*));
    cudaError_t err2 = cudaMalloc((void**)&g_eki_meteo.device_flex_unis_data, num_files * sizeof(FlexUnis*));
    cudaError_t err3 = cudaMalloc((void**)&g_eki_meteo.device_flex_hgt_data, num_files * sizeof(float*));
    
    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
        std::cerr << "[ERROR] GPU 포인터 배열 할당 실패" << std::endl;
        g_eki_meteo.cleanup();
        return false;
    }
    
    // Allocate GPU memory and transfer data for each timestep
    std::vector<FlexPres*> temp_pres_ptrs(num_files, nullptr);
    std::vector<FlexUnis*> temp_unis_ptrs(num_files, nullptr);
    std::vector<float*> temp_hgt_ptrs(num_files, nullptr);
    
    bool gpu_allocation_success = true;
    
    for (int i = 0; i < num_files; i++) {
        // Pres data
        if (cudaMalloc((void**)&temp_pres_ptrs[i], g_eki_meteo.pres_data_size) != cudaSuccess) {
            std::cerr << "[ERROR] GPU Pres memory allocation failed (file " << i << ")" << std::endl;
            gpu_allocation_success = false;
            break;
        }
        if (cudaMemcpy(temp_pres_ptrs[i], g_eki_meteo.host_flex_pres_data[i], 
                       g_eki_meteo.pres_data_size, cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cerr << "[ERROR] GPU Pres data transfer failed (file " << i << ")" << std::endl;
            gpu_allocation_success = false;
            break;
        }
        
        // Unis data
        if (cudaMalloc((void**)&temp_unis_ptrs[i], g_eki_meteo.unis_data_size) != cudaSuccess) {
            std::cerr << "[ERROR] GPU Unis memory allocation failed (file " << i << ")" << std::endl;
            gpu_allocation_success = false;
            break;
        }
        if (cudaMemcpy(temp_unis_ptrs[i], g_eki_meteo.host_flex_unis_data[i], 
                       g_eki_meteo.unis_data_size, cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cerr << "[ERROR] GPU Unis data transfer failed (file " << i << ")" << std::endl;
            gpu_allocation_success = false;
            break;
        }
        
        // Height data
        if (cudaMalloc((void**)&temp_hgt_ptrs[i], g_eki_meteo.hgt_data_size) != cudaSuccess) {
            std::cerr << "[ERROR] GPU Height memory allocation failed (file " << i << ")" << std::endl;
            gpu_allocation_success = false;
            break;
        }
        if (cudaMemcpy(temp_hgt_ptrs[i], g_eki_meteo.host_flex_hgt_data[i].data(), 
                       g_eki_meteo.hgt_data_size, cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cerr << "[ERROR] GPU Height data transfer failed (file " << i << ")" << std::endl;
            gpu_allocation_success = false;
            break;
        }
        
        std::cout << "GPU data transfer completed: file " << i << std::endl;
    }
    
    if (!gpu_allocation_success) {
        // Clean up partially allocated memory
        for (int i = 0; i < num_files; i++) {
            if (temp_pres_ptrs[i]) cudaFree(temp_pres_ptrs[i]);
            if (temp_unis_ptrs[i]) cudaFree(temp_unis_ptrs[i]);
            if (temp_hgt_ptrs[i]) cudaFree(temp_hgt_ptrs[i]);
        }
        g_eki_meteo.cleanup();
        return false;
    }
    
    // 포인터 배열을 GPU로 복사
    err1 = cudaMemcpy(g_eki_meteo.device_flex_pres_data, temp_pres_ptrs.data(), 
                      num_files * sizeof(FlexPres*), cudaMemcpyHostToDevice);
    err2 = cudaMemcpy(g_eki_meteo.device_flex_unis_data, temp_unis_ptrs.data(), 
                      num_files * sizeof(FlexUnis*), cudaMemcpyHostToDevice);
    err3 = cudaMemcpy(g_eki_meteo.device_flex_hgt_data, temp_hgt_ptrs.data(), 
                      num_files * sizeof(float*), cudaMemcpyHostToDevice);
    
    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
        std::cerr << "[ERROR] GPU 포인터 배열 복사 실패" << std::endl;
        g_eki_meteo.cleanup();
        return false;
    }
    
    auto gpu_end_time = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end_time - gpu_start_time);
    
    // Allocate existing LDM GPU memory slots in EKI mode (device_meteorological_flex_pres0/1, unis0/1)
    std::cout << "Allocating existing LDM GPU memory slots..." << std::endl;
    
    // Slot 0 (for past data)
    cudaError_t err_alloc1 = cudaMalloc((void**)&g_eki_meteo.ldm_pres0_slot, g_eki_meteo.pres_data_size);
    cudaError_t err_alloc2 = cudaMalloc((void**)&g_eki_meteo.ldm_unis0_slot, g_eki_meteo.unis_data_size);
    
    // Slot 1 (for future data)  
    cudaError_t err_alloc3 = cudaMalloc((void**)&g_eki_meteo.ldm_pres1_slot, g_eki_meteo.pres_data_size);
    cudaError_t err_alloc4 = cudaMalloc((void**)&g_eki_meteo.ldm_unis1_slot, g_eki_meteo.unis_data_size);
    
    // 전역 변수에 포인터 복사
    device_meteorological_flex_pres0 = g_eki_meteo.ldm_pres0_slot;
    device_meteorological_flex_unis0 = g_eki_meteo.ldm_unis0_slot;
    device_meteorological_flex_pres1 = g_eki_meteo.ldm_pres1_slot;
    device_meteorological_flex_unis1 = g_eki_meteo.ldm_unis1_slot;
    
    if (err_alloc1 != cudaSuccess || err_alloc2 != cudaSuccess || 
        err_alloc3 != cudaSuccess || err_alloc4 != cudaSuccess) {
        std::cerr << "[ERROR] Failed to allocate existing LDM GPU memory slots" << std::endl;
        std::cerr << "  device_meteorological_flex_pres0: " << cudaGetErrorString(err_alloc1) << std::endl;
        std::cerr << "  device_meteorological_flex_unis0: " << cudaGetErrorString(err_alloc2) << std::endl;
        std::cerr << "  device_meteorological_flex_pres1: " << cudaGetErrorString(err_alloc3) << std::endl;
        std::cerr << "  device_meteorological_flex_unis1: " << cudaGetErrorString(err_alloc4) << std::endl;
        g_eki_meteo.cleanup();
        return false;
    }
    
    // 초기 상태로 첫 번째 기상자료 로드 (과거/미래 동일하게)
    if (g_eki_meteo.num_time_steps > 0) {
        FlexPres* first_pres_ptr;
        FlexUnis* first_unis_ptr;
        
        cudaMemcpy(&first_pres_ptr, &g_eki_meteo.device_flex_pres_data[0], 
                   sizeof(FlexPres*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&first_unis_ptr, &g_eki_meteo.device_flex_unis_data[0], 
                   sizeof(FlexUnis*), cudaMemcpyDeviceToHost);
        
        // Past slot
        cudaMemcpy(g_eki_meteo.ldm_pres0_slot, first_pres_ptr, 
                   g_eki_meteo.pres_data_size, cudaMemcpyDeviceToDevice);
        cudaMemcpy(g_eki_meteo.ldm_unis0_slot, first_unis_ptr, 
                   g_eki_meteo.unis_data_size, cudaMemcpyDeviceToDevice);
        
        // Future slot (initially same data)
        cudaMemcpy(g_eki_meteo.ldm_pres1_slot, first_pres_ptr, 
                   g_eki_meteo.pres_data_size, cudaMemcpyDeviceToDevice);
        cudaMemcpy(g_eki_meteo.ldm_unis1_slot, first_unis_ptr, 
                   g_eki_meteo.unis_data_size, cudaMemcpyDeviceToDevice);
        
        std::cout << "Initial meteorological data loaded (index 0)" << std::endl;
    }
    
    g_eki_meteo.is_initialized = true;
    
    std::cout << "GPU memory transfer completed (" << gpu_duration.count() << "ms)" << std::endl;
    std::cout << "Total preloading completed (" << (duration.count() + gpu_duration.count()) << "ms)" << std::endl;
    std::cout << "Memory usage: " << std::endl;
    std::cout << "  - Pres data: " << (g_eki_meteo.pres_data_size * num_files / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  - Unis data: " << (g_eki_meteo.unis_data_size * num_files / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  - Height data: " << (g_eki_meteo.hgt_data_size * num_files / 1024) << " KB" << std::endl;
    
    return true;
}

void LDM::cleanupEKIMeteorologicalData() {
    std::cout << "Starting EKI meteorological data memory cleanup..." << std::endl;
    g_eki_meteo.cleanup();
    std::cout << "EKI meteorological data memory cleanup completed" << std::endl;
}