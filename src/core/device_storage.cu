// Device storage for global constant arrays
// These are allocated once in device memory and passed as pointers to kernels
// to avoid invalid device symbol errors in non-RDC mode

__device__ float d_flex_hgt[50];
__device__ float d_T_const[3600];
