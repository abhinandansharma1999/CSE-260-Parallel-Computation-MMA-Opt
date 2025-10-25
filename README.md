# Optimizing Dense Matrix Multiplication on ARM Graviton3
• Implemented a double-precision GEMM kernel using SVE intrinsics, surpassing OpenBLAS performance [average gflops/s achieved = 21/5].
• Designed cache-aware blocking and packing routines along with loop unrolling to maximize data reuse.
• Conducted performance benchmarking and auto-tuning to evaluate GFLOPS scalability across parameter configurations.

***********
Optimized kernel file: cse260_hw1/dgemm_mykernel.cpp
***********
