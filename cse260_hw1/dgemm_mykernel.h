#ifndef DGEMM_MYKERNEL_H
#define DGEMM_MYKERNEL_H

#include "../dgemm/dgemm.h"
#include <arm_sve.h>
#include <stdexcept>
// #include <iostream>
// using namespace std;

class DGEMM_mykernel : public DGEMM {
public:
    void compute(const Mat& A, const Mat& B, Mat& C) override;
    string name() override;
private:
    void my_dgemm_w_packing(int m, int n, int k, const double *A, int lda, const double *B, int ldb, double *C, int ldc);
    void my_dgemm_wo_packing(int m, int n, int k, const double *A, int lda, const double *B, int ldb, double *C, int ldc);
    void my_macro_kernel_w_packing(int ib, int jb, int pb, const double *packA, const double *packB, double *C, int ldc);
    void my_macro_kernel_wo_packing(int ib, int jb, int pb, const double *packA, const double *packB, double *C, int ldc);
    void my_dgemm_ukr_w_packing_w_simd(int kc, int mr, int nr, const double *a, const double *b, double *c, int ldc);
    void my_dgemm_ukr_wo_packing(int kc, int mr, int nr, const double *a, const double *b, double *c, int ldc);
    void my_dgemm_8x8x8(int kc, int mr, int nr, const double *a, const double *b, double *c, int ldc);

    void pack_A(int m, int k, const double * A, int lda, double * packed_A);
    void pack_B(int k, int n, const double * B, int ldb, double * packed_B);
};

#endif // DGEMM_MYKERNEL_H