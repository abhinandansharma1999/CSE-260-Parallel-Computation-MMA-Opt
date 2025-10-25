#ifndef DGEMM_MYKERNEL_H
#define DGEMM_MYKERNEL_H

#include "../dgemm/dgemm.h"

class DGEMM_mykernel : public DGEMM {
public:
    void compute(const Mat& A, const Mat& B, Mat& C) override;
    string name() override;

private:
    void my_dgemm(int m, int n, int k, const double *A, int lda, const double *B, int ldb, double *C, int ldc);
    void my_macro_kernel(int ib, int jb, int pb, const double *packA, const double *packB, double *C, int ldc);
    void my_dgemm_ukr(int kc, int mr, int nr, const double *a, const double *b, double *c, int ldc);

    void pack_A(int m, int k, const double * A, int lda, double * packed_A);
    void pack_B(int k, int n, const double * B, int ldb, double * packed_B);
    
    void my_dgemm_ukr_scalar_4x4(int kc, const double *a, const double *b, double *c, int ldc);
    void handle_arbitrary_size( int    kc,
                                  int    mr,
                                  int    nr,
                                  const double *a,
                                  const double *b,
                                  double *c,
                                  int ldc);
    double* packedA = nullptr;
    double* packedB = nullptr;
};

#endif // DGEMM_MYKERNEL_H
