#include "dgemm_mykernel.h"
#include "parameters.h"
#include <arm_sve.h>
#include <stdexcept>

void DGEMM_mykernel::compute(const Mat& A, const Mat& B, Mat& C) {
    int m = A.rows();
    int k = A.cols();
    int n = B.cols();

    my_dgemm(m, n, k, A.data(), k, B.data(), n, C.data(), n);
}

string DGEMM_mykernel::name() {
    return "my_kernel";
}

void DGEMM_mykernel::my_dgemm(
        int    m,
        int    n,
        int    k,
        const double *XA,
        int    lda,
        const double *XB,
        int    ldb,
        double *C,       
        int    ldc       
        )
{
    int    ic, ib, jc, jb, pc, pb;
    const double *packA, *packB;
    double* packedA = new double[param_mc * param_kc];
    double* packedB = new double[param_kc * param_nc];

    // Using NOPACK option for simplicity
    // #define NOPACK

    for ( ic = 0; ic < m; ic += param_mc ) {              // 5-th loop around micro-kernel
        ib = min( m - ic, param_mc );
        for ( pc = 0; pc < k; pc += param_kc ) {          // 4-th loop around micro-kernel
            pb = min( k - pc, param_kc );
            
            packA = &XA[pc + ic * lda ];
            double *current_packedA = packedA;
            pack_A(ib, pb, &XA[pc + ic * lda], lda, current_packedA);
            packA = current_packedA;

            for ( jc = 0; jc < n; jc += param_nc ) {        // 3-rd loop around micro-kernel
                jb = min( n - jc, param_nc );

                double *current_packedB = packedB;
                pack_B(pb, jb, &XB[ldb * pc + jc], ldb, current_packedB);
                packB = current_packedB;

                // Implement your macro-kernel here
                my_macro_kernel(
                        ib,
                        jb,
                        pb,
                        packA,
                        packB,
                        &C[ ic * ldc + jc ], 
                        ldc
                        );
            }                                               // End 3.rd loop around micro-kernel
        }                                                 // End 4.th loop around micro-kernel
    }                                                     // End 5.th loop around micro-kernel
}

#define a(i, j, ld) a[ (i)*(ld) + (j) ]
#define b(i, j, ld) b[ (i)*(ld) + (j) ]
#define c(i, j, ld) c[ (i)*(ld) + (j) ]

//
// C-based microkernel (NOPACK version)
//
// Implement your micro-kernel here
void DGEMM_mykernel::my_dgemm_ukr( int    kc,
                                  int    mr,
                                  int    nr,
                                  const double *a,
                                  const double *b,
                                  double *c,
                                  int ldc)
{
    int l, j, i;
    double cloc[param_mr][param_nr] = {{0}};
    
    // Load C into local array
    for (i = 0; i < mr; ++i) {
        for (j = 0; j < nr; ++j) {
            cloc[i][j] = c(i, j, ldc);
        }
    }
    
    int vector_size = 4;

    // Perform matrix multiplication
    for ( l = 0; l < kc; ++l ) {                 
        for ( i = 0; i < mr; ++i ) { 
            // double as = a(i, l, ldc);
            double as = a(l, i, mr);
            svfloat64_t a_vec = svdup_f64(as);

            for ( j = 0; j <= nr - vector_size; j += vector_size ) { 
                // cloc[i][j] +=  as * b(l, j, ldc);
                svfloat64_t c_vec = svld1_f64(svptrue_b64(), &cloc[i][j]);
                svfloat64_t b_vec = svld1_f64(svptrue_b64(), &b[(l) *(nr) + (j)]);
                c_vec = svmla_f64_m(svptrue_b64(), c_vec, a_vec, b_vec);
                svst1_f64(svptrue_b64(), &cloc[i][j], c_vec);
            }

            if (j < nr) {
                svbool_t pred = svwhilelt_b64(j, nr);
                svfloat64_t c_vec = svld1_f64(pred, &cloc[i][j]);
                svfloat64_t b_vec = svld1_f64(pred, &b[(l) *(nr) + (j)]);
                c_vec = svmla_f64_m(pred, c_vec, a_vec, b_vec);
                svst1_f64(pred, &cloc[i][j], c_vec); 
            }
        }
    }
    
    // Store local array back to C
    for (i = 0; i < mr; ++i) {
        for (j = 0; j < nr; ++j) {
            c(i, j, ldc) = cloc[i][j];
        }
    }
}

// Implement your macro-kernel here
void DGEMM_mykernel::my_macro_kernel(
        int    ib,
        int    jb,
        int    pb,
        const double * packA,
        const double * packB,
        double * C,
        int    ldc
        )
{
    int    i, j;

    for ( i = 0; i < ib; i += param_mr ) {                      // 2-th loop around micro-kernel
        for ( j = 0; j < jb; j += param_nr ) {                  // 1-th loop around micro-kernel
            my_dgemm_ukr (
                        pb,
                        min(ib-i, param_mr),
                        min(jb-j, param_nr),
                        &packA[i * pb],          // assumes sq matrix, otherwise use lda
                        &packB[j * pb],                
                        &C[ i * ldc + j ],
                        ldc
                        );
        }                                                       // 1-th loop around micro-kernel
    }
}

void DGEMM_mykernel::pack_A(int m, int k, const double * A, int lda, double * packed_A){
    int buffer_counter = 0;
    for(int i = 0; i < m; i += param_mr){
        int ib = min(m - i, param_mr);
        for(int p = 0; p < k; p++){
            for(int ii = 0; ii < ib; ii++){
                packed_A[buffer_counter] = A[(i + ii) * lda + p];
                buffer_counter++;
            }
        }
    }

}

void DGEMM_mykernel::pack_B(int k, int n, const double * B, int ldb, double * packed_B){
    int buffer_counter = 0;
    for(int j = 0; j < n; j += param_nr){
        int ib = min(n - j, param_nr);
        for(int p = 0; p < k; p++){
            for(int jj = 0; jj < ib; jj++){
                packed_B[buffer_counter] = B[p * ldb + j + jj];
                buffer_counter++;
            }
        }

    }
}
