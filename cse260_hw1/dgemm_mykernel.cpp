#include "dgemm_mykernel.h"
#include "parameters.h"

#include <stdexcept>

#include <vector>
#include <cstring>

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

    // Using NOPACK option for simplicity
    // #define NOPACK
    
    for ( ic = 0; ic < m; ic += param_mc ) {              // 5-th loop around micro-kernel
        ib = min( m - ic, param_mc );
        for ( pc = 0; pc < k; pc += param_kc ) {          // 4-th loop around micro-kernel
            pb = min( k - pc, param_kc );
            
            #ifdef NOPACK
            packA = &XA[pc + ic * lda ];
            #else

            // Pack A: copy ib x pb block starting at (ic, pc) into register array
            double packA_reg[ib * pb];  // Static size for register optimization
            int buf_index = 0;
            for (int ii = 0; ii < ib; ii += param_mr) {  // Loop over actual rows of mc x kc block
                int iib = min(ib - ii, param_mr);
                for (int kk = 0; kk < pb; ++kk) {  // Loop over all columns of mr x kc block
                    for (int iii = 0; iii < iib; ++iii) {  // Loop over rows of mr x kc block
                        packA_reg[buf_index] = XA[(ic + ii + iii) * lda + (pc + kk)];
                        buf_index += 1;
                    }
                }
            }
            packA = &packA_reg[0];

            #endif

            for ( jc = 0; jc < n; jc += param_nc ) {        // 3-rd loop around micro-kernel
                jb = min( n - jc, param_nc );

                #ifdef NOPACK
                packB = &XB[ldb * pc + jc ];
                #else

                // Pack B: copy pb x jb block starting at (pc, jc) into register array
                double packB_reg[pb * jb];  // Use max size for safety
                int buf_index = 0;
                for (int jj = 0; jj < jb; jj += param_nr) {  // Loop over cols of kc x nc block
                    int jjb = min(jb - jj, param_nr);
                    for (int kk = 0; kk < pb; ++kk) {  // Loop over rows of kc x nr block
                        for (int jjj = 0; jjj<jjb; ++jjj) {   // Loop over rows of kc x nr block
                            packB_reg[buf_index] = XB[(pc + kk) * ldb + (jc + jj + jjj)];
                            buf_index += 1;
                        }
                    }
                }
                packB = &packB_reg[0];

                #endif

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
    
    // Perform matrix multiplication
    for ( l = 0; l < kc; ++l ) {                 
        for ( i = 0; i < mr; ++i ) { 
            // double as = a(i, l, ldc);
            double as = a(l, i, mr);
            for ( j = 0; j < nr; ++j ) { 
                // cloc[i][j] +=  as * b(l, j, ldc);
                cloc[i][j] +=  as * b(l, j, nr);
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

                        #ifdef NOPACK
                        &packA[i * ldc],          // assumes sq matrix, otherwise use lda
                        #else
                        &packA[i * pb],
                        #endif

                        #ifdef NOPACK
                        &packB[j], 
                        #else
                        &packB[j * pb],
                        #endif

                        &C[ i * ldc + j ],
                        ldc
                        );
        }                                                       // 1-th loop around micro-kernel
    }
}

