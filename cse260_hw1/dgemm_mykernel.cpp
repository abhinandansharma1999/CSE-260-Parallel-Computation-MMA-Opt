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
    // pre-allocate packed vectors.
    double* packedA = new double[param_mc * param_kc];
    double* packedB = new double[param_kc * param_nc];

    // Using NOPACK option for simplicity
    // #define NOPACK
    for ( ic = 0; ic < m; ic += param_mc ) {              // 5-th loop around micro-kernel
        ib = min( m - ic, param_mc );
        for ( pc = 0; pc < k; pc += param_kc ) {          // 4-th loop around micro-kernel
            pb = min( k - pc, param_kc );
            
            // pack A such that it has more cache efficient memory access
            pack_A(ib, pb, &XA[pc + ic * lda], lda, packedA);
            packA = packedA;

            for ( jc = 0; jc < n; jc += param_nc ) {        // 3-rd loop around micro-kernel
                jb = min( n - jc, param_nc );
                
                // pack B such that it has more efficient memory access.
                pack_B(pb, jb, &XB[ldb * pc + jc], ldb, packedB);
                packB = packedB;

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
// Implement your micro-kernel here -- based on code from discussion session.
void DGEMM_mykernel::my_dgemm_ukr(int kc, int mr, int nr,
                                  const double *a, const double *b,
                                  double *c, int ldc) {
    
    // We assume mr = 8 and nr = 4, almost all of the time, and handle it efficiently.
    if (mr == 8 && nr == 8) {
        // Small enough to fit in registers
        svfloat64_t c00, c10, c20, c30, c40, c50, c60, c70;
        svfloat64_t c01, c11, c21, c31, c41, c51, c61, c71;
        
        // Load all Cs into registers
        svbool_t all_pred = svptrue_b64();
        c00 = svld1(all_pred, &c(0, 0, ldc));
        c10 = svld1(all_pred, &c(1, 0, ldc));
        c20 = svld1(all_pred, &c(2, 0, ldc));
        c30 = svld1(all_pred, &c(3, 0, ldc));
        c40 = svld1(all_pred, &c(4, 0, ldc));
        c50 = svld1(all_pred, &c(5, 0, ldc));
        c60 = svld1(all_pred, &c(6, 0, ldc));
        c70 = svld1(all_pred, &c(7, 0, ldc));

        c01 = svld1(all_pred, &c(0, 4, ldc));
        c11 = svld1(all_pred, &c(1, 4, ldc));
        c21 = svld1(all_pred, &c(2, 4, ldc));
        c31 = svld1(all_pred, &c(3, 4, ldc));
        c41 = svld1(all_pred, &c(4, 4, ldc));
        c51 = svld1(all_pred, &c(5, 4, ldc));
        c61 = svld1(all_pred, &c(6, 4, ldc));
        c71 = svld1(all_pred, &c(7, 4, ldc));
        
        for (int l = 0; l < kc; ++l) {
            // get 8 values of a.
            double a0 = a(l, 0, 8);
            double a1 = a(l, 1, 8);
            double a2 = a(l, 2, 8);
            double a3 = a(l, 3, 8);
            double a4 = a(l, 4, 8);
            double a5 = a(l, 5, 8);
            double a6 = a(l, 6, 8);
            double a7 = a(l, 7, 8);
            
            // Get 4 row values of b
            svfloat64_t b0 = svld1(all_pred, &b(l, 0, 8));
            svfloat64_t b1 = svld1(all_pred, &b(l, 4, 8));
            
            // Fused multiply add on every row to get values of c
            c00 = svmla_f64_m(all_pred, c00, svdup_f64(a0), b0);
            c10 = svmla_f64_m(all_pred, c10, svdup_f64(a1), b0);
            c20 = svmla_f64_m(all_pred, c20, svdup_f64(a2), b0);
            c30 = svmla_f64_m(all_pred, c30, svdup_f64(a3), b0);
            c40 = svmla_f64_m(all_pred, c40, svdup_f64(a4), b0);
            c50 = svmla_f64_m(all_pred, c50, svdup_f64(a5), b0);
            c60 = svmla_f64_m(all_pred, c60, svdup_f64(a6), b0);
            c70 = svmla_f64_m(all_pred, c70, svdup_f64(a7), b0);

            c01 = svmla_f64_m(all_pred, c01, svdup_f64(a0), b1);
            c11 = svmla_f64_m(all_pred, c11, svdup_f64(a1), b1);
            c21 = svmla_f64_m(all_pred, c21, svdup_f64(a2), b1);
            c31 = svmla_f64_m(all_pred, c31, svdup_f64(a3), b1);
            c41 = svmla_f64_m(all_pred, c41, svdup_f64(a4), b1);
            c51 = svmla_f64_m(all_pred, c51, svdup_f64(a5), b1);
            c61 = svmla_f64_m(all_pred, c61, svdup_f64(a6), b1);
            c71 = svmla_f64_m(all_pred, c71, svdup_f64(a7), b1);
        }
        
        // Write values back to C.
        svst1(all_pred, &c(0, 0, ldc), c00);
        svst1(all_pred, &c(1, 0, ldc), c10);
        svst1(all_pred, &c(2, 0, ldc), c20);
        svst1(all_pred, &c(3, 0, ldc), c30);
        svst1(all_pred, &c(4, 0, ldc), c40);
        svst1(all_pred, &c(5, 0, ldc), c50);
        svst1(all_pred, &c(6, 0, ldc), c60);
        svst1(all_pred, &c(7, 0, ldc), c70);

        svst1(all_pred, &c(0, 4, ldc), c01);
        svst1(all_pred, &c(1, 4, ldc), c11);
        svst1(all_pred, &c(2, 4, ldc), c21);
        svst1(all_pred, &c(3, 4, ldc), c31);
        svst1(all_pred, &c(4, 4, ldc), c41);
        svst1(all_pred, &c(5, 4, ldc), c51);
        svst1(all_pred, &c(6, 4, ldc), c61);
        svst1(all_pred, &c(7, 4, ldc), c71);
    }
    else {
        // Marginal case handling...
        handle_arbitrary_size(kc, mr, nr, a, b, c, ldc);
    }
}

/**
 * Handles the marginal cases of the microkernel where mr != 8 and nr != 4.
 * Basic matrix vector multiply.
 */
void DGEMM_mykernel::handle_arbitrary_size( int    kc,
                                  int    mr,
                                  int    nr,
                                  const double *a,
                                  const double *b,
                                  double *c,
                                  int ldc)
{
    int l, j, i;
    // load c values into registers.
    double cloc[param_mr][param_nr] = {{0}};
    int vlen = svcntd();

    // Load C into local array
    for (i = 0; i < mr; ++i) {
        for (j = 0; j < nr; ++j) {
            cloc[i][j] = c(i, j, ldc);
        }
    }

    
    // Perform matrix multiplication
    for ( l = 0; l < kc; ++l ) {                 
        for ( i = 0; i < mr; ++i ) { 
            // project a over 4 values
            svfloat64_t ax = svdup_f64(a(l, i, mr));
            for ( j = 0; j < nr; j += vlen) {
                // mask for loop control
                svbool_t pred = svwhilelt_b64(j, nr);
                // load b into registers
                svfloat64_t bx = svld1(pred, &b(l, j, nr));
                // load c into vector registers
                svfloat64_t cx = svld1(pred, &cloc[i][j]);
                // fused multiply add
                cx = svmla_f64_m(pred, cx, ax, bx);
                
                // store results back.
                svst1(pred, &cloc[i][j], cx);
            }
        }
    }
    
    // Store local array back to RAM
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

/* 
* function that packs A into a single dimmension vector that will be efficiently unraveled in 
* execution.
* You divide A into Mr x Kc strips, and traverse each in top to bottom, then left to right.
*/
void DGEMM_mykernel::pack_A(int m, int k, const double * A, int lda, double * packed_A){
    int buffer_counter = 0;
    // traverse over Mc x Kc strips of A
    for(int i = 0; i < m; i += param_mr){
        int ib = min(m - i, param_mr);
        // traverse left to right
        for(int p = 0; p < k; p++){
            // traverse top to bottom
            for(int ii = 0; ii < ib; ii++){
                packed_A[buffer_counter++] = A[(i + ii) * lda + p];
            }
        }
    }

}

/* 
* function that packs B into a single dimmension vector that will be efficiently unraveled in 
* execution.
* You divide B into Kc x Nr strips, and traverse each left to right, then top to bottom.
*/
void DGEMM_mykernel::pack_B(int k, int n, const double * B, int ldb, double * packed_B){
    int buffer_counter = 0;
    // traverse over Kc x Nr strips of B
    for(int j = 0; j < n; j += param_nr){
        int ib = min(n - j, param_nr);
        // traverse top to bottom
        for(int p = 0; p < k; p++){
            // traverse left to right
            for(int jj = 0; jj < ib; jj++){
                packed_B[buffer_counter++] = B[p * ldb + j + jj];
            }
        }

    }
}
