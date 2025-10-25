#include "dgemm_mykernel.h"
#include "parameters.h"

void DGEMM_mykernel::compute(const Mat& A, const Mat& B, Mat& C) {
    int m = A.rows();
    int k = A.cols();
    int n = B.cols();

    // my_dgemm_wo_packing(m, n, k, A.data(), k, B.data(), n, C.data(), n);
    my_dgemm_w_packing(m, n, k, A.data(), k, B.data(), n, C.data(), n);
}

string DGEMM_mykernel::name() {
    return "my_kernel";
}

void DGEMM_mykernel::my_dgemm_wo_packing(
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

    for ( ic = 0; ic < m; ic += param_mc ) {              // 5-th loop around micro-kernel
        ib = min( m - ic, param_mc );
        for ( pc = 0; pc < k; pc += param_kc ) {          // 4-th loop around micro-kernel
            pb = min( k - pc, param_kc );
            
            packA = &XA[pc + ic * lda ];

            for ( jc = 0; jc < n; jc += param_nc ) {        // 3-rd loop around micro-kernel
                jb = min( n - jc, param_nc );

                packB = &XB[ldb * pc + jc ];

                // Implement your macro-kernel here
                my_macro_kernel_wo_packing(
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

void DGEMM_mykernel::my_dgemm_w_packing(
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

    // int lanes = svcntd(); // Returns the number of double (float64) lanes in the SVE vector register
    // cout << lanes << endl;   // Output: 4
    // size_t vector_register_size = svcntb();
    // cout << vector_register_size << endl;    // output: 32 (bytes)

    
    for ( ic = 0; ic < m; ic += param_mc ) {              // 5-th loop around micro-kernel
        ib = min( m - ic, param_mc );
        for ( pc = 0; pc < k; pc += param_kc ) {          // 4-th loop around micro-kernel
            pb = min( k - pc, param_kc );

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

            for ( jc = 0; jc < n; jc += param_nc ) {        // 3-rd loop around micro-kernel
                jb = min( n - jc, param_nc );

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

                // Implement your macro-kernel here
                my_macro_kernel_w_packing(
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

/*
// C-based microkernel (PACK version)
void DGEMM_mykernel::my_dgemm_8x8x8(
    int    kc,
    int    mr,
    int    nr,
    const double *a,
    const double *b,
    double *c,
    int ldc
    )
{   
    svfloat64_t a0x, a1x, a2x, a3x, a4x, a5x, a6x, a7x;
    svfloat64_t bx_1, bx_2;
    svfloat64_t c0x_1, c0x_2, c1x_1, c1x_2, c2x_1, c2x_2, c3x_1, c3x_2, c4x_1, c4x_2, c5x_1, c5x_2, c6x_1, c6x_2, c7x_1, c7x_2;

    svbool_t npred_1 = svwhilelt_b64_u64(0,nr);
    svbool_t npred_2 = svwhilelt_b64_u64(4,nr);

    c0x_1 = svld1_f64(npred_1, c + 0 * ldc);
    c0x_2 = svld1_f64(npred_2, c + 0 * ldc + 4);
    c1x_1 = svld1_f64(npred_1, c + 1 * ldc);
    c1x_2 = svld1_f64(npred_2, c + 1 * ldc + 4);
    c2x_1 = svld1_f64(npred_1, c + 2 * ldc);
    c2x_2 = svld1_f64(npred_2, c + 2 * ldc + 4);
    c3x_1 = svld1_f64(npred_1, c + 3 * ldc);
    c3x_2 = svld1_f64(npred_2, c + 3 * ldc + 4);
    c4x_1 = svld1_f64(npred_1, c + 4 * ldc);
    c4x_2 = svld1_f64(npred_2, c + 4 * ldc + 4);
    c5x_1 = svld1_f64(npred_1, c + 5 * ldc);
    c5x_2 = svld1_f64(npred_2, c + 5 * ldc + 4);
    c6x_1 = svld1_f64(npred_1, c + 6 * ldc);
    c6x_2 = svld1_f64(npred_2, c + 6 * ldc + 4);
    c7x_1 = svld1_f64(npred_1, c + 7 * ldc);
    c7x_2 = svld1_f64(npred_2, c + 7 * ldc + 4);

    for (int kk=0; kk<kc; ++kk){
        float64_t aval = *(a + 0 + kk*mr);
        a0x = svdup_f64(aval);
        bx_1 = svld1_f64(npred_1, b + kk*nr);
        c0x_1 = svmla_f64_m(npred_1, c0x_1, bx_1, a0x); 
        bx_2 = svld1_f64(npred_2, b + 4 + kk*nr);
        c0x_2 = svmla_f64_m(npred_2, c0x_2, bx_2, a0x); 
    
        aval = *(a + 1 + kk*mr);
        a1x = svdup_f64(aval);
        bx_1 = svld1_f64(npred_1, b + kk*nr);
        c1x_1 = svmla_f64_m(npred_1, c1x_1, bx_1, a1x); 
        bx_2 = svld1_f64(npred_2, b + 4 + kk*nr);
        c1x_2 = svmla_f64_m(npred_2, c1x_2, bx_2, a1x); 
    
        aval = *(a + 2 + kk*mr);
        a2x = svdup_f64(aval);
        bx_1 = svld1_f64(npred_1, b + kk*nr);
        c2x_1 = svmla_f64_m(npred_1, c2x_1, bx_1, a2x); 
        bx_2 = svld1_f64(npred_2, b + 4 + kk*nr);
        c2x_2 = svmla_f64_m(npred_2, c2x_2, bx_2, a2x);
    
        aval = *(a + 3 + kk*mr);
        a3x = svdup_f64(aval);
        bx_1 = svld1_f64(npred_1, b + kk*nr);
        c3x_1 = svmla_f64_m(npred_1, c3x_1, bx_1, a3x); 
        bx_2 = svld1_f64(npred_2, b + 4 + kk*nr);
        c3x_2 = svmla_f64_m(npred_2, c3x_2, bx_2, a3x);
    
        aval = *(a + 4 + kk*mr);
        a4x = svdup_f64(aval);
        bx_1 = svld1_f64(npred_1, b + kk*nr);
        c4x_1 = svmla_f64_m(npred_1, c4x_1, bx_1, a4x); 
        bx_2 = svld1_f64(npred_2, b + 4 + kk*nr);
        c4x_2 = svmla_f64_m(npred_2, c4x_2, bx_2, a4x);
    
        aval = *(a + 5 + kk*mr);
        a5x = svdup_f64(aval);
        bx_1 = svld1_f64(npred_1, b + kk*nr);
        c5x_1 = svmla_f64_m(npred_1, c5x_1, bx_1, a5x); 
        bx_2 = svld1_f64(npred_2, b + 4 + kk*nr);
        c5x_2 = svmla_f64_m(npred_2, c5x_2, bx_2, a5x);
    
        aval = *(a + 6 + kk*mr);
        a6x = svdup_f64(aval);
        bx_1 = svld1_f64(npred_1, b + kk*nr);
        c6x_1 = svmla_f64_m(npred_1, c6x_1, bx_1, a6x); 
        bx_2 = svld1_f64(npred_2, b + 4 + kk*nr);
        c6x_2 = svmla_f64_m(npred_2, c6x_2, bx_2, a6x);
    
        aval = *(a + 7 + kk*mr);
        a7x = svdup_f64(aval);
        bx_1 = svld1_f64(npred_1, b + kk*nr);
        c7x_1 = svmla_f64_m(npred_1, c7x_1, bx_1, a7x); 
        bx_2 = svld1_f64(npred_2, b + 4 + kk*nr);
        c7x_2 = svmla_f64_m(npred_2, c7x_2, bx_2, a7x);
    }

    svst1_f64(npred_1, c + 0*ldc, c0x_1);
    svst1_f64(npred_2, c + 0*ldc + 4, c0x_2);
    svst1_f64(npred_1, c + 1*ldc, c1x_1);
    svst1_f64(npred_2, c + 1*ldc + 4, c1x_2);
    svst1_f64(npred_1, c + 2*ldc, c2x_1);
    svst1_f64(npred_2, c + 2*ldc + 4, c2x_2);
    svst1_f64(npred_1, c + 3*ldc, c3x_1);
    svst1_f64(npred_2, c + 3*ldc + 4, c3x_2);
    svst1_f64(npred_1, c + 4*ldc, c4x_1);
    svst1_f64(npred_2, c + 4*ldc + 4, c4x_2);
    svst1_f64(npred_1, c + 5*ldc, c5x_1);
    svst1_f64(npred_2, c + 5*ldc + 4, c5x_2);
    svst1_f64(npred_1, c + 6*ldc, c6x_1);
    svst1_f64(npred_2, c + 6*ldc + 4, c6x_2);
    svst1_f64(npred_1, c + 7*ldc, c7x_1);
    svst1_f64(npred_2, c + 7*ldc + 4, c7x_2);
}
*/

void DGEMM_mykernel::my_dgemm_ukr_w_packing_w_simd( 
    int    kc,
    int    mr,
    int    nr,
    const double *a,
    const double *b,
    double *c,
    int ldc)
{   
    // for (int k = 0; k < kc; k+=8){
    //     int kb = min(kc-k, 8);
    //     my_dgemm_8x8x8( 
    //         kb,
    //         mr,
    //         nr,
    //         &a[k*mr],
    //         &b[k*nr],
    //         &c[k*nr],
    //         ldc                
    //     );
    // }

    svfloat64_t a0x, a1x, a2x, a3x, a4x, a5x, a6x, a7x;
    svfloat64_t bx_1, bx_2;
    svfloat64_t c0x_1, c0x_2, c1x_1, c1x_2, c2x_1, c2x_2, c3x_1, c3x_2, c4x_1, c4x_2, c5x_1, c5x_2, c6x_1, c6x_2, c7x_1, c7x_2;

    svbool_t npred_1 = svwhilelt_b64_u64(0,nr);
    svbool_t npred_2 = svwhilelt_b64_u64(4,nr);

    c0x_1 = svld1_f64(npred_1, c + 0 * ldc);
    c0x_2 = svld1_f64(npred_2, c + 0 * ldc + 4);
    c1x_1 = svld1_f64(npred_1, c + 1 * ldc);
    c1x_2 = svld1_f64(npred_2, c + 1 * ldc + 4);
    c2x_1 = svld1_f64(npred_1, c + 2 * ldc);
    c2x_2 = svld1_f64(npred_2, c + 2 * ldc + 4);
    c3x_1 = svld1_f64(npred_1, c + 3 * ldc);
    c3x_2 = svld1_f64(npred_2, c + 3 * ldc + 4);
    c4x_1 = svld1_f64(npred_1, c + 4 * ldc);
    c4x_2 = svld1_f64(npred_2, c + 4 * ldc + 4);
    c5x_1 = svld1_f64(npred_1, c + 5 * ldc);
    c5x_2 = svld1_f64(npred_2, c + 5 * ldc + 4);
    c6x_1 = svld1_f64(npred_1, c + 6 * ldc);
    c6x_2 = svld1_f64(npred_2, c + 6 * ldc + 4);
    c7x_1 = svld1_f64(npred_1, c + 7 * ldc);
    c7x_2 = svld1_f64(npred_2, c + 7 * ldc + 4);

    for (int kk=0; kk<kc; ++kk){
        float64_t aval = *(a + 0 + kk*mr);
        a0x = svdup_f64(aval);
        bx_1 = svld1_f64(npred_1, b + kk*nr);
        c0x_1 = svmla_f64_m(npred_1, c0x_1, bx_1, a0x); 
        bx_2 = svld1_f64(npred_2, b + 4 + kk*nr);
        c0x_2 = svmla_f64_m(npred_2, c0x_2, bx_2, a0x); 
    
        aval = *(a + 1 + kk*mr);
        a1x = svdup_f64(aval);
        bx_1 = svld1_f64(npred_1, b + kk*nr);
        c1x_1 = svmla_f64_m(npred_1, c1x_1, bx_1, a1x); 
        bx_2 = svld1_f64(npred_2, b + 4 + kk*nr);
        c1x_2 = svmla_f64_m(npred_2, c1x_2, bx_2, a1x); 
    
        aval = *(a + 2 + kk*mr);
        a2x = svdup_f64(aval);
        bx_1 = svld1_f64(npred_1, b + kk*nr);
        c2x_1 = svmla_f64_m(npred_1, c2x_1, bx_1, a2x); 
        bx_2 = svld1_f64(npred_2, b + 4 + kk*nr);
        c2x_2 = svmla_f64_m(npred_2, c2x_2, bx_2, a2x);
    
        aval = *(a + 3 + kk*mr);
        a3x = svdup_f64(aval);
        bx_1 = svld1_f64(npred_1, b + kk*nr);
        c3x_1 = svmla_f64_m(npred_1, c3x_1, bx_1, a3x); 
        bx_2 = svld1_f64(npred_2, b + 4 + kk*nr);
        c3x_2 = svmla_f64_m(npred_2, c3x_2, bx_2, a3x);
    
        aval = *(a + 4 + kk*mr);
        a4x = svdup_f64(aval);
        bx_1 = svld1_f64(npred_1, b + kk*nr);
        c4x_1 = svmla_f64_m(npred_1, c4x_1, bx_1, a4x); 
        bx_2 = svld1_f64(npred_2, b + 4 + kk*nr);
        c4x_2 = svmla_f64_m(npred_2, c4x_2, bx_2, a4x);
    
        aval = *(a + 5 + kk*mr);
        a5x = svdup_f64(aval);
        bx_1 = svld1_f64(npred_1, b + kk*nr);
        c5x_1 = svmla_f64_m(npred_1, c5x_1, bx_1, a5x); 
        bx_2 = svld1_f64(npred_2, b + 4 + kk*nr);
        c5x_2 = svmla_f64_m(npred_2, c5x_2, bx_2, a5x);
    
        aval = *(a + 6 + kk*mr);
        a6x = svdup_f64(aval);
        bx_1 = svld1_f64(npred_1, b + kk*nr);
        c6x_1 = svmla_f64_m(npred_1, c6x_1, bx_1, a6x); 
        bx_2 = svld1_f64(npred_2, b + 4 + kk*nr);
        c6x_2 = svmla_f64_m(npred_2, c6x_2, bx_2, a6x);
    
        aval = *(a + 7 + kk*mr);
        a7x = svdup_f64(aval);
        bx_1 = svld1_f64(npred_1, b + kk*nr);
        c7x_1 = svmla_f64_m(npred_1, c7x_1, bx_1, a7x); 
        bx_2 = svld1_f64(npred_2, b + 4 + kk*nr);
        c7x_2 = svmla_f64_m(npred_2, c7x_2, bx_2, a7x);
    }

    svst1_f64(npred_1, c + 0*ldc, c0x_1);
    svst1_f64(npred_2, c + 0*ldc + 4, c0x_2);
    svst1_f64(npred_1, c + 1*ldc, c1x_1);
    svst1_f64(npred_2, c + 1*ldc + 4, c1x_2);
    svst1_f64(npred_1, c + 2*ldc, c2x_1);
    svst1_f64(npred_2, c + 2*ldc + 4, c2x_2);
    svst1_f64(npred_1, c + 3*ldc, c3x_1);
    svst1_f64(npred_2, c + 3*ldc + 4, c3x_2);
    svst1_f64(npred_1, c + 4*ldc, c4x_1);
    svst1_f64(npred_2, c + 4*ldc + 4, c4x_2);
    svst1_f64(npred_1, c + 5*ldc, c5x_1);
    svst1_f64(npred_2, c + 5*ldc + 4, c5x_2);
    svst1_f64(npred_1, c + 6*ldc, c6x_1);
    svst1_f64(npred_2, c + 6*ldc + 4, c6x_2);
    svst1_f64(npred_1, c + 7*ldc, c7x_1);
    svst1_f64(npred_2, c + 7*ldc + 4, c7x_2);
}


// C-based microkernel (NOPACK version)
void DGEMM_mykernel::my_dgemm_ukr_wo_packing(
    int    kc,
    int    mr,
    int    nr,
    const double *a,
    const double *b,
    double *c,
    int ldc
    )
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
            double as = a(i, l, ldc);
            for ( j = 0; j < nr; ++j ) { 
                cloc[i][j] +=  as * b(l, j, ldc);
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

// C-based macrokernel (PACK version)
void DGEMM_mykernel::my_macro_kernel_w_packing(
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
            my_dgemm_ukr_w_packing_w_simd (
                pb,
                min(ib - i, param_mr),
                min(jb - j, param_nr),
                &packA[i * pb],
                &packB[j * pb],
                &C[ i * ldc + j ],
                ldc
            );
        }                                                       // 1-th loop around micro-kernel
    }
}

// C-based macrokernel (UNPACK version)
void DGEMM_mykernel::my_macro_kernel_wo_packing(
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
            my_dgemm_ukr_wo_packing (
                pb,
                min(ib-i, param_mr),
                min(jb-j, param_nr),
                &packA[i * ldc],          // assumes sq matrix, otherwise use lda
                &packB[j], 
                &C[ i * ldc + j ],
                ldc
            );
        }                                                       // 1-th loop around micro-kernel
    }
}
