#include "dgemm_mykernel.h"
#include "parameters.h"

void DGEMM_mykernel::compute(const Mat& A, const Mat& B, Mat& C) {
    int m = A.rows();
    int k = A.cols();
    int n = B.cols();

    if(m<128)  my_dgemm_wo_packing(m, n, k, A.data(), k, B.data(), n, C.data(), n);
    if(m>=128) my_dgemm_w_packing(m, n, k, A.data(), k, B.data(), n, C.data(), n);
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


// void DGEMM_mykernel::my_dgemm_ukr_w_packing_w_simd( 
//     int    kc,
//     int    mr,
//     int    nr,
//     const double *a,
//     const double *b,
//     double *c,
//     int ldc)
// {   
//     // for (int k = 0; k < kc; k+=8){
//     //     int kb = min(kc-k, 8);
//     //     my_dgemm_8x8x8( 
//     //         kb,
//     //         mr,
//     //         nr,
//     //         &a[k*mr],
//     //         &b[k*nr],
//     //         &c[k*nr],
//     //         ldc                
//     //     );
//     // }

//     int vlen = svcntd();

//     svfloat64_t a0x, a1x, a2x, a3x, a4x, a5x, a6x, a7x;
//     svfloat64_t bx_1, bx_2;
//     svfloat64_t c0x_1, c0x_2, c1x_1, c1x_2, c2x_1, c2x_2, c3x_1, c3x_2, c4x_1, c4x_2, c5x_1, c5x_2, c6x_1, c6x_2, c7x_1, c7x_2;
    
//     float64_t aval;

//     svbool_t npred_1 = svwhilelt_b64((uint64_t)0, (uint64_t)nr);  
//     svbool_t npred_2 = svwhilelt_b64((uint64_t)vlen, (uint64_t)nr); 
//     // svbool_t npred_1_1 = svwhilelt_b64_u64(1,nr);
//     // svbool_t npred_1_2 = svwhilelt_b64_u64(vlen,nr);
//     // svbool_t npred_2_1 = svwhilelt_b64_u64(2,nr);
//     // svbool_t npred_2_2 = svwhilelt_b64_u64(vlen,nr);
//     // svbool_t npred_3_1 = svwhilelt_b64_u64(3,nr);
//     // svbool_t npred_3_2 = svwhilelt_b64_u64(vlen,nr);
//     // svbool_t npred_4_1 = svwhilelt_b64_u64(0,nr);
//     // svbool_t npred_4_2 = svwhilelt_b64_u64(vlen,nr+0);
//     // svbool_t npred_5_1 = svwhilelt_b64_u64(0,nr);
//     // svbool_t npred_5_2 = svwhilelt_b64_u64(vlen,nr+1);
//     // svbool_t npred_6_1 = svwhilelt_b64_u64(0,nr);
//     // svbool_t npred_6_2 = svwhilelt_b64_u64(vlen,nr+2);
//     // svbool_t npred_7_1 = svwhilelt_b64_u64(0,nr);
//     // svbool_t npred_7_2 = svwhilelt_b64_u64(vlen,nr+3);

//     // svbool_t npred_1 = svptrue_b64();
//     // svbool_t npred_2 = svptrue_b64();
//     if(nr>0){
//     c0x_1 = svld1_f64(npred_1, c + 0 * ldc);
//     c0x_2 = svld1_f64(npred_2, c + 0 * ldc + vlen);
//     }
//     if(nr>1){
//     c1x_1 = svld1_f64(npred_1, c + 1 * ldc);
//     c1x_2 = svld1_f64(npred_2, c + 1 * ldc + vlen);
//     }
//     if(nr>2){
//     c2x_1 = svld1_f64(npred_1, c + 2 * ldc);
//     c2x_2 = svld1_f64(npred_2, c + 2 * ldc + vlen);
//     }
//     if(nr>3){
//     c3x_1 = svld1_f64(npred_1, c + 3 * ldc);
//     c3x_2 = svld1_f64(npred_2, c + 3 * ldc + vlen);
//     }
//     if(nr>4){
//     c4x_1 = svld1_f64(npred_1, c + 4 * ldc);
//     c4x_2 = svld1_f64(npred_2, c + 4 * ldc + vlen);
//     }
//     if(nr>5){
//     c5x_1 = svld1_f64(npred_1, c + 5 * ldc);
//     c5x_2 = svld1_f64(npred_2, c + 5 * ldc + vlen);
//     }
//     if(nr>6){
//     c6x_1 = svld1_f64(npred_1, c + 6 * ldc);
//     c6x_2 = svld1_f64(npred_2, c + 6 * ldc + vlen);
//     }
//     if(nr>7){
//     c7x_1 = svld1_f64(npred_1, c + 7 * ldc);
//     c7x_2 = svld1_f64(npred_2, c + 7 * ldc + vlen);
//     }

//     for (int kk=0; kk<kc; ++kk){
//         if(nr>0){
//         aval = *(a + 0 + kk*mr);
//         a0x = svdup_f64(aval);
//         bx_1 = svld1_f64(npred_1, b + kk*nr);
//         c0x_1 = svmla_f64_m(npred_1, c0x_1, bx_1, a0x); 
//         bx_2 = svld1_f64(npred_2, b + vlen + kk*nr);
//         c0x_2 = svmla_f64_m(npred_2, c0x_2, bx_2, a0x); 
//         }        
//         if (nr>1){
//         aval = *(a + 1 + kk*mr);
//         a1x = svdup_f64(aval);
//         bx_1 = svld1_f64(npred_1, b + kk*nr);
//         c1x_1 = svmla_f64_m(npred_1, c1x_1, bx_1, a1x); 
//         bx_2 = svld1_f64(npred_2, b + vlen + kk*nr);
//         c1x_2 = svmla_f64_m(npred_2, c1x_2, bx_2, a1x); 
//         }        
//         if (nr>2){        
//         aval = *(a + 2 + kk*mr);
//         a2x = svdup_f64(aval);
//         bx_1 = svld1_f64(npred_1, b + kk*nr);
//         c2x_1 = svmla_f64_m(npred_1, c2x_1, bx_1, a2x); 
//         bx_2 = svld1_f64(npred_2, b + vlen + kk*nr);
//         c2x_2 = svmla_f64_m(npred_2, c2x_2, bx_2, a2x);
//         }        
//         if (nr>3){
//         aval = *(a + 3 + kk*mr);
//         a3x = svdup_f64(aval);
//         bx_1 = svld1_f64(npred_1, b + kk*nr);
//         c3x_1 = svmla_f64_m(npred_1, c3x_1, bx_1, a3x); 
//         bx_2 = svld1_f64(npred_2, b + vlen + kk*nr);
//         c3x_2 = svmla_f64_m(npred_2, c3x_2, bx_2, a3x);
//         }               
//         if (nr>4){
//         aval = *(a + 4 + kk*mr);
//         a4x = svdup_f64(aval);
//         bx_1 = svld1_f64(npred_1, b + kk*nr);
//         c4x_1 = svmla_f64_m(npred_1, c4x_1, bx_1, a4x); 
//         bx_2 = svld1_f64(npred_2, b + vlen + kk*nr);
//         c4x_2 = svmla_f64_m(npred_2, c4x_2, bx_2, a4x);
//         }        
//         if (nr>5){
//         aval = *(a + 5 + kk*mr);
//         a5x = svdup_f64(aval);
//         bx_1 = svld1_f64(npred_1, b + kk*nr);
//         c5x_1 = svmla_f64_m(npred_1, c5x_1, bx_1, a5x); 
//         bx_2 = svld1_f64(npred_2, b + vlen + kk*nr);
//         c5x_2 = svmla_f64_m(npred_2, c5x_2, bx_2, a5x);
//         }        
//         if (nr>6){
//         aval = *(a + 6 + kk*mr);
//         a6x = svdup_f64(aval);
//         bx_1 = svld1_f64(npred_1, b + kk*nr);
//         c6x_1 = svmla_f64_m(npred_1, c6x_1, bx_1, a6x); 
//         bx_2 = svld1_f64(npred_2, b + vlen + kk*nr);
//         c6x_2 = svmla_f64_m(npred_2, c6x_2, bx_2, a6x);
//         }        
//         if (nr>7){
//         aval = *(a + 7 + kk*mr);
//         a7x = svdup_f64(aval);
//         bx_1 = svld1_f64(npred_1, b + kk*nr);
//         c7x_1 = svmla_f64_m(npred_1, c7x_1, bx_1, a7x); 
//         bx_2 = svld1_f64(npred_2, b + vlen + kk*nr);
//         c7x_2 = svmla_f64_m(npred_2, c7x_2, bx_2, a7x);
//         }
//     }

//     if (nr>0){
//     svst1_f64(npred_1, c + 0*ldc, c0x_1);
//     svst1_f64(npred_2, c + 0*ldc + vlen, c0x_2);
//     }        
//     if (nr>1){
//     svst1_f64(npred_1, c + 1*ldc, c1x_1);
//     svst1_f64(npred_2, c + 1*ldc + vlen, c1x_2);
//     }        
//     if (nr>2){
//     svst1_f64(npred_1, c + 2*ldc, c2x_1);
//     svst1_f64(npred_2, c + 2*ldc + vlen, c2x_2);
//     }        
//     if (nr>3){
//     svst1_f64(npred_1, c + 3*ldc, c3x_1);
//     svst1_f64(npred_2, c + 3*ldc + vlen, c3x_2);
//     }        
//     if (nr>4){
//     svst1_f64(npred_1, c + 4*ldc, c4x_1);
//     svst1_f64(npred_2, c + 4*ldc + vlen, c4x_2);
//     }        
//     if (nr>5){
//     svst1_f64(npred_1, c + 5*ldc, c5x_1);
//     svst1_f64(npred_2, c + 5*ldc + vlen, c5x_2);
//     }        
//     if (nr>6){
//     svst1_f64(npred_1, c + 6*ldc, c6x_1);
//     svst1_f64(npred_2, c + 6*ldc + vlen, c6x_2);
//     }        
//     if (nr>7){
//     svst1_f64(npred_1, c + 7*ldc, c7x_1);
//     svst1_f64(npred_2, c + 7*ldc + vlen, c7x_2);
//     }
// }

void DGEMM_mykernel::my_dgemm_ukr_w_packing_w_simd( 
    int    kc,
    int    mr,
    int    nr,
    const double *a,
    const double *b,
    double *c,
    int ldc)
{   
    int vlen = svcntd();  // Typically 4 for 256-bit SVE with doubles
    
    svfloat64_t a0x, a1x, a2x, a3x, a4x, a5x, a6x, a7x;
    svfloat64_t bx_1, bx_2;
    svfloat64_t c0x_1, c0x_2, c1x_1, c1x_2, c2x_1, c2x_2, c3x_1, c3x_2;
    svfloat64_t c4x_1, c4x_2, c5x_1, c5x_2, c6x_1, c6x_2, c7x_1, c7x_2;
    
    float64_t aval;

    svbool_t npred_1 = svwhilelt_b64_u64(0, nr);
    svbool_t npred_2 = svwhilelt_b64_u64(vlen, nr);

    // Load C values (only for rows that exist: mr can be < 8)
    if (mr > 0) {
        c0x_1 = svld1_f64(npred_1, c + 0 * ldc);
        if (nr > vlen) c0x_2 = svld1_f64(npred_2, c + 0 * ldc + vlen);
    }
    if (mr > 1) {
        c1x_1 = svld1_f64(npred_1, c + 1 * ldc);
        if (nr > vlen) c1x_2 = svld1_f64(npred_2, c + 1 * ldc + vlen);
    }
    if (mr > 2) {
        c2x_1 = svld1_f64(npred_1, c + 2 * ldc);
        if (nr > vlen) c2x_2 = svld1_f64(npred_2, c + 2 * ldc + vlen);
    }
    if (mr > 3) {
        c3x_1 = svld1_f64(npred_1, c + 3 * ldc);
        if (nr > vlen) c3x_2 = svld1_f64(npred_2, c + 3 * ldc + vlen);
    }
    if (mr > 4) {
        c4x_1 = svld1_f64(npred_1, c + 4 * ldc);
        if (nr > vlen) c4x_2 = svld1_f64(npred_2, c + 4 * ldc + vlen);
    }
    if (mr > 5) {
        c5x_1 = svld1_f64(npred_1, c + 5 * ldc);
        if (nr > vlen) c5x_2 = svld1_f64(npred_2, c + 5 * ldc + vlen);
    }
    if (mr > 6) {
        c6x_1 = svld1_f64(npred_1, c + 6 * ldc);
        if (nr > vlen) c6x_2 = svld1_f64(npred_2, c + 6 * ldc + vlen);
    }
    if (mr > 7) {
        c7x_1 = svld1_f64(npred_1, c + 7 * ldc);
        if (nr > vlen) c7x_2 = svld1_f64(npred_2, c + 7 * ldc + vlen);
    }

    // Main computation loop
    for (int kk = 0; kk < kc; ++kk) {
        // Load B row vectors
        bx_1 = svld1_f64(npred_1, b + kk * nr);
        if (nr > vlen) {
            bx_2 = svld1_f64(npred_2, b + kk * nr + vlen);
        }

        // Row 0
        if (mr > 0) {
            aval = a[kk * mr + 0];
            a0x = svdup_f64(aval);
            c0x_1 = svmla_f64_m(npred_1, c0x_1, bx_1, a0x);
            if (nr > vlen) c0x_2 = svmla_f64_m(npred_2, c0x_2, bx_2, a0x);
        }
        
        // Row 1
        if (mr > 1) {
            aval = a[kk * mr + 1];
            a1x = svdup_f64(aval);
            c1x_1 = svmla_f64_m(npred_1, c1x_1, bx_1, a1x);
            if (nr > vlen) c1x_2 = svmla_f64_m(npred_2, c1x_2, bx_2, a1x);
        }
        
        // Row 2
        if (mr > 2) {
            aval = a[kk * mr + 2];
            a2x = svdup_f64(aval);
            c2x_1 = svmla_f64_m(npred_1, c2x_1, bx_1, a2x);
            if (nr > vlen) c2x_2 = svmla_f64_m(npred_2, c2x_2, bx_2, a2x);
        }
        
        // Row 3
        if (mr > 3) {
            aval = a[kk * mr + 3];
            a3x = svdup_f64(aval);
            c3x_1 = svmla_f64_m(npred_1, c3x_1, bx_1, a3x);
            if (nr > vlen) c3x_2 = svmla_f64_m(npred_2, c3x_2, bx_2, a3x);
        }
        
        // Row 4
        if (mr > 4) {
            aval = a[kk * mr + 4];
            a4x = svdup_f64(aval);
            c4x_1 = svmla_f64_m(npred_1, c4x_1, bx_1, a4x);
            if (nr > vlen) c4x_2 = svmla_f64_m(npred_2, c4x_2, bx_2, a4x);
        }
        
        // Row 5
        if (mr > 5) {
            aval = a[kk * mr + 5];
            a5x = svdup_f64(aval);
            c5x_1 = svmla_f64_m(npred_1, c5x_1, bx_1, a5x);
            if (nr > vlen) c5x_2 = svmla_f64_m(npred_2, c5x_2, bx_2, a5x);
        }
        
        // Row 6
        if (mr > 6) {
            aval = a[kk * mr + 6];
            a6x = svdup_f64(aval);
            c6x_1 = svmla_f64_m(npred_1, c6x_1, bx_1, a6x);
            if (nr > vlen) c6x_2 = svmla_f64_m(npred_2, c6x_2, bx_2, a6x);
        }
        
        // Row 7
        if (mr > 7) {
            aval = a[kk * mr + 7];
            a7x = svdup_f64(aval);
            c7x_1 = svmla_f64_m(npred_1, c7x_1, bx_1, a7x);
            if (nr > vlen) c7x_2 = svmla_f64_m(npred_2, c7x_2, bx_2, a7x);
        }
    }

    // Store results back to C
    if (mr > 0) {
        svst1_f64(npred_1, c + 0 * ldc, c0x_1);
        if (nr > vlen) svst1_f64(npred_2, c + 0 * ldc + vlen, c0x_2);
    }
    if (mr > 1) {
        svst1_f64(npred_1, c + 1 * ldc, c1x_1);
        if (nr > vlen) svst1_f64(npred_2, c + 1 * ldc + vlen, c1x_2);
    }
    if (mr > 2) {
        svst1_f64(npred_1, c + 2 * ldc, c2x_1);
        if (nr > vlen) svst1_f64(npred_2, c + 2 * ldc + vlen, c2x_2);
    }
    if (mr > 3) {
        svst1_f64(npred_1, c + 3 * ldc, c3x_1);
        if (nr > vlen) svst1_f64(npred_2, c + 3 * ldc + vlen, c3x_2);
    }
    if (mr > 4) {
        svst1_f64(npred_1, c + 4 * ldc, c4x_1);
        if (nr > vlen) svst1_f64(npred_2, c + 4 * ldc + vlen, c4x_2);
    }
    if (mr > 5) {
        svst1_f64(npred_1, c + 5 * ldc, c5x_1);
        if (nr > vlen) svst1_f64(npred_2, c + 5 * ldc + vlen, c5x_2);
    }
    if (mr > 6) {
        svst1_f64(npred_1, c + 6 * ldc, c6x_1);
        if (nr > vlen) svst1_f64(npred_2, c + 6 * ldc + vlen, c6x_2);
    }
    if (mr > 7) {
        svst1_f64(npred_1, c + 7 * ldc, c7x_1);
        if (nr > vlen) svst1_f64(npred_2, c + 7 * ldc + vlen, c7x_2);
    }
}

void DGEMM_mykernel::my_dgemm_ukr_wo_packing_w_simd( 
    int    kc,
    int    mr,
    int    nr,
    const double *a,
    const double *b,
    double *c,
    int ldc)
{   
    int vlen = svcntd();  // Typically 4 for 256-bit SVE with doubles
    
    svfloat64_t a0x, a1x, a2x, a3x, a4x, a5x, a6x, a7x;
    svfloat64_t bx_1, bx_2;
    svfloat64_t c0x_1, c0x_2, c1x_1, c1x_2, c2x_1, c2x_2, c3x_1, c3x_2;
    svfloat64_t c4x_1, c4x_2, c5x_1, c5x_2, c6x_1, c6x_2, c7x_1, c7x_2;
    
    float64_t aval;

    svbool_t npred_1 = svwhilelt_b64_u64(0, nr);
    svbool_t npred_2 = svwhilelt_b64_u64(vlen, nr);

    // Load C values (only for rows that exist: mr can be < 8)
    if (mr > 0) {
        c0x_1 = svld1_f64(npred_1, c + 0 * ldc);
        if (nr > vlen) c0x_2 = svld1_f64(npred_2, c + 0 * ldc + vlen);
    }
    if (mr > 1) {
        c1x_1 = svld1_f64(npred_1, c + 1 * ldc);
        if (nr > vlen) c1x_2 = svld1_f64(npred_2, c + 1 * ldc + vlen);
    }
    if (mr > 2) {
        c2x_1 = svld1_f64(npred_1, c + 2 * ldc);
        if (nr > vlen) c2x_2 = svld1_f64(npred_2, c + 2 * ldc + vlen);
    }
    if (mr > 3) {
        c3x_1 = svld1_f64(npred_1, c + 3 * ldc);
        if (nr > vlen) c3x_2 = svld1_f64(npred_2, c + 3 * ldc + vlen);
    }
    if (mr > 4) {
        c4x_1 = svld1_f64(npred_1, c + 4 * ldc);
        if (nr > vlen) c4x_2 = svld1_f64(npred_2, c + 4 * ldc + vlen);
    }
    if (mr > 5) {
        c5x_1 = svld1_f64(npred_1, c + 5 * ldc);
        if (nr > vlen) c5x_2 = svld1_f64(npred_2, c + 5 * ldc + vlen);
    }
    if (mr > 6) {
        c6x_1 = svld1_f64(npred_1, c + 6 * ldc);
        if (nr > vlen) c6x_2 = svld1_f64(npred_2, c + 6 * ldc + vlen);
    }
    if (mr > 7) {
        c7x_1 = svld1_f64(npred_1, c + 7 * ldc);
        if (nr > vlen) c7x_2 = svld1_f64(npred_2, c + 7 * ldc + vlen);
    }

    // Main computation loop
    for (int kk = 0; kk < kc; ++kk) {
        // Load B row vectors
        bx_1 = svld1_f64(npred_1, b + kk * ldc);
        if (nr > vlen) {
            bx_2 = svld1_f64(npred_2, b + kk * ldc + vlen);
        }

        // Row 0
        if (mr > 0) {
            aval = a[kk + ldc * 0];
            a0x = svdup_f64(aval);
            c0x_1 = svmla_f64_m(npred_1, c0x_1, bx_1, a0x);
            if (nr > vlen) c0x_2 = svmla_f64_m(npred_2, c0x_2, bx_2, a0x);
        }
        
        // Row 1
        if (mr > 1) {
            aval = a[kk + ldc * 1];
            a1x = svdup_f64(aval);
            c1x_1 = svmla_f64_m(npred_1, c1x_1, bx_1, a1x);
            if (nr > vlen) c1x_2 = svmla_f64_m(npred_2, c1x_2, bx_2, a1x);
        }
        
        // Row 2
        if (mr > 2) {
            aval = a[kk + ldc * 2];
            a2x = svdup_f64(aval);
            c2x_1 = svmla_f64_m(npred_1, c2x_1, bx_1, a2x);
            if (nr > vlen) c2x_2 = svmla_f64_m(npred_2, c2x_2, bx_2, a2x);
        }
        
        // Row 3
        if (mr > 3) {
            aval = a[kk + ldc * 3];
            a3x = svdup_f64(aval);
            c3x_1 = svmla_f64_m(npred_1, c3x_1, bx_1, a3x);
            if (nr > vlen) c3x_2 = svmla_f64_m(npred_2, c3x_2, bx_2, a3x);
        }
        
        // Row 4
        if (mr > 4) {
            aval = a[kk + ldc * 4];
            a4x = svdup_f64(aval);
            c4x_1 = svmla_f64_m(npred_1, c4x_1, bx_1, a4x);
            if (nr > vlen) c4x_2 = svmla_f64_m(npred_2, c4x_2, bx_2, a4x);
        }
        
        // Row 5
        if (mr > 5) {
            aval = a[kk + ldc * 5];
            a5x = svdup_f64(aval);
            c5x_1 = svmla_f64_m(npred_1, c5x_1, bx_1, a5x);
            if (nr > vlen) c5x_2 = svmla_f64_m(npred_2, c5x_2, bx_2, a5x);
        }
        
        // Row 6
        if (mr > 6) {
            aval = a[kk + ldc * 6];
            a6x = svdup_f64(aval);
            c6x_1 = svmla_f64_m(npred_1, c6x_1, bx_1, a6x);
            if (nr > vlen) c6x_2 = svmla_f64_m(npred_2, c6x_2, bx_2, a6x);
        }
        
        // Row 7
        if (mr > 7) {
            aval = a[kk + ldc * 7];
            a7x = svdup_f64(aval);
            c7x_1 = svmla_f64_m(npred_1, c7x_1, bx_1, a7x);
            if (nr > vlen) c7x_2 = svmla_f64_m(npred_2, c7x_2, bx_2, a7x);
        }
    }

    // Store results back to C
    if (mr > 0) {
        svst1_f64(npred_1, c + 0 * ldc, c0x_1);
        if (nr > vlen) svst1_f64(npred_2, c + 0 * ldc + vlen, c0x_2);
    }
    if (mr > 1) {
        svst1_f64(npred_1, c + 1 * ldc, c1x_1);
        if (nr > vlen) svst1_f64(npred_2, c + 1 * ldc + vlen, c1x_2);
    }
    if (mr > 2) {
        svst1_f64(npred_1, c + 2 * ldc, c2x_1);
        if (nr > vlen) svst1_f64(npred_2, c + 2 * ldc + vlen, c2x_2);
    }
    if (mr > 3) {
        svst1_f64(npred_1, c + 3 * ldc, c3x_1);
        if (nr > vlen) svst1_f64(npred_2, c + 3 * ldc + vlen, c3x_2);
    }
    if (mr > 4) {
        svst1_f64(npred_1, c + 4 * ldc, c4x_1);
        if (nr > vlen) svst1_f64(npred_2, c + 4 * ldc + vlen, c4x_2);
    }
    if (mr > 5) {
        svst1_f64(npred_1, c + 5 * ldc, c5x_1);
        if (nr > vlen) svst1_f64(npred_2, c + 5 * ldc + vlen, c5x_2);
    }
    if (mr > 6) {
        svst1_f64(npred_1, c + 6 * ldc, c6x_1);
        if (nr > vlen) svst1_f64(npred_2, c + 6 * ldc + vlen, c6x_2);
    }
    if (mr > 7) {
        svst1_f64(npred_1, c + 7 * ldc, c7x_1);
        if (nr > vlen) svst1_f64(npred_2, c + 7 * ldc + vlen, c7x_2);
    }
}

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
            // my_dgemm_ukr_wo_packing (
            my_dgemm_ukr_wo_packing_w_simd (
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
