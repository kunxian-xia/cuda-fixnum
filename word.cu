#include "util/cuda_wrap.h"
// #include "fixnum/word_fixnum.cu"
#include "fixnum/warp_fixnum.cu"
#include "functions/modinv.cu"
#include "modnum/modnum_monty_redc.cu"
#include "modnum/modnum_monty_cios.cu"
#include "quadratic_ext.cu"
using namespace cuFIXNUM;

#include <iostream>
using namespace std;

const int bytes_per_fixnum = 128;

typedef warp_fixnum<bytes_per_fixnum, u64_fixnum> fixnum;
typedef modnum_monty_cios<fixnum> modnum_redc;

typedef quad_ext_element<fixnum, modnum_redc> quad;

__global__ void dispatch(fixnum *alpha, fixnum *modulus, quad *a, quad *b, quad *c)
{
    int blk_tid_offset = blockDim.x * blockIdx.x;
    int tid_in_blk = threadIdx.x;
    int idx = (blk_tid_offset + tid_in_blk) / fixnum::SLOT_WIDTH;

    if (idx < 1) {
        // TODO: Find a way to load each argument into a register before passing
        // it to fn, and then unpack the return values where they belong. This
        // will guarantee that all operations happen on registers, rather than
        // inadvertently operating on memory.

        // Func<fixnum> fn;
        // TODO: This offset calculation is entwined with fixnum layout and so
        // belongs somewhere else.
        int off = idx * fixnum::layout::WIDTH + fixnum::layout::laneIdx();
        modnum_redc mod(modulus[off]);
        
        quad_ext_element<fixnum, modnum_redc>::to_modnum(mod, a[off]);
        quad_ext_element<fixnum, modnum_redc>::to_modnum(mod, b[off]);

        quad_ext<fixnum, modnum_redc> ext(modulus[off], alpha[off]);
        ext.add(c[off], a[off], b[off]);

        quad_ext_element<fixnum, modnum_redc>::from_modnum(mod, c[off]);
        // quad_ext_element<fixnum, modnum_redc>::from_modnum(mod, a[off]);
        // quad_ext_element<fixnum, modnum_redc>::from_modnum(mod, b[off]);
        // TODO: This is hiding a sin against memory aliasing / management /
        // type-safety.
        // fn(args[off]...);
    }
}

int main()
{
    uint8_t mnt4_modulus[bytes_per_fixnum] = {1,128,94,36,222,99,144,94,159,17,221,44,82,84,157,227,240,37,196,154,113,16,136,99,164,84,114,118,233,204,90,104,56,126,83,203,165,13,15,184,157,5,24,242,118,231,23,177,157,247,90,161,217,36,209,153,141,237,160,232,37,185,253,7,115,216,151,108,249,232,183,94,237,175,143,91,80,151,249,183,173,205,226,238,34,144,34,16,17,196,146,45,198,196,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    uint8_t b_alpha[bytes_per_fixnum] = {13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    // b_alpha[0] = (uint8_t) 13;
    // quad_ext<fixnum, modnum_redc> ext(mod, alpha);
    fixnum *alpha, *modulus;
    cuda_malloc_managed((void**)&alpha, bytes_per_fixnum);
    cuda_malloc_managed((void**)&modulus, bytes_per_fixnum);

    quad *a, *b, *c;
    cuda_malloc_managed((void**)&a, 2*bytes_per_fixnum);
    cuda_malloc_managed((void**)&b, 2*bytes_per_fixnum);
    cuda_malloc_managed((void**)&c, 2*bytes_per_fixnum);

    uint8_t ba[bytes_per_fixnum*2];
    uint8_t bb[bytes_per_fixnum*2]; 
    uint8_t bc[bytes_per_fixnum*2];

    memset(ba, 0, sizeof(ba));
    memset(bb, 0, sizeof(bb));
    memset(bc, 0, sizeof(bc));

    // ba[0] = 10, ba[bytes_per_fixnum+0] = 11;
    // bb[0] = 10, bb[bytes_per_fixnum+0] = 11;
    ba[0] = 115, ba[1] = 11; ba[16] = 11; ba[17] = 105;
    ba[8] = 10, ba[8+1] = 11; ba[24] = 5; ba[25] = 6;
    bb[0] = 10, bb[1] = 11; bb[16] = 77; bb[17] = 35;
    bb[8] = 7; bb[9] = 73;

    fixnum::from_bytes(reinterpret_cast<uint8_t*>(alpha), b_alpha, fixnum::BYTES);
    fixnum::from_bytes(reinterpret_cast<uint8_t*>(modulus), mnt4_modulus, fixnum::BYTES);
    fixnum::from_bytes(reinterpret_cast<uint8_t*>(a), ba, fixnum::BYTES*2);
    fixnum::from_bytes(reinterpret_cast<uint8_t*>(b), bb, fixnum::BYTES*2);

    dispatch<<<1, 256>>>(alpha, modulus, a, b, c);

    cuda_device_synchronize();
    uint8_t bd[bytes_per_fixnum*2];
    fixnum::to_bytes(bc,fixnum::BYTES*2, reinterpret_cast<uint8_t*>(c));
    // fixnum::to_bytes(bd, fixnum::BYTES*2, reinterpret_cast<uint8_t*>(modulus));

    for (int i = 0, j = 8, t = 0; i < (fixnum::BYTES)*2; i+=16, j+=16, t++) {
        for (int k = 0; k < 8; k++)
            printf("%d: %d\t%d\t%d \t %d\t%d\t%d\n", t*8+k, ba[i + k], ba[j + k], bb[i + k], bb[j + k], bc[i + k], bc[j + k]);
        // printf("%d\t%d\t%d, \t %d\n", ba[i], bb[i], bc[i], bd[i]);
    }
    cuda_free(a);
    cuda_free(b);
}