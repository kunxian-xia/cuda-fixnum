
#include <cstdio>
// #include <cstring>
// #include <cassert>
#include <vector>

#include "fixnum/warp_fixnum.cu"
#include "array/fixnum_array.h"
#include "functions/modexp.cu"
#include "functions/multi_modexp.cu"
#include "modnum/modnum_monty_redc.cu"
#include "modnum/modnum_monty_cios.cu"
#include "quadratic_ext.cu"
#include "cubic_ext.cu"

const unsigned int bytes_per_elem = 128;
const unsigned int io_bytes_per_elem = 96;

using namespace std;
using namespace cuFIXNUM;

// 1. read_mnt4_fq2
// 2. allocate cuda memory for input vectors
// 3. dispatch the work to input
// 4. get output back 
// 5. write_mnt4_fq2

// template< typename fixnum >
// struct quad_mul {
//     typedef modnum_monty_redc<fixnum> modnum_redc;
//     typedef quad_ext_element<fixnum> quad;
//     typedef quad_ext<fixnum, modnum_redc> quad_ext;
//     __device__ void operator()(fixnum alpha, fixnum modulus, quad a, quad b, quad &r) {
//         quad_ext ext(modulus, alpha);

//         ext.to_modnum(a);
//         ext.to_modnum(b);
//         ext.mul(r, a, b);

//         ext.from_modnum(r);
//     }
// };

template< typename fixnum >
struct cubic_mul {
    typedef modnum_monty_redc<fixnum> modnum_redc;
    typedef cubic_ext_element<fixnum> cubic;
    typedef cubic_ext<fixnum, modnum_redc> cubic_ext;
    __device__ void operator()(fixnum alpha, fixnum modulus, fixnum rr, cubic a, cubic b, cubic &r) {
        cubic_ext ext(modulus, alpha);

        // ext.to_modnum(a);
        // ext.to_modnum(b);
        cubic t;
        ext.mul(t, a, b);

        modnum_redc mod(modulus);
        fixnum t0, t1, t2;
        mod.mul(r.a0, t.a0, rr);
        mod.mul(r.a1, t.a1, rr);
        mod.mul(r.a2, t.a2, rr);
        // r.a0 = t0; r.a1 = t1; r.a2 = t2;
        // cubic tmp;
        // tmp.a0 = rr; tmp.a1 = rr; tmp.a2 = rr;
        // ext.mul(r, t, tmp);
        // ext.from_modnum(r);
    }
};

template <typename fixnum, template<typename> class Func > 
__global__ void dispatch(int nelts, fixnum *alpha, fixnum *modulus, fixnum *rr, cubic_ext_element<fixnum> *a, 
    cubic_ext_element<fixnum> *b, cubic_ext_element<fixnum> *c)
{
    int blk_tid_offset = blockDim.x * blockIdx.x;
    int tid_in_blk = threadIdx.x;
    int idx = (blk_tid_offset + tid_in_blk) / fixnum::SLOT_WIDTH;

    if (idx < nelts) {
        Func<fixnum> fn;
        // TODO: This offset calculation is entwined with fixnum layout and so
        // belongs somewhere else.
        int off = idx * fixnum::layout::WIDTH + fixnum::layout::laneIdx();
        
        // TODO: This is hiding a sin against memory aliasing / management /
        // type-safety.
        fn(alpha[off], modulus[off], rr[off], a[off], b[off], c[off]);
    }
}

template <typename fixnum>
void mnt_fq3_to_cubic_element(uint8_t *fq3, cubic_ext_element<fixnum> *ele) {
  uint8_t* data = reinterpret_cast<uint8_t*>(ele);
  int bytes = fixnum::BYTES;
  int word_size = sizeof(fixnum);

  for (int i = 0; i < bytes/word_size; i++) {
      for (int j = 0; j < word_size; j++) {
          data[3*i*word_size+j] = fq3[i*word_size+j];
          data[(3*i+1)*word_size+j] = fq3[bytes_per_elem+i*word_size+j];
          data[(3*i+2)*word_size+j] = fq3[2*bytes_per_elem+i*word_size+j];
      }
  }
}

template <typename fixnum >
uint8_t* cubic_element_to_mnt_fq3(cubic_ext_element<fixnum> *ele) {
    uint8_t *r = new uint8_t[fixnum::BYTES*3];
    uint8_t *data = reinterpret_cast<uint8_t*>(ele);
    int word_size = sizeof(fixnum);

    for (unsigned int i = 0; i < bytes_per_elem/word_size; i++) {
        for (int j = 0; j < word_size; j++) {
            r[i*word_size+j] = data[3*i*word_size+j];
            r[bytes_per_elem+i*word_size+j] = data[(3*i+1)*word_size+j];
            r[2*bytes_per_elem+i*word_size+j] = data[(3*i+2)*word_size+j];
        }
    }
    return r;
}

template <int fn_bytes, typename word_fixnum, template <typename> class Func>
std::vector<uint8_t*> compute_product(std::vector<uint8_t*> a, std::vector<uint8_t*> b, uint8_t *modulus, uint8_t *alpha, uint8_t *rr) {
    typedef warp_fixnum<fn_bytes, word_fixnum> fixnum;
    typedef cubic_ext_element<fixnum> cubic;
    typedef cubic_ext<fixnum, modnum_monty_redc<fixnum>> ext;

    int n = a.size();
    cubic *inputs_a, *inputs_b, *output;
    cuda_malloc_managed((void**)&inputs_a, fn_bytes*3*n);
    cuda_malloc_managed((void**)&inputs_b, fn_bytes*3*n);
    cuda_malloc_managed((void**)&output, fn_bytes*3*n);
  
    for (int i = 0; i < n; i++) {
        mnt_fq3_to_cubic_element<fixnum>(a[i], &inputs_a[i*fn_bytes*3/sizeof(cubic)]);
        mnt_fq3_to_cubic_element<fixnum>(b[i], &inputs_b[i*fn_bytes*3/sizeof(cubic)]);
    }

    fixnum *inputs_mod, *inputs_alpha, *inputs_rr;
    cuda_malloc_managed(&inputs_mod, fn_bytes*n);
    cuda_malloc_managed(&inputs_alpha, fn_bytes*n);
    cuda_malloc_managed(&inputs_rr, fn_bytes*n);
    for (int i = 0; i < n; i++) {
        fixnum::from_bytes(reinterpret_cast<uint8_t*>(
            &inputs_mod[i*fn_bytes/sizeof(fixnum)]), modulus, fixnum::BYTES);
        fixnum::from_bytes(reinterpret_cast<uint8_t*>(
            &inputs_alpha[i*fn_bytes/sizeof(fixnum)]), alpha, fixnum::BYTES);
        fixnum::from_bytes(reinterpret_cast<uint8_t*>(
            &inputs_rr[i*fn_bytes/sizeof(fixnum)]), rr, fixnum::BYTES);
    }

    constexpr int BLOCK_SIZE = 192;
    constexpr int fixnums_per_block = BLOCK_SIZE / fixnum::SLOT_WIDTH;
    int nblocks = ceilquo(n, fixnums_per_block);
    cudaStream_t stream;

    cuda_check(cudaStreamCreate(&stream), "create stream");
    cuda_check(cudaStreamSynchronize(stream), "stream sync");

    dispatch<fixnum, Func><<<nblocks, BLOCK_SIZE, 0, stream>>>(n, inputs_alpha, inputs_mod, inputs_rr, inputs_a, inputs_b, output);
    cuda_check(cudaPeekAtLastError(), "kernel invocation/run");
    cuda_check(cudaStreamSynchronize(stream), "stream sync");
    cuda_check(cudaStreamDestroy(stream), "stream destroy");

    cuda_device_synchronize();

    std::vector<uint8_t *> ret;
    for (int i = 0; i < n; i++) {
        ret.emplace_back( cubic_element_to_mnt_fq3<fixnum>(&output[i*fn_bytes*3/sizeof(cubic)]) );
    }

    cuda_free(inputs_a);
    cuda_free(inputs_b);
    cuda_free(output);
    cuda_free(inputs_mod); cuda_free(inputs_alpha); cuda_free(inputs_rr);
    return ret;
}

uint8_t* read_mnt_fq3(FILE *inputs) {
  uint8_t* buf = (uint8_t*)calloc(bytes_per_elem*3, sizeof(uint8_t));
  fread((void*)(buf+bytes_per_elem-io_bytes_per_elem), io_bytes_per_elem*sizeof(uint8_t), 1, inputs);
  fread((void*)(buf+2*bytes_per_elem-io_bytes_per_elem), io_bytes_per_elem*sizeof(uint8_t), 1, inputs);
  fread((void*)(buf+3*bytes_per_elem-io_bytes_per_elem), io_bytes_per_elem*sizeof(uint8_t), 1, inputs);
  // fread((void*) buf, io_bytes_per_elem*sizeof(uint8_t), 1, inputs);
  // fread((void*) (buf+bytes_per_elem), io_bytes_per_elem*sizeof(uint8_t), 1, inputs);
  // fread((void*) (buf+2*bytes_per_elem), io_bytes_per_elem*sizeof(uint8_t), 1, inputs);
  return buf;
}

void write_mnt_fq3(uint8_t* fq3, FILE* outputs) {
  fwrite((void *) fq3, io_bytes_per_elem * sizeof(uint8_t), 1, outputs);
  fwrite((void *) (fq3+bytes_per_elem), io_bytes_per_elem * sizeof(uint8_t), 1, outputs);
  fwrite((void *) (fq3+2*bytes_per_elem), io_bytes_per_elem * sizeof(uint8_t), 1, outputs);
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
      printf("usage: ./main compute inputs output\n");
      exit(1);
  }

  setbuf(stdout, NULL);

  // mnt4_q
  uint8_t mnt4_modulus[bytes_per_elem] = {1,128,94,36,222,99,144,94,159,17,221,44,82,84,157,227,240,37,196,154,113,16,136,99,164,84,114,118,233,204,90,104,56,126,83,203,165,13,15,184,157,5,24,242,118,231,23,177,157,247,90,161,217,36,209,153,141,237,160,232,37,185,253,7,115,216,151,108,249,232,183,94,237,175,143,91,80,151,249,183,173,205,226,238,34,144,34,16,17,196,146,45,198,196,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  uint8_t rr[bytes_per_elem] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  // mnt6_q
  uint8_t mnt6_modulus[bytes_per_elem] = {1,0,0,64,226,118,7,217,79,58,161,15,23,153,160,78,151,87,0,63,188,129,195,214,164,58,153,52,118,249,223,185,54,38,33,41,148,202,235,62,155,169,89,200,40,92,108,178,157,247,90,161,217,36,209,153,141,237,160,232,37,185,253,7,115,216,151,108,249,232,183,94,237,175,143,91,80,151,249,183,173,205,226,238,34,144,34,16,17,196,146,45,198,196,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  printf("%d\n", rr[96]);
  uint8_t alpha[bytes_per_elem] = {11};

  auto inputs = fopen(argv[2], "r");
  auto outputs = fopen(argv[3], "w");

  size_t n;

   while (true) {
    size_t elts_read = fread((void *) &n, sizeof(size_t), 1, inputs);
    if (elts_read == 0) { break; }
    printf("%lu\n", n);
    std::vector<uint8_t*> x0;
    for (size_t i = 0; i < n; ++i) {
      x0.emplace_back(read_mnt_fq3(inputs));
    }

    std::vector<uint8_t*> x1;
    for (size_t i = 0; i < n; ++i) {
      x1.emplace_back(read_mnt_fq3(inputs));
    }

    std::vector<uint8_t*> ret = compute_product<bytes_per_elem, u64_fixnum, cubic_mul>(x0, x1, mnt6_modulus, alpha, rr);
    for (size_t i = 0; i < n; ++i) {
        write_mnt_fq3(ret[i], outputs);
    }
    for (size_t i = 0; i < n;++i) {
        free(x0[i]);
        free(x1[i]);
        free(ret[i]);
    }
   }
   fclose(inputs);
   fclose(outputs);
}
