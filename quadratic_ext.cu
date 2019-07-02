#pragma once

// #include "fixnum/word_fixnum.cu"

template < typename fixnum, typename monty > 
class quad_ext_element {
public:
    typedef fixnum modnum;
    modnum a0;
    modnum a1;

    // __device__ quad_ext_element(fixnum z0, fixnum z1) : a0(z0), a1(z1) { }
    __device__ quad_ext_element(modnum z0, modnum z1) : a0(z0), a1(z1) { }
};

template < typename fixnum, typename monty >
class quad_ext {
public:
    typedef fixnum modnum;
    monty mod;
    modnum alpha; 

    typedef quad_ext_element<fixnum, monty> quad_ext_element;

    __device__ quad_ext(monty modulus, modnum _alpha) : mod(modulus), alpha(_alpha) {}

    __device__ void add(quad_ext_element &z, quad_ext_element x, quad_ext_element y) {
        mod.add(z.a0, x.a0, y.a0);
        mod.add(z.a1, x.a1, y.a1);
    }

    __device__ void mul(quad_ext_element &z, quad_ext_element x, quad_ext_element y) {
        modnum t0, t1, t2;

        // c0 = a0*b0 + 13*a1*b1
        mod.mul(t0, x.a0, y.a0);
        mod.mul(t1, x.a1, y.a1);
        mod.mul(t2, alpha, t1);
        mod.add(z.a0, t0, t2);

        // c1 = a0*b1 + a1*b0
        mod.mul(t0, x.a0, y.a1);
        mod.mul(t1, x.a1, y.a0);
        mod.add(z.a1, t0, t1);
    }
};
