#pragma once

// #include "fixnum/word_fixnum.cu"

template < typename fixnum, typename monty > 
class quad_ext_element {
public:
    typedef fixnum modnum;
    modnum a0;
    modnum a1;

    __device__ quad_ext_element() { }

    __device__ static void to_modnum(monty mod, quad_ext_element &z) {
        modnum t0, t1;
        mod.to_modnum(t0, z.a0);
        mod.to_modnum(t1, z.a1);

        z.a0 = t0; z.a1 = t1;
    }

    __device__ static void from_modnum(monty mod, quad_ext_element &z) {
        fixnum t0, t1;
        mod.from_modnum(t0, z.a0);
        mod.from_modnum(t1, z.a1);
        z.a0 = t0; z.a1 = t1;
    }
    /*
    __device__ quad_ext_element(monty mod, fixnum z0, fixnum z1) {
        mod.to_modnum(a0, z0);
        mod.to_modnum(a1, z1);
    }
    */
    // __device__ quad_ext_element(modnum z0, modnum z1) : a0(z0), a1(z1) {  printf("cons"); }
};

template < typename fixnum, typename monty >
class quad_ext {
public:
    typedef fixnum modnum;
    monty mod;
    modnum alpha; 

    typedef quad_ext_element<fixnum, monty> quad_ext_element;

    __device__ quad_ext(fixnum modulus, modnum _alpha) : mod(modulus), alpha(_alpha) {}

    __device__ void add(quad_ext_element &z, quad_ext_element &x, quad_ext_element &y) {
        modnum t0, t1;
        mod.add(t0, x.a0, y.a0);
        mod.add(t1, x.a1, y.a1);
        z.a0 = t0; z.a1 = t1;
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
