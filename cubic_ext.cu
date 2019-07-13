#pragma once

// #include "fixnum/word_fixnum.cu"

template < typename fixnum> 
class cubic_ext_element {
public:
    typedef fixnum modnum;
    modnum a0;
    modnum a1;
    modnum a2;

    __device__ cubic_ext_element() { }
};

template < typename fixnum, typename monty >
class cubic_ext {
public:
    typedef fixnum modnum;
    monty mod;
    modnum alpha; 

    typedef cubic_ext_element<fixnum> ext_element;

    __device__ cubic_ext(fixnum modulus, fixnum _alpha) : mod(modulus), alpha(_alpha) {
        modnum t;
        mod.to_modnum(t, alpha);
        alpha = t;
    }

    __device__ void to_modnum(ext_element &z) {
        modnum t0, t1, t2;
        mod.to_modnum(t0, z.a0);
        mod.to_modnum(t1, z.a1);
        mod.to_modnum(t2, z.a2);

        z.a0 = t0; z.a1 = t1; z.a2 = t2;
    }

    __device__ void from_modnum(ext_element &z) {
        fixnum t0, t1, t2;
        mod.from_modnum(t0, z.a0);
        mod.from_modnum(t1, z.a1);
        mod.from_modnum(t2, z.a2);

        z.a0 = t0; z.a1 = t1; z.a2 = t2;
    }

    __device__ void add(ext_element &z, ext_element &x, ext_element &y) {
        modnum t0, t1, t2;
        mod.add(t0, x.a0, y.a0);
        mod.add(t1, x.a1, y.a1);
        mod.add(t2, x.a2, y.a2);

        z.a0 = t0; z.a1 = t1; z.a2 = t2;
    }

    __device__ void mul(ext_element &z, ext_element x, ext_element y) {
        modnum t0, t1, t2;
        modnum a0b0, a0b1, a0b2;
        modnum a1b0, a1b1, a1b2;
        modnum a2b0, a2b1, a2b2;

        mod.mul(a0b0, x.a0, y.a0);
        mod.mul(a0b1, x.a0, y.a1);
        mod.mul(a0b2, x.a0, y.a2);
        mod.mul(a1b0, x.a1, y.a0);
        mod.mul(a1b1, x.a1, y.a1);
        mod.mul(a1b2, x.a1, y.a2);
        mod.mul(a2b0, x.a2, y.a0);
        mod.mul(a2b1, x.a2, y.a1);
        mod.mul(a2b2, x.a2, y.a2);

        // c0 = a0*b0 + 11*(a1*b2 + a2*b1)
        mod.add(t0, a1b2, a2b1);
        mod.mul(t1, alpha, t0);
        mod.add(z.a0, a0b0, t1);

        // c1 = a0*b1 + a1*b0 + alpha*a2*b2
        mod.mul(t0, alpha, a2b2);
        mod.add(t1, t0, a0b1);
        mod.add(z.a1, t1, a1b0);

        // c2 = a0*b2 + a1*b1 + a2*b0
        mod.add(t0, a0b2, a1b1);
        mod.add(z.a2, t0, a2b0);
    }
};
