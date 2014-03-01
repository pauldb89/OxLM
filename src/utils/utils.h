// Copyright 2013 Karl Moritz Hermann
// File: utils.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 07-11-2013
// Last Update: Thu 07 Nov 2013 02:38:14 PM GMT

#ifndef UTILS_UTILS_H
#define UTILS_UTILS_H

template <typename Type>
struct modvars
{
    Type T;
    Type S;
    Type C;
    Type R;
    Type Q;
    Type F;
    Type FB;
    Type B;

    modvars() {
        init();
    }
    void init() {};
};

template<> inline void modvars<bool>::init() {
    T = true;
    S = true;
    C = true;
    R = true;
    Q = true;
    F = true;
    FB = true;
    B = true;
}

typedef modvars<double> Lambdas;
typedef modvars<bool>  Bools;
typedef modvars<int>   Counts;

#endif  // UTILS_UTILS_H
