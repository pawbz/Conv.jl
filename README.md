### Operations

This package aims for a fast implementation of the following operations:
$$d_{i}=QF^{-1}[(F P g_i)(F P s_i)], \text{where}$$

* `s`: source signature in time
* `g`: Green's functions in time
* `d`: data in time
* `F`: FFT operator
* `P`: zero-padding matrix
* `Q`: truncation matrix

### Usage
```julia
nt=10; nr=10
g=randn(nt,nr); s=ones(nt, nr)
```
Create parameter variable and allocate memory.
```
pa=Conv.Param(dsize=[nt,nr], ssize=[nt,nr], gsize=[nt,nr], g=g, s=s) # memory allocation
```
And finally, perform a convolution.
```
Conv.mod!(pa, :d) #updates the data matrix in pa.d
```

[![Build Status](https://travis-ci.org/pawbz/Conv.jl.svg?branch=master)](https://travis-ci.org/pawbz/Conv.jl)
[![codecov](https://codecov.io/gh/pawbz/Conv.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/pawbz/Conv.jl)

