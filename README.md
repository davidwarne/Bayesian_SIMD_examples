# Bayesian computations using SIMD operations

Supplementary Material to accompany the paper, 

DJ Warne, SA Sisson, C Drovandi (2019) Acceleration of expensive computations in
Bayesian statistics using vector operations. ArXiv pre-print TBA

## Summary

Contains example C code using OpenMP and Intel MKL/VSL to parallelise and vectorise Bayesian computations.

Example applications are:
1. Sampling of prior predicitve distributions for approximate Bayesian computation;
2. Computation of Bayesian p-values for testing prior weak informativity. 

## Developers
David J. Warne (david.warne@qut.edu.au), School of Mathematical Sciences, Queensland University of Technology.

Christopher C. Drovandi (c.drovandi@qut.edu.au), School of Mathematical Sciences, Queensland University of Technology.

Scott A. Sisson (scott.sisson@unsw.edu.au), School of Mathematics and Statistics, University of New South Wales.

## License

Bayesian computations using SIMD operations
Copyright (C) 2019  David J. Warne, Christopher C. Drovandi, Scott A. Sisson

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.



## Requirements

1. Intel Math Kernel Library (MKL) version >= 18 Update 1;
2. Intel C Compiler  version >= 18.0.1 (Also compatible with GNU C Compiler version >= 7.0.0, but not tested);
3. CPU supporting  Advance Vector Extensions instruction sets (either AVX, AVX2 or AVX512).

## Compile and run Benchmarks

The build scripts and benchmarks assume a Linux Operating system using the Bourne-again shell 

1. `cd path/to/example/`
2. `make`
3. `./run_bench.sh`

Please note, the benchmark results presented in the Paper were performed using 4 cores of the Intel Xeon E5-2680v3 and 4 cores of the Intel Xeon Gold 6140. Speed-up factors may vary across CPU models, core counts and vector widths. 
