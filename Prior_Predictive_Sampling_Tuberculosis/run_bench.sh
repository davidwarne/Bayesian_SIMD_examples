#!/bin/bash

N=16000
SEED=1337

echo "Tuberculosis transmission prior predicive sampling benchmark"
echo "N = $N"

#run and time vectorised parallel version
echo 'timing parallel+SIMD'
time ./tuberculosis_ABC_pps_vec_par $N $SEED > /dev/null
echo 'done'

#run and time scalar parallel version 
# (still uses OpenMP and MKL/VSL, but SIMD and auto-vectorisation disabled)
echo 'timing parallel+scalar'
time ./tuberculosis_ABC_pps_novec_par $N $SEED > /dev/null
echo 'done'

#run and time scalar sequential version 
# (still uses MKL/VSL, but OpenMP and auto-vectorisation disabled )
echo 'timing sequential+scalar'
time ./tuberculosis_ABC_pps_novec_nopar $N $SEED  > /dev/null
echo 'done'

