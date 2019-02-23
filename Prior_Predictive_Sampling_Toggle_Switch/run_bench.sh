#!/bin/bash

N=4000
C=8000
T=600
SEED=1337

echo "Toggle switch prior predicive sampling benchmark"
echo "N = $N C = $C T = $T"

#run and time vectorised parallel version
echo 'timing parallel+SIMD'
time ./toggle_switch_ABC_pps_vec_par $N $SEED $C $T > /dev/null
echo 'done'

#run and time scalar parallel version 
# (still uses OpenMP and MKL/VSL, but SIMD and auto-vectorisation disabled)
echo 'timing parallel+scalar'
time ./toggle_switch_ABC_pps_novec_par $N $SEED $C $T > /dev/null
echo 'done'

#run and time scalar sequential version 
# (still uses MKL/VSL, but OpenMP and auto-vectorisation disabled )
echo 'timing sequential+scalar'
time ./toggle_switch_ABC_pps_novec_nopar $N $SEED $C $T > /dev/null
echo 'done'


