#!/bin/bash

DAT=MonthlyReturns2018.csv
SEED=1337

echo "Prior weak informativity test benchmark"

#run and time vectorised parallel version
echo 'timing parallel+SIMD'
time ./weak_info_test_vec_par $N $K $SEED > /dev/null
echo 'done'

#run and time scalar parallel version 
# (still uses OpenMP and MKL/VSL, but SIMD and auto-vectorisation disabled)
echo 'timing parallel+scalar'
time ./weak_info_test_novec_par $N $K $SEED > /dev/null
echo 'done'

#run and time scalar sequential version 
# (still uses MKL/VSL, but OpenMP and auto-vectorisation disabled )
echo 'timing sequential+scalar'
time ./weak_info_test_novec_nopar $N $K $SEED  > /dev/null
echo 'done'
