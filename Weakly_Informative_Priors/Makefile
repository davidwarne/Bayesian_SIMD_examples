
all:
	make weak_info_test_vec_par
	make weak_info_test_novec_nopar
	make weak_info_test_novec_par

weak_info_test_vec_par: weak_info_test_vec_par.c Makefile
	icc -mkl -qopenmp -O2 -xhost weak_info_test_vec_par.c -o weak_info_test_vec_par

weak_info_test_novec_par: weak_info_test_vec_par.c Makefile
	icc -mkl -qopenmp -O2 -xhost -qno-openmp-simd -no-vec weak_info_test_vec_par.c -o weak_info_test_novec_par

weak_info_test_novec_nopar: weak_info_test_vec_par.c Makefile
	icc -mkl -O2 -xhost -no-vec weak_info_test_vec_par.c -o weak_info_test_novec_nopar

clean:
	rm -f weak_info_test_vec_par weak_info_test_novec_nopar weak_info_test_novec_par
