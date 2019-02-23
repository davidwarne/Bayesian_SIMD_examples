
all:
	make tuberculosis_ABC_pps_vec_par
	make tuberculosis_ABC_pps_novec_nopar
	make tuberculosis_ABC_pps_novec_par

tuberculosis_ABC_pps_vec_par: tuberculosis_ABC_pps_vec_par.c Makefile
	icc -mkl -qopenmp -O2 -xhost tuberculosis_ABC_pps_vec_par.c -o tuberculosis_ABC_pps_vec_par

tuberculosis_ABC_pps_novec_par: tuberculosis_ABC_pps_vec_par.c Makefile
	icc -mkl -qopenmp -O2 -xhost -qno-openmp-simd -no-vec tuberculosis_ABC_pps_vec_par.c -o tuberculosis_ABC_pps_novec_par

tuberculosis_ABC_pps_novec_nopar: tuberculosis_ABC_pps_vec_par.c Makefile
	icc -mkl -O2 -xhost -no-vec tuberculosis_ABC_pps_vec_par.c -o tuberculosis_ABC_pps_novec_nopar

clean:
	rm -f tuberculosis_ABC_pps_vec_par tuberculosis_ABC_pps_novec_nopar tuberculosis_ABC_pps_novec_par
