
all:
	make toggle_switch_ABC_pps_vec_par
	make toggle_switch_ABC_pps_novec_nopar
	make toggle_switch_ABC_pps_novec_par

toggle_switch_ABC_pps_vec_par: toggle_switch_ABC_pps_vec_par.c Makefile
	icc -mkl=sequential -qopenmp -O2 -xHost -fcf-protection=none toggle_switch_ABC_pps_vec_par.c -o toggle_switch_ABC_pps_vec_par

toggle_switch_ABC_pps_novec_par: toggle_switch_ABC_pps_vec_par.c Makefile
	icc -mkl=sequential -qopenmp -O2 -xHost -qno-openmp-simd -no-vec toggle_switch_ABC_pps_vec_par.c -o toggle_switch_ABC_pps_novec_par

toggle_switch_ABC_pps_novec_nopar: toggle_switch_ABC_pps_vec_par.c Makefile
	icc -mkl=sequential -qno-openmp -O2 -xHost -no-vec toggle_switch_ABC_pps_novec_nopar.c -o toggle_switch_ABC_pps_novec_nopar

clean:
	rm -f toggle_switch_ABC_pps_vec_par toggle_switch_ABC_pps_novec_nopar toggle_switch_ABC_pps_novec_par
