
all:
	make toggle_switch_ABC_pps_vec_par
	make toggle_switch_ABC_pps_novec_nopar
	make toggle_switch_ABC_pps_novec_par

toggle_switch_ABC_pps_vec_par: toggle_switch_ABC_pps_vec_par.c Makefile
	icc -mkl -qopenmp -O2 -xhost toggle_switch_ABC_pps_vec_par.c -o toggle_switch_ABC_pps_vec_par

toggle_switch_ABC_pps_novec_par: toggle_switch_ABC_pps_vec_par.c Makefile
	icc -mkl -qopenmp -O2 -xhost -qno-openmp-simd -no-vec toggle_switch_ABC_pps_vec_par.c -o toggle_switch_ABC_pps_novec_par

toggle_switch_ABC_pps_novec_nopar: toggle_switch_ABC_pps_vec_par.c Makefile
	icc -mkl -O2 -xhost -no-vec toggle_switch_ABC_pps_vec_par.c -o toggle_switch_ABC_pps_novec_nopar

clean:
	rm -f toggle_switch_ABC_pps_vec_par toggle_switch_ABC_pps_novec_nopar toggle_switch_ABC_pps_novec_par
