
all:
	make SMC_RW_LikeAnneal_BEGE_vec_par
	make SMC_RW_LikeAnneal_BEGE_novec_par
	make SMC_RW_LikeAnneal_BEGE_novec_nopar
	make SMC_RW_LikeAnneal_BEGE_vec_nopar

SMC_RW_LikeAnneal_BEGE_vec_par: SMC_RW_LikeAnneal_BEGE_vec_par.c Makefile
	icc -mkl -qopenmp -O2 -xhost SMC_RW_LikeAnneal_BEGE_vec_par.c -o SMC_RW_LikeAnneal_BEGE_vec_par 

SMC_RW_LikeAnneal_BEGE_novec_par: SMC_RW_LikeAnneal_BEGE_novec_par.c Makefile
	icc -mkl -qopenmp -O2 -xhost -no-vec SMC_RW_LikeAnneal_BEGE_novec_par.c -o SMC_RW_LikeAnneal_BEGE_novec_par 

SMC_RW_LikeAnneal_BEGE_vec_nopar: SMC_RW_LikeAnneal_BEGE_vec_nopar.c Makefile
	icc -mkl -qopenmp -O2 -xhost SMC_RW_LikeAnneal_BEGE_vec_nopar.c -o SMC_RW_LikeAnneal_BEGE_vec_nopar

SMC_RW_LikeAnneal_BEGE_novec_nopar: SMC_RW_LikeAnneal_BEGE_novec_nopar.c Makefile
	icc -mkl -O2 -xhost -no-vec SMC_RW_LikeAnneal_BEGE_novec_nopar.c -o SMC_RW_LikeAnneal_BEGE_novec_nopar

clean:
	rm -f SMC_RW_LikeAnneal_BEGE_vec_par SMC_RW_LikeAnneal_BEGE_novec_par SMC_RW_LikeAnneal_BEGE_novec_nopar SMC_RW_LikeAnneal_BEGE_vec_nopar 
