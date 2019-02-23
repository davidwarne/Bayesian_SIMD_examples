/* Bayesian computations using SIMD operations
 * Copyright (C) 2019  David J. Warne, Christopher C. Drovandi
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
 /**
 * @file toggle_switch_ABC_pps.c
 *
 * @brief Demonstration of vectorisation for approximate Bayesian computation
 *
 * @details Efficient prior predictive sampling for the genetic toggle switch 
 * model. Cells are evoluted in groups of equal to the vector processing unit 
 * capacity for double precision floating point. Standard Gaussian random 
 * variates are generated for the entire evolution of the group to efficiently 
 * utilise the MKL VSL generator.
 *
 * @author David J. Warne (david.warne@qut.edu.au)
 *         School of Mathematical Sciences
 *         Queensland University of Technology
 *
 * @author Chris C. Drovandi (c.drovandi@qut.edu.au)
 *         School of Mathematical Sciences
 *         Queensland University of Technology
 *
 * @date 12 Oct 2018
 *
 */

/* standard C headers*/
#include <stdio.h>
#include <math.h>
#include <string.h>

/* Intel headers */
#include <mkl.h>
#include <mkl_vsl.h>

/* OpenMP header */
#include <omp.h>

/* length of vector processing units and ideal memory alignement*/
#if defined(__AVX512BW__)
    #define VECLEN 8
    #define ALIGN 64
#elif defined(__AVX2__)
    #define VECLEN 4
    #define ALIGN 64
#elif defined(__AVX__)
    #define VECLEN 4
    #define ALIGN 32
#elif defined(__SSE4_2__)
    #define VECLEN 2
    #define ALIGN 16
#endif

/** 
 * @brief vectorised toggle switch stochastic simulation
 *
 * @details processes cells in blocks of VECLEN to exploit vectorisation and 
 * cache. Utilises MKL VSL routine to generate all Gaussian random variates for
 * the entire evolution of the block. 
 *
 * @param stream Pointer to RNG state
 * @param mu basal observation noise level
 * @param sigma,gamma increase the observation noise at low expression levels 
 * @param alpha, beta logistic-like repression of gene expression
 * @param T simulation time and observation time
 * @param C number of cells
 * @param zeta memory location to store gaussian random variaates
 * @param y vector of C observations of u-gene expression levels
 *
 * @note assumes argument data arrays are aligned on ALIGN-byte boundary. 
 * The restrict keyword is used to ensure the compile knows there is no pointer 
 * aliasing.
 *
 * @warning For optimal performance, the number of cells, C, should be a 
 * multiple of VECLEN
 */
void 
simulate_toggle_switch(VSLStreamStatePtr stream, 
                       double mu, double sigma, double gamma, 
                       double * restrict alpha, double * restrict beta, 
                       int T, int C,double * restrict zeta, double * restrict y)
{

    /* process cells in blocks of VECLEN*/
    for (int c=0;c<C;c+=VECLEN)
    {
        /* Generate all the random variates for these realisation*/
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2,stream,2*VECLEN*T,zeta,0.0,1.0);
        
        /* compute state trajectories for this block in SIMD*/
        #pragma omp simd aligned(zeta:ALIGN, y:ALIGN) 
        for (int c2=0;c2<VECLEN;c2++)
        {
            double u_t, v_t, alpha_u, alpha_v, beta_u, beta_v;
            double _gamma, _sigma, _mu;

            /* copy parameters to ensure in cache/registers*/
            _gamma = gamma;
            _sigma = sigma;
            _mu = mu;
            alpha_u = alpha[0];
            alpha_v = alpha[1];
            beta_u = beta[0];
            beta_v = beta[1];
            
            /*set initial conditions*/
            u_t = 10;
            v_t = 10;
            
            /* evolve u/v pairs for this cell  */
            for (int j=1;j<T;j++)
            {
                double p_u = pow(v_t,beta_u);
                double p_v = pow(u_t,beta_v);
                u_t *= 0.97;
                u_t += alpha_u/(1.0 + p_u) - 1.0;
                v_t *= 0.97;
                v_t += alpha_v/(1.0 + p_v) - 1.0;
                double zeta_u = zeta[(j-1)*VECLEN*2 + c2]; 
                double zeta_v = zeta[(j-1)*VECLEN*2 + 4 + c2]; 
                u_t += 0.5*zeta_u;
                u_t  = (u_t >= 1.0) ? u_t : 1.0; 
                v_t += 0.5*zeta_v;
                v_t  = (v_t >= 1.0) ? v_t : 1.0; 
            }
            
            /* make noisy observation */ 
            y[c+c2] = u_t + _mu + _sigma*_mu*zeta[(T-1)*VECLEN*2 + c2]/pow(u_t,_gamma);
            y[c+c2] = (y[c+c2] >= 1.0) ? y[c+c2]: 1.0;
        }
    }
}

/**
 * @brief program entry point
 * @details Generates prior predictive samples for the genetic toggle switch 
 * model. Distributes simulations across availablle cores and utilises fine grain
 * SIMD operations for groups of cells within each simulation.
 *
 * @param argc number of command line arguments
 * @param argv vector of argument strings
 */
int 
main(int argc,char **argv)
{ 
    int T, C, K;
    double *theta, *obs_vals;
    int seed, sims;
    
    /* default number of simulation timesteps*/
    T = 300;

    /* default number of cells*/
    C = 2000;

    /* number of parameters*/
    K = 7;

    /* get command line arguments*/
    if (argc < 2)
    {
        fprintf(stderr,"Usage : [%s] sims seed [C] [T]\n",argv[0]);
        exit(1);
    }
    sims = (int)atoi(argv[1]);
    seed = (int)atoi(argv[2]);

    /* check if C was specified*/
    if (argc > 3)
    {
        C = (int)atoi(argv[3]);
        if (C%VECLEN != 0)
        {
            fprintf(stderr,"Warning: For optimal performance C must be a multiple of %d\n",VECLEN);
        }
    }
    /* check if T was specified*/
    if (argc > 4)
    {
        T = (int)atoi(argv[4]);
    }
   
    /*allocate aligned memory for noisy observations*/
    obs_vals = (double *)_mm_malloc(C*sims*sizeof(double),ALIGN);
    
    /* allocate aligned memory prior samples that generated the data*/
    theta = (double *)_mm_malloc(K*sims*sizeof(double),ALIGN); 
    
    /* compute simulations in parallel*/
    #pragma omp parallel shared(seed,sims,C,obs_vals,theta)
    {
        VSLStreamStatePtr stream;
        int thread_id, num_threads, sims_per_thread; 
        double *zeta;
        
        /* get thread information and assign workload*/
        thread_id = omp_get_thread_num();
        num_threads = omp_get_num_threads(); 
        sims_per_thread = sims/num_threads; 
        
        /* initialise RNG stream for this thread*/
        vslNewStream(&stream,VSL_BRNG_MT2203+thread_id,seed);
        
        /*allocate aligned memory for Gaussian random variates*/
        zeta = (double *)_mm_malloc(2*VECLEN*T*sizeof(double),ALIGN);

        /* compute simulations in this threads workload*/
        for (int k=thread_id*sims_per_thread;k<(thread_id+1)*sims_per_thread;k++)
        {
            /* model parameters */
            double mu,sigma,gamma; /* measurement error parameters */
            double alpha[2], beta[2]; /* parameters for gene expression*/
            
            /*sample prior */
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,1,&mu,250.0,400.0);
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,1,&sigma,0.05,0.5);
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,1,&gamma,0.05,0.35);
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,2,alpha,0.0,50.0);
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,2,beta,0.0,7.0);
            
            /*run simulation and store observations*/
            simulate_toggle_switch(stream,mu,sigma,gamma,alpha,beta,T,C,zeta,
                                   obs_vals +k*C);
            /*store prior sample*/
            theta[k*7] = mu;
            theta[k*7 + 1] = sigma;
            theta[k*7 + 2] = gamma;
            theta[k*7 + 3] = alpha[0];
            theta[k*7 + 4] = beta[0];
            theta[k*7 + 5] = alpha[1];
            theta[k*7 + 6] = beta[1];
        }
        /*clean up memory*/
        vslDeleteStream(&stream);
        _mm_free(zeta);
    }
   
    /*output prior predicitive samples for postprocessing for ABC*/
    fprintf(stdout,"\"Sample\",\"mu\",\"sigma\",\"gamma\",\"alpha_u\",\"beta_u\",\"alpha_v\",\"beta_v\"");
    for (int j=0;j<C;j++)
    {
        fprintf(stdout,",\"y_%d\"",j);
    }
    fprintf(stdout,"\n");
    for (int i=0;i<sims;i++)
    {
        fprintf(stdout,"%d",i);
        for (int j=0;j<K;j++)
        {
            fprintf(stdout,",%g",theta[i*K + j]);
        }
        for (int j=0;j<C;j++)
        {
            fprintf(stdout,",%f",obs_vals[i*C + j]);
        }
        fprintf(stdout,"\n");
    }
    exit(0);
}
