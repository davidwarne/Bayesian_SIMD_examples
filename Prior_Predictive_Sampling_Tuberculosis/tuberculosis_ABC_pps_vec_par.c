/* Bayesian computations using SIMD operations
 * Copyright (C) 2019  David J. Warne, Christopher C. Drovandi, Scott A. Sisson
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
 * @file tuderculosis_ABC_pps.c 
 * @brief Demonstration of vectorisation for approximate Bayesian computation
 * @details Uses the Disease transmission and mutation model
 *
 * @author David J. Warne (david.warne@qut.edu.au)
 *         School of Mathematical Sciences
 *         Queensland University of Technology
 *
 * @author Chris C. Drovandi (c.drovandi@qut.edu.au)
 *         School of Mathematical Sciences
 *         Queensland University of Technology
 *
 * @author Scott A. Sisson (scott.sisson@unsw.edu.au)
 *         School of Mathematics and Statistics
 *         University of New South Whales
 *
 * @date 1 Nov 2018
 *
 */

/* standard C headers */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Intel headers */
#include <mkl.h>
#include <mkl_vsl.h>

/* OpenMP header */
#include <omp.h>

/* length of vector processing units and ideal memory alignment*/
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

#define BIRTH 0
#define DEATH 1
#define MUTATION 2

#define MAXN 10000
#define MAXG 1000

/**
 * @brief discrete random variable sampler using lookup method
 *
 * @param stream Pointer to RNG state
 * @param p array of probabilities
 * @param n number of possible outcomes
 * @param ix pointer to store the selected outcome
 */
void 
sample(VSLStreamStatePtr stream,double * restrict p,int n, int *ix)
{
    double sum,u;
    int k;
   
    k = 0;
    sum = p[k];
    /* generate a uniform random variable and see where it lies in the cumulative prob */
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,1,&u, 0.0,1.0);
    while (u > sum && k < n-1)
    {
        k++;
        sum += p[k];
    }
    *ix = k;
    return;
}

/**
 * @brief partly vectorised TB dynamics stochastic simulation
 *
 * @details uses Gillespies direct method with some update operations being
 * vectorised. Example of an algorithm that is diffisult to vectorise without
 * introduction of approximations.
 *
 * @param stream Pointer to RNG state
 * @param alpha birth rate
 * @param delta death rate
 * @param mutation rate
 * @param g_ret pointer to store distinct genotype count
 * @param H_ret pointer to store genetic diveristy
 
 * @note assumes argument data arrays are aligned on ALIGN-byte boundary. 
 * The restrict keyword is used to ensure the compile knows there is no pointer 
 * aliasing.
 */
void 
simulate(VSLStreamStatePtr stream, double alpha, double delta, double mu, 
         double * restrict g_ret, double * restrict H_ret)
{

    /* the current number of genotypes and population*/
    int G=1, N=1;
    
    /*maximum population*/
    int Nstop = MAXN;
    
    /*maximum genotypes*/
    int maxG = MAXN;
    
    /*birth/death/mutation probabilities*/
    __declspec(align(ALIGN)) double probs[3];
    double sumProbs; 
    
    /*genotype probabilities*/
    __declspec(align(ALIGN)) double probs_G[MAXN];
    
    /* number of individuals in each genotype in population */
    __declspec(align(ALIGN)) double X[MAXN]; 
    
    /* number of individuals in each genotype in sample */
    __declspec(align(ALIGN)) double x[MAXN]; 
    double g,H; 
    int event_val = 0, event_val_G = 0;
    
    /* initialise the numbers in each genotype */
    #pragma omp simd
    for (int i=0; i<maxG; i++)
    {
        X[i] = 0;
        x[i] = 0;
        probs_G[i] = 0.0;
    }

    /*  start with 1 indvidual in first genotype */
    X[0] = 1; 

    sumProbs = alpha + delta + mu;
    probs[0] = alpha/sumProbs; 
    probs[1] = delta/sumProbs; 
    probs[2] = mu/sumProbs;  
    
    /* simulate until population has died out or has become too large */
    while (N > 0 && N < Nstop && G < maxG)
    {
        float Nf;
        Nf = (double)N;
        
        /* which event occurs?*/
        sample(stream,probs,3,&event_val);
        
        #pragma omp simd 
        for (int i=0; i<G; i++)
        {
            probs_G[i] = X[i]/Nf;
        }
       
        /* which genotype population does the event happen to?*/
        sample(stream,probs_G,G,&event_val_G);
        
        /* now based on the event sampled update the states */
        switch(event_val)
        {
            case BIRTH: /* birth event */
                /*just increase the genotype and total population*/
                X[event_val_G]++;
                N++;
                break;
            case DEATH: /* death event */
                /*decrease the genotype and total population*/
                X[event_val_G]--;
                N--;
                /* if genotype has gone extinct then remove it */
                if (X[event_val_G] == 0)
                {
                    /* shift numbers of other genotypes down */
                    #pragma omp simd
                    for (int k=event_val_G; k<(G-1); k++)
                    {
                        X[k] = X[k+1];
                    }
                    X[G-1] = 0; /* remove genotype */
                    G--;
                }
                break;
            case MUTATION: /* mutation event */
                /*decrease the original genotype population*/
                X[event_val_G]--;
                /* Create new genotype with population 1*/
                X[G] = 1;
                G++;
                /* if genotype has gone extinct then remove it */
                if (X[event_val_G] == 0)
                {
                    /* shift numbers of other genotypes down */
                    #pragma omp simd
                    for (int k=event_val_G; k<(G-1); k++)
                    {
                        X[k] = X[k+1];
                    }
                    X[G-1] = 0; /* remove genotype */
                    G--;
                }
                break;
        }
    }
    
    /* compute summary statistics from random sub-sample*/
    if (N > 0)
    {
        int numSamples = 473;
        int index;
        /* take a sample from the population */
        for (int i = 0; i<numSamples; i++)
        {
            while(1)
            {
                viRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,1,&index, 0,G);
                if (X[index] > 0)
                {
                    /* remove sample from population (sampling without replacement) */
                    X[index] = X[index]-1; 
                    /* add invidual into the sample */
                    x[index] = x[index]+1; 
                    break;
                }
            }
        }
         
        
        /* compute the distinct genotypes */
        g = 0; 
        #pragma omp simd reduction(+:g)
        for (int i=0; i<G; i++)
        {
            g += (x[i] > 0);
        }
        
        /* compute the genetic diversity */
        H = 0;
        #pragma omp simd reduction(+:H)
        for (int i=0;i<G;i++)
        {
            H += x[i]*x[i];
        }
        
        H = 1.0 - H/((double)(numSamples*numSamples));
    }
    else
    {
        g = 0;
        H = 0;
    }
    /* return final results */
    *H_ret = H;
    *g_ret = g;
    return;
}

/**
 * @brief program entry point
 * @details Generates prior predictive samples for the tuberculosis transmission
 * model. distributes simulations across available cores and utilises fine grain
 * SIMD operations where possible within the Gillespie simulation.
 *
 * @param argc the number of command line arguments
 * @param argv vector of argument strings
 */
int main(int argc,char ** argv) 
{
    double *theta;
    double *S;
    int seed;
    int sims;

    /* get command line arguments*/
    if (argc < 3)
    {
        fprintf(stderr,"Usage: [%s] sims seed",argv[0]);
    }
    else
    {
        sims = (int)atoi(argv[1]);
        seed = (int)atoi(argv[2]);
    }

    /* alloctate aligned memory for prior samples*/
    theta = (double *)_mm_malloc(sims*3*sizeof(double),ALIGN);
    
    /* alloctate aligned memory for summary statistics*/
    S = (double *)_mm_malloc(sims*2*sizeof(double),ALIGN);
   
    #pragma omp parallel shared(seed,sims,S,theta)
    {
        VSLStreamStatePtr stream;
        int thread_id, num_threads, sims_per_thread;

        /* get thread information and assign workload */
        thread_id = omp_get_thread_num();
        num_threads = omp_get_num_threads();
        sims_per_thread = sims/num_threads;

        /* initialise RNG stream for this thread*/
        vslNewStream(&stream,VSL_BRNG_MT2203+thread_id,seed);

        for (int k=thread_id*sims_per_thread;k<(thread_id+1)*sims_per_thread;k++)
        {
                /*model parameters birth,death and mutation rates*/
                double alpha, delta, mu; 
                
                /*summary stats, number of genotypes and genetic diversity*/
                double g_ret = 0.0, H_ret = 0.0;

                /* generate from prior */
                vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,1,&alpha, 0.0,5.0);
                vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,1,&delta, 0.0,alpha);
                
                /*prior for mutation rate is a truncated Gaussian on (0,infty)*/
                while(1)
                {
                    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER,stream,1,&mu,0.198,0.06735);
                    if (mu > 0)
                    {
                        break;
                    }
                
                }
                /* simulate pseudo-data*/ 
                simulate(stream,alpha, delta, mu, &g_ret, &H_ret);
                
                /*store prior samples*/
                theta[k*3] = alpha; 
                theta[k*3 + 1] = delta; 
                theta[k*3 + 2] = mu;

                /*store summary stats*/
                S[k*2] = g_ret; 
                S[k*2 +1] = H_ret;
        }
        /*clean up memory*/
        vslDeleteStream(&stream);
    }

    /*output prior predictive samples for postprocessing for ABC*/
    fprintf(stdout,"\"Sample\",\"alpha\",\"delta\",\"mu\",\"G\",\"H\"\n");
    for (int j=0;j<sims;j++)
    {
        fprintf(stdout,"%d,%g,%g,%g,%g,%g\n",j,theta[j*3],theta[j*3+1],
                                                  theta[j*3+2],S[j*2],S[j*2+1]);
    }
    exit(0);
}




