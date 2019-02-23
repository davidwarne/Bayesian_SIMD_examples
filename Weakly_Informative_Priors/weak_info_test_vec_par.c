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
 * @file weakInfo_bioassy.c 
 * @brief Demonstration of Intel Xeon Phi weakly informative prior selection.
 * @details The approach is based on the prior predictive p-value aproach
 * applied to the bioassy application.
 *
 * @author Chris C. Drovandi (c.drovandi@qut.edu.au)
 *         School of Mathematical Sciences
 *         Queensland University of Technology
 *
 * @author David J. Warne (david.warne@qut.edu.au)
 *         School of Mathematical Sciences
 *         Queensland University of Technology
 *
 * @date 13 Nov 2018
*/

/* standard C headers */
#include <stdio.h>
#include <math.h>
#include <time.h>
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

/* macro definitions*/
#define SIZE_D 4
#define SIGMA0_BASE 10.0
#define SIGMA1_BASE 2.5
#define NUM_PARTICLES 500

/* function prototype declarations*/
void 
compute_pvalues(VSLStreamStatePtr, double, double, unsigned int, 
                double * restrict, double * restrict);
void 
simulate_bioassay(VSLStreamStatePtr, int * restrict, double , double, 
                  double * restrict);
unsigned int 
nCk(unsigned int, unsigned int);
void 
loglike_bioassay(unsigned int, double * restrict, int * restrict, 
                 double * restrict, double * restrict);
void 
bvnpdf(unsigned int, double * restrict, double * restrict, double * restrict,
       double * restrict);
double 
SMC_RW(VSLStreamStatePtr, unsigned int, int* restrict, double* restrict, double,
       double);
double 
quantile(unsigned int, double * restrict, double);
void 
insertionSort(unsigned int, double * restrict);
double 
compute_ESS_diff(double, double, double * restrict, double * restrict, 
                 unsigned int);
double 
logsumexp(double * restrict, unsigned int );


/**
 * @brief program entry point
 *
 * @details computes weak information test for a set of prior hyperparameters.
 * p-values computations are distributed across cores. Each P-value calculation
 * further parallelised through SIMD operations withing the SMC step.
 *
 * @param argc number of command line arguments
 * @param argv vector of argument strings
 */
int 
main(int argc, char ** argv)
{
    /*dose levels (standardised)*/
    __declspec(align(ALIGN)) double d[SIZE_D]; 
    unsigned int num_datasets,  sims, seed;

    /*prior predictive p-values*/
    double *pvals; 
    
    /*our hyperparameters*/
    double *sigma0, *sigma1; 
    
    /*Intel MKL VSL random stream */
    VSLStreamStatePtr stream;

    /*initialise RNG stream for sampling hyperparameter space*/
    vslNewStream(&stream,VSL_BRNG_MT19937,1337);
    
    /*  dose data*/
    d[0] = -0.86;
    d[1] = -0.30;
    d[2] = -0.05;
    d[3] = -0.73;
    
    /* read commandline arguments*/
    if (argc < 4)
    {
        fprintf(stderr,"Usage : [%s] numdatasets sims seed\n",argv[0]);
        exit(1);
    }
    num_datasets = (unsigned int)atoi(argv[1]); 
    sims = (unsigned int)atoi(argv[2]); 
    seed = (unsigned int)atoi(argv[3]);
    
    /* allcote aligned memory for hyperparameters and p-values*/
    sigma0 = (double *)_mm_malloc(sims*sizeof(double),ALIGN);
    sigma1 = (double *)_mm_malloc(sims*sizeof(double),ALIGN);
    pvals = (double *)_mm_malloc(sims*sizeof(double),ALIGN);
    
    /* sample hyperparameter space*/
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,sims,sigma0,0.1,10.0);    
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,sims,sigma1,0.1,20.0); 
    
    /* the base prior representing current best information is appended*/
    sigma0[sims-1] = SIGMA0_BASE;
    sigma1[sims-1] = SIGMA1_BASE;
 
    /* create threads */
    #pragma omp parallel shared(num_datasets,sims,sigma0,sigma1,pvals,d)
    {
        /*Intel MKL VSL random stream*/
        VSLStreamStatePtr stream_sims;
        int thread_id, num_threads;
        
        /* get thread information*/
        thread_id = omp_get_thread_num();
        num_threads = omp_get_num_threads();

        /*initialise independent MT RNG stream for this thread*/
        vslNewStream(&stream_sims,VSL_BRNG_MT2203+thread_id,seed);

        /* distribute work among threads*/
        #pragma omp for schedule(guided)
        for (int i=0;i<sims;i++)
        {
            double *p;
            p = (double*)_mm_malloc(num_datasets*sizeof(double),ALIGN);
            
            /* compute prior predictive p-values for testing weak informativity*/
            compute_pvalues(stream_sims,sigma0[i],sigma1[i],num_datasets,d,p);
            
            /* obtain p-value at cutoff level 0.05*/
            pvals[i] = quantile(num_datasets,p,0.05);
            
            /* clean up memory*/
            _mm_free(p);
        }

        /*clean up thread private RNG state*/
        vslDeleteStream(&stream_sims);
    }


    /* write results*/
    fprintf(stdout,"\"R\",\"Sigma0\",\"Sigma1\",\"pvalue\"\n");
    for (int i=0;i<sims;i++)
    {
        fprintf(stdout,"%d,%lg,%lg,%lg\n",i,sigma0[i],sigma1[i],pvals[i]);
    }
    
    /*clean up memory*/
    vslDeleteStream(&stream);
    _mm_free(sigma0);
    _mm_free(sigma1);
    _mm_free(pvals);
}

/**
 * @brief compute prior predictive p-values
 *
 * @details Computes p(\lambda) = Pr(D(t|\lambda) > D(t)) where 
 * D(t|\lambda) = 1/p(t|\lambda) and D(t) = 1/p_base(t)
 * p(t|\lambda) and p_base(t) are prior predictive distributions under the 
 * priors p(\theta | \lambda) and base prior p(\theta) respectively.
 *
 * @param stream state pointer to this threads RNG
 * @param sigma0, sigma1 hyperparameters to test for weak informativity
 * @param num_datasets number of datasets to compute p-values over
 * @param d known parameters of forwards simulationn
 * @param p pointer to array to store p-value for each hyperparameter
 *
 * @note numerical overflow/underflow is avoided through computing with logs, 
 * that is we actually compute Pr(log D(t|\lambda) < log D(t))
 */
void 
compute_pvalues(VSLStreamStatePtr stream,double sigma0, double sigma1,
                unsigned int num_datasets, double * restrict d, 
                double * restrict p)
{
    __declspec(align(ALIGN)) int y[SIZE_D];
    double *theta;
    double *log_evidences_base;
    double *log_evidences_prior;

    /*allocate aligned memory for prior samples and evidences*/
    theta = (double*)_mm_malloc(2*num_datasets*sizeof(double),ALIGN);
    log_evidences_base = (double*)_mm_malloc(num_datasets*sizeof(double),ALIGN);
    log_evidences_prior = (double*)_mm_malloc(num_datasets*sizeof(double),ALIGN);

    /*compute evidences
     * under proposad prior based on data simulated from the base prior
     */
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2,stream,num_datasets,
                  theta,0,SIGMA0_BASE);
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2,stream,num_datasets,
                  theta+num_datasets,0,SIGMA1_BASE);
    
    /* generate data sets and compute evidences using SMC */              
    for (unsigned int i=0;i<num_datasets;i++)
    {
        /* generate bio-assay data Y ~ p (y | \theta)*/
        simulate_bioassay(stream,y,theta[i],theta[num_datasets + i],d);
        
        /* compute evidence using SMC 
         * p(y) = \int p(y | \theta) p(\theta) d \theta 
         */
        log_evidences_base[i] = SMC_RW(stream,NUM_PARTICLES,y,d,sigma0,sigma1);
    }

    /*compute evidences 
     * under proposed prior based on data simulated from the proposed prior
     */
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2,stream,num_datasets,
                  theta,0,sigma0);
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2,stream,num_datasets,
                  theta+num_datasets,0,sigma1);

    /* generate data sets and compute evidences using SMC */              
    for (unsigned int i=0;i<num_datasets;i++)
    {
        /* generate bio-assay data Y ~ p (y | \theta)*/
        simulate_bioassay(stream,y,theta[i],theta[num_datasets + i],d);
        
        /* compute evidence using SMC 
         * p(y | \lambda) = \int p(y | \theta) p(\theta | \lambda) d \theta 
         */
        log_evidences_prior[i] = SMC_RW(stream,NUM_PARTICLES,y,d,sigma0,sigma1);
    }

    /*compute prior predicitive p-values*/
    for (unsigned int i=0;i<num_datasets;i++)
    {
        unsigned int psum = 0;
        #pragma omp simd reduction(+:psum) 
        for (unsigned int j=0;j<num_datasets;j++)
        {
            psum += (log_evidences_prior[j] < log_evidences_base[i]);
        }
        p[i] = ((double)psum)/((double)num_datasets);
    }

    /* clean up memory*/
    _mm_free(theta);
    _mm_free(log_evidences_base);
    _mm_free(log_evidences_prior);
}

/**
 * @brief simulation of bioassay experiment
 * @details records y_i number of animmal deaths at dose level d_i
 *
 * @param stream state pointer to this threads RNG
 * @param y pointer to output array of number of deaths per does level
 * @param theta0,theta1 paramters of death event probability
 * @param d pointer to array of dose levels
 */
void 
simulate_bioassay(VSLStreamStatePtr stream, int * restrict y, double theta0,
                      double theta1, double * restrict d)
{
    __declspec(align(ALIGN)) double p[SIZE_D];
    
    /* compute death proabilities at each dose level*/
    #pragma omp simd 
    for (int i=0;i<SIZE_D;i++)
    {
        p[i] = 1.0/(1.0 + exp(-theta0 - theta1*d[i]));
    }
    
    /* Generate number of death events y_i ~ Binom(5,p_i)*/
    for (int i=0;i<SIZE_D;i++)
    {
        viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE,stream,1,y+i,5,p[i]);
    }
}

/** 
 * @brief computes bionomial coefficients N choose K.
 *
 * @param n number of elements
 * @param k number of elements to select
 */
unsigned int 
nCk(unsigned int n, unsigned int k)
{
    unsigned int r,d;
    r = 1; d = 1;
    for (unsigned int i=1;i<=k;i++)
    {
        r *= (n - (k -i));
    }
    for (unsigned int i=1;i<=k;i++)
    {
        d *= i;
    }
    return r/d;
}

/** 
 * @brief computes log-likelihood of bioassay data for a collection of paramter
 * combinations
 *
 * @param N number of prior samples
 * @param theta array of prior samples
 * @param y bioassay data
 * @param d bioassay dose levels
 * @param f array to store log-likelihood evaluations
 */
void
loglike_bioassay(unsigned int N, double* restrict theta, int *restrict y, 
                 double * restrict d,double * restrict f)
{
    double p;
    for (int i=0;i<N;i++)
    {
        f[i] = 0;
        double s = 0;
        #pragma omp simd reduction(+:s)
        for (int j=0;j<SIZE_D;j++)
        {
            double yd = (double)y[j];
            double nck = (double)nCk(5,y[j]);
            p = 1.0/(1.0+exp(-theta[i*2] - theta[i*2+1]*d[j]));
            s += log(nck*pow(p,yd)*pow(1.0-p,5.0 - yd));
        }
        f[i] = s;
    }
}


/**
 * @brief computes the bivariat normal probability density function
 *
 * @param N number of samples to evalutate the PDF for
 * @param X array of samples
 * @param mu mean vector
 * @param Sigma covariance matrix 
 * @param f array to store PDF values 
 */
void
bvnpdf(unsigned int N, double * restrict X,double * restrict mu,
           double * restrict Sigma,double * restrict f)
{
    double rho,sigma1,sigma2,D,M,z;
    
    sigma1 = sqrt(Sigma[0]);
    sigma2 = sqrt(Sigma[3]);
    rho = Sigma[1]/(sigma1*sigma2);
    D = (2*M_PI*sigma1*sigma2*sqrt(1-rho*rho));
    M = (-2*(1-rho*rho));
    
    /* compute in SIMD the PDF values for each sample*/
    #pragma omp simd 
    for (unsigned int i=0;i<N;i++)
    {
        double x1,x2;
        x1 = X[i*2] -mu[0];
        x2 = X[i*2+1] - mu[1];
        z = (x1*x1)/(Sigma[0]) -2*(x1*x2)/(sigma1*sigma2) + (x2*x2)/Sigma[3];
        f[i] = exp(z/M)/D;
    }
}

/**
 * @brief computes for array x_1:N, computes log(sum_i=1^N exp(x_i))
 * @details subtracts out largest x_i to avoid numerical overflow/underflow
 * e.g., let M = max(x_1:N) and y_i = x_i - M, then
 *       log(sum_i=1^N exp(x_i)) = log(sum_i=1^N exp(y_i)) + M
 *
 * @param x array of logs
 * @param N number of elements in x
 */
double 
logsumexp(double * restrict x , unsigned int N)
{
    double the_max;
    the_max = x[0];
    double sum_exp;
    
    /* obtain the maximum logarithm*/
    for (unsigned int i=0;i<N;i++)
    {
        the_max = (the_max < x[i]) ? x[i] : the_max;
    }
    sum_exp = 0;
    
    /*subtract the max and compute the sum*/
    #pragma omp simd reduction(+:sum_exp) 
    for (unsigned int i=0;i<N;i++)
    {
        x[i] -= the_max;
        sum_exp += exp(x[i]);
    }
    
    /*add the substracted logarithm to rescale*/
    return the_max + log(sum_exp);
}

/**
 * @brief compute ESS - N/2 for differnce in two temperature values
 *
 * @param gammavarnew proposel new temperature
 * @param current temperature
 * @param weigth particle weights
 * @param loglike particle loglikelihood
 * @param N number of particles
 *
 * @returns ESS - N/2
 */
double 
compute_ESS_diff(double gammavarnew,double gammavar, double * restrict weight,
                       double * restrict loglike, unsigned int N)
{
    /* re-compute weights*/
    #pragma omp simd 
    for (unsigned int i=0;i<N;i++)
    {
        weight[i] = (gammavarnew - gammavar)*loglike[i];
    }
    double max_w,sum_w;
    max_w = weight[0];
    for (unsigned int i=0;i<N;i++)
    {
        max_w = (max_w < weight[i]) ? weight[i] : max_w;
    }
    sum_w = 0;
    
    /* numerically stable log sum exp*/
    #pragma omp simd reduction(+:sum_w)
    for (unsigned int i=0;i<N;i++)
    {
        weight[i] -= max_w;
        weight[i] = exp(weight[i]);
        sum_w += weight[i];
    }
    
    /* normalise weights*/
    #pragma omp simd 
    for (unsigned int i=0;i<N;i++)
    {
       weight[i] /= sum_w;
    }
    sum_w = 0;
    
    /* compute ESS*/
    #pragma omp simd reduction(+:sum_w) 
    for (unsigned int i=0;i<N;i++)
    {
        sum_w += weight[i]*weight[i];
    }
    
    /* return final difference*/
    return 1.0/sum_w - ((double)N)/2.0;
}


/**
 * @brief Compute empirical convariance matrix
 *
 * @param X array of sample k-dimensional random vectors
 * @param N the number of samples
 * @param k dimenstion of samples
 * @param Sigma pointer to array to store output empirical convariance matrix 
 */
void 
cov(double * restrict X,unsigned int N,unsigned int k,double* restrict Sigma)
{
    __declspec(align(ALIGN)) double mu[10];
    
    /*initialise mean*/
    #pragma omp simd
    for (unsigned int j=0;j<k;j++)
    {
        mu[j] = 0;
    }
    
    /* compute sample mean*/
    for (unsigned int i=0;i<N;i++) 
    {
        #pragma omp simd 
        for (unsigned int j=0;j<k;j++)
        {
            mu[j] += X[i*k+j];
        }
    }

    double Nd = (double)N;
    #pragma omp simd
    for (unsigned int j=0;j<k;j++)
    {
        mu[j] /= Nd;
    }
    
    /*initialise covariance matrix*/
    #pragma omp simd
    for (unsigned int j=0;j<k*k;j++)
    {
        Sigma[j] = 0;
    }

    /* compute sample covariance matrix*/
    for (unsigned int i=0;i<N;i++)
    {
        #pragma omp simd collapse(2) 
        for (unsigned int j=0;j<k;j++)
        {
            for (unsigned int jj=0;jj<k;jj++)
            {
                Sigma[j*k + jj] += (X[i*k + j] - mu[j])*(X[i*k+jj] - mu[jj]);
            }
        }
    }
    
    /* compensate for bias from using sample mean*/ 
    Nd = (double)(N-1);
    #pragma omp simd
    for (unsigned int j=0;j<k*k;j++)
    {
        Sigma[j] /= Nd;
    }
}

/**
 * @brief computes log evidence using SMC
 * 
 * @details adaptive SMC scheme with tuned random-walk Metropolis-Hasting 
 * proposal kernel
 *
 * @param stream state pointer to this threads RNG
 * @param N number of particles
 * @param y bioassay data
 * @param d bioassay dose levels
 * @param sigma0, sigma1 prior parameters
 */
double 
SMC_RW(VSLStreamStatePtr stream,unsigned int N,int* restrict y, 
              double* restrict d, double sigma0, double sigma1)
{
    double log_evidence,gammavar_t, gammavar_tp1;
    __declspec(align(ALIGN)) double Sigma[4];
    __declspec(align(ALIGN)) double T[4];
    __declspec(align(ALIGN)) double mu[2];
    __declspec(align(ALIGN)) double cov_rw[4];
    double opt_h,c;
    double *theta, *w,*w_temp, *logprior_curr, *loglike, *u, *rd;
    double *theta_prop, *theta_rs, *logprior_curr_rs, *loglike_rs;
    double * acc_probs;
    unsigned int *r;
    
    /* current particles, proposals and resampled */
    theta = (double *)_mm_malloc(N*2*sizeof(double),ALIGN);
    theta_prop = (double *)_mm_malloc(N*2*sizeof(double),ALIGN);
    theta_rs = (double *)_mm_malloc(N*2*sizeof(double),ALIGN);
    
    /* current log prior and resampled*/
    logprior_curr = (double *)_mm_malloc(N*sizeof(double),ALIGN);
    logprior_curr_rs = (double *)_mm_malloc(N*sizeof(double),ALIGN);
    
    /* current log likelihood and resampled*/
    loglike = (double *)_mm_malloc(N*sizeof(double),ALIGN);
    loglike_rs = (double *)_mm_malloc(N*sizeof(double),ALIGN);
    
    /*particle weight*/
    w = (double *)_mm_malloc(N*sizeof(double),ALIGN);
    w_temp = (double *)_mm_malloc(N*sizeof(double),ALIGN);
    
    /*uniform random numbers*/
    u = (double*)_mm_malloc(N*sizeof(double),ALIGN);
    r = (unsigned int*)_mm_malloc(N*sizeof(unsigned int),ALIGN);
    rd = (double*)_mm_malloc(N*sizeof(double),ALIGN);
    
    /* acceptance probabilities*/
    acc_probs = (double *)_mm_malloc(N*sizeof(double),ALIGN);

    /*initialising*/
    
    /*for tuned proposal kernel of the random walk Metropolis-Hastings*/
    opt_h = 2.38/sqrt(2.0);
    /*for choosing number of MCMC repeats. We choose so that particles have 
     * probablitity of at least 1-c of moving
     */
    c = 0.01; 

    log_evidence = 0.0;
    
    /* power for initial high temperature*/
    gammavar_t = 0.0; 

    /* prior covariance*/
    Sigma[0] = sigma0*sigma0; Sigma[1] = 0;
    Sigma[2] = 0;             Sigma[3] = sigma1*sigma1;

    /* prior mean*/
    mu[0] = 0.0; 
    mu[1] = 0.0;
    
    /* prior Cholesky decomposition*/
    T[0] = sigma0; T[1] =0; /*Sigma = T*T'*/
    T[2] = 0;      T[3] = sigma1;
    
    /*draws from prior to obtain initial set of particles*/
    vdRngGaussianMV(VSL_RNG_METHOD_GAUSSIANMV_BOXMULLER2,stream,N,theta,2,
                    VSL_MATRIX_STORAGE_FULL,mu,T);
    
    /*initialise log-prior for each particle*/
    bvnpdf(N,theta,mu,Sigma,logprior_curr);
    #pragma omp simd 
    for (unsigned int i=0;i<N;i++)
    {
        logprior_curr[i] = log(logprior_curr[i]);
    }

    /*initialise log-likelihood for each particle*/
    loglike_bioassay(N,theta,y,d,loglike);

    /* Adaptive SMC sampler*/
    while (gammavar_t < 1.0)
    {
        double expected_acc_probs = 0;
        unsigned int R_t = 0;
        double w_sum = 0;
        double w_sum2= 0;
        double ESS1=0;
        double w_max = 0;
        
        /* update gamma_t such that ESS >= N/2*/
        /*compute normalised weight of large step in gamma*/
        #pragma omp simd 
        for (unsigned int i=0;i<N;i++) 
        {
            w[i] = (1.0-gammavar_t)*loglike[i];
        }
        
        /* log sum exp */
        w_max = w[0];
        for (unsigned int i=0;i<N;i++)
        {
            w_max = (w[i] > w_max) ? w[i] : w_max;
        }
        w_sum = 0;
        #pragma omp simd reduction(+:w_sum) 
        for (unsigned int i=0;i<N;i++)
        {
            w[i] = exp(w[i]-w_max);
            w_sum += w[i];
        }
        #pragma omp simd 
        for (unsigned int i=0;i<N;i++)
        {
            w[i] /= w_sum;
        }
        w_sum2 = 0;
        
        /* compute Effective sample size (ESS)*/
        #pragma omp simd reduction(+:w_sum2)
        for (unsigned int i=0;i<N;i++)
        {
             w_sum2 += w[i]*w[i];
        }
        ESS1 = 1.0/w_sum2;
        
        /*choosing next temperature*/
        if (ESS1 >= ((double)N)/2.0)
        {
            gammavar_tp1 = 1.0;
        }
        else
        {
            /*smaller temperature step required*/
            double a = 0;
            double b = 0;
            double p = 0;
            double fa = 0;
            double fp = 0;
            double err = 0;
        
            /* find optimum step using bisection method*/
            a = gammavar_t + 1e-6;
            b = 1.0;
            p = (a+b)/2.0;
            fp = compute_ESS_diff(p,gammavar_t,w_temp,loglike,N);
            err =  fabs(fp);
            while (err > 1e-5)
            {
                fa = compute_ESS_diff(a,gammavar_t,w_temp,loglike,N);
                if (fa*fp < 0) 
                {
                    b = p;
                } 
                else  
                {
                    a = p;
                }
                p = (a+b)/2.0;
                fp = compute_ESS_diff(p,gammavar_t,w_temp,loglike,N);
                err = fabs(fp);
            }
            gammavar_tp1 = p;
        }

        /*re-weighting particles*/
        #pragma omp simd 
        for (unsigned int i=0;i<N;i++)
        {
            w[i] = (gammavar_tp1 - gammavar_t)*loglike[i];
        }

        /* compute log_evidence update*/
        log_evidence = log_evidence + logsumexp(w,N) - log((double)N);
        w_max = w[0];
        for (unsigned int i=0;i<N;i++)
        {
            w_max = (w_max < w[i]) ? w[i] : w_max;
        }
        w_sum = 0;
        #pragma omp simd reduction(+:w_sum) 
        for (unsigned int i=0;i<N;i++)
        {
            w[i] -= w_max;
            w[i] = exp(w[i]);
            w_sum += w[i];
        }
        #pragma omp simd 
        for (unsigned int i=0;i<N;i++)
        {
            w[i] = w[i]/w_sum;
        }

        /*sampling with replacement according to weights*/
        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,N,u,0,1);
        
        /*lookup method*/
        for (unsigned int i=0;i<N;i++)
        {
            int k;
            k = 0;
            double w_sum3 = w[k];
            while (u[i] > w_sum3 && k < N-1)
            {
                k++;
                w_sum3 += w[k];
            }
            r[i] = k;
        }
        
        /* re-assign particles*/
        for (unsigned int i=0;i<N;i++){
            theta_rs[i*2] = theta[r[i]*2];
            theta_rs[i*2+1] = theta[r[i]*2+1];
            loglike_rs[i] = loglike[r[i]]; 
            logprior_curr_rs[i] = logprior_curr[r[i]]; 
        }
        
        /*copy back*/
        memcpy(theta,theta_rs,N*2*sizeof(double));
        memcpy(loglike,loglike_rs,N*sizeof(double));
        memcpy(logprior_curr,logprior_curr_rs,N*sizeof(double));
        
        /*compute sample covariances for tuned MCMC proposal kernel*/
        cov(theta,N,2,cov_rw);
        #pragma omp simd 
        for (unsigned int i=0;i<4;i++)
        {
            cov_rw[i] *= opt_h*opt_h;
        }
        
        /*we require Cholesky Factorisation for MKL VSL*/
        T[0] = sqrt(cov_rw[0]);    T[1] = 0;
        T[2] = cov_rw[2] / T[0];   T[3] = sqrt(cov_rw[3] - T[2]*T[2]);
        mu[0] = 0;
        mu[1] = 0;
        
        /*generate N bivariate normal for MCMC proposals*/
        vdRngGaussianMV(VSL_RNG_METHOD_GAUSSIANMV_BOXMULLER2,stream,N,
                        theta_prop,2,VSL_MATRIX_STORAGE_FULL,mu,T);
        
        /*generate N uniform variates for acceptance steps*/
        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,N,u,0,1);
        
        /*perform initial MCMC step for each particle in SIMD*/
        for (unsigned int i=0;i<N-VECLEN;i+=VECLEN)
        {
            __declspec(align(ALIGN)) double loglike_prop[VECLEN];
            __declspec(align(ALIGN)) double logprior_prop[VECLEN];
            __declspec(align(ALIGN)) double log_mh[VECLEN];
        
            /* generate proposal*/
            #pragma omp simd 
            for (unsigned int j = 0; j<2*VECLEN;j++)
            {
                theta_prop[i*2 + j] += theta[i*2+j];
            }
            
            /* compute log-prior and log-likelihood for proposal*/
            loglike_bioassay(VECLEN,theta_prop+i*2,y,d,loglike_prop);
            bvnpdf(VECLEN,theta_prop+i*2,mu,Sigma,logprior_prop);
           
            /* compute Metropolis-Hastings acceptance probabilities*/
            #pragma omp simd                    
            for (unsigned int j=0;j<VECLEN;j++)
            {
                logprior_prop[j] = log(logprior_prop[j]);
                log_mh[j] = gammavar_tp1*(loglike_prop[j] - loglike[i + j]) 
                            + logprior_prop[j] - logprior_curr[i + j];
                acc_probs[i+j] = exp(log_mh[j]);
            }
            
            /*accept/reject proposal*/
            #pragma omp simd
            for (unsigned int j=0;j<VECLEN;j++)
            {
                unsigned int tf = (u[i+j] < acc_probs[i+j]);
                theta[i*2 +j*2] = (tf) ? theta_prop[i*2 + j*2] 
                                       : theta[i*2 + j*2];
                theta[i*2+1+j*2] = (tf) ? theta_prop[i*2+1+j*2] 
                                        : theta[i*2+1 + j*2];
                loglike[i+j] = (tf) ? loglike_prop[j] 
                                    : loglike[i+j];
                logprior_curr[i+j] = (tf) ? logprior_prop[j] 
                                          : logprior_curr[i+j];
            }
        }

        /*determine remainin repeats*/
        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,N,u,0,1);
        expected_acc_probs = 0;
        unsigned int sum_acc_probs = 0;
        
        /*compute acceptance probability estimate*/
        #pragma omp simd reduction(+:sum_acc_probs)
        for (unsigned int i=0;i<N;i++)
        {
            sum_acc_probs += (acc_probs[i] > u[i]);
        }
        expected_acc_probs = ((double)sum_acc_probs)/((double)N);
        if (expected_acc_probs <= 0.0)
        {
            expected_acc_probs =1e-6;
        } 
        else if (expected_acc_probs >= 1.0)
        {
            expected_acc_probs = 1.0 - 1e-6;
        }
        
        /* determine the number of MCMC steps to ensure probability of 1-c that 
         * a particle moves
         */
        R_t = (unsigned int)ceil(log(0.01)/log(1.0 - expected_acc_probs ));
        
        /*perform remaining repeats in blocks of VECLEN*/
        for (unsigned int i=0;i<N-VECLEN;i+=VECLEN)
        {
            __declspec(align(ALIGN)) double loglike_prop[VECLEN];
            __declspec(align(ALIGN)) double logprior_prop[VECLEN];
            __declspec(align(ALIGN)) double log_mh[VECLEN];
            __declspec(align(ALIGN)) double acc_probs_temp[VECLEN];
            __declspec(align(ALIGN)) double theta_temp[VECLEN*2];
            __declspec(align(ALIGN)) double loglike_temp[VECLEN];
            __declspec(align(ALIGN)) double logprior_curr_temp[VECLEN];
            
            /* generate Gaussian random variates for proposals in this block*/
            vdRngGaussianMV(VSL_RNG_METHOD_GAUSSIANMV_BOXMULLER2,stream,
                          VECLEN*R_t,theta_prop,2,VSL_MATRIX_STORAGE_FULL,mu,T);
            
            /* generate uniform random variates for accept/reject steps*/
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,VECLEN*R_t,u,0,1);
            
            /* copy partical state for this block to ensure in cache*/
            #pragma omp simd 
            for (unsigned int j = 0; j<2*VECLEN;j++)
            {
                theta_temp[j] = theta[i*2 + j];
            }
            #pragma omp simd                      
            for (unsigned int j = 0; j<VECLEN;j++)
            {
                acc_probs_temp[j] = acc_probs[i + j];
                loglike_temp[j] = loglike[i + j];
                logprior_curr_temp[j] = logprior_curr[i + j];
            }

            /* evolve a block of Markov chains together using SIMD*/
            for (unsigned int k=0;k<R_t;k++)
            {
                /* generate proposal (using pre-computed Gaussian samples)*/
                #pragma omp simd  
                for (unsigned int j = 0; j<2*VECLEN;j++)
                {
                    theta_prop[k*VECLEN*2 + j] += theta_temp[j];
                }
                
                /* compute log-prior and log-likelihood for the proposal*/
                loglike_bioassay(VECLEN,theta_prop+k*VECLEN*2,y,d,loglike_prop);
                bvnpdf(VECLEN,theta_prop+k*VECLEN*2,mu,Sigma,logprior_prop);
                
                /* Compute Metropolis-Hastings acceptance probabilities*/
                #pragma omp simd
                for (unsigned int j=0;j<VECLEN;j++)
                {
                    logprior_prop[j] = log(logprior_prop[j]);
                    log_mh[j] = gammavar_tp1*(loglike_prop[j] - loglike_temp[j]) 
                                + logprior_prop[j] - logprior_curr_temp[j];
                    acc_probs_temp[j] = exp(log_mh[j]);
                }

                /*perform accept/reject step*/
                #pragma omp simd 
                for (unsigned int j=0;j<VECLEN;j++)
                {
                    unsigned int tf = (u[k*VECLEN+j] < acc_probs_temp[j]);
                    theta_temp[j*2] = (tf) ? theta_prop[k*VECLEN*2 + j*2] 
                                           : theta_temp[j*2];
                    theta_temp[1+j*2] = (tf) ? theta_prop[k*VECLEN*2+1+j*2] 
                                             : theta_temp[1 + j*2];
                    loglike_temp[j] = (tf) ? loglike_prop[j] 
                                           : loglike_temp[j];
                    logprior_curr_temp[j] = (tf) ? logprior_prop[j] 
                                                 : logprior_curr_temp[j];
                }
            }

            /* write temp data of final state back to main particle state*/
            #pragma omp simd 
            for (unsigned int j = 0; j<2*VECLEN;j++)
            {
                theta[i*2 + j] = theta_temp[j];
            }
            #pragma omp simd 
            for (unsigned int j = 0; j<VECLEN;j++)
            {
                acc_probs[i + j] = acc_probs_temp[j]; 
                loglike[i + j] = loglike_temp[j]; 
                logprior_curr[i + j] = logprior_curr_temp[j]; 
            }
        }

        /* update temperature*/
        gammavar_t =  gammavar_tp1;
    }
    
    /*clean up memory*/
    _mm_free(theta);
    _mm_free(theta_prop);
    _mm_free(theta_rs);
    _mm_free(logprior_curr);
    _mm_free(logprior_curr_rs);
    _mm_free(loglike);
    _mm_free(loglike_rs);
    _mm_free(w);
    _mm_free(w_temp);
    _mm_free(u);
    _mm_free(r);
    _mm_free(rd);
    _mm_free(acc_probs);
    
    /*return final log_evidence estimate*/
    return log_evidence; 
}

/**
 * @brief compute the q-quantile
 *
 * @param n number of samples
 * @param x samples
 * @param q the q-th quantile level
 * @return the value of quantile
 */
double 
quantile(unsigned int n, double *x,double q)
{
    double u,l;
    unsigned int i;
    
    /*sort samples */
    insertionSort(n,x);

    /* pick the samples closes to the q quantile*/
    
    /* upper and lower bound of the quantile value*/
    u = ceil(((double)n)*q);
    l = floor(((double)n)*q);
    
    /* nearest neighbour interpolation */
    i = (unsigned int)((u - ((double)n)*q < ((double)n)*q - l) ? u : l);
    return x[i];
}

/**
 * @brief standard sorting algorithm
 *
 * @param n number of elements to be sorted
 * @param x array to be sorted in-place
 */
void 
insertionSort(unsigned int n, double * x)
{
    for (int i=1; i<n;i++)
    {
        double A;
        int j;
        A = x[i];
        j = i-1;
        while (j >= 0 && x[j] > A)
        {
            x[j+1] = x[j];
            j--;
        }
        x[j+1] = A;
    }
}

