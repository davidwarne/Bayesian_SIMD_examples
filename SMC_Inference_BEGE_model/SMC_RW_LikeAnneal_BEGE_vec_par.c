/**
 * @file SMC_RW_LikeAnneal_BEGE.c
 * @brief Demonstration of vectorisation and parallelisation for inference on 
 * the Bad Environment, Good Evironment (BEGE) model.
 * 
 * @details Random walk adaptive SMC with likelihood annealing with an expensive
 * likelihood.
 *
 * @author Chris C. Drovandi (c.drovandi@qut.edu.au)
 *         School of Mathematical Sciences
 *         Queensland University of Technology
 * @author David J. Warne (david.warne@qut.edu.au)
 *         School of Mathematical Sciences
 *         Queensland University of Technology
 *
 * @date 4 Feb 2019
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Intel headers */
#include <mkl.h>
#include <mkl_vsl.h>
#include <mathimf.h>

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

#define ITMAX 100       /*maximum allowed number of iterations*/
#define EPS 3.0e-7      /*relative accuracy*/

#define FPMIN 1.0e-30   /*number near the smallest representable 
                            floating point number*/

#define NUM_PARAM 11

/*function prototypes for gamma function and incomplete gamma functions*/

/* error message function*/
void
nrerror(char *);

/* series approximation */
void
gser_vec(double * restrict, double * restrict, double * restrict, 
         double * restrict);

/* compute P(a,x) = \gamma(a,x)/Gamma(a)*/
void
gammp_vec(double, double, int, double * restrict, double * restrict);

/* compute Q(a,x) = 1 - P(a,x) = \Gamma(a,x)/Gamma(a)*/
void
gammq_vec(double, double, int, double * restrict, double * restrict);

/* compute ln Gamma(a) @note possibly use the intel imf lgamma*/
void
gammln_vec(double * restrict, double * restrict);

double*
loadData(char *, int *);

double 
compute_ESS_diff(double, double, double * restrict, double * restrict, 
                 unsigned int);

double
SMC_RW_LikeAnneal(unsigned int, unsigned int, int,double* restrict, 
                  double * restrict, double * restrict, double* restrict);

double
log_prior(double * restrict, double * restrict, double * restrict);

double
bege_gjrgarch_likelihood(double * restrict, int, double * restrict, 
                         double * restrict, double * restrict);
double
loglikedgam_vec(double, double, double, double, double, double);

double 
quantile(unsigned int, double *,double);


VSLStreamStatePtr stream;
unsigned int NperThread;
int thread_id;
#pragma omp threadprivate(stream,NperThread,thread_id)

/**
 * @brief main entry point of program 
 */
int 
main(int argc, char ** argv)
{
    char *filename;
    double *rate_return, *theta;
    int ndat;
    unsigned int seed;
    int N, NUM_PARTICLES, num_threads;

    /*limits for uniform priors*/
    __declspec(align(ALIGN)) double prior_l[NUM_PARAM] = {1e-4,1e-4,1e-4,1e-4,
                                            1e-4,1e-4,1e-4,1e-4,-0.2,1e-4,-0.9};
    __declspec(align(ALIGN)) double prior_u[NUM_PARAM] = { 0.5, 0.3,0.99, 0.5, 
                                            0.5, 1.0, 0.3,0.99, 0.1,0.75, 0.9};

    if (argc < 4)
    {
        fprintf(stderr,"Usage: [%s] N datafile seed\n",argv[0]);
        exit(1);
    }
    NUM_PARTICLES = (int)atoi(argv[1]); 
    filename = argv[2];
    seed = (unsigned int)atoi(argv[3]);
    
    /* load finance data */
    rate_return = loadData(filename, &ndat); 
   
    /* to ensure that particles pack neatly into vectors with no leftovers*/
    num_threads = omp_get_max_threads();
    if (NUM_PARTICLES%(VECLEN*num_threads) == 0)
    {
        N = NUM_PARTICLES;
    }
    else
    {
        N = ((NUM_PARTICLES / (VECLEN*num_threads)) + 1)*(VECLEN*num_threads);
    }
    fprintf(stderr,"Particles: %d, Vector length: %d, threads: %d\n",N,
                                                            VECLEN,num_threads);

    /* allocate memory for particles */ /**@note theta[N][NUM_PARAM]*/
    theta = (double*)_mm_malloc(NUM_PARAM*N*sizeof(double),ALIGN); 

    /* perform Random Walk SMC with Likelihood Annealling*/
    SMC_RW_LikeAnneal(seed, N, ndat, rate_return, prior_l, prior_u, theta);
    
    /*output particles*/
    for (int i=0;i<N;i++)
    {
        fprintf(stdout,"%d",i);
        for (int j=0;j<NUM_PARAM;j++)
        {
            fprintf(stdout,",%lg", theta[i*NUM_PARAM+j]);
        }
        fprintf(stdout,"\n");
    }
    return 0;
}

/**
 * @brief load return rate data
 * @param data filename
 */
double*
loadData(char * filename, int *ndat)
{
    FILE *fp;
    int N;
    double * data;
    char buf[255];

    /*open file for reading*/
    if ((fp = fopen(filename,"r")) == NULL)
    {
        fprintf(stderr,"Could not open file [%s]\n.",filename);
        exit(1);
    }

    /* the first entry is the number of elements in the time series*/
    fscanf(fp,"%d",&N);
    /* allocate memory for data*/
    data = (double *)_mm_malloc(N*sizeof(double),ALIGN);

    for (int i=0;i<N;i++)
    {
        fscanf(fp,"%lg",data+i);
    }
    fclose(fp);

    *ndat = N;
    return data;
}

/**
 * @brief print message to stderr
 * @param msg error message to print
 */
void 
nrerror(char *msg)
{
    fprintf(stderr,"%s\n",msg);
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
    #pragma omp parallel shared(loglike,weight) firstprivate(gammavarnew,gammavar)
    {
        #pragma omp simd
        for (int i=thread_id*NperThread;i<((thread_id+1)*NperThread);i++)
        {
            weight[i] = (gammavarnew - gammavar)*loglike[i];
        }
    }
    double max_w,sum_w;
    max_w = weight[0];
    for (unsigned int i=0;i<N;i++)
    {
        max_w = (max_w < weight[i]) ? weight[i] : max_w;
    }
    sum_w = 0;
    /* umerically stable log sum exp*/
    #pragma omp parallel shared(weight) firstprivate(max_w) reduction(+:sum_w)
    {
        double sum_w_temp = 0;
        #pragma omp simd reduction(+:sum_w_temp)
        for (int i=thread_id*NperThread;i<((thread_id+1)*NperThread);i++)
        {
            weight[i] -= max_w;
            weight[i] = exp(weight[i]);
            sum_w_temp += weight[i];
        }
        sum_w += sum_w_temp;
    }
    /* normalise weights*/
    #pragma omp parallel shared(weight) firstprivate(sum_w)
    {
        #pragma omp simd
        for (int i=thread_id*NperThread;i<((thread_id+1)*NperThread);i++)
        {
            weight[i] /= sum_w;
        }
    }
    sum_w = 0;
    /* compute ESS*/
    #pragma omp parallel shared(weight) reduction(+:sum_w)
    {
        double sum_w_temp = 0;
        #pragma omp simd reduction(+:sum_w_temp)
        for (int i=thread_id*NperThread;i<((thread_id+1)*NperThread);i++)
        {
            sum_w_temp += weight[i]*weight[i];
        }
        sum_w += sum_w_temp;
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
 * @param mu pointer to array to store output empirical mean
 * @param Sigma pointer to array to store output empirical convariance matrix 
 */
void 
cov(double * restrict X,unsigned int N,unsigned int k, double *restrict mu, 
    double* restrict Sigma)
{
    /*initialise mean*/
    for (unsigned int j=0;j<k;j++)
    {
        mu[j] = 0;
    }
    
    /* compute sample mean*/
    for (unsigned int i=0;i<N;i++) 
    {
        for (unsigned int j=0;j<k;j++)
        {
            mu[j] += X[i*k+j];
        }
    }

    double Nd = (double)N;
    for (unsigned int j=0;j<k;j++)
    {
        mu[j] /= Nd;
    }
    
    /*initialise covariance matrix*/
    for (unsigned int j=0;j<k*k;j++)
    {
        Sigma[j] = 0;
    }

    /* compute sample covariance matrix*/
    for (unsigned int i=0;i<N;i++)
    {
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
    for (unsigned int j=0;j<k*k;j++)
    {
        Sigma[j] /= Nd;
    }
}

/**
 * @brief computed the mahalanobis squared distance between two point X and Y
 * @param K dimenision of vectors X, Y
 * @param X point 1
 * @param Y point 2
 * @param Sigma_inv Lower diagonal portion of the inverse covariance matrix
 */
double
mahalanobis_sq(int K,double * restrict X, double * restrict Y,double * Sigma_inv)
{
    __declspec(align(ALIGN)) double diff[K];
    double dist;
    
    for (int i=0;i<K;i++)
    {
        double _diff = 0.0;
        for (int j=0;j<K;j++)
        {
            _diff += (X[j] - Y[j])*Sigma_inv[i*K + j];
        }
        diff[i] = _diff;
    }
    dist = 0.0;
    for (int i=0;i<K;i++)
    {
        dist += diff[i]*(X[i] - Y[i]);
    }
    return dist;
}

/**
 * @brief Adaptive Sequential Monte Carlo with Likelihood Annealing
 * @details An implementation of the adaptive SMC method for inference on the 
 * BEGE econometrics model with an 11-dimensional parameter space. Assumes 
 * uniform priors.
 *
 * @param seed seed value to initialise RNG streams
 * @param N the number of particles to use
 * @param rate_return the econometric data (monthly returns on some market)
 * @param prior_l lower limit of each parameter for the uniform prior
 * @param prior_u upper limit for each parameter for the uniform prior
 * @param theta output array of final particles ( NxD)
 * @returns the log evidence
 */
double
SMC_RW_LikeAnneal(unsigned int seed, unsigned int N, int ndat, 
                  double* restrict rate_return, double * restrict prior_l, 
                  double * restrict prior_u, double* restrict theta)
{

    double *logw_previous, *logw, *w, *w_temp, *loglike; 
    double *theta_particle, *theta_particle_prop, *logprior;
    double *loglike_rs, *theta_particle_rs, *logprior_rs;
    double *acc_prob, *dist, *ESJD; /*expected squared jump distance*/
    int *r;
    __declspec(align(ALIGN)) double h[10] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
    __declspec(align(ALIGN)) double median_ESJD[10];
    __declspec(align(ALIGN)) double Sigma[NUM_PARAM*NUM_PARAM];
    __declspec(align(ALIGN)) double T[NUM_PARAM*NUM_PARAM];
    __declspec(align(ALIGN)) double mu[NUM_PARAM];
    __declspec(align(ALIGN)) double cov_rw[NUM_PARAM*NUM_PARAM];
    __declspec(align(ALIGN)) double cov_inv[NUM_PARAM*NUM_PARAM];
    int t = 1;
    double log_evidence = 0.0;
    double gamma_t = 0.0;
    double gamma_tp1;
    double Nf = (double)N;
    double median_dist;
    double h_opt, *h_all, *u;
    int *h_ind;
    int h_N, ind;
    double *ESJD_h_tmp;
    double *dist_move;
    
    int belowThreshold;
    int R_move;

    logw_previous = (double*) _mm_malloc(N*sizeof(double),ALIGN);
    loglike = (double*) _mm_malloc(N*sizeof(double),ALIGN);
    loglike_rs = (double*) _mm_malloc(N*sizeof(double),ALIGN);
    logw = (double*) _mm_malloc(N*sizeof(double),ALIGN);
    w = (double*) _mm_malloc(N*sizeof(double),ALIGN);
    w_temp = (double*) _mm_malloc(N*sizeof(double),ALIGN);
    logprior = (double*) _mm_malloc(N*sizeof(double),ALIGN);
    logprior_rs = (double*) _mm_malloc(N*sizeof(double),ALIGN);
    theta_particle = (double*) _mm_malloc(NUM_PARAM*N*sizeof(double),ALIGN);
    theta_particle_rs = (double*) _mm_malloc(NUM_PARAM*N*sizeof(double),ALIGN);
    theta_particle_prop = (double*) _mm_malloc(NUM_PARAM*N*sizeof(double),ALIGN);
    r = (int*) _mm_malloc(N*sizeof(int),ALIGN);
    h_ind = (int*) _mm_malloc(N*sizeof(int),ALIGN);
    h_all = (double*) _mm_malloc(N*sizeof(double),ALIGN);
    u = (double*) _mm_malloc(N*sizeof(double),ALIGN);
    ESJD = (double*) _mm_malloc(N*sizeof(double),ALIGN); 
    acc_prob = (double*) _mm_malloc(N*sizeof(double),ALIGN);
    ESJD_h_tmp = (double*)_mm_malloc(N*sizeof(double),ALIGN);
    dist_move = (double*)_mm_malloc(N*sizeof(double),ALIGN);
    dist = (double*)_mm_malloc(N*N*sizeof(double),ALIGN); 
   
    /* initialise RNGs for threads*/
    #pragma omp parallel firstprivate(seed)
    {
        thread_id = omp_get_thread_num();
        NperThread = N / omp_get_num_threads();
        /*initialise independent MT RNG stream*/
        vslNewStream(&stream,VSL_BRNG_MT2203+thread_id,seed);
    }

    /* initialise particle weights*/
    #pragma omp parallel shared(logw_previous) firstprivate(Nf)
    {
        #pragma omp simd 
        for (int i=thread_id*NperThread;i<((thread_id+1)*NperThread);i++)
        {
            logw_previous[i] = -log(Nf);
        }
    }

    /* initialise particles*/
    #pragma omp parallel shared(theta,theta_particle,prior_u,prior_l)
    {
        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,NperThread*NUM_PARAM,
                    theta+thread_id*NperThread*NUM_PARAM,0,1);
        for (int i=thread_id*NperThread;i<((thread_id+1)*NperThread);i++)
        {
            for (int j=0;j<NUM_PARAM;j++)
            {
                theta[i*NUM_PARAM + j] *= (prior_u[j]-prior_l[j]);
                theta[i*NUM_PARAM + j] += prior_l[j];
                theta_particle[i*NUM_PARAM + j] = 
                                   log((theta[i*NUM_PARAM + j] - prior_l[j])
                                        /(prior_u[j] - theta[i*NUM_PARAM + j]));
            }
        }
    }
    
    /* evaluate log prior of particles*/
    #pragma omp parallel shared(logprior,theta,theta_particle,prior_l,prior_u)
    {
        #pragma omp for schedule(guided)
        for(int i=0;i<N;i++)
        {
            logprior[i] = log_prior(theta_particle+i*NUM_PARAM, prior_l, prior_u);
            /*this while loop is used to eliminate and infinity values from simulation*/
            while (isinf(logprior[i]))
            {
                vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,NUM_PARAM,theta+i*NUM_PARAM,0,1);
                for (int j=0;j<NUM_PARAM;j++)
                {
                    theta[i*NUM_PARAM + j] *= (prior_u[j]-prior_l[j]);
                    theta[i*NUM_PARAM + j] += prior_l[j];
                    theta_particle[i*NUM_PARAM + j] = 
                                         log((theta[i*NUM_PARAM + j] - prior_l[j])
                                            /(prior_u[j] - theta[i*NUM_PARAM + j]));
                }
                logprior[i] = log_prior(theta_particle+i*NUM_PARAM, prior_l, prior_u);
            }
        }
    }
 
    fprintf(stderr,"Computing initial log-likelihood...\n");
    /* evaluate log likelihood under the BEGE model */
    #pragma omp parallel shared(loglike,theta_particle,prior_l,prior_u,rate_return) \
                            firstprivate(ndat)
    {
        #pragma omp for schedule(guided)
        for(int i=0;i<N;i++)
        {
            loglike[i] = bege_gjrgarch_likelihood(theta_particle+i*NUM_PARAM, ndat,
                                              rate_return, prior_l, prior_u);
        }
    }
    
    fprintf(stderr,"Starting SMC...\n");
    /*Adaptive SMC sampler*/
    while (gamma_t < 1.0)
    {
        double w_sum = 0;
        double w_sum2 = 0;
        double ESS1 = 0;
        double w_max = 0;
       
        #pragma omp parallel shared(logw_previous,logw,loglike) firstprivate(gamma_t)
        {
            #pragma omp simd 
            for (int i=thread_id*NperThread;i<((thread_id+1)*NperThread);i++)
            {
                logw[i] = logw_previous[i] + (1.0 - gamma_t)*loglike[i];
            }
        }
        
        w_sum = 0;
        w_sum2 = 0;
        /* log sum exp trick for numerical stability */
        w_max = logw[0];
        for (int i=0;i<N;i++)
        {
            w_max = (logw[i] > w_max) ? logw[i] : w_max; 
        }

        #pragma omp parallel shared(w,logw) firstprivate(w_max) reduction(+:w_sum)
        {
            double w_sum_temp = 0;
            #pragma omp simd reduction (+:w_sum_temp)
            for (int i=thread_id*NperThread;i<((thread_id+1)*NperThread);i++)
            {
                w[i] = logw[i] - w_max;
                w[i] = exp(w[i]);
                w_sum_temp += w[i];
            }
            w_sum += w_sum_temp;
        }
        
        /* computing Effective sample size*/
        #pragma omp parallel shared(w) firstprivate(w_sum) reduction(+:w_sum2)
        {
            double w_sum2_temp = 0;
            #pragma omp simd reduction (+:w_sum2_temp)
            for (int i=thread_id*NperThread;i<((thread_id+1)*NperThread);i++)
            {
                w[i] /= w_sum;
                w_sum2_temp += w[i]*w[i];
            }
            w_sum2 += w_sum2_temp;
        }
        ESS1 = 1.0/w_sum2;
        /* choosing next temperature*/
        if (ESS1 >= Nf/2.0)
        {
            gamma_tp1 = 1.0;
        }
        else
        {
            double  a = 0;
            double  b = 0;
            double  p = 0;
            double  fa = 0;
            double  fp = 0;
            double  err = 0;

            /* find optimum temperature stepp using bisection method */
            a = gamma_t + 1e-6;
            b = 1.0;
            p = (a + b)/2.0;
            fp = compute_ESS_diff(p,gamma_t,w_temp,loglike,N);
            err = fabs(fp);
            while (err > 1e-5)
            {
                fa = compute_ESS_diff(a,gamma_t,w_temp,loglike,N);
                if (fa*fp < 0)
                {
                    b = p;
                }
                else
                {
                    a = p;
                }
                p = (a + b)/2.0;
                fp = compute_ESS_diff(p,gamma_t,w_temp,loglike,N);
                err = fabs(fp);
            }
            gamma_tp1 = p;
        }

        fprintf(stderr,"gamma(t) = %g next gamma(t+1) = %g\n",gamma_t,gamma_tp1);
        /* substitute the value of just calculated gamma */
        #pragma omp parallel shared(logw,logw_previous,loglike) firstprivate(gamma_tp1,gamma_t)
        {
            #pragma omp simd
            for (int i=thread_id*NperThread;i<((thread_id+1)*NperThread);i++)
            {
                logw[i] = logw_previous[i] + (gamma_tp1 - gamma_t)*loglike[i];
            }
        }
        /* log sum exp trick for numerical stability */
        w_max = logw[0];
        for (int i=0;i<N;i++)
        {
            w_max = (logw[i] > w_max) ? logw[i] : w_max; 
        }
        w_sum = 0;
        w_sum2 = 0;
        #pragma omp parallel shared(logw,w) firstprivate(w_max) reduction(+:w_sum)
        {
            double w_sum_temp = 0;
            #pragma omp simd reduction(+:w_sum_temp)
            for (int i=thread_id*NperThread;i<((thread_id+1)*NperThread);i++)
            {
                w[i] = logw[i] - w_max;
                w[i] = exp(w[i]);
                w_sum_temp += w[i];
            }
            w_sum += w_sum_temp;
        }
        #pragma omp parallel shared(w) firstprivate(w_sum)
        {
            #pragma omp simd
            for (int i=thread_id*NperThread;i<((thread_id+1)*NperThread);i++)
            {
                w[i] /= w_sum;
            }
        }
        fprintf(stderr,"Re-sampling...");
        /** @todo could be large enough to consider Murray et al.'s methods*/
        /*sampling with replacement according to weights*/
        #pragma omp parallel shared(u)
        {
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,NperThread,
                        u + thread_id*NperThread,0,1);
        }
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
        for (int i=0;i<N;i++)
        {
            loglike_rs[i] = loglike[r[i]];
            logprior_rs[i] = logprior[r[i]];
            for (int j=0;j<NUM_PARAM;j++)
            {
                theta_particle_rs[i*NUM_PARAM + j] = theta_particle[r[i]*NUM_PARAM + j];
            }
        }
        /*copy back*/
        memcpy(loglike,loglike_rs,N*sizeof(double));
        memcpy(logprior,logprior_rs,N*sizeof(double));
        memcpy(theta_particle,theta_particle_rs,NUM_PARAM*N*sizeof(double));

        fprintf(stderr,"done.\n");
        fprintf(stderr,"Detemining optimal scaling...\n");
        
        /*compute covariance of resampled particles */
        cov(theta_particle, N, NUM_PARAM, mu, Sigma);
        memset(T,0,NUM_PARAM*NUM_PARAM*sizeof(double));
        for (int i=0;i<NUM_PARAM;i++)
        {
            for (int j=0;j<=i;j++)
            {
                T[i*NUM_PARAM + j] = Sigma[i*NUM_PARAM + j];
            }
        }
        
        /*cholesky factorise and invert*/
        LAPACKE_dpotrf(LAPACK_ROW_MAJOR,'L',NUM_PARAM,T,NUM_PARAM);
        memcpy(cov_inv,T,NUM_PARAM*NUM_PARAM*sizeof(double));
        LAPACKE_dpotri(LAPACK_ROW_MAJOR,'L',NUM_PARAM,cov_inv,NUM_PARAM);
        
        /* copy lower diagonal to upper*/
        for (int i=0;i<NUM_PARAM;i++)
        {
            for (int j=0;j<i;j++)
            {
                cov_inv[j*NUM_PARAM + i] = cov_inv[i*NUM_PARAM + j];
            }
        }

        /* compute mahalanobis distance before moving */
        fprintf(stderr,"Computing dists...\n");
        #pragma omp parallel shared(theta_particle,dist,cov_inv)
        {
            #pragma omp for schedule(static) 
            for (int i=0;i<N;i++)
            {
                for (int j=0;j<N-VECLEN;j+=VECLEN)
                {
                    for (int k=0;k<VECLEN;k++)
                    {
                        dist[i*N+j+k] = mahalanobis_sq(NUM_PARAM, theta_particle+i*NUM_PARAM,
                                            theta_particle+(j+k)*NUM_PARAM,cov_inv);
                    }
                    #pragma omp simd
                    for (int k=0;k<VECLEN;k++)
                    {
                        dist[i*N +j+k] = sqrt(dist[i*N+j+k]);
                    }
                }
            }
        }
        fprintf(stderr,"Computing quantile...\n");
        median_dist = quantile(N*N,dist,0.5);

        #pragma omp simd
        for (int i=0;i<N;i++)
        {
            h_ind[i] = i;
        }

        /* permute using the Knuth shuffle*/
        #pragma omp parallel shared(u)
        {
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,NperThread,
                        u + thread_id*NperThread,0,1);
        }
        for (int i=0;i<N;i++)
        {
            /* for each element i, let j ~ U(i,n-1) or j = ceil((n-1-i)*u + i), u ~ U(0,1)*/
            int j = (int)ceil((Nf - 1.0 -((double)i))*u[i]+ ((double)i));
            int temp;
            /* swap i,j element*/
            temp = h_ind[i];
            h_ind[i] = h_ind[j];
            h_ind[j] = temp;
        }

        for (int i=0;i<N;i++)
        {
            h_all[i] = h[h_ind[i]%10];
        }
        memset(ESJD,0,N*sizeof(double));
        
        /* MVN RW */
        fprintf(stderr,"trial MCMC steps...\n");
        /*generate N MCMC proposals*/
        memset(mu,0,NUM_PARAM*sizeof(double));
        #pragma omp parallel shared(theta_particle_prop,mu,T,u)
        {
            vdRngGaussianMV(VSL_RNG_METHOD_GAUSSIANMV_BOXMULLER2,stream,NperThread, 
                            theta_particle_prop + thread_id*NperThread*NUM_PARAM, 
                            NUM_PARAM, VSL_MATRIX_STORAGE_FULL,mu,T);
            /* N uniforms for accept/reject step*/
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,NperThread,
                        u + thread_id*NperThread,0,1);
        }
        #pragma omp parallel shared(theta_particle,theta_particle_prop,\
                                prior_u,prior_l,h_all,rate_return,loglike,logprior,\
                                acc_prob,cov_inv,ESJD) \
                             firstprivate(ndat,gamma_tp1)
        {
            for (int i=thread_id*NperThread;i<((thread_id+1)*NperThread-VECLEN);i+=VECLEN)
            {
                __declspec(align(ALIGN)) double loglike_prop[VECLEN];
                __declspec(align(ALIGN)) double logprior_prop[VECLEN];
                __declspec(align(ALIGN)) double log_mh[VECLEN];
                /*generate proposal*/
                for (int j=0;j<NUM_PARAM*VECLEN;j++)
                {
                    theta_particle_prop[i*NUM_PARAM + j] *= h_all[i]; 
                    theta_particle_prop[i*NUM_PARAM + j] += theta_particle[i*NUM_PARAM + j];
                }
                /*compute log prior and log likelihood*/
                for (int j=0;j<VECLEN;j++)
                {
                    logprior_prop[j] = log_prior(theta_particle_prop+(i+j)*NUM_PARAM, prior_l, prior_u);
                    loglike_prop[j] = bege_gjrgarch_likelihood(theta_particle_prop+(i+j)*NUM_PARAM, 
                                         ndat, rate_return, prior_l, prior_u);
                }
                
                #pragma omp simd
                for (int j=0;j<VECLEN;j++)
                {
                    /* compute metropolis-Hastings acceptance probability*/
                    log_mh[j] = gamma_tp1*(loglike_prop[j] - loglike[i+j]) 
                         + logprior_prop[j] - logprior[i+j];
                    acc_prob[i+j] = fmin(exp(log_mh[j]),1.0);
                }

                for (int j=0;j<VECLEN;j++)
                {
                    ESJD[i+j] = mahalanobis_sq(NUM_PARAM, theta_particle + (i+j)*NUM_PARAM,
                                      theta_particle_prop + (i+j)*NUM_PARAM, cov_inv);
                }

                #pragma omp simd
                for (int j=0;j<VECLEN;j++)
                {
                    ESJD[i+j] = sqrt(ESJD[i+j])*acc_prob[i+j];
                }

                for (int k=0;k<VECLEN;k++)
                {
                    unsigned int tf = (u[i+k] <= acc_prob[i+k]);
                    for (int j=0;j<NUM_PARAM;j++)
                    {
                        theta_particle[(i+k)*NUM_PARAM + j] = (tf) ? theta_particle_prop[(i+k)*NUM_PARAM + j] : theta_particle[(i+k)*NUM_PARAM + j];
                    }
                    loglike[i+k] = (tf) ? loglike_prop[k] : loglike[i+k];
                    logprior[i+k] = (tf) ? logprior_prop[k] : logprior[i+k];
                }

            }
        }        

        /* Median value of ESJD for different h indices from 1 to 10*/
        ind = 0;
        for (int j=0;j<10;j++)
        {
            h_N = 0;
            for (int i=0;i<N;i++)
            {
                if (h_ind[i] == j)
                {
                    ESJD_h_tmp[h_N] = ESJD[i];
                    h_N++;
                }
            }
            median_ESJD[j] = quantile(h_N,ESJD_h_tmp,0.5);
            if (median_ESJD[j] > median_ESJD[ind])
            {
                ind = j;
            }
        }

        h_opt = h[ind];
        fprintf(stderr,"The scale is %f\n",h_opt);
        
        memset(dist_move,0,N*sizeof(double));
        belowThreshold = 1;
        R_move = 0;

        
        fprintf(stderr,"MCMC proposals for particle mutation...\n");
        /* Performing remaining repeats */

        while (belowThreshold)
        {
            int sum_cond = 0;
            R_move++;
            /*generate N MCMC proposals*/
            memset(mu,0,NUM_PARAM*sizeof(double));
            #pragma omp parallel shared(theta_particle_prop,mu,T,u)
            {
                vdRngGaussianMV(VSL_RNG_METHOD_GAUSSIANMV_BOXMULLER2,stream,NperThread, 
                                theta_particle_prop + thread_id*NperThread*NUM_PARAM,
                                NUM_PARAM, VSL_MATRIX_STORAGE_FULL,mu,T);
                /* N uniforms for accept/reject step*/
                vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD,stream,NperThread,
                                u + thread_id*NperThread,0,1);
            }
            #pragma omp parallel shared(theta_particle,theta_particle_prop,\
                                    prior_u,prior_l,rate_return,loglike,logprior,\
                                    acc_prob,cov_inv,ESJD,dist_move) \
                                 firstprivate(ndat,gamma_tp1,h_opt) reduction(+:sum_cond)
            {
                int sum_cond_temp = 0;
                for (int i=thread_id*NperThread;i<((thread_id+1)*NperThread-VECLEN);i+=VECLEN)
                {
                    __declspec(align(ALIGN)) double loglike_prop[VECLEN];
                    __declspec(align(ALIGN)) double logprior_prop[VECLEN];
                    __declspec(align(ALIGN)) double log_mh[VECLEN];
                    /*generate proposal*/
                    for (int j=0;j<NUM_PARAM*VECLEN;j++)
                    {
                        theta_particle_prop[i*NUM_PARAM + j] *= h_opt; 
                        theta_particle_prop[i*NUM_PARAM + j] += theta_particle[i*NUM_PARAM + j];
                    }
                    /*compute log prior and log likelihood*/
                    for (int j=0;j<VECLEN;j++)
                    {
                        logprior_prop[j] = log_prior(theta_particle_prop+(i+j)*NUM_PARAM, prior_l, prior_u);
                        loglike_prop[j] = bege_gjrgarch_likelihood(theta_particle_prop+(i+j)*NUM_PARAM, 
                                             ndat, rate_return, prior_l, prior_u);
                    }

                    #pragma omp simd
                    for (int j=0;j<VECLEN;j++)
                    {
                        /* compute metropolis-Hastings acceptance probability*/
                        log_mh[j] = gamma_tp1*(loglike_prop[j] - loglike[i+j]) 
                             + logprior_prop[j] - logprior[i+j];
                        acc_prob[i+j] = fmin(exp(log_mh[j]),1.0);
                    }

                    for (int j=0;j<VECLEN;j++)
                    {
                        ESJD[i+j] = mahalanobis_sq(NUM_PARAM, theta_particle + (i+j)*NUM_PARAM,
                                          theta_particle_prop + (i+j)*NUM_PARAM, cov_inv);
                    }

                    #pragma omp simd
                    for (int j=0;j<VECLEN;j++)
                    {
                        ESJD[i+j] = sqrt(ESJD[i+j])*acc_prob[i+j];
                    }

                    for (int k=0;k<VECLEN;k++)
                    {
                        unsigned int tf = (u[i+k] <= acc_prob[i+k]);
                        for (int j=0;j<NUM_PARAM;j++)
                        {
                            theta_particle[(i+k)*NUM_PARAM + j] = (tf) ? theta_particle_prop[(i+k)*NUM_PARAM + j] : theta_particle[(i+k)*NUM_PARAM + j];
                        }
                        loglike[i+k] = (tf) ? loglike_prop[k] : loglike[i+k];
                        logprior[i+k] = (tf) ? logprior_prop[k] : logprior[i+k];
                        dist_move[i+k] = (tf) ? dist_move[i+k] + ESJD[i+k]/acc_prob[i+k] : dist_move[i+k];
                    }
                
                    #pragma omp simd reduction(+:sum_cond_temp)
                    for (int j=0;j<VECLEN;j++)
                    {
                        sum_cond_temp += (dist_move[i+j] > median_dist);
                    }
                }

                sum_cond += sum_cond_temp;
            } 
            belowThreshold = (sum_cond < (int)ceil(Nf*0.5));
        }
        fprintf(stderr,"The value of R_move was %d\n",R_move);

        gamma_t = gamma_tp1;
    }

    /* transform theta*/
    #pragma omp parallel shared(theta,theta_particle,prior_u,prior_l)
    {
        for (int i=thread_id*NperThread;i<((thread_id+1)*NperThread);i++)
        {
            for (int j=0;j<NUM_PARAM;j++)
            {
                theta[i*NUM_PARAM + j] = (prior_u[j]*exp(theta_particle[i*NUM_PARAM + j]) 
                                    + prior_l[j])/(exp(theta_particle[i*NUM_PARAM + j]) + 1.0);   
 

            }
        }
    }
    /* clean up memory*/
    #pragma omp parallel 
    {
        vslDeleteStream(&stream);
    }
    _mm_free(logw_previous);
    _mm_free(loglike);
    _mm_free(loglike_rs);
    _mm_free(logw);
    _mm_free(w);
    _mm_free(w_temp);
    _mm_free(logprior);
    _mm_free(logprior_rs);
    _mm_free(theta_particle);
    _mm_free(theta_particle_rs);
    _mm_free(theta_particle_prop);
    _mm_free(r);
    _mm_free(h_ind);
    _mm_free(h_all);
    _mm_free(u);
    _mm_free(ESJD);
    _mm_free(acc_prob);
    _mm_free(ESJD_h_tmp);
    _mm_free(dist_move);
}

/**
 * @brief computes the log likelihood of the time series under BEGE-GJR-GARCH
 * dynnamics, given observed data and model parameters.
 */
double
bege_gjrgarch_likelihood(double * restrict theta, int ndat, double * restrict data, 
                         double * restrict prior_l, double * restrict prior_u)
{
    __declspec(align(ALIGN)) double params[NUM_PARAM];
    double r_bar, p_bar, tp, rho_p, phi_pp, phi_pn, n_bar, tn, rho_n, phi_np, phi_nn;
    double loglikelihood, p_t, n_t, t1;
    
    #pragma omp simd
    for (int j=0;j<NUM_PARAM;j++)
    {
        params[j] = (prior_u[j]*exp(theta[j]) + prior_l[j])/(exp(theta[j])+1.0);
    }

    /* setting parameters */
    r_bar = params[10];
    p_bar = params[0];
    tp = params[1];
    rho_p = params[2];
    phi_pp = params[3];
    phi_pn = params[4];
    n_bar = params[5];
    tn = params[6];
    rho_n = params[7];
    phi_np = params[8];
    phi_nn = params[9];

    /* computing the log-likelihood */
    loglikelihood = 0.0;
    t1 = 10e-1;

    p_t = fmax(p_bar/(1.0 -rho_p - (phi_pp + phi_pn)/2.0),t1); 
    n_t = fmax(n_bar/(1.0 -rho_n - (phi_np + phi_nn)/2.0),t1); 

    loglikelihood += loglikedgam_vec(data[0] - r_bar, p_t,n_t,tp,tn,0.001);
    for (int t=1;t<ndat;t++)
    {
        double obs;
        if ((data[t-1] - r_bar) < 0.0)
        {
            p_t = fmax(p_bar + rho_p*p_t + phi_pn*( 
                     ((data[t-1]-r_bar)*(data[t-1]-r_bar))/(2.0*tp*tp)), t1);
            n_t = fmax(n_bar + rho_n*n_t + phi_nn*( 
                     ((data[t-1]-r_bar)*(data[t-1]-r_bar))/(2.0*tn*tn)), t1);
        }
        else
        {
            p_t = fmax(p_bar + rho_p*p_t + phi_pp*( 
                     ((data[t-1]-r_bar)*(data[t-1]-r_bar))/(2.0*tp*tp)), t1);
            n_t = fmax(n_bar + rho_n*n_t + phi_np*( 
                     ((data[t-1]-r_bar)*(data[t-1]-r_bar))/(2.0*tn*tn)), t1);
        }
        loglikelihood += loglikedgam_vec(data[t] - r_bar,p_t,n_t,tp,tn,0.001);
    }

    return loglikelihood;
}


/**
 * @brief Numerically estimates the likelihood of an observation unnder the 
 * BEGE density.
 * @detail Numerically evaluates the CDF of the distribution at two points above 
 * and below the observed data point and then taking a first order finite 
 * difference approximationn (i.e., the pdf is the derivative of the cdf).
 *
 * @param z  the point at which the pdf is evaluated
 * @param p good evironment shape parameter
 * @param n bad evironment shape parameter
 * @param tp good envvironment scale parameter
 * @param tn bad envvironment scale parameter
 * @param zint the finite difference approximation interval. 
 * @note In the application to the monthly US aggregate stock market returns,
 * we have found the value of 0.001 and smaller is reasonablle.
 *
 * @returns the log likelihood of the observation 
 *
 */
double
loglikedgam_vec(double z, double p, double n, double tp, double tn, double zint)
{
    #define NP 100
    __declspec(align(ALIGN)) double zi[2];
    __declspec(align(ALIGN)) double cz[2];
   
    double pz = 0;
    cz[0] = 0; cz[1] = 0;
    zi[0] = z - zint/2.0;
    zi[1] = z + zint/2.0;

    /* define a grid of points for pt over which to integrate*/
    double pmin = -p*tp + 1e-4;
    double pmax = 10.0*sqrt(p)*tp;
    double dp = (pmax-pmin)/((double)NP-1);
    for (int i=0;i<2;i++)
    {
        __declspec(align(ALIGN)) double cp[NP];
        __declspec(align(ALIGN)) double cpm1[NP];
        __declspec(align(ALIGN)) double pp[NP];
        __declspec(align(ALIGN)) double cn[NP];
        __declspec(align(ALIGN)) double pgrid[NP];
        __declspec(align(ALIGN)) double ngrid[NP];
        #pragma omp simd
        for (int j=0;j<NP;j++)
        {
            pgrid[j] = pmin + j*dp;
            /* for each pt, nt must be equal to pt - z*/
            ngrid[j] = pgrid[j] - dp - zi[i];
            pgrid[j] += p*tp;
            ngrid[j] += n*tn;
        }

        /*use vectorise incomplete gamma functions*/
        gammp_vec(p,tp,NP,pgrid,cp);
        gammq_vec(n,tn,NP,ngrid,cn);
       
        cpm1[0] = 0.0;
        #pragma omp simd
        for (int j=1;j<NP;j++)
        {
            cpm1[j] = cp[j-1];
        }

        #pragma omp simd
        for (int j=0;j<NP;j++)
        {
            pp[j] = cp[j] - cpm1[j];
        }


        double _cz = 0.0;
        #pragma omp simd reduction(+:_cz)
        for (int j=0;j<NP;j++)
        {
            _cz += cn[j]*pp[j];
        }
        cz[i] = _cz;
    }
    
    /* finite difference approximation*/
    pz = (cz[1] - cz[0])/zint;
    return log(fmin(fmax(pz,1e-20),1e20));
}

/**
 * @brief Computes the log prior
 *
 * @param phi transformed particle
 * @param prior_l lower prior limit
 * @param prior_u upper prior limit
 * @return log of prior PDF
 *
 * @note takes transformed parameters as inputs
 * @todo will probably vectorise better across all N particles
 */
double
log_prior(double * restrict phi, double * restrict prior_l, double * restrict prior_u)
{
    __declspec(align(ALIGN)) double sumB[24] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.9, 0.5, 0.3, 0.99, 0.5, 0.5, 1.0, 0.3, 0.995, 0.1, 0.75, 0.9, 0.995, 0.995};
    __declspec(align(ALIGN)) double theta[NUM_PARAM];
    __declspec(align(ALIGN)) double logprior;

    /* Transforming back to original scale */
    #pragma omp simd
    for (int j=0;j<NUM_PARAM;j++)
    {
        theta[j] = (prior_u[j]*exp(phi[j]) + prior_l[j])/(exp(phi[j]) + 1.0);
    }

    /** @todo not sure why this condition is needed? */
    unsigned char cond = 1;
    for (int j=0;j<NUM_PARAM;j++)
    {
        cond = cond && (-theta[j] <= sumB[j]);
    }
    for (int j=0;j<NUM_PARAM;j++)
    {
        cond = cond && (theta[j] <= sumB[NUM_PARAM + j]);
    }
    cond = cond && ((1.0*theta[2] + 0.5*theta[3] + 0.5*theta[4]) <= sumB[22]);
    cond = cond && ((1.0*theta[7] + 0.5*theta[8] + 0.5*theta[9]) <= sumB[23]);
    if (cond)
    {
        logprior = 0.0;
        #pragma omp simd
        for (int j=0;j<NUM_PARAM;j++)
        {
            logprior += -phi[j] - 2.0*log(1.0 + exp(-phi[j]));
        }
    }
    else
    {
        logprior = -INFINITY;
    }
    return logprior;
}

/**
 * @brief vectorised ln[Gamma(a_[1:n])]
 */
void
gammln_vec(double * restrict a, double * restrict gln)
{
    __declspec(align(ALIGN)) double cof[6] = {76.18009172947146, -86.50532032941677,
                     24.01409824083091, -1.231739572450155,
                     0.1208650973866179e-2, -0.5395239384953e-5};
    __declspec(align(ALIGN)) double ser[VECLEN];
    __declspec(align(ALIGN)) double y[VECLEN];
    __declspec(align(ALIGN)) double x[VECLEN];
    __declspec(align(ALIGN)) double tmp[VECLEN];
    #pragma omp simd
    for (int i=0;i<VECLEN;i++)
    {
        ser[i] =  1.000000000190015;
    }
    #pragma omp simd
    for (int i=0;i<VECLEN;i++)
    {
        y[i] = a[i];
    }
    #pragma omp simd
    for (int i=0;i<VECLEN;i++)
    {
        x[i] = a[i];
    }
    #pragma omp simd
    for (int i=0;i<VECLEN;i++)
    {
        tmp[i] = x[i] + 5.5;
        tmp[i] -= (x[i] + 0.5)*log(tmp[i]);
    }
    for (int j=0;j<=5;j++)
    {
        #pragma omp simd
        for (int i=0;i<VECLEN;i++)
        {
            ser[i] += cof[j]/(++(y[i]));
        } 
    }
    #pragma omp simd
    for (int i=0;i<VECLEN;i++)
    {
        gln[i] = -tmp[i] + log(2.5066282746310005*ser[i]/x[i]);
    }
}

/**
 * @brief computes P(a[1:n],x[1:n]) = \gamma(a[1:n],x[1:n])/\Gamma(a[1:n]) evaluated by its series 
 * representation. Also returns ln[\Gamma(a[1:n])].
 */
void
gser_vec(double * restrict gamser, double * restrict  a, double * restrict  x, 
     double * restrict gln)
{
    __declspec(align(ALIGN)) double sum[VECLEN]; 
    __declspec(align(ALIGN)) double del[VECLEN]; 
    __declspec(align(ALIGN)) double ap[VECLEN]; 

    gammln_vec(a,gln);

    int s1 = 0;
    #pragma omp simd reduction(+:s1)
    for (int i=0;i<VECLEN;i++)
    {
        s1 += (x[i] <= 0);
    }
    if(s1 == VECLEN)
    {
        #pragma omp simd
        for (int i=0;i<VECLEN;i++)
        {
            gamser[i] = 0.0;
        }
        return;
    }
    else
    {
        
        #pragma omp simd
        for (int i=0;i<VECLEN;i++)
        {
            ap[i] = a[i];
            sum[i] = 1.0/a[i];
            del[i] = sum[i];
        }
        for (int n=1;n<=ITMAX;n++)
        {
            #pragma omp simd
            for (int i=0;i<VECLEN;i++)
            {
                ++(ap[i]);
                del[i] *= x[i]/ap[i];
                sum[i] += del[i];
            }
            int s2 = 0;
            #pragma omp simd reduction(+:s2)
            for (int i=0;i<VECLEN;i++)
            {
                s2 += (fabs(del[i]) < fabs(sum[i])*EPS);
            }
            if (s2 == VECLEN)
            {
                break;
            }
        }
        #pragma omp simd
        for (int i=0;i<VECLEN;i++)
        {
            gamser[i] = (x[i] > 0) ? sum[i]*exp(-x[i] + a[i]*log(x[i]) - (gln[i])) : 0.0;
        }
        return;
    }
}


/**
 * @brief computes P(a,x[1:N]/b) = \gamma(a,x[1:N]/b)/\Gamma(a)
 * @note assumes the array x contains a monotonically increasing sequence.
 */
void
gammp_vec(double a, double b, int N, double * restrict x, double * restrict P)
{
    __declspec(align(ALIGN)) double _a[VECLEN];    
    #pragma omp simd
    for(int i=0;i<VECLEN;i++)
    {
       _a[i] = a;
    }
    /*compute first block with series representation*/
    for (int k=0;k<N-VECLEN;k+=VECLEN)
    {
        __declspec(align(ALIGN)) double gamser[VECLEN];    
        __declspec(align(ALIGN)) double gln[VECLEN];    
        __declspec(align(ALIGN)) double _x[VECLEN];
        #pragma omp simd
        for (int i=0;i<VECLEN;i++)
        {
            _x[i] = x[k+i]/b;
        }
        gser_vec(gamser,_a,_x, gln);
        #pragma omp simd
        for (int i=0;i<VECLEN;i++)
        {
            P[k+i] = gamser[i];
        }
    }
}


/**
 * @brief computes Q(a,x[i]/b) = 1 - P(a,x[i]/b)
 * @note assumes the array x contains a monotonically increasing sequence.
 */
void
gammq_vec(double a, double b, int N, double * restrict x, double * restrict Q)
{
    __declspec(align(ALIGN)) double _a[VECLEN];    
    #pragma omp simd
    for(int i=0;i<VECLEN;i++)
    {
       _a[i] = a;
    }
    
    /*compute first block with series representation*/
    for (int k=0;k<N-VECLEN;k+=VECLEN)
    {
        __declspec(align(ALIGN)) double gamser[VECLEN];    
        __declspec(align(ALIGN)) double gln[VECLEN];    
        __declspec(align(ALIGN)) double _x[VECLEN];
        #pragma omp simd
        for (int i=0;i<VECLEN;i++)
        {
            _x[i] = x[k+i]/b;
        }
        gser_vec(gamser,_a,_x, gln);
        #pragma omp simd
        for (int i=0;i<VECLEN;i++)
        {
            Q[k+i] = 1.0 - gamser[i];
        }
    }
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


