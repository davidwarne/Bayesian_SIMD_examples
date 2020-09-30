library(tictoc)
library(doParallel)

# Naive implementation of toggle switch simulation
simulate_toggle_switch <- function(mu,sigma,gam, alpha, beta, T, C) {
    
    # create ouput array
    y <- numeric(C)
    # evolve each cell one-by-one
    for (i in 1:C) {
        # initialise
        alpha_u <- alpha[1]
        alpha_v <- alpha[2]
        beta_u <- beta[1]
        beta_v <- beta[2]
        u_t <- 10
        v_t <- 10
        for (j in 2:T){
            p_u <- v_t^beta_u
            p_v <- u_t^beta_v
            u_t <- 0.97*u_t + alpha_u/(1+p_u) - 1.0 + 0.5*rnorm(1,0,1)
            v_t <- 0.97*v_t + alpha_v/(1+p_v) - 1.0 + 0.5*rnorm(1,0,1)
            if (u_t < 1.0 ){
                u_t <- 1.0
            }
            if (v_t < 1.0) {
                v_t <- 1.0
            }
        }
        y[i] <- u_t + sigma*mu*rnorm(1,0,1)/(u_t^gam)
        if (y[i] < 1.0) {
            y <- 1.0
        }
    }
    return(y)
}

# prior predictive sampling 
T <- 600
C <- 8000
N <- 240

for (P in cores) {
    # set up level of parallelism
    cl <- makeCluster(P)
    registerDoParallel(cl)
    # run optimised R simulations
    tic()
    obs_vals2 <- foreach(k = 1:N) %dopar%  {
        theta <- runif(7,c(250.0,0.05,0.05,0.0,0.0,0.0,0.0),
                         c(400.0,0.5,0.35,50.0,50.0,7.0,7.0))
        mu <- theta[1]
        sigma <- theta[2]
        gam <- theta[3]
        alpha <- theta[4:5]
        beta <- theta[6:7]
    
        c(theta,simulate_toggle_switch(mu,sigma,gam,alpha,beta,T,C))
    }
    print(c(P,N,C,T))
    toc()
    stopCluster(cl)
}
