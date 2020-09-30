library(tictoc)
library(doParallel)

# optimized R-base implementation of the toggle switch model
simulate_toggle_switch_vec <- function(mu,sigma,gam, alpha, beta, T, C) {
    
    # create ouput array
    u_t <- numeric(C)
    v_t <- numeric(C)
    
    # initialise
    alpha_u <- alpha[1]
    alpha_v <- alpha[2]
    beta_u <- beta[1]
    beta_v <- beta[2]
    u_t[1:C] <- 10
    v_t[1:C] <- 10
    # generate random variates
    zeta <- matrix(nrow=C,ncol=2*(T-1)+1) 
    zeta[,] <- rnorm(C*(2*(T-1)+1),0,1)
    # evolve all cells together
    for (j in 2:T) {
        p_u <- v_t^beta_u
        p_v <- u_t^beta_v
        u_t <- 0.97*u_t + alpha_u/(1+p_u) - 1.0 + 0.5*zeta[1:C,2*(j-1)]
        v_t <- 0.97*v_t + alpha_v/(1+p_v) - 1.0 + 0.5*zeta[1:C,2*(j-1) + 1]
        u_t[u_t < 1.0] <- 1.0;
        v_t[v_t < 1.0] <- 1.0;
        
    }
    y <- u_t + sigma*mu*zeta[1:C,1]/(u_t^gam)
    y[y < 1.0] <- 1.0
    return(y)
}

# prior predictive sampling 
T <- 600
C <- 8000
N <- 8064

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
    
        c(theta,simulate_toggle_switch_vec(mu,sigma,gam,alpha,beta,T,C))
    }
    print(c(P,N,C,T))
    toc()
    stopCluster(cl)
}
