
"ls_add" <- function(x,y)
{
	if (x>y){
		return(x+log(1.0+exp(y-x)))
	} else {
		return(y+log(1.0+exp(x-y)))
	}
	return()
}

"nested.sample" <- function(data,log.likelihood,prior.samp,prior.args,replace.samp,
                            N=1000,inc.eps=1e-8,
                            grid.method=c("crude","sample"),
                            int.method=c("simple","trapezoidal"),
                            store.samples=FALSE,
                            verbose=FALSE)
{
  # Arguments:
  # ==========
  # prior.samp     :: function(N,prior.args=list(...),verbose)
  # log.likelihood :: function(theta,data,verbose)
  # replace.samp   :: function(N,L,prior.samp,prior.args,log.likelihood,data,verbose)
  
  # Sample N points from the prior:
  if (verbose){
  	cat("Sampling from prior...\n")
  }
  theta.set <- prior.samp(N,prior.args,verbose)
  if (store.samples){
  	posterior.samples <- vector("list",N)
  }

  # Compute log-likelihood for sampled points:
  if (verbose){
  	cat("Evaluating log-likelihood...\n")
  }
  theta.loglike <- log.likelihood(theta.set,data,verbose)
  
  # Initialize:
  logZ <- -Inf
  logW <- rep(NA,N)
  lw <- rep(NA,N)
  X <- rep(NA,N)
  L <- rep(NA,N)
  I <- rep(NA,N)

  logw <- log(1.0 - exp(-1.0/N))
  X[1] <- 1.0
  H <- 0.0

  if (verbose){
  	cat("Beginning nested sampling loop...\n")
  }

  for (i in 1:N){
    
    # Store bin:
    lw[i] <- logw
    
    # Find lowest value point:
    k <- which.min(theta.loglike)
    L[i] <- theta.loglike[k]

    # Compute X_{i}:
    if ("crude" %in% grid.method){
      # Crude method:
      X[i+1] <- exp(-i/N)
    } else if ("sample" %in% grid.method){
      # Sample...
      stop("'grid.method'='sample' not yet implemented!")
    } else {
      stop("Invalid 'grid.method'")
    }
    
    # The integral:
    if ("simple" %in% int.method){
      # Simple method:
      #w[i] <- X[i] - X[i+1]
    } else if ("trapezoidal" %in% int.method){
      # Trapezoidal:
      stop("'int.method'=='trapezoidal' not yet implemented!")
    } else{
      stop("Invalid 'int.method' supplied")
    }
  
    # Increment the integral (converting log-likelihood back to likelihood):
    log_inc <- logw + L[i]
    logZnew <- ls_add(logZ,log_inc)
    # Store the log-increment:
    I[i] <- log_inc
    # The information:
    if (is.infinite(logZ) && logZ<0.0){
    	# Avoid first iteration (-Inf)*exp(-Inf)
    	H <- exp(log_inc-logZnew)*L[i] - logZnew
    } else {
    	H <- exp(log_inc-logZnew)*L[i] + exp(logZ-logZnew)*(H+logZ) - logZnew
    }
    # Update Z:
    logZ <- logZnew
    # Store samples:
    if (store.samples){
    	posterior.samples[[i]] <- theta.set
    }

    # Check for sufficient accuracy:
    #if (inc < inc.eps){
    #  # Converged...
    #  break
    #}
    
    # Replace lowest point with a new one sampled with constraint:
    if (verbose){
    	cat("Sampling replacement values...\n")
    }
    theta.set[k,] <- replace.samp(N=1,L[i],prior.samp=prior.samp,prior.args=prior.args,log.likelihood=log.likelihood,data=data,verbose=verbose)
    
    # Compute log-likelihood for newly sampled point:
    if (verbose){
    	cat("Evaluating new log-likelihood...\n")
    }
    theta.loglike[k] <- log.likelihood(theta=theta.set[k,,drop=FALSE],data=data,verbose=verbose)
    
    # Shrink the interval:
    logw <- logw - 1.0/N
    
    if (verbose){
    	cat(paste0("Finished i=",i,"/",N,"...\n"))
    }

  }
  
  # Final patch:
  j <- i
  #logZ <- ls_add(logZ,sum(L)*X[j+1]/N

  # Return:  
  ret <- list("logZ"=logZ,"steps"=j,"X"=X,"L"=L,"logw"=lw, 
              "I"=I,"H"=H,"N"=N,"err"=sqrt(H/N))
  if (store.samples){
  	ret$posterior.samples <- posterior.samples
  }
  return(ret)
}
