# prior.samp     :: function(N,prior.args=list(...),verbose)
# log.likelihood :: function(theta,data,verbose)
# replace.samp   :: function(N,L,prior.samp,prior.args,log.likelihood,data,verbose)


"mix.data.samp" <- function(n,mu,sigma,alpha,verbose=FALSE)
{
  require(MCMCpack)
	K <- length(mu)
	# Simulate mixture proportions:
	p <- rdirichlet(n=1,alpha=alpha)
	# Simulate mixture indicators:
	I <- rmultinom(n=n,size=1,prob=p)
	I.vec <- apply(I,2,function(x){which(x==1)})
	# Simulate the y's:
	y <- rnorm(n=n,mean=mu[I.vec],sd=sigma[I.vec])
	# Bag up the parameters:
	pars <- c(K,p,mu)
	names(pars) <- c("K",paste0("p_",1:K),paste0("mu_",1:K))
	return(list("y"=y,"pars"=pars))
}

n <- 1000
mu <- c(-1.5,1.5)
sigma <- c(1.0,1.0)
alpha <- c(1.0,2.0)
K <- length(mu)
# mu <- c(-1.5,0.0,1.5)
# sigma <- c(1.0,1.0,1.0)
# alpha <- c(1.0,1.0,2.0)

test.mix.data <- mix.data.samp(n=n,mu=mu,sigma=sigma,alpha=alpha)

"mix.prior.samp" <- function(N,prior.args,verbose)
{
  require(MCMCpack) # for rdirichlet
  require(mvtnorm)  # for rmvnorm
  K <- length(prior.args$alpha_0)
  p <- rdirichlet(N,prior.args$alpha_0)
  mu <- rmvnorm(n=N,mean=mu_0,sigma=prior.args$V_0)
  pars <- cbind(p,mu)
  colnames(pars) <- c(paste0("p_",1:K),paste0("mu_",1:K))
  return(pars)
}

N <- 100
alpha_0 <- rep(2,K)
mu_0 <- rep(0,K)
V_0 <- diag(rep(1,K))
mix.prior.args <- list("mu_0"=mu_0,"V_0"=V_0,"alpha_0"=alpha_0)

# Test:
test.mix.prior <- mix.prior.samp(N=N,prior.args=mix.prior.args)

"mix.log.likelihood" <- function(theta,data,verbose=FALSE)
{
  # Likelihood:
  # \prod_{i=1}^{n} \sum_{j=1}^{K} p_{j} \phi(y_{i};\mu_{j})
  # = \sum_{i=1}^{n} \log( \sum_{j=1}^{K} p_{j}\phi(y_{i};\mu_{j}) )
  "ind.mix.log.likelihood" <- function(x,data,K)
  {
  	mix_terms <- matrix(NA,nrow=length(data),ncol=K)
  	for (j in 1:K){
  		mix_terms[,j] <- x[j]*dnorm(x=data,mean=x[K+j],sd=1.0,log=FALSE)
  	}
  	ret <- sum(log(apply(mix_terms,1,sum)))
  	return(ret)
  }
  K <- ncol(theta)/2
  if (abs(K-as.integer(K))>.Machine$double.eps){
  	stop("'theta' must have 2K columns")
  }
  ret <- apply(theta,1,ind.mix.log.likelihood,data=data,K=K)
  return(ret)
}

# Test:
test.mix.ll <- mix.log.likelihood(theta=test.mix.prior,data=test.mix.data$y)

"mix.replace.samp" <- function(N,L,prior.samp,prior.args,log.likelihood,data,n.abort=100000,verbose=FALSE)
{
  # Sample from the prior:
  done_yet <- FALSE
  nsamples <- 0
  ntries <- 0
  ret <- NULL
  while(!done_yet){
  	# Sample from prior:
  	ntries <- ntries+1
  	tmp <- prior.samp(N=1,prior.args=prior.args)
  	# Evaluate likelihood:
  	tmp.ll <- log.likelihood(theta=tmp,data=data)
  	# Check if it is above the threshold:
  	if (tmp.ll > L){
  		# Stash the sample:
  		ret <- rbind(ret,tmp)
  		nsamples <- nsamples+1
  		if (nsamples >= N){
  			done_yet <- TRUE
  		}
  	}
  	if (ntries >= n.abort){
  		stop("Took too many tries to generate replacement samples: bailing...")
  	}
  }
  if (verbose){
  	cat(paste0("Replacement sampling took ",ntries," attempts to generate ",nsamples," samples...\n"))
  }
  return(ret)
}

# Test:
N <- 10
n.abort <- 20000
verbose <- TRUE

system.time({
test.mix.replace <- mix.replace.samp(N=N,
	L=max(test.mix.ll),
	prior.samp=mix.prior.samp,
	prior.args=mix.prior.args,
	log.likelihood=mix.log.likelihood,
	data=test.mix.data$y,
  n.abort=n.abort,
	verbose=TRUE)  
})["elapsed"] # ~300 attempts per success, ~ 2 secs per sample

N <- 1000
inc.eps <- 1e-8
grid.method <- "crude"
int.method <- "simple"
store.samples <- TRUE
verbose <- TRUE

source("nested_sampling.R")

mix.nested <- nested.sample(data=test.mix.data$y,
                            log.likelihood=mix.log.likelihood,
                            prior.samp=mix.prior.samp,
                            prior.args=mix.prior.args,
                            replace.samp=mix.replace.samp,
                            N=N,
                            inc.eps=inc.eps,
                            grid.method=grid.method,
                            int.method=int.method,
                            store.samples=store.samples,
                            verbose=verbose)

# Visualize some stuff (not sure what yet...)
plot(mix.nested$L,type="l")

#plot(mcmc(t(apply(mix.nested$theta[,1:3],1,sort))))
#plot(mcmc(t(apply(mix.nested$theta[,4:6],1,sort))))
test.mix.data$pars
