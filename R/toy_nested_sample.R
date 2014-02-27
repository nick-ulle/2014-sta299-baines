# prior.samp     :: function(N,prior.args=list(...),verbose)
# log.likelihood :: function(theta,data,verbose)
# replace.samp   :: function(N,L,prior.samp,prior.args,log.likelihood,data,verbose)

set.the.wd <- TRUE
if (set.the.wd){
  wdir <- "~/Dropbox/Documents/Davis_Teaching/Individual_Study/Nicholas_Ulle/STA299Stuff/R" 
  setwd(wdir)
}
set.seed(2367575)

"toy.evidence.gq" <- function(data,prior.args,n.nodes)
{
  require(statmod)
  # GQ: \int w(x)f(x) \approx \sum w(x_i)*f(x_i)
  # Here w(x) = exp(-x^2), with w(x) corresponding to Hermite polynomials
  mu.grid <- gauss.quad(n=n.nodes,kind="hermite")
  prior.terms <- dnorm(mu.grid$nodes,mean=prior.args$mu_0,sd=sqrt(prior.args$V_0))
  like.terms <- rep(NA,n.nodes)
  ybar <- mean(data) ; n <- length(data)
  for (i in 1:n.nodes){
    like.terms[i] <- prod(dnorm(x=data,mean=mu.grid$nodes[i],sd=1)) # hideously unstable
  }
  renorm.terms <- exp((mu.grid$nodes^2))
  return(log(sum(mu.grid$weights*renorm.terms*prior.terms*like.terms)))
}

#toy.evidence.gq(data=toy.data,prior.args=toy.prior.args,n.nodes=100)

"toy.data.samp" <- function(n,mu,verbose=FALSE)
{
  stopifnot(length(mu)==1)
	# Simulate the y's:
	y <- rnorm(n=n,mean=mu,sd=1)
	# Bag up the parameters:
	return(list("y"=y,"pars"=mu))
}

# Data parameters:
n <- 1000
mu <- c(1.0)

# Prior for mu:
mu.0 <- 0.0
V.0 <- 1.0

test.toy.data <- toy.data.samp(n=n,mu=mu)

"toy.prior.samp" <- function(N,prior.args,verbose)
{
  mu <- rnorm(n=N,mean=prior.args$mu_0,sd=sqrt(prior.args$V_0))
  return(matrix(mu,ncol=1))
}

N <- 100
toy.prior.args <- list("mu_0"=mu.0,"V_0"=V.0)

# Test:
test.toy.prior <- toy.prior.samp(N=N,prior.args=toy.prior.args)

"toy.log.likelihood" <- function(theta,data,verbose=FALSE)
{
  # Likelihood:
  # \prod_{i=1}^{n} \phi(y_{i};\mu)
  "ind.toy.log.likelihood" <- function(x,data)
  {
  	ret <- sum(dnorm(x=data,mean=x,log=TRUE))
  	return(ret)
  }
  ret <- apply(theta,1,ind.toy.log.likelihood,data=data)
  return(ret)
}

# Test:
test.toy.ll <- toy.log.likelihood(theta=test.toy.prior,data=test.toy.data$y)

"toy.replace.samp" <- function(N,L,prior.samp,prior.args,log.likelihood,data,n.abort=100000,verbose=FALSE)
{
  # Sample from the prior:
  done_yet <- FALSE
  nsamples <- 0
  ntries <- 0
  ret <- NULL
  if (verbose){
    cat(paste0("Replacement sampling ",N," samples with threshold ",L,"...\n"))
  }
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
test.toy.replace <- toy.replace.samp(N=N,
	L=max(test.toy.ll),
	prior.samp=toy.prior.samp,
	prior.args=toy.prior.args,
	log.likelihood=toy.log.likelihood,
	data=test.toy.data$y,
  n.abort=n.abort,
	verbose=TRUE)  
})["elapsed"] # ~300 attempts per success, ~ 2 secs per sample

"toy.analytical" <- function(data,prior.args,log=TRUE)
{
  n <- length(data)
  mu_0 <- prior.args$mu_0
  V_0 <- prior.args$V_0
  y <- data
  if (log){
    ret <- -(n/2)*log(2*pi) - 0.5*log(V_0) - 0.5*log(n+(1/V_0)) - 0.5*((mu_0^2)/V_0 + sum(y*y)) + 0.5*((n+(1/V_0))^(-1))*((mu_0/V_0 + sum(y))^2)
  } else {
    ret <- ((2*pi)^(n/2))*((V_0)^0.5)*((n+(1/V_0))^(-0.5))*
      exp(-0.5*((mu_0^2)/V_0 + sum(y*y)) + 0.5*((n+(1/V_0))^(-1))*((mu_0/V_0 + sum(y))^2))
  }
  return(ret)
}

N <- 1000
mu.0 <- 0.0
V.0 <- 1.0
toy.data <- 5 # test.toy.data$y
toy.prior.args <- list("mu_0"=mu.0,"V_0"=V.0)
inc.eps <- 1e-8
grid.method <- "crude"
int.method <- "simple"
store.samples <- TRUE
verbose <- TRUE

source("nested_sampling.R")

toy.nested <- nested.sample(data=toy.data,
                            log.likelihood=toy.log.likelihood,
                            prior.samp=toy.prior.samp,
                            prior.args=toy.prior.args,
                            replace.samp=toy.replace.samp,
                            N=N,
                            inc.eps=inc.eps,
                            grid.method=grid.method,
                            int.method=int.method,
                            store.samples=store.samples,
                            verbose=verbose)

# n = 1000
# -1706

toy.true <- toy.analytical(data=toy.data,prior.args=toy.prior.args,log=TRUE)
toy.true # -1406


toy.gq <- toy.evidence.gq(data=toy.data,prior.args=toy.prior.args,n.nodes=100)
toy.gq

cat(paste0("    Nested sampling estimate of log(Z) = ",toy.nested$logZ,"\n")) 
cat(paste0("           Analytic estimate of log(Z) = ",toy.true,"\n")) 
cat(paste0("Gaussian Quadrature estimate of log(Z) = ",toy.gq,"\n")) 

xmin <- -3
xmax <- 5
for (i in 1:N){
  jpeg(sprintf("nested_sample_%05d.jpg",i))
  plot(function(x){dnorm(x,mean=toy.data,sd=1.0)},xmin,xmax,ylab="p(x)",main=paste0("Iteration = ",i))
  plot(function(x){dnorm(x,mean=toy.prior.args$mu_0,sd=toy.prior.args$V_0)},xmin,xmax,col="red",add=T)
  points(y=rep(0.0,N),x=toy.nested$posterior.samples[[i]],pch=3)
  dev.off()
}

dim(toy.nested$posterior.samples[[1]])


# Visualize some stuff (not sure what yet...)
plot(toy.nested$L,type="l")

#plot(mcmc(t(apply(toy.nested$theta[,1:3],1,sort))))
#plot(mcmc(t(apply(toy.nested$theta[,4:6],1,sort))))
test.toy.data$pars
