
library(MCMCpack) # for rdirichlet

set.seed(1312313)

# Simulate from a mixture of normals:

n <- 1000
mu <- c(-1.5,0.0,1.5)
sigma <- c(1.0,1.0,1.0)
K <- length(mu)

alpha <- c(1.0,1.0,2.0)

# Simulate mixture proportions:
p <- rdirichlet(n=1,alpha=alpha)

# Simulate mixture indicators:
I <- rmultinom(n=n,size=1,prob=p)
I.vec <- apply(I,2,function(x){which(x==1)})

# Simulate the y's:
y <- rnorm(n=n,mean=mu[I.vec],sd=sigma[I.vec])

# Bag up the parameters:
pars <- c(K,p,mu)
names(pars) <- c("K",paste0("p_",1:3),paste0("mu_",1:3))

densplot(mcmc(y)) # just to visualize

write.table(y,file=paste0("sim_data_k_",K,".txt"),row.names=F,col.names=F)
write.table(data.frame(pars),file=paste0("sim_pars_k_",K,".txt"),row.names=T,col.names=F,quote=T)



