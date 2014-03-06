#!/usr/bin/env python

import numpy as np

def rejection_sample(cutoff, propose, log_likelihood):
    '''Sample from a (truncated) distribution using rejection sampling.

    Args:
        cutoff          threshold sampled values must exceed
        propose         function which returns a proposal
        log_likelihood  vectorized log-likelihood function
    '''
    for i in range(1, 100001): 
        proposal = propose(1)
        if log_likelihood(proposal) > cutoff:
            break
        elif i == 100000:
            raise Exception("Rejection sampling failed after " +
                            "{} iterations.\nCutoff {}.".format(i, cutoff))

    return log_likelihood(proposal)

def nested_sample(log_likelihood, prior_sample, npts, niter):
    '''Evaluate evidence for a given model using nested sampling.

    Args:
        log_likelihood  vectorized log-likelihood function (ln L)
        prior_sample    function to sample from prior
        npts            number of prior points at each step (N)
        niter           number of iterative steps (j)
    '''

    # Sample initial likelihood points.
    lpts = log_likelihood(prior_sample(npts))

    # Initialize starting parameters.
    evidence = -float("Inf")

    for i in range(0, niter):
        # Get smallest likelihood point.
        id = np.argmin(lpts)
        smallest = lpts[id]

        # Calculate weight.
        wt = np.exp(-i / npts) - np.exp(-(i + 2) / npts)
        wt = np.log(0.5 * wt)

        # Increment evidence by weighted likelihood point.
        evidence_incr = wt + smallest # width * likelihood
        evidence = log_sum(evidence_incr, evidence)

        # Sample a new likelihood point to replace the old one.
        lpts[id] = rejection_sample(smallest, prior_sample, log_likelihood)

        if (i + 1) % 100 == 0:
            print("Finished iteration {} of {} with threshold {:.4f}."
                  .format(i + 1, niter, smallest))

    # Add final correction to evidence.
    for i in range(0, npts):
        evidence_incr = -np.log(npts) + lpts[i] - (niter / npts)
        evidence = log_sum(evidence_incr, evidence)

    return evidence

def log_sum(x, y):
    '''Compute a logarithmic sum in a numerically stable way.
    '''
    if x > y:
        return x + np.log(1 + np.exp(y - x))
    else:
        return y + np.log(1 + np.exp(x - y))

def toy_example():
    '''Run the toy example.
    '''
    # Initialization:
    sigma2 = 1.
    mu_0 = 0.
    tau2_0 = 1.
    y = 5

    def toy_prior_sample(n):
        return np.random.normal(mu_0, tau2_0, n)

    def toy_log_like(mu):
        return -0.5 * np.log(2 * np.pi) - 0.5 * (y - mu)**2 / sigma2

    evidence = nested_sample(toy_log_like, toy_prior_sample, 200, 1500)
    print(4 * ' ' + "Nested sampling estimate of log(Z) = {}".format(evidence))

def main():
    toy_example()

if __name__ == '__main__':
    main()

