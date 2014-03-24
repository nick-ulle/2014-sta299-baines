#!/usr/bin/env python

import numpy as np

def rejection_sample(threshold, propose, log_likelihood):
    '''Sample from a (truncated) distribution using rejection sampling.

    Args:
        threshold       threshold sampled values must exceed
        propose         function which returns a proposal
        log_likelihood  vectorized log-likelihood function
    '''
    for i in range(1, 100001): 
        proposal = propose(1)
        if log_likelihood(proposal) > threshold:
            break
        elif i == 100000:
            raise Exception("Rejection sampling failed after " +
                            "{} iterations.\n".format(i) +
                            "Threshold {}.".format(threshold))
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

        # Calculate trapezoidal weights.
        wt = np.exp(-i / npts) - np.exp(-(i + 2) / npts)
        wt = np.log(0.5 * wt)

        # Increment evidence by weighted likelihood point.
        evidence_incr = wt + smallest # width * likelihood
        evidence = np.logaddexp(evidence_incr, evidence)

        # Sample a new likelihood point to replace the old one.
        lpts[id] = rejection_sample(smallest, prior_sample, log_likelihood)

        if (i + 1) % 100 == 0:
            print("Finished iteration {} of {} with threshold {:.4f}."
                  .format(i + 1, niter, smallest))

    # Add final correction to evidence.
    for i in range(0, npts):
        evidence_incr = -np.log(npts) + lpts[i] - (niter / npts)
        evidence = np.logaddexp(evidence_incr, evidence)

    return evidence

def toy_example():
    '''Run the toy example.
    '''
    # Initialization:
    np.random.seed(1220)

    sigma2 = 1.0
    mu_0 = 0.0
    tau2_0 = 1.0
    y = np.array([5.0])

    def toy_prior_sample(n):
        return np.random.normal(mu_0, tau2_0, n)

    def toy_log_like(mu):
        out = 0
        for y_ in y:
            out += -0.5 * (np.log(2 * np.pi * sigma2) + (y_ - mu)**2 / sigma2)
        return out

    evidence = nested_sample(toy_log_like, toy_prior_sample, 200, 1500)
    print(4 * ' ' + "Nested sampling estimate of log(Z) = {}".format(evidence))

def main():
    toy_example()

if __name__ == '__main__':
    main()

