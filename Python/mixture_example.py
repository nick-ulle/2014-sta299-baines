#!/usr/bin/env python

import collections
import functools as ft
import argparse as ap

import numpy as np
import nested_sampling as ns

Model = collections.namedtuple('Model', 'mu0 tau0 alpha')
InitialValues = collections.namedtuple('Model', 'mu p')

def gibbs_sampler(y, init, model, max_iter):
    ''' Runs a Gibbs sampler for the mixture model posterior.

    Args:
        y           data vector
        init        InitialValues object
        model       Model object
        max_iter    maximum number of iterations to run

    Returns:
        mu, p   samples from posterior mu and p, respectively
    '''
    # Iterate through the three conditional distributions, sampling from each.
    # Store the resulting chain.

    # Variables:
    #   I   class indicators (n by K)
    #   mu  class means (max_iter by K)
    #   p   class probabilities (max_iter by K)
    
    #   i   indexes observations, i = 0, ..., n - 1
    #   j   indexes classes (mixture terms), j = 0, ..., K - 1
    
    # Get dimensions.
    n = y.size
    k = model.alpha.size

    # Resize y to (n by K) for faster calculations.
    y = np.repeat(y[:, None], k, axis = 1)

    # Set up starting values.
    mu = np.empty((max_iter, k))
    mu[0, :] = init.mu

    p = np.empty((max_iter, k))
    p[0, :] = init.p

    I = np.empty((n, k))

    # Run the Gibbs sampler.
    # TODO Vectorize more of this loop's internals.
    for iter in range(1, max_iter):
        # First, sample n observations from the conditional posterior of I.
        probs = np.exp(-0.5 * (y - mu[iter - 1, :])**2) * p[iter - 1, :]
        probs = probs / np.sum(probs, axis = 1)[:, None]
        for i in range(0, n):
            I[i, :] = np.random.multinomial(1, probs[i, :])

        # Next up, sample from the conditional posterior of mu.
        nn = np.sum(I, axis = 0)

        for j in range(0, k):
            # Calculate mean of the observations in the class.
            if nn[j] == 0:
                mean_j = 0.0
            else:
                mean_j = np.mean( y[I[:, j] == 1, 0] )

            # Calculate posterior mean and variance, then sample.
            variance = model.tau0 / (1.0 + model.tau0 * nn[j])
            mean = (model.mu0 / model.tau0 + nn[j] * mean_j) * variance

            mu[iter, j] = np.random.normal(mean, np.sqrt(variance))

        if (iter % 500) == 0:
            print('Completed iteration {}.'.format(iter))

        # Finally, sample from the conditional posterior of p.
        p[iter, :] = np.random.dirichlet(model.alpha + nn)

    return mu, p

def log_like(y, mu, p):
    # Repeat y to speed up computation.
    y = np.repeat(y[:, None], mu.size, axis = 1)

    # TODO Use numerically stable log-addition.
    ll = np.log( np.sum(np.exp(-0.5 * (y - mu)**2) * p, axis = 1) )

    return np.sum(ll)

def est_z1(y, mu, p):
    ''' Implements Z_1 evidence estimator described in documentation.
    '''
    m, _ = mu.shape
    z = float('-Inf')
    for i in range(0, m):
        # Calculate log( 1 / f(x | mu, p) ) = -log( f(x | mu, p) ).
        term = -log_like(y, mu[i, :], p[i, :])
        # Accumulate sum of inverse log-likelihoods.
        z = np.logaddexp(z, term)

    # Take the average (log-scale).
    z -= np.log(m)
    # Invert and return.
    return -z

def main():
    parser = ap.ArgumentParser(description = 'Run the nested sampling mixture'
                               + ' model example.')
    parser.add_argument('-s', '--sample', action = 'store_true',
                        help = 'Sample from the mixture model posterior.')
    args = parser.parse_args()

    if args.sample:
        # Run the sampler.
        np.random.seed(100)

        model = Model(mu0 = 0.0, 
                      tau0 = 1.0,
                      alpha = np.array([1.0, 1.0, 1.0]))
        init = InitialValues(mu = np.array([-1.0, 0.0, 1.0]),
                             p = np.array([1.0, 1.0, 1.0]) / 3.0)

        # Load the data.
        y = np.loadtxt('../sim_data_k_3.txt')

        mu, p = gibbs_sampler(y, init, model, 15000)

        np.savetxt('../out/mu.txt', mu)
        np.savetxt('../out/p.txt', p)
    else:
        # Do post-processing.
        y = np.loadtxt('../sim_data_k_3.txt')
        mu = np.loadtxt('../out/mu.txt')
        p = np.loadtxt('../out/p.txt')

        z1 = est_z1(y, mu, p)
        print('Estimated log evidence Z1 = {}'.format(z1))

if __name__ == '__main__':
    main()
