#!/usr/bin/env python

import collections

import numpy as np
import nested_sampling as ns

Model = collections.namedtuple('Model', 'mu0 tau0 alpha0')
InitialValues = collections.namedtuple('Model', 'mu p')

def gibbs_sampler(y, init, model, max_iter):
    # Iterate through the three conditional distributions, sampling from each.
    # Store the resulting chain.

    # Variables:
    #   I   class indicators (n by K)
    #   mu  class means (max_iter by K)
    #   p   class probabilities (max_iter by K)
    
    #   i   indexes observations, i = 0, ..., n - 1
    #   j   indexes classes (mixture terms), j = 0, ..., K - 1
    
    # Initialize all of the variables.
    n = y.size
    k = init.p.size

    mu = np.empty((max_iter, k))
    mu[0, :] = init.mu

    p = np.empty((max_iter, k))
    p[0, :] = init.p

    # Run the Gibbs sampler.
    for iter in range(1, max_iter):
        # First, sample n observations from the conditional posterior of I.
        I = np.empty((n, k))
        for i in range(0, n):
            # Calculate posterior probabilities, then sample.
            prob = np.exp(-0.5 * (y[i] - mu[iter - 1, :])**2) * p[iter - 1, :]
            prob = prob / np.sum(prob)

            I[i, :] = np.random.multinomial(1, prob)

        # Next up, sample from the conditional posterior of mu.
        nn = np.sum(I, axis = 0)
        for j in range(0, k):
            # Calculate mean of the observations in the class.
            if nn[j] == 0:
                mean_j = 0
                print('Class {} empty on iteration {}!'.format(j, iter))
            else:
                class_j = y[ I[:, j] == 1 ]
                mean_j = np.mean(class_j)

            # Calculate posterior mean and variance, then sample.
            variance = 1 / (1 / model.tau0 + nn[j])
            mean = (model.mu0 / model.tau0 + nn[j] * mean_j) * variance

            if variance < 0:
                print('Erroneous variance {}'.format(variance))
                print('nn is {}'.format(nn))

            mu[iter, j] = np.random.normal(mean, np.sqrt(variance))

        if (iter % 500) == 0:
            print('Completed iteration {}.'.format(iter))

        # Finally, sample from the conditional posterior of p.
        p[iter, :] = np.random.dirichlet(model.alpha0 + nn)

    return mu, p


def main():
    np.random.seed(100)

    model = Model(mu0 = 0.0, 
                  tau0 = 2.0,
                  alpha0 = np.array([1.0, 1.0, 2.0]))
    init = InitialValues(mu = np.array([0.0, 0.0, 0.0]),
                         p = np.array([1.0, 1.0, 1.0]) / 3.0)

    # Load the data.
    y = np.loadtxt('../sim_data_k_3.txt')

    mu, p = gibbs_sampler(y, init, model, 10000)

    np.savetxt('../out/mu.txt', mu)
    np.savetxt('../out/p.txt', p)

if __name__ == '__main__':
    main()
