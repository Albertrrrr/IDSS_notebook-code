import numpy as np


def metropolis(fx, q, x_init, n):
    # Perform Metropolis MCMC sampling.
    # p(x): a function that can be evaluated anywhere. p(x) returns the value of p at x
    # q(): a function q that draws a sample from a symmetric distribution and returns it
    # x_init: a starting point
    # n: number of samples
    x = x_init

    samples = []
    rejected = []  # we only keep the rejected samples to plot them later
    for i in range(n):
        # find a new candidate spot to jump to
        x_prime = q(x)
        p_r = fx(x_prime) / fx(x)
        r = np.random.uniform(0, 1)
        # if it's better, go right away
        if r < p_r:
            x = x_prime
            samples.append(x_prime)
        else:
            samples.append(x)
            rejected.append(x_prime)

    return np.array(samples), np.array(rejected)


def log_metropolis(log_fx, q, x_init, n):
    # Perform Metropolis MCMC sampling.
    # log_fx(x): a function that can be evaluated anywhere.
    # p(x) returns the log probability of at x
    # q(x): a function q that draws a sample from a symmetric distribution and returns it
    # x_init: a starting point
    # n: number of samples
    x = x_init
    samples = []
    accepts = 0
    for i in range(n):
        # find a new candidate spot to jump to
        x_prime = q(x)
        p_r = log_fx(x_prime) - log_fx(x)
        r = np.random.uniform(0, 1)
        if r < np.exp(p_r):
            x = x_prime
            samples.append(x_prime)
            accepts += 1
        else:
            samples.append(x)
    print(f"Acceptance ratio: {accepts/n:.3f}")
    return np.array(samples)
