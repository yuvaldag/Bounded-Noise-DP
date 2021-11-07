# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 20:27:43 2021

@author: Yuval
"""

import numpy as np
from numpy import log, exp
from scipy.stats import norm
from bounded_noise import realBinarySearch

def deltaExact(k, sigma, eps):
    '''
    Given k and sigma, find the minimal delta such that the Gaussian mechanism
    that answers k queries with Gaussian noise N(0,sigma**2) is (eps, delta)-DP

    Parameters
    ----------
    k : natural number
        number of queries
    sigma : positive number
        Standard deviation of the Gaussian mechanism
    eps : positive number
        Privacy parameter

    Returns
    -------
    Positive number
        The minimal delta as explained above

    '''
    sens = np.sqrt(k)
    return (
        norm.cdf(sens/(2*sigma) - eps*sigma/sens) 
        - exp(eps) * norm.cdf(-sens/(2*sigma) - eps*sigma/sens)        
        )

def getSigmaExact(eps, delta, k):
    '''
    Given epsilon, delta and k, find the minimal value of sigma such that the
    Gaussian mechanism with variance sigma**2 is (eps,delta)-DP for answering
    k 1-sensitive queries

    Parameters
    ----------
    eps : nonnegative number
        Privacy parameter
    delta : positive number
        Privacy parameter
    k : integer
        Number of queries

    Returns
    -------
    sigma : positive
        The standard deviation of the mechanism defined above

    '''
    deltaFun = lambda sigma: -log(deltaExact(k,sigma,eps))
    sigma = realBinarySearch(deltaFun, -log(delta), 0, np.inf, multCond=True)[0]
    return sigma

def gaussHighProb(sigma, beta, k):
    '''
    Given a standard deviation sigma and x_1,...,x_k drawn iid N(0,sigma**2),
    outputs the value t such that
    Pr[max_{i=1}^k |g_i| > t] = beta.

    Parameters
    ----------
    sigma : positive
        Gaussian standard deviation
    beta : number in (0,1)
        A deviation probability
    k : natural numbr
        A number of queries

    Returns
    -------
    t : positive
        The threshold definded above

    '''
    quantile = 2*(1-beta)**(1/k)-1
    return sigma * norm.ppf(quantile)


def findGaussMaxErr(eps, k, delta, beta):
    '''
    Given (eps,delta) and k, finds the number t, such that the
    gaussian mechanism which is (eps,delta)-DP for answering k 1-sensitive queries,
    has all noises bounded by t with probability 1-beta

    Parameters
    ----------
    eps : nonnegative
        Privacy parameter
    k : natural number
        Number of queries
    delta : number in (0,1)
        Privacy parameter
    beta : number in (0,1)
        Failure probability

    Returns
    -------
    t : positive
        The number defined above

    '''
    sigma = getSigmaExact(eps,delta,k)
    return gaussHighProb(sigma, beta, k)