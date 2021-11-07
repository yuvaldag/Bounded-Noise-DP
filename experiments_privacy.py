# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 20:36:36 2021

@author: Yuval
"""

'''
Simulations - privacy
'''

import numpy as np
import matplotlib.pyplot as plt
import time
from bounded_noise import Prob, pFunc, expFunc, optimalNoise, truncThresh
from gaussian_mechanism import getSigmaExact, findGaussMaxErr, gaussHighProb
from numpy import log, exp, sqrt


def textToPr(prText):
    '''
    Given a text description of a bounded-noise mechanism, translate it to
    a Prob object

    Parameters
    ----------
    prText : str
        A text description of a mechanism

    Returns
    -------
    Prob
        The corresponding Prob object

    '''
    if prText[:5] == 'pFunc':
        num = float(prText[5:])
        return Prob(pFunc(num))
    elif prText[:7] == 'expFunc':
        c = float(prText[7:])
        return Prob(expFunc(c))
    else:
        print('unidentified probability function {}'.format(prText))
        raise

def testOneVal(k, epsilon, delta, prText, accurate, beta=0.05, gaussStd=False):
    '''
    Tests the optimal noise in one setting, comparing a bounded-noise mechanism
    to the Gaussian mechanism.

    Parameters
    ----------
    k : positive integer
        number of queries
    epsilon : positive
        privacy parameter
    delta : in (0,1)
        privacy parameter
    prText : str
        A string description of the bounded noise mechanism.
        Examples: 
            'pFunc1': probability is proportional to exp(-f(x)), here
                f(x) = 1/(1-x**2)**1
            'pFunc2': similarly with f(x) = 1/(1-x**2)**2
            'expFunc': with f(x) = exp(1/(1-x**2))
    accurate : bool
        If True, the bound on the bounded noise mechanism is better and the
        computation time is longer
    beta: in (0,1)
        A failure probability for the Gaussian mechanism
    gaussStd: bool
        Controls the output for the gaussian mechanism

    Prints
    ------
    Normalized bounded noise: optimal noise level for the bounded noise mechanism,
        divided by sqrt(k * log(1/delta))/eps
    Normalized gaussian noise: 
        If gaussStd:
            the standard deviation of the gaussian mechanism that is (eps,delta)-DP,
            divided by sqrt(k * log(1/delta))/eps
        else:
            t = a threshold such that the maximal noise of the
            gaussian mechanism over the k queries is bounded by t w.pr. 1-beta;
            Output value is t divided by sqrt(k * log(1/delta))/eps
    Ratio: ratio between noise in the bounded-mechanism and gaussian noises

    Returns
    -------
    Normalized bounded noise, normalized gaussian noise
    '''
    startTime = time.time()
    pr = textToPr(prText)
    
    print('experiment: k={}, eps={}, delta={}, prob={}, beta={}, gaussStd={}, accurate={}'.format(
        k,epsilon, delta, prText, beta, gaussStd, accurate))

    asympNoise = np.sqrt(log(1/delta)* k)/epsilon
    gaussNoise = getSigmaExact(epsilon,delta,k)
    if not gaussStd:
        gaussNoise = gaussHighProb(gaussNoise,beta,k)
    noise = optimalNoise(pr, k, epsilon, delta, 4*gaussNoise,
                     accurate)
    
    print('normalized bounded noise: {}, normalized gauss noise: {}, ratio: {}'.format(
        noise / asympNoise, gaussNoise/asympNoise, noise/gaussNoise))
    if gaussStd:
        print('Note: the value for the Gaussan mechanism corresponds to the ' +
              'standard deviation and not the maximal value')
    elapsedTime = time.time() - startTime
    print('time passed (s):', elapsedTime)
    
    return noise/asympNoise, gaussNoise/asympNoise


def experimentKs(eps=0.1, delta=10**-10, prText='pFunc2', 
                 minK=2, maxK=10, numKs=8, accurate=True):
    '''
    Multiple experiments of testOneVal, for different values of K +
    generating a plot

    Parameters
    ----------
    eps, delta, prText, accurate : as in testOneVal
    minK, maxK, numKs : determining the number of K values and their (logarithmic)
        range
    '''
    betaAlg = 0.05
    
    ks = np.logspace(minK, maxK, num=numKs)
    noises = np.zeros(ks.shape)
    gaussNoises = np.zeros(ks.shape)
    confLs = np.zeros(ks.shape)
    
    for i, k in enumerate(ks):
        noise, gaussNoise = testOneVal(
            k, eps, delta, prText, accurate, gaussStd=True)
        noises[i] = noise
        gaussNoises[i] = gaussNoise
        confLs[i] = truncThresh(textToPr(prText), betaAlg/k)
    
    plotVals(ks,noises,gaussNoises,confLs, ks)
    
    return noises, gaussNoises

def experimentDeltas(eps=0.1, k=10**6, prText='pFunc2', minDelta=2, 
                     maxDelta=250, accurate=True, numDeltas=8):
    '''
    Experiments different values of delta. Arguments similar to experimentKs
    '''
    betaAlg = 0.05
    
    logDeltas=np.logspace(log(minDelta), log(maxDelta), num=numDeltas,base=np.e)
    deltas = 10**-logDeltas
    
    noises = np.zeros(deltas.shape)
    gaussNoises = np.zeros(deltas.shape)
    confLs = (np.ones(deltas.shape) * 
              truncThresh(textToPr(prText), betaAlg/k))
    ks = np.ones(deltas.shape) * k
    
    for i, delta in enumerate(deltas):
        noise, gaussNoise = testOneVal(
            k, eps, delta, prText, accurate, gaussStd=True)
        noises[i] = noise
        gaussNoises[i] = gaussNoise
    
    plotVals(logDeltas,noises,gaussNoises,confLs, ks)
    
    return deltas, noises, gaussNoises


def experimentEps(k=10**6, delta=10**-10, prText='pFunc2', minEps=-7, 
                     maxEps=0, accurate=True, numEps=8):
    '''
    Experiments different values of epsilon. Arguments similar to experimentKs
    '''
    betaAlg = 0.05
    
    epss=np.logspace(minEps, maxEps, num=numEps)
    noises = np.zeros(epss.shape)
    gaussNoises = np.zeros(epss.shape)
    confLs = (np.ones(epss.shape) * 
              truncThresh(textToPr(prText), betaAlg/k))
    ks = np.ones(epss.shape) * k

    for i, eps in enumerate(epss):
        noise, gaussNoise = testOneVal(
            k, eps, delta, prText, accurate, gaussStd=True)
        noises[i] = noise
        gaussNoises[i] = gaussNoise
   
    plotVals(epss, noises, gaussNoises, confLs, ks)
    
    return noises, gaussNoises
 

def plotVals(x, noises, gaussNoises, confLs, k,
             betas=[0.5, 0.05, 0.001, 10**-6]):
    plt.plot(x, noises,
             x, noises*confLs,
             x, gaussHighProb(gaussNoises, betas[0], k), '--',
             x, gaussHighProb(gaussNoises, betas[1], k), '--',
             x, gaussHighProb(gaussNoises, betas[2], k), '--',
             x, gaussHighProb(gaussNoises, betas[3], k), '--')
    plt.xscale('log')
    plt.ylim(ymin=0)
    plt.show()
    
if __name__ == '__main__':
    print('Experimenting one parameter value\n---------------------')
    testOneVal(10**6,0.1,10**-10,'pFunc2',True)
    print('\ntesting multiple K values and generating plot')
    print('---------------------------------------------')
    experimentKs()
    print('\ntesting multiple delta values and generating plot')
    print('---------------------------------------------')
    experimentDeltas()
    print('\ntesting multiple epsilon values and generating plot')
    print('---------------------------------------------')
    experimentEps()    