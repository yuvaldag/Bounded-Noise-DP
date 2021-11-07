# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 16:09:28 2021

@author: Yuval
"""

import numpy as np
from numpy import log, exp, sqrt
import scipy.integrate as integrate
from scipy.optimize import minimize

verbose = False


def realBinarySearch(f, val, a, b, err=0.000001, multCond=False, guess=None):
    ''' Performs a binary search over the real line to find a point $x$
    such that f(x) = val, where f is nondecreasing. More concretely, we return
    and interval that contains x.
    
    Params:
        f: a function f : [a,b] \to R. It is nondecreasing
        val: a target value
        [a,b]: the domain of f, a >= 0, b in (0, np.inf]
        err: the absolute error. Namely, the returned interval [x_1,x_2]
            satisfied x_2 - x_1 \le 1+err
        multCond: if true, then we instead guarantee that x_2/x_1 \le 1+err
        Guess: an initial guess of a point that might lie close to the
            preimage of val
    Returns:
        A pair of points (x_1, x_2) such that the preimage of val under f lies
        in [x_1,x_2]
    '''
    initialA = a
    initialB = b
    
    if multCond == False:
        stopCond = lambda a,b: b-a < err
    else:
        stopCond = lambda a,b: a != 0 and b/a < (1+err)
    
    if guess == None:
        if b != np.inf:
            guess = (a+b)/2
        else:
            guess = 1
    
    while not stopCond(a,b):
        if f(guess) < val:
            a = guess
            guess = np.min((2*guess, (guess + b)/2))
        else:
            b = guess
            guess = (a+guess)/2
    
    if a == initialA or b == initialB:
        print('limits for binary search, are two small: ' +
            'input limits are [{}, {}] '.format(initialA,initialB) +
            'while output is [{},{}]'.format(a,b)
            )
    
    return a,b
 


class Prob:
    '''
    A probability distribution over [-1,1]. Its density equals
    exp(-f(x))/Z where Z = int_{-1}^1 exp(-f(x))dx.
    
    Attributes:
        f: the function f such that Pr[x] = exp(-f(x))/Z
        Z: the normalizing constant Z
    '''
    def __init__(self, f, eps = 0.00001):
        '''
        Parameters
        ----------
        f : function over (-1,1)
            To be attributed to the instance class
        eps : real valued
            A small err. The normalizing constant is integrated over 
            [-1+eps, 1-eps]

        Returns
        -------
        None.

        '''
        self.f = f
        unscaledDensity = lambda x: exp(-f(x))
        self.Z = integrate.quad(unscaledDensity, -1+eps, 1-eps)[0]
    
    def __call__(self, x):
        '''

        Parameters
        ----------
        x : number in (-1,1)

        Returns
        -------
        Nonnegative number
            The density of x
        '''
        if x < -900: #If x is too small then we'll get an error after
                     # exponentiating
            return 0
        return exp(-self.f(x))/self.Z
    
    def logPr(self, x):
        '''
        Parameters
        ----------
        x : number in (-1,1)

        Returns
        -------
        Real number
            Log density at x

        '''
        return -self.f(x) - log(self.Z)

'''
Candidates for a function f = negative log density of some distribution
'''
def expFunc(x):
    return exp(1/(1-x*x))

def pFunc(p):
    '''
    Returns a different function for each p > 0
    '''
    return lambda x: 1/(1-x*x)**p


def truncThresh(pr, delta):
    '''
    Find a threshold L such that pr([-L,L]) >= 1-delta.

    Parameters
    ----------
    pr : Prob
        A probability meausure
    delta : nonegative number
        Probability of deviation

    Returns
    -------
    Number in [0,1]
        A number L such that pr([-L,L]) >= 1-delta

    '''
    def negLogBnd(L):
        logPr = - pr.f(L) - log(pr.Z) + log(2-2*L)
        return -logPr
    
    return realBinarySearch(negLogBnd, -log(delta), 0, 1)[1]


def logMGF(pr, g, theta, L, truncProb):
    '''
    Computes the log moment generating function. Returns the log of
    E[exp(theta * g(x) 1(|x| <= L))]
    = int_{-L}^L exp(theta * g(x)) pr(x) dx 
        + int_{-1}^{-L} exp(0) pr(x) dx + int_L^1 exp(0) pr(x) dx
    where the expectation is over x drawn from pr, and pr also denotes the
    density above.

    Parameters
    ----------
    pr : Prob
        A probability density
    g : A real valued function, g : (-1,1) \to R
    theta : positive number
    L : A real number in (0,1)
    truncProb : a number between 0 and 1
        the probability that |x| > L

    Returns
    -------
    A positive number
        The logarithm of the integral defined above
    '''
    if theta < 0:
        return 0
    pointVal = lambda x: exp(pr.logPr(x) + theta * g(x))
    I = integrate.quad(pointVal, -L, L)
    MGF = I[0] + I[1] + truncProb # here I(0) is the integral result and
                                # I(1) is an estimate of the absolute error
                                # Note that I(1) may be wrong some times!
                                # Make sure to verify!
    return log(MGF)


coeffTheta=0.065 # This is an auxiliary parameter that is used to find a good
                # initialization to a minimization problem that is executed
                # in the function below. This value is stored from execution
                # to execution to improve the initialization
def logDeviationProb(pr, g, L, k, t, truncProb, Rinv):
    '''
    Given k iid random draws from pr, denoted by x_1,...,x_k, 
    bound the *log* of the probability that sum_i g(x_i)*1(|x_i|<L) > t, 
    where g(x) = pr.f(x+1/R)-pr.f(x).
    The bound is obtained by using the moment generating function of the 
    truncated variable
    g(x) 1(|x| < L), by invoking the function logMGF.
    The function g is intended to be g(x) = pr.f(x+1/R)-pr.f(x),
    where R is the magnitude of noise that is being tested

    Parameters
    ----------
    pr : Type: Prob
        A probability distribution
    g : A function from (-1,1) to the real numbers
    L : A number in [0,1] (to be sent as an argument to logMGF)
    k : An integer
        The number of iid draws
    t : A positive number
        The threshold such that we want to bound Pr[sum_i g(x_i)>t]
    truncProb : a number in [0,1]
        The probability that |x|>L (to be sent to logMGF)
    Rinv : A positive number
        Equals 1/R where R is the bound on the noise that we currently analyze

    Returns
    -------
    A real number
        A bound on the curresponding log-probability

    '''
    global coeffTheta
    
    thetaBnd = lambda theta: logMGF(pr, g, theta, L, truncProb) * k - theta * t
    #bestBnd = minimize(thetaBnd, 0, maxTheta, 0.1)
    
    asympTheta = t/(k*Rinv**2)
    bestBnd = minimize(thetaBnd, asympTheta * coeffTheta, method='Nelder-Mead')
    
    theta = bestBnd.x[0]
    coeffTheta = theta/asympTheta
    if verbose:
        result = bestBnd.fun
        lMGF = (result + theta*t)/k
        coeffMGF = lMGF / (theta*Rinv)**2
        print('minimization: Rinv={}, t={}, k={}, coeffTheta={}, theta={}, coeffMGF={}'.format(
            Rinv, t, k, coeffTheta, theta, coeffMGF
        ))
    return bestBnd


def deviationInt(pr, L, k, Rinv, eps, truncProb, accurate):
    '''
    Bounds the integral
    int_eps^np.infty Pr[sum_{i=1}^k g(x_i)*1(|x_i|<L) > t] e^{eps-t} dt,
    where x_1,...,x_k are iid draws from pr,
    and g(x) = pr.f(x+1/Rinv) - pr.f(x)

    Parameters
    ----------
    pr, L, k, truncProb, Rinv : same as in deviationProb
        
    eps : Positive number
        The privacy parameter. Appears in the above formula
    accurate : Bool
        Indicates whether the calculation should be heavy or lighter.
        If False, then we bound the integral simply with
        Pr[sum_i g(x_i) > eps].
        If true, we still discretize the integral, but we have a better estimate

    Returns
    -------
    A real number
        The integral defined above
    '''
    if Rinv > (1-L)/2:
        return 0
    
    g = lambda x: pr.f(x+Rinv) - pr.f(x)
    
    devFunc = lambda t: logDeviationProb(pr,g,L,k,t,truncProb,Rinv).fun
    if accurate:
        return deviationIntManual(devFunc,eps)
    else:
        return exp(devFunc(eps))


jump = 1.01 # An auxiliary ranom variable to speed up computation.
            # It is updated from execution to execution
def deviationIntManual(func, eps, err=0.1):
    '''
    An approximation (upper bound) of the integral 
    int_eps^infty Pr[Z>t] e^{eps-t} dt, where Z is a ranom variable
    (it will be replaced with sum_i g(x_i)1(|x_i|<L))

    Parameters
    ----------
    func : A function from the positive numbers to the reals
        This function that receives t and outputs log Pr[Z>t] for a random
        variable Z
    eps : a positive number
        The privacy parameter
    err : a positive number
        An error parameter in computing the integral. The smaller it is, the
        more accurate it is.

    Returns
    -------
    A real number
        The approximation of the above integral.

    '''
    global jump
    while exp(func(jump*eps) - func(eps)) < 1-err:
        jump = sqrt(jump)
    while exp(func(jump**2 * eps) - func(eps)) >= 1-err:
        jump = jump**2
    
    I = 0
    t = eps
    while True:
        val = exp(func(t))
        if val < I * err:
            return I + val * exp(eps-t)
        nextT = t * jump
        I += val * (exp(eps-t) - exp(eps - nextT))
        t = nextT


def optimalNoise(pr, k, epsilon, delta, startR, accurate):
    '''
    Given an epsilon, delta, a number of queries k, and a probability distribution pr,
    find an upper bound on the lowest R such that the 
    mechanism that answers k 1-sensitive queries using noise pr with magnitue 
    scaled by R is (epsilon,delta)-DP.

    Parameters
    ----------
    pr : Prob
        A probability distribution
    k : a natural number
        Numer of quries
    epsilon : positive integer
        Privacy parameter
    delta : number in (0,1)
        Privacy parameter
    startR : Positive number
        A guess for the noise to try first
    accurate : Bool
        An indication whether the computation shoul be accurate. If false, the
        computation will take less time but the answer will be given faster

    Returns
    -------
    A positive number
        A noise R such that the protocol with noise supported by [-R,R] is
        (eps, delta)-DP

    '''
    delta1 = delta/100
    delta2 = delta - delta1
    truncProb = delta1/k
    L = truncThresh(pr, truncProb)
    if verbose:
        print('L', L)
    dev = lambda Rinv: deviationInt(pr, L, k, Rinv, epsilon, truncProb,
                                           accurate)
    Rinv = realBinarySearch(dev, delta2, 0, np.inf, err=0.01, 
                            multCond=True, guess=1/startR)[0]
    return 1/Rinv
