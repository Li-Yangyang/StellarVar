#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Defines VarGPPred classes"""

import numpy as np
import exoplanet as xo
import pymc3 as pm
import theano.tensor as tt
from scipy.stats import sigmaclip

class VarGPPred(object):
    """
    Simple class for predicting period via gaussian process
    Attributes:
    ---------
    """
    def __init__(self, lc):
        self.lc = lc
        self._trace = None
        self.min_period = np.max(sigmaclip(np.diff(self.lc.lcf.time))[0])
        self.max_period = 0.5 * (self.lc.lcf.time.max() - self.lc.lcf.time.min())
        
    def period_prior(self):
        """
        Returns the peak of lomb-scargle periodigram estimator
        """
        results = xo.estimators.lomb_scargle_estimator(
        self.lc.lcf.time, self.lc.lcf.flux, self.lc.lcf.flux_err, max_peaks=1, min_period=self.min_period, max_period=self.max_period,
        samples_per_peak=100)

        peak = results["peaks"][0]
        
        return peak
    
    def predict(self):
        """
        Predict the period of stellar variability via Gaussian Process fitting
        
        Returns all samples of paramters after mcmc fitting
        """
        peak = self.period_prior()
        x = self.lc.lcf.time
        y = self.lc.lcf.flux
        yerr = self.lc.lcf.flux_err
        with pm.Model() as model:
            # The mean flux of the time series
    	    mean = pm.Normal("mean", mu=0.0, sd=10.0)

            # A jitter term describing excess white noise
    	    logs2 = pm.Normal("logs2", mu=2*np.log(np.min(sigmaclip(yerr)[0])), sd=1.0)

            # The parameters of the RotationTerm kernel
    	    logamp = pm.Normal("logamp", mu=np.log(np.var(y)), sd=5.0)
    	    logperiod = pm.Bound(pm.Normal, lower=np.log(self.min_period),
                        upper=np.log(self.max_period))("logperiod", mu=np.log(peak["period"]), sd=2.0)
    	    logQ0 = pm.Uniform("logQ0", lower=-15, upper=5)
    	    logdeltaQ = pm.Uniform("logdeltaQ", lower=-15, upper=5)
    	    mix = pm.Uniform("mix", lower=0, upper=1.0)

    	    # Track the period as a deterministic
    	    period = pm.Deterministic("period", tt.exp(logperiod))
    
    	    kernel = xo.gp.terms.RotationTerm(
        	    log_amp=logamp,
        	    period=period,
                    log_Q0=logQ0,
        	    log_deltaQ=logdeltaQ,
        	    mix=mix)
    	    gp = xo.gp.GP(kernel, x, yerr**2 + tt.exp(logs2), J=4)

    	    # Compute the Gaussian Process likelihood and add it into the
    	    # the PyMC3 model as a "potential"
    	    pm.Potential("loglike", gp.log_likelihood(y - mean))

    	    # Compute the mean model prediction for plotting purposes
    	    pm.Deterministic("pred", gp.predict())

    	    # Optimize to find the maximum a posteriori parameters
    	    map_soln = xo.optimize(start=model.test_point)
            
        np.random.seed(42)
        sampler = xo.PyMC3Sampler(finish=200)
        with model:
            sampler.tune(tune=2000, start=map_soln, step_kwargs=dict(target_accept=0.9), progressbar=False)
            trace = sampler.sample(draws=2000, progressbar=False)
            
        self._trace = trace
        return trace
    
    def corner(self):
        """
        Plot corner of all paramters distributions
        
        Returns
        -------
        figure:
            matplotlib.figure.Figure
        """
        if self._trace == None:
            raise ValueError("Do not sample the posteriors yet!")
        else:
            import corner
            sample = pm.trace_to_dataframe(self._trace, varnames=["mix", "logdeltaQ", "logQ0", 
                                                                  "logperiod", "logamp", "logs2", "mean"])
            figure = corner.corner(sample)
            
        return figure
