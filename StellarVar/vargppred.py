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
        self.lctype = lc.lctype

    def rotation_model(self):
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
        return model, map_soln

    def granulation_model(self):
        peak = self.period_prior()
        x = self.lc.lcf.time
        y = self.lc.lcf.flux
        yerr = self.lc.lcf.flux_err
        with pm.Model() as model:
            # The mean flux of the time series
            mean = pm.Normal("mean", mu=0.0, sd=10.0)

            # A jitter term describing excess white noise
            logs2 = pm.Normal("logs2", mu=2*np.log(np.min(sigmaclip(yerr)[0])), sd=1.0)

            logw0 = pm.Bound(pm.Normal, lower=-0.5, upper=np.log(2 * np.pi / self.min_period))("logw0", mu=0.0, sd=5)
            logSw4 = pm.Normal("logSw4", mu=np.log(np.var(y)), sd=5)
            kernel = xo.gp.terms.SHOTerm(log_Sw4=logSw4, log_w0=logw0, Q=1 / np.sqrt(2))

            #GP model
            gp = xo.gp.GP(kernel, x, yerr**2 + tt.exp(logs2))

            # Compute the Gaussian Process likelihood and add it into the
    	    # the PyMC3 model as a "potential"
            pm.Potential("loglike", gp.log_likelihood(y - mean))

    	    # Compute the mean model prediction for plotting purposes
            pm.Deterministic("pred", gp.predict())

    	    # Optimize to find the maximum a posteriori parameters
            map_soln = xo.optimize(start=model.test_point)
        return model, map_soln

    def hybrid_model(self):
        peak = self.period_prior()
        x = self.lc.lcf.time
        y = self.lc.lcf.flux
        yerr = self.lc.lcf.flux_err

        y1 = self.lc.lcf.flatten(window_length=7, return_trend=True)[1].flux
        y2 = y - y1
	#y2 = self.lc.lcf.flatten(window_length=401, return_trend=False).flux
        with pm.Model() as model:
            # The mean flux of the time series
            mean = pm.Normal("mean", mu=0.0, sd=10.0)

            # A jitter term describing excess white noise
            logs21 = pm.Normal("logs21", mu=np.log(np.var(y1)), sd=1.0)

            # The parameters of the RotationTerm kernel
            logamp = pm.Normal("logamp", mu=np.log(np.var(y1)), sd=5.0)
            logperiod = pm.Bound(pm.Normal, lower=np.log(self.min_period),
                        upper=np.log(self.max_period))("logperiod", mu=np.log(peak["period"]), sd=2.0)
            logQ0 = pm.Uniform("logQ0", lower=-15, upper=5)
            logdeltaQ = pm.Uniform("logdeltaQ", lower=-15, upper=5)
            mix = pm.Uniform("mix", lower=0, upper=1.0)

            # Track the period as a deterministic
            period = pm.Deterministic("period", tt.exp(logperiod))
    
            kernel1 = xo.gp.terms.RotationTerm(
                log_amp=logamp,
                period=period,
                log_Q0=logQ0,
                log_deltaQ=logdeltaQ,
                mix=mix)
            
            gp1 = xo.gp.GP(kernel1, x, yerr**2 + tt.exp(logs21))

            # Compute the Gaussian Process likelihood and add it into the
            # the PyMC3 model as a "potential"
            pm.Potential("loglike1", gp1.log_likelihood(y1 - mean))

            # Compute the mean model prediction for plotting purposes
            pred1  = pm.Deterministic("pred1", gp1.predict())

            # The parameters of SHOTerm kernel for non-periodicity granulation
            logw0 = pm.Bound(pm.Normal, lower=-0.5, upper=np.log(2 * np.pi / self.min_period))("logw0", mu=0.0, sd=5)
            logSw4 = pm.Normal("logSw4", mu=np.log(np.var(y2)), sd=5)
            logs22 = pm.Normal("logs22", mu=np.log(np.var(y2)), sd=1.0)
    
            kernel2 = xo.gp.terms.SHOTerm(log_Sw4=logSw4, log_w0=logw0, Q=1 / np.sqrt(2))
            gp2 = xo.gp.GP(kernel2, x, yerr**2 + tt.exp(logs22))

            pm.Potential("loglike2", gp2.log_likelihood(y2 - mean))
            pm.Deterministic("pred2", gp2.predict())

            # Optimize to find the maximum a posteriori parameters
            map_soln = xo.optimize(start=model.test_point)
        return model, map_soln
        
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
        if self.lctype == "rotation":
            model, map_soln = self.rotation_model()
        elif self.lctype == "granulation":
            model, map_soln = self.granulation_model()
        elif self.lctype == "hybrid":
            model, map_soln = self.hybrid_model()
            
        np.random.seed(42)
        with model:
            trace = pm.sample(
            tune=2000,
            draws=2000,
            start=map_soln,
            step=xo.get_dense_nuts_step(target_accept=0.9),
            progressbar=False
            )
            
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
            if self.lctype == "rotation":
                sample = pm.trace_to_dataframe(self._trace, varnames=["mix", "logdeltaQ", "logQ0", 
                                                                  "logperiod", "logamp", "logs2", "mean"])
            elif self.lctype == "granulation":
                sample = pm.trace_to_dataframe(self._trace, varnames=["logSw4", "logw0", "logs2", "mean"])
            elif self.lctype == "hybrid":
                sample = pm.trace_to_dataframe(self._trace, varnames=["mix", "logdeltaQ", "logQ0", "logperiod", "logamp", "logs21", "logw0", "logSw4", "logs22", "mean"])
            figure = corner.corner(sample)
            
        return figure

    def GP_fitting_plots(self):
        import matplotlib.pylab as plt
        y1 = self.lc.lcf.flatten(window_length=9, return_trend=True)[1].flux
        y2 = self.lc.lcf.flux - y1
        fig, ax = plt.subplots(4,2, figsize=(18, 13), gridspec_kw={'width_ratios': [3, 1]})
        #pred_tot = np.percentile(trace["pred1"][:]+trace["pred2"], [16, 50, 84], axis=0)
        pred_rot = np.percentile(self._trace["pred1"][:], [16, 50, 84], axis=0)
        pred_gra = np.percentile(self._trace["pred2"][:], [16, 50, 84], axis=0)
        pred_tot = pred_rot + pred_gra

        #ax[0][0]
        ax[0][0].plot(self.lc.lcf.time, pred_tot[1], color="C1", label="model_tot")
        art1 = ax[0][0].fill_between(self.lc.lcf.time, pred_tot[0], pred_tot[2], color="C1", alpha=0.3)
        art1.set_edgecolor("none")
        ax[0][0].errorbar(self.lc.lcf.time, self.lc.lcf.flux, self.lc.lcf.flux_err, fmt=".k", label="data_tot")
        ax[0][0].set_ylabel("flux[ppm]", fontsize=15)

        #ax[0][1]
        results = xo.estimators.lomb_scargle_estimator(
              self.lc.lcf.time, self.lc.lcf.flux, self.lc.lcf.flux_err, max_peaks=1, min_period=self.min_period, max_period=self.max_period,
              samples_per_peak=100)
        peak = results["peaks"][0]
        freq, power = results["periodogram"]
        ax[0][1].plot(-np.log10(freq), power, "k")
        ax[0][1].axvline(np.log10(peak["period"]), color="k", lw=2, alpha=0.5, label="$P_{{LS}}:{period:.2f}d$".format(LS = 'LS', period=peak['period']))
        ax[0][1].set_ylabel("power", fontsize=15)

        #ax[1][0]
        ax[1][0].plot(self.lc.lcf.time, pred_rot[1], color="C2", label="model_rot")
        art2 = ax[1][0].fill_between(self.lc.lcf.time, pred_rot[0], pred_rot[2], color="C2", alpha=0.3)
        art2.set_edgecolor("none")
        ax[1][0].errorbar(self.lc.lcf.time, y1, self.lc.lcf.flux_err, fmt=".k", label="data_rot")
        ax[1][0].set_ylabel("rot_flux[ppm]", fontsize=15)

        #ax[1][1]
        results = xo.estimators.lomb_scargle_estimator(
              self.lc.lcf.time, self.lc.lcf.flux - pred_gra[1], self.lc.lcf.flux_err, max_peaks=1, min_period=self.min_period, max_period=self.max_period,
              samples_per_peak=100)
        peak = results["peaks"][0]
        freq, power = results["periodogram"]
        ax[1][1].plot(-np.log10(freq), power, "k")
        ax[1][1].axvline(np.log10(peak["period"]), color="k", lw=2, alpha=0.5, label="$P_{{LS}}:{period:.2f}d$".format(LS = 'LS', period=peak['period']))
        ax[1][1].set_ylabel("power", fontsize=15)

        #ax[2][0]
        ax[2][0].plot(self.lc.lcf.time, pred_gra[1], color="r", label="model_gra")
        art3 = ax[2][0].fill_between(self.lc.lcf.time, pred_gra[0], pred_gra[2], color="r", alpha=0.3)
        art3.set_edgecolor("none")
        ax[2][0].errorbar(self.lc.lcf.time, self.lc.lcf.flux-pred_rot[1], self.lc.lcf.flux_err, fmt=".k", label="data_gra")
        ax[2][0].set_ylabel("gra_flux[ppm]", fontsize=15)

        #ax[2][1]
        results = xo.estimators.lomb_scargle_estimator(
              self.lc.lcf.time, self.lc.lcf.flux - pred_rot[1], self.lc.lcf.flux_err, max_peaks=1, min_period=self.min_period, max_period=self.max_period,
              samples_per_peak=100)
        peak = results["peaks"][0]
        freq, power = results["periodogram"]
        ax[2][1].plot(-np.log10(freq), power, "k")
        ax[2][1].axvline(np.log10(peak["period"]), color="k", lw=2, alpha=0.5, label="$P_{{LS}}:{period:.2f}d$".format(LS = 'LS', period=peak['period']))
        ax[2][1].set_ylabel("power", fontsize=15)

        #ax[3][0]
        ax[3][0].errorbar(self.lc.lcf.time, self.lc.lcf.flux-pred_tot[1], self.lc.lcf.flux_err, fmt=".k", label="residual")
        ax[3][0].set_ylabel("residual[ppm]", fontsize=15)
        ax[3][0].set_xlabel("Time-2454833[BKJD]", fontsize=15)

        #ax[3][1]
        results = xo.estimators.lomb_scargle_estimator(
              self.lc.lcf.time, self.lc.lcf.flux - pred_tot[1], self.lc.lcf.flux_err, max_peaks=1, min_period=self.min_period, max_period=self.max_period,
              samples_per_peak=100)
        peak = results["peaks"][0]
        freq, power = results["periodogram"]
        ax[3][1].plot(-np.log10(freq), power, "k")
        ax[3][1].axvline(np.log10(peak["period"]), color="k", lw=2, alpha=0.5, label="$P_{{LS}}:{period:.2f}d$".format(LS = 'LS', period=peak['period']))
        ax[3][1].set_ylabel("power", fontsize=15)
        ax[3][1].set_xlabel("$log_{10}(period)$", fontsize=15)

        return fig
