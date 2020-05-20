#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Defines VarGPPred classes"""

import numpy as np
import exoplanet as xo
import pymc3 as pm
import theano.tensor as tt
from scipy.stats import sigmaclip

from .utils import window_rms


class VarGPPred(object):
    """
    Simple class for predicting period via gaussian process
    Attributes:
    ---------
    """
    def __init__(self, lc, window_width):
        self.lc = lc
        self.window_width = window_width
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
        def submodel1(x,y,yerr,parent):
            with pm.Model(name="rotation_", model=parent) as submodel:
                # The mean flux of the time series
                mean1 = pm.Normal("mean1", mu=0.0, sd=10.0)

                # A jitter term describing excess white noise
                logs21 = pm.Normal("logs21", mu=np.log(np.mean(yerr)), sd=2.0)

                # The parameters of the RotationTerm kernel
                logamp = pm.Normal("logamp", mu=np.log(np.var(y)), sd=5.0)
                #logperiod = pm.Uniform("logperiod", lower=np.log(vgp.min_period), upper=np.log(vgp.max_period))
                logperiod = pm.Bound(pm.Normal, lower=np.log(self.min_period),
                    upper=np.log(self.max_period))("logperiod", mu=np.log(peak["period"]), sd=1.0)
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
                loglike1 = gp1.log_likelihood(y - mean1)
                #pred1  = pm.Deterministic("pred1", gp1.predict())
        
            return logperiod, logQ0, gp1, loglike1

        def submodel2(x,y,yerr,parent):
            with pm.Model(name="granulation", model=parent) as submodel:
                # The parameters of SHOTerm kernel for non-periodicity granulation
                mean2 = pm.Normal("mean2", mu=0.0, sd=10.0)
                #logz = pm.Uniform("logz", lower=np.log(2 * np.pi / 4), upper=np.log(2*np.pi/vgp.min_period))
                #sigma = pm.HalfCauchy("sigma", 3.0)
                #logw0 = pm.Normal("logw0", mu=logz, sd=2.0)
                logw0 = pm.Bound(pm.Normal, lower=np.log(2 * np.pi / 2.5), upper=np.log(2 * np.pi / self.min_period))("logw0", mu=np.log(2 * np.pi / 0.8), sd=1)
                logSw4 = pm.Normal("logSw4", mu=np.log(np.var(y)*(2 * np.pi / 2)**4), sd=5)
                logs22 = pm.Normal("logs22", mu=np.log(np.mean(yerr)), sd=2.0)
                logQ = pm.Bound(pm.Normal, lower=np.log(1/2), upper=np.log(2))("logQ", mu=np.log(1/np.sqrt(2)), sd=1)
    
                kernel2 = xo.gp.terms.SHOTerm(log_Sw4=logSw4, log_w0=logw0, log_Q=logQ)
                gp2 = xo.gp.GP(kernel2, x, yerr**2 + tt.exp(logs22))

                loglike2 = gp2.log_likelihood(y - mean2)
        
            return logw0, logQ, gp2, loglike2


        peak = self.period_prior()
        x = self.lc.lcf.time
        y = self.lc.lcf.flux
        yerr = self.lc.lcf.flux_err

        y1_old = self.lc.lcf.flatten(window_length=self.window_width, return_trend=True)[1].flux
        y2_old = y - y1_old
        y2 = sigmaclip(y2_old)[0]
        idx = np.in1d(y2_old, y2)
        self.y_new = y[idx]
        self.yerr_new = yerr[idx]
        self.x_new = x[idx]

        y1_old_ave = np.nanmean(y1_old[idx])
        #y1_old_std = np.nanstd(y1_old[idx])
        self.y1 = (y1_old[idx] - y1_old_ave) * 1e6
        self.y1_err = window_rms(yerr, self.window_width)[idx] * 1e6
        #y1 = y1[idx]

        y2_old = self.y_new - y1_old[idx]
        y2_old_ave = np.nanmean(y2_old)
        #y2_old_std = np.nanstd(y2_old)
        self.y2 = (y2_old - y2_old_ave) * 1e6#/y2_old_ave
        y2_err = np.sqrt(self.yerr_new**2 - window_rms(yerr, self.window_width)[idx]**2)
        y2_err[np.isnan(y2_err)] = 0
        self.y2_err = y2_err * 1e6#/ y2_old_ave 
	#y2 = self.lc.lcf.flatten(window_length=401, return_trend=False).flux

        y_class = [self.y1, self.y2]
        yerr_class = [self.y1_err, self.y2_err]
        submodel_class = [submodel1, submodel2]
        with pm.Model() as model:
            gp_class = []
            loglikes = []
            logtaus = []
            logQs = []
            for i in range(1,3):
                logtau, log_Q, gp, loglike = submodel_class[i-1](self.x_new, y_class[i-1], yerr_class[i-1], model)
                gp_class.append(gp)
                loglikes.append(loglike)
                logtaus.append(logtau)
                logQs.append(log_Q)
            #loglikes = tt.stack(loglikes)
            #pm.Potential("loglike", pm.math.logsumexp(loglikes))
            pm.Potential("loglike_rot", loglikes[0])
            pm.Potential("loglike_gra", loglikes[1])
            predrot = pm.Deterministic("pred_rot", gp_class[0].predict()/1e6+y1_old_ave)
            predgra = pm.Deterministic("pred_gra", gp_class[1].predict()/1e6+y2_old_ave)
            predtot = pm.Deterministic("pred_tot", gp_class[0].predict()/1e6+y1_old_ave+gp_class[1].predict()/1e6+y2_old_ave)
            # Optimize to find the maximum a posteriori parameters

            map_soln = xo.optimize(start=model.test_point, vars=[logtaus[0], logQs[0]])
            map_soln = xo.optimize(start=model.test_point, vars=[logtaus[1], logQs[1]])
            map_soln = xo.optimize(start=model.test_point)
        return model, map_soln
        
    def period_prior(self):
        """
        Returns the peak of lomb-scargle periodigram estimator
        """
        results = xo.estimators.lomb_scargle_estimator(
        self.lc.lcf.time, self.lc.lcf.flux, self.lc.lcf.flux_err, max_peaks=1, min_period=self.min_period, max_period=self.max_period,
        samples_per_peak=100)
  
        if len(results["peaks"])==0:
            results = xo.estimators.lomb_scargle_estimator(
                self.lc.lcf.time, self.lc.lcf.flux, max_peaks=1, min_period=self.min_period, max_period=self.max_period,
                samples_per_peak=100)

        peak = results["peaks"][0]
        
        return peak
    
    def predict(self, model_type=None):
        """
        Predict the period of stellar variability via Gaussian Process fitting
        
        Returns all samples of paramters after mcmc fitting
        """
        if model_type == None:
            if self.lctype == "rotation":
                model, map_soln = self.rotation_model()
            elif self.lctype == "granulation":
                model, map_soln = self.granulation_model()
            elif self.lctype == "hybrid":
                model, map_soln = self.hybrid_model()
        else:
            if model_type == "rotation":
                model, map_soln = self.rotation_model()
            elif model_type == "granulation":
                model, map_soln = self.granulation_model()
            elif model_type == "hybrid":
                model, map_soln = self.hybrid_model()
            
        np.random.seed(42)
        with model:
            trace = pm.sample(
            tune=2000,
            draws=2000,
            start=map_soln,
            step=xo.get_dense_nuts_step(target_accept=0.99),
            progressbar=True
            )
            
        self._trace = trace
        return trace
    
    def corner(self, model_type=None):
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
            if model_type == None:
                if self.lctype == "rotation":
                    sample = pm.trace_to_dataframe(self._trace, varnames=["mix", "logdeltaQ", "logQ0", 
                                                                  "logperiod", "logamp", "logs2", "mean"])
                elif self.lctype == "granulation":
                    sample = pm.trace_to_dataframe(self._trace, varnames=["logSw4", "logw0", "logs2", "mean"])
                elif self.lctype == "hybrid":
                    sample = pm.trace_to_dataframe(self._trace, varnames= ['granulation_logs22', 'granulation_logQ',#'granulation_logz', #'granulation_sigma', 
                                                                          'granulation_logSw4', 'granulation_logw0', 'granulation_mean2', 'rotation__mix', 
                                                                          'rotation__logdeltaQ', 'rotation__logQ0', 'rotation__logperiod', 
                                                                          'rotation__logamp', 'rotation__logs21', 'rotation__mean1'])
            else:
                if model_type == "rotation":
                    sample = pm.trace_to_dataframe(self._trace, varnames=["mix", "logdeltaQ", "logQ0", 
                                                                  "logperiod", "logamp", "logs2", "mean"])
                elif model_type == "granulation":
                    sample = pm.trace_to_dataframe(self._trace, varnames=["logSw4", "logw0", "logs2", "mean"])
                elif model_type == "hybrid":
                    sample = pm.trace_to_dataframe(self._trace, varnames=['granulation_logs22', 'granulation_logQ',#'granulation_logz', #'granulation_sigma', 
                                                                          'granulation_logSw4', 'granulation_logw0', 'granulation_mean2', 'rotation__mix', 
                                                                          'rotation__logdeltaQ', 'rotation__logQ0', 'rotation__logperiod', 
                                                                          'rotation__logamp', 'rotation__logs21', 'rotation__mean1'])
                
            figure = corner.corner(sample)
            
        return figure

    def GP_fitting_plots(self, summary):
        import matplotlib.pylab as plt
        import matplotlib.ticker as tck
        from astropy.timeseries import LombScargle
        fig, ax = plt.subplots(4,2, figsize=(18, 13), gridspec_kw={'width_ratios': [3, 1]})
        for a in ax.flatten():
            a.axes.linewidth = 2.5
            a.tick_params(axis="x", which="major", size=5, width=2.5, labelsize=10, direction="in")
            a.tick_params(axis="y", which="major", size=5, width=2.5, labelsize=10, direction="in")
            a.tick_params(axis='x', which='minor', size=3, width=1.5, direction="in")
            a.tick_params(axis='y', which='minor', size=3, width=1.5, direction="in")
        #Here we take ppm form to look clearly
        pred_rot = np.percentile(self._trace["pred_rot"][:], [16, 50, 84], axis=0)*1e6
        pred_gra = np.percentile(self._trace["pred_gra"][:], [16, 50, 84], axis=0)*1e6
        pred_tot = np.percentile(self._trace["pred_tot"][:], [16, 50, 84], axis=0)*1e6

        #ax[0][0]
        #ax[0][0].set_rasterization_zorder(1)
        ax[0][0].errorbar(self.x_new, self.y_new*1e6, self.yerr_new*1e6, fmt=".k", alpha=0.3, label="data_tot")
        ax[0][0].plot(self.x_new, pred_tot[1], color="C1", alpha=0.7, label="model_tot")
        art1 = ax[0][0].fill_between(self.x_new, pred_tot[0], pred_tot[2], color="C1", alpha=0.5)
        art1.set_edgecolor("none")
        ax[0][0].set_ylabel("flux[ppm]", fontsize=15)
        ax[0][0].legend(loc="lower left")
        ax[0][0].xaxis.set_minor_locator(tck.AutoMinorLocator())
        ax[0][0].yaxis.set_minor_locator(tck.AutoMinorLocator())

        #ax[0][1]
        results = xo.estimators.lomb_scargle_estimator(
              self.x_new, self.y_new*1e6, self.yerr_new*1e6, max_peaks=1, min_period=self.min_period, max_period=self.max_period,
              samples_per_peak=100)

        if len(results["peaks"])==0:
            results = xo.estimators.lomb_scargle_estimator(
                self.x_new, self.y_new*1e6, max_peaks=1, min_period=self.min_period, max_period=self.max_period,
                samples_per_peak=100)

        peak = results["peaks"][0]
        freq, power = LombScargle(self.x_new, self.y_new*1e6, self.yerr_new*1e6).autopower(minimum_frequency=1/self.max_period, maximum_frequency=1/self.min_period)
        if np.any(np.isnan(power)):
            freq, power = LombScargle(self.x_new, self.y_new*1e6).autopower(minimum_frequency=1/self.max_period, maximum_frequency=1/self.min_period)
        ax[0][1].plot(-np.log10(freq), power, "k")
        ax[0][1].axvline(np.log10(peak["period"]), color="crimson", linestyle="--", lw=2, alpha=0.5, label="$P_{{LS}}:{period:.2f}d$".format(LS = 'LS', period=peak['period']))
        ax[0][1].set_ylabel("power", fontsize=15)
        ax[0][1].legend(loc="upper left")
        ax[0][1].xaxis.set_minor_locator(tck.AutoMinorLocator())
        ax[0][1].yaxis.set_minor_locator(tck.AutoMinorLocator())
        ylim = np.max(power) + 0.1
        ax[0][1].set_ylim(ymax=ylim)

        #ax[1][0]
        ax[1][0].errorbar(self.x_new, self.y1, self.y1_err, fmt=".k", alpha=0.3, label="data_rot")
        ax[1][0].plot(self.x_new, pred_rot[1], color="C2", alpha=0.7, label="model_rot")
        art2 = ax[1][0].fill_between(self.x_new, pred_rot[0], pred_rot[2], color="C2", alpha=0.5)
        art2.set_edgecolor("none")
        ax[1][0].set_ylabel("rot_flux[ppm]", fontsize=15)
        ax[1][0].legend(loc="lower left")
        ax[1][0].xaxis.set_minor_locator(tck.AutoMinorLocator())
        ax[1][0].yaxis.set_minor_locator(tck.AutoMinorLocator())

        #ax[1][1]
        # error inject into the lombscargle should be carefully calculated again oringal - fitted error
        results = xo.estimators.lomb_scargle_estimator(
              self.x_new, self.y_new*1e6 - pred_gra[1], pred_gra[2]-pred_gra[0], max_peaks=1, min_period=self.min_period, max_period=self.max_period,
              samples_per_peak=100)

        if len(results["peaks"])==0:
            results = xo.estimators.lomb_scargle_estimator(
                self.x_new, self.y_new*1e6 - pred_gra[1], max_peaks=1, min_period=self.min_period, max_period=self.max_period,
                samples_per_peak=100)

        peak = results["peaks"][0]
        freq, power = LombScargle(self.x_new, self.y_new*1e6 - pred_gra[1], pred_gra[2]-pred_gra[0]).autopower(minimum_frequency=1/self.max_period, maximum_frequency=1/self.min_period)
        if np.any(np.isnan(power)):
            freq, power = LombScargle(self.x_new, self.y_new*1e6 - pred_gra[1]).autopower(minimum_frequency=1/self.max_period, maximum_frequency=1/self.min_period)

        ax[1][1].plot(-np.log10(freq), power, "k")
        ax[1][1].axvline(np.log10(peak["period"]), color="crimson", linestyle="--", lw=2, alpha=0.5, label="$P_{{LS}}:{period:.2f}d$".format(LS = 'LS', period=peak['period']))
        ax[1][1].axvline(np.log10(np.exp(summary.loc["rotation__logperiod"]["mean"])), color="navy", linestyle="--", lw=2, alpha=0.5, label="$P_{{GP}}:{period:.2f}d$".format(LS = 'LS', period=np.exp(summary.loc["rotation__logperiod"]["mean"])))
        ax[1][1].set_ylabel("power", fontsize=15)
        ax[1][1].legend(loc="upper left")
        ax[1][1].xaxis.set_minor_locator(tck.AutoMinorLocator())
        ax[1][1].yaxis.set_minor_locator(tck.AutoMinorLocator())
        ax[1][1].set_ylim(ymax=ylim)

        #ax[2][0]
        ax[2][0].errorbar(self.x_new, self.y2, self.y2_err, fmt=".k", alpha=0.3, label="data_gra")
        ax[2][0].plot(self.x_new, pred_gra[1], color="r", alpha=0.7, label="model_gra")
        art3 = ax[2][0].fill_between(self.x_new, pred_gra[0], pred_gra[2], color="r", alpha=0.5)
        art3.set_edgecolor("none")
        ax[2][0].set_ylabel("gra_flux[ppm]", fontsize=15)
        ax[2][0].legend(loc="lower left")
        ax[2][0].xaxis.set_minor_locator(tck.AutoMinorLocator())
        ax[2][0].yaxis.set_minor_locator(tck.AutoMinorLocator())

        #ax[2][1]
        results = xo.estimators.lomb_scargle_estimator(
              self.x_new, self.y_new*1e6 - pred_rot[1], pred_rot[2]-pred_rot[0], max_peaks=1, min_period=self.min_period, max_period=self.max_period,
              samples_per_peak=100)

        if len(results["peaks"])==0:
            results = xo.estimators.lomb_scargle_estimator(
                self.x_new, self.y_new*1e6 - pred_rot[1], max_peaks=1, min_period=self.min_period, max_period=self.max_period,
                samples_per_peak=100)

        peak = results["peaks"][0]
        freq, power = LombScargle(self.x_new, self.y_new*1e6 - pred_rot[1], pred_rot[2]-pred_rot[0]).autopower(minimum_frequency=1/self.max_period, maximum_frequency=1/self.min_period)
        if np.any(np.isnan(power)):
            freq, power = LombScargle(self.x_new, self.y_new*1e6 - pred_rot[1]).autopower(minimum_frequency=1/self.max_period, maximum_frequency=1/self.min_period)

        ax[2][1].plot(-np.log10(freq), power, "k")
        ax[2][1].axvline(np.log10(peak["period"]), color="crimson", linestyle="--", lw=2, alpha=0.5, label="$P_{{LS}}:{period:.2f}d$".format(LS = 'LS', period=peak['period']))
        Q = np.exp(summary.loc["granulation_logQ"]["mean"])
        w0 = np.exp(summary.loc["granulation_logw0"]["mean"])
        if Q>0.5:
            w0 = w0/(24*3600) *np.sqrt(4*Q**2-1)/(2*Q)
        else:
            w0 = w0/(24*3600)
        ax[2][1].axvline(np.log10(2*np.pi/w0/24/3600), color="navy", linestyle="--", lw=2, alpha=0.5, label="$\\tau_{{GP}}:{period:.2f}d$".format(LS = 'LS', period=2*np.pi/w0/24/3600))
        ax[2][1].set_ylabel("power", fontsize=15)
        ax[2][1].legend(loc="upper left")
        ax[2][1].xaxis.set_minor_locator(tck.AutoMinorLocator())
        ax[2][1].yaxis.set_minor_locator(tck.AutoMinorLocator())
        ax[2][1].set_ylim(ymax=ylim)

        #ax[3][0]
        ax[3][0].errorbar(self.x_new, self.y_new*1e6-pred_tot[1], pred_tot[2]-pred_tot[0], fmt=".k", alpha=0.3, label="residual")
        ax[3][0].set_ylabel("residual[ppm]", fontsize=15)
        ax[3][0].set_xlabel("Time-2454833[BKJD]", fontsize=15)
        ax[3][0].legend(loc="lower left")
        ax[3][0].xaxis.set_minor_locator(tck.AutoMinorLocator())
        ax[3][0].yaxis.set_minor_locator(tck.AutoMinorLocator())

        #ax[3][1]
        results = xo.estimators.lomb_scargle_estimator(
              self.x_new, self.y_new*1e6 - pred_tot[1], pred_tot[2]-pred_tot[0], max_peaks=1, min_period=self.min_period, max_period=self.max_period,
              samples_per_peak=100)

        if len(results["peaks"])==0:
            results = xo.estimators.lomb_scargle_estimator(
                self.x_new, self.y_new*1e6 - pred_tot[1], max_peaks=1, min_period=self.min_period, max_period=self.max_period,
                samples_per_peak=100)

        peak = results["peaks"][0]
        freq, power = LombScargle(self.x_new, self.y_new*1e6 - pred_tot[1], pred_tot[2]-pred_tot[0]).autopower(minimum_frequency=1/self.max_period, maximum_frequency=1/self.min_period)
        if np.any(np.isnan(power)):
            freq, power = LombScargle(self.x_new, self.y_new*1e6 - pred_tot[1]).autopower(minimum_frequency=1/self.max_period, maximum_frequency=1/self.min_period)

        ax[3][1].plot(-np.log10(freq), power, "k")
        ax[3][1].axvline(np.log10(peak["period"]), color="crimson", linestyle="--", lw=2, alpha=0.5, label="$P_{{LS}}:{period:.2f}d$".format(LS = 'LS', period=peak['period']))
        ax[3][1].set_ylabel("power", fontsize=15)
        ax[3][1].set_xlabel("$log_{10}(period)$", fontsize=15)
        ax[3][1].legend(loc="upper left")
        ax[3][1].xaxis.set_minor_locator(tck.AutoMinorLocator())
        ax[3][1].yaxis.set_minor_locator(tck.AutoMinorLocator())
        ax[3][1].set_ylim(ymax=ylim)
        
        return fig
