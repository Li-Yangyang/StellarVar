import numpy as np
from scipy.signal import find_peaks
from scipy.stats import sigmaclip
import matplotlib.pylab as plt
from interpacf import interpolated_acf, dominant_period
import exoplanet as xo

def fig_local_view(lc, Prot):
    """
    display the local view of target lightcurve,
    the red vertical lines have separation as the mean fitting rotation period

    Returns
    -------
    figure:
        matplotlib.figure.Figure 
    """
    x = lc.lcf.time
    y = lc.lcf.flux
    yerr = lc.lcf.flux_err
    lower = np.quantile(x, [1/6.0, 1/2.0, 5/6.0]) - 100.0
    upper = np.quantile(x, [1/6.0, 1/2.0, 5/6.0]) + 100.0
    fig, ax = plt.subplots(3,1, figsize=(20, 8.5))
    for i in range(len(ax)):
        x_sub = x[(x>lower[i])&(x<upper[i])]
        y_sub = y[(x>lower[i])&(x<upper[i])]
        yerr_sub = yerr[(x>lower[i])&(x<upper[i])]
        ax[i].errorbar(x_sub, y_sub, yerr_sub, fmt='-k')
        if Prot >= 1.:
            peaks, _ = find_peaks(y_sub, distance=Prot)
        else:
            continue
        for p in peaks:
            ax[i].axvline(x_sub[p], color='r')
        ax[i].set_xlim(lower[i], upper[i])
        ax[i].set_ylabel("flux[ppm]")
    plt.xlabel("Time-2454833[BKJD]")

    return fig

def fig_acf(lc):
    """
    display the acf results of light curve. However, this package is not robust as 
    Mcquillian method. So we just take it as a reference. The acf accurate results are
    adopted from Mcq+ catalog
    
    Return:
    -------
    figure:
        matplotlib.figure.Figure
    """
    x = lc.lcf.time
    y = lc.lcf.flux
    min_period = np.max(sigmaclip(np.diff(x))[0])
    max_period = 0.5*(x.max() - x.min())
    lag, acf = interpolated_acf(x, y)
    period, fig = dominant_period(lag, acf, min=min_period, max=max_period, plot=True)

    return period, fig

def fig_lombscargle(lc, min_period=None, max_period=None, Pgp=None, Pmcq=None):
    """
    display the lombscargle plots towards the light curve.
    """
    x = lc.lcf.time
    y = lc.lcf.flux
    yerr = lc.lcf.flux_err
    if min_period == None and max_period == None:
        min_period = np.max(sigmaclip(np.diff(x))[0])
        max_period = 0.5*(x.max() - x.min())

    results = xo.estimators.lomb_scargle_estimator(
              x, y, yerr, max_peaks=1, min_period=min_period, max_period=max_period,
              samples_per_peak=100)

    if len(results["peaks"])==0:
        results = xo.estimators.lomb_scargle_estimator(
              x, y, max_peaks=1, min_period=min_period, max_period=max_period,
              samples_per_peak=100)

    peak = results["peaks"][0]
    freq, power = results["periodogram"]
    fig = plt.figure(figsize=(6.9, 5.5))
    plt.plot(-np.log10(freq), power, "k")
    plt.axvline(np.log10(peak["period"]), color="k", lw=2, alpha=0.5, label="$P_{{LS}}:{period:.2f}d$".format(LS = 'LS', period=peak['period']))
    if Pgp!=None:
       plt.axvline(np.log10(Pgp), color="C1", lw=2, alpha=0.5, label="$P_{{GP}}:{period:.2f}d$".format(GP = 'GP', period=Pgp))
    if Pmcq!=None:
       plt.axvline(np.log10(Pmcq), color="C2", lw=2, alpha=0.5, label="$P_{{Mcq}}:{period:.2f}d$".format(Mcq = 'Mcq', period=Pmcq))
    plt.xlim((-np.log10(freq)).min(), (-np.log10(freq)).max())
    plt.yticks([])
    plt.xticks(fontsize=20)
    plt.xlabel("log10(period)", fontsize=15)
    plt.ylabel("power", fontsize=15)
    plt.legend()

    return peak['period'], fig

def fig_psd(vgp, summary, lctype="granulation"):
    from scipy import signal
    fig = plt.figure(figsize=(6.9, 5.5))
    if lctype == "granulation":
        fluxes = vgp.y2
        d = np.median(np.diff(vgp.lc.lcf.time)) * 24* 3600
        fft = np.fft.rfft(fluxes)
        power = (fft * np.conj(fft)).real / len(fluxes)
        freq = np.fft.rfftfreq(len(fluxes), d)
        freq *= 1e6
        w0 = np.exp(summary.loc["granulation_logw0"]["mean"])
        w0 = w0/(24*3600) 
        S0 = np.exp(summary.loc["granulation_logSw4"]["mean"])/w0**4
        Q = np.exp(summary.loc["granulation_logQ"]["mean"])
        
        w = freq*1e-6 * 2*np.pi
        S_w = np.sqrt(8*np.pi) * S0*w0**4 / ((w**2-w0**2)**2+w0**2*w**2/Q**2)
        plt.loglog(freq, power/np.median(power), color="k")
        plt.loglog(freq, (S_w/np.median(S_w)), color="red")
        plt.ylim(np.min(S_w/np.median(S_w))*1e-2,np.max(S_w/np.median(S_w))*10)
        plt.ylabel("power")
        plt.xlabel("$\mu Hz$")
        plt.axvspan(1/(2.5*24*3600)*1e6, np.max(freq), lw=2, alpha=0.5)
    if lctype == "rotation":
        f, Pxx_den = signal.periodogram(vgp.y1, 1/np.median(np.diff(vgp.lc.lcf.time))/24/3600, detrend=False)
    if lctype == "total":
        f, Pxx_den = signal.periodogram(vgp.lc.lcf.flux, 1/np.median(np.diff(vgp.lc.lcf.time))/24/3600, detrend=False)
    #plt.xlim([1e-7, 1/1.5/24/3600])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('$PSD [V^{2}/Hz]$')

    return fig
