# -*- coding: utf-8 -*-
"""Defines StellarLightCurveFile classes, i.e. files that contain StellarLightCurves."""

from __future__ import print_function, division
import pkg_resources

import numpy as np
import warnings
from utils import *
from lightkurve import search_lightcurvefile
import copy

import matplotlib.pylab as plt

MPLSTYLE = "/data/lightcurve.mplstyle"
MPLSTYLE = pkg_resources.resource_filename(__name__, MPLSTYLE)

class StellarLightCurve(object):
    """
    Implements a simple class for a generic light curve.
    Attributes:
    -----------
    (Cont.)
    """
    def __init__(self, mission, xoi_id = -1.0, xic_id = -1.0, mask_planets = True):
        self.xic_id = xic_id
        self.xoi_id = xoi_id
        self.mission = mission
        if self.xic_id == -1.0:
            if self.xoi_id == -1.0:
                raise SyntaxError("You miss the input id of the star")
            else:
                self.xic_id = koi2kid(self.xoi_id)
        self.mask_planets = mask_planets
        
        start = 0
        lcf = None
        if mission == "Kepler":
            prefix = "KIC"
        if mission == "TESS":
            prefix = "TIC"
        while type(lcf) is type(None):
            try:
                search_lc = search_lightcurvefile(prefix + str(self.xic_id), quarter=start)
                if  len(search_lc.target_name)!=0:
                    lcf = search_lightcurvefile(prefix + str(self.xic_id), quarter=start).download(target=str(self.xic_id)).PDCSAP_FLUX.normalize()
                    break
                else:
                    start = start + 1
            except AttributeError:
                start = start + 1
                #print(lcf, type(lcf))
                pass
            
        for q in range(start+1,17):
            try:
                lcf = lcf.append(search_lightcurvefile(prefix + str(self.xic_id), quarter=q).download(target=str(self.xic_id)).PDCSAP_FLUX.normalize())
            except AttributeError:
                continue
            
        self.lcf = lcf
        self._mask = None
        
    def copy(self):
        """Returns a copy of the LightCurve object.

        This method uses the `copy.deepcopy` function to ensure that all
        objects stored within the LightCurve are copied (e.g. time and flux).

        Returns
        -------
        lc_copy : LightCurve
            A new `LightCurve` object which is a copy of the original.
        """
        return copy.deepcopy(self)
        
    def planets(self, koi_id):
        planets = kepler_planet(koi_id)
        return planets
    
    @property
    def mask(self):
        time_min = self.lcf.time.min()
        time_max = self.lcf.time.max()
        t_masked = self.lcf.time
        p = self.planets(self.xoi_id)
        for i in range(len(p)):
            no_epoch_up = int(np.ceil((time_max - p[i].epoch)/p[i].period))
            for j in range(no_epoch_up):
                tt = p[i].epoch + j*p[i].period
                t_bound1 = tt - (p[i].duration + p[i].e1_duration) * 2
                t_bound2 = tt + (p[i].duration + p[i].e1_duration) * 2
                t_masked = np.ma.masked_inside(t_masked, t_bound1, t_bound2)
                
            no_epoch_low = int(np.ceil((p[i].epoch - time_min)/p[i].period))
            for j in range(no_epoch_low):
                tt = p[i].epoch - j*p[i].period
                t_bound1 = tt - (p[i].duration + p[i].e1_duration) * 2
                t_bound2 = tt + (p[i].duration + p[i].e1_duration) * 2
                t_masked = np.ma.masked_inside(t_masked, t_bound1, t_bound2)
                
        self._mask = t_masked.mask
        
        return self._mask
    
    def masked(self):
        lc = self.copy()
        mask = lc.mask
        lc.lcf.time = np.ma.array(lc.lcf.time, mask=mask).compressed()
        lc.lcf.flux = np.ma.array(lc.lcf.flux, mask=mask).compressed()
        lc.lcf.flux_err = np.ma.array(lc.lcf.flux_err, mask=mask).compressed()
        
        return lc
    
    def scale(self, scale_rate=1e3):
        lc = self.copy()
        x = np.ascontiguousarray(lc.lcf.time, dtype=np.float64)
        y = np.ascontiguousarray(lc.lcf.flux, dtype=np.float64)
        yerr = np.ascontiguousarray(lc.lcf.flux_err, dtype=np.float64)
        mu = np.nanmean(y)
        y = (y / mu - 1) * scale_rate
        yerr = yerr * scale_rate / mu
        lc.lcf.time = x
        lc.lcf.flux = y
        lc.lcf.flux_err = yerr
        
        return lc

    
    def bin(self, binsize = 13, method = 'median'):
        """
        Ｔhis module is from lightkurve.
        
        Bins a lightcurve in blocks of size `binsize`.

        The value of the bins will contain the mean (`method='mean'`) or the
        median (`method='median'`) of the original data.  The default is mean.

        Parameters
        ----------
        binsize : int
            Number of cadences to include in every bin.
        method: str, one of 'mean' or 'median'
            The summary statistic to return for each bin. Default: 'mean'.

        Returns
        -------
        binned_lc : LightCurve object
            Binned lightcurve.

        Notes
        -----
        - If the ratio between the lightcurve length and the binsize is not
          a whole number, then the remainder of the data points will be
          ignored.
        - If the original lightcurve contains flux uncertainties (flux_err),
          the binned lightcurve will report the root-mean-square error.
          If no uncertainties are included, the binned curve will return the
          standard deviation of the data.
        - If the original lightcurve contains a quality attribute, then the
          bitwise OR of the quality flags will be returned per bin.
        """
        available_methods = ['mean', 'median']
        if method not in available_methods:
            raise ValueError("method must be one of: {}".format(available_methods))
        methodf = np.__dict__['nan' + method]
        
        n_bins = self.lcf.flux.size // binsize
        binned_lc = self.copy()
        indexes = np.array_split(np.arange(len(self.lcf.time)), n_bins)
        binned_lc.lcf.time = np.array([methodf(self.lcf.time[a]) for a in indexes])
        binned_lc.lcf.flux = np.array([methodf(self.lcf.flux[a]) for a in indexes])
        
        if np.any(np.isfinite(self.lcf.flux_err)):
            # root-mean-square error
            binned_lc.lcf.flux_err = np.array(
                [np.sqrt(np.nansum(self.lcf.flux_err[a]**2))
                 for a in indexes]
            ) / binsize
        else:
            # Make them zeros.
            binned_lc.lcf.flux_err = np.zeros(len(binned_lc.lcf.flux))
            
        if hasattr(binned_lc.lcf, 'quality'):
            # Note: np.bitwise_or only works if there are no NaNs
            binned_lc.lcf.quality = np.array(
                [np.bitwise_or.reduce(a) if np.all(np.isfinite(a)) else np.nan
                 for a in np.array_split(self.lcf.quality, n_bins)])
        if hasattr(binned_lc.lcf, 'cadenceno'):
            binned_lc.lcf.cadenceno = np.array([np.nan] * n_bins)
        if hasattr(binned_lc.lcf, 'centroid_col'):
            # Note: nanmean/nanmedian yield a RuntimeWarning if a slice is all NaNs
            binned_lc.lcf.centroid_col = np.array(
                [methodf(a) if np.any(np.isfinite(a)) else np.nan
                 for a in np.array_split(self.lcf.centroid_col, n_bins)])
        if hasattr(binned_lc.lcf, 'centroid_row'):
            binned_lc.lcf.centroid_row = np.array(
                [methodf(a) if np.any(np.isfinite(a)) else np.nan
                 for a in np.array_split(self.lcf.centroid_row, n_bins)])

        return binned_lc
    
    def removed_nans(self):
        """
        Remove cadences where flux is NaN.
        
        Returns
        -------
        clean_lightcurve : StelalrLightCurve object
            A new ``StellarLightCurve`` from which NaNs fluxes have been removed.
        """
        lc = self.copy()
        lc.lcf = lc.lcf.remove_nans()
        return lc  # This will return a sliced copy

    def sub_lc(self, t0, te):
        """
        Take the sub time stamps of the lightcurve.
        
        Returns
        -------
        pieced_lightcurve: StellarLightCurve object
            A new ``StellarLightCurve`` which have is the subdiary of the origin light curve.
        """
        lc = self.copy()
        idx = np.where((lc.lcf.time>=t0)&(lc.lcf.time<=te))
        lc.lcf.time = lc.lcf.time[idx]
        lc.lcf.flux = lc.lcf.flux[idx]
        lc.lcf.flux_err = lc.lcf.flux_err[idx]

        return lc
        
    def _create_plot(self, method='plot', ax=None, data='origin',
                     xlabel=None, ylabel=None, title='', style='lightkurve',
                     show_colorbar=True, colorbar_label='',
                     **kwargs):
        """
        This module is from lightkurve
        
        Implements `plot()`, `scatter()`, and `errorbar()` to avoid code duplication.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """
        # Configure the default style
        if style is None or style == 'lightkurve':
            style = MPLSTYLE
        # Default xlabel
        if xlabel is None:
            if self.lcf.time_format == 'bkjd':
                xlabel = 'Time - 2454833 [BKJD days]'
            elif self.lcf.time_format == 'btjd':
                xlabel = 'Time - 2457000 [BTJD days]'
            elif self.lcf.time_format == 'jd':
                xlabel = 'Time [JD]'
            else:
                xlabel = 'Time'
        # Default ylabel
        ylabel = 'Normalized Flux'
            
        # Default legend label
        if ('label' not in kwargs):
            kwargs['label'] = self.lcf.label

        # Choose which set of data to plot
        if data == 'origin':
            time, flux, flux_err = self.lcf.time, self.lcf.flux, self.lcf.flux_err
        elif data == 'masked':
            time, flux, flux_err = self.masked().lcf.time, self.masked().lcf.flux, self.masked().lcf.flux_err

        # Make the plot
        with plt.style.context(style):
            if ax is None:
                fig, ax = plt.subplots(1)
            if method == 'scatter':
                sc = ax.scatter(time, flux, **kwargs)
                # Colorbars should only be plotted if the user specifies, and there is
                # a color specified that is not a string (e.g. 'C1') and is iterable.
                if show_colorbar and ('c' in kwargs) and \
                   (not isinstance(kwargs['c'], str)) and hasattr(kwargs['c'], '__iter__'):
                    cbar = plt.colorbar(sc, ax=ax)
                    cbar.set_label(colorbar_label)
                    cbar.ax.yaxis.set_tick_params(tick1On=False, tick2On=False)
                    cbar.ax.minorticks_off()
            elif method == 'errorbar':
                ax.errorbar(x=time, y=flux, yerr=flux_err, **kwargs)
            else:
                ax.plot(time, flux, **kwargs)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            # Show the legend if labels were set
            legend_labels = ax.get_legend_handles_labels()
            if (np.sum([len(a) for a in legend_labels]) != 0):
                ax.legend()

        return ax
    
    def plot(self, **kwargs):
        """
        This module is from lightkurve
        
        Plot the light curve using matplotlib's `plot` method.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.
        normalize : bool
            Normalize the lightcurve before plotting?
        xlabel : str
            Plot x axis label
        ylabel : str
            Plot y axis label
        title : str
            Plot set_title
        style : str
            Path or URL to a matplotlib style file, or name of one of
            matplotlib's built-in stylesheets (e.g. 'ggplot').
            Lightkurve's custom stylesheet is used by default.
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.plot`.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """
        return self._create_plot(method='plot', **kwargs)

    def scatter(self, colorbar_label='', show_colorbar=True, **kwargs):
        """
        This module is from lightkurve
        
        Plots the light curve using matplotlib's `scatter` method.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.
        normalize : bool
            Normalize the lightcurve before plotting?
        xlabel : str
            Plot x axis label
        ylabel : str
            Plot y axis label
        title : str
            Plot set_title
        style : str
            Path or URL to a matplotlib style file, or name of one of
            matplotlib's built-in stylesheets (e.g. 'ggplot').
            Lightkurve's custom stylesheet is used by default.
        colorbar_label : str
            Label to show next to the colorbar (if `c` is given).
        show_colorbar : boolean
            Show the colorbar if colors are given using the `c` argument?
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.scatter`.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """
        return self._create_plot(method='scatter', colorbar_label=colorbar_label,
                                 show_colorbar=show_colorbar, **kwargs)

    def errorbar(self, linestyle='', **kwargs):
        """
        This module is from lightkurve
        
        Plots the light curve using matplotlib's `errorbar` method.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.
        normalize : bool
            Normalize the lightcurve before plotting?
        xlabel : str
            Plot x axis label
        ylabel : str
            Plot y axis label
        title : str
            Plot set_title
        style : str
            Path or URL to a matplotlib style file, or name of one of
            matplotlib's built-in stylesheets (e.g. 'ggplot').
            Lightkurve's custom stylesheet is used by default.
        linestyle : str
            Connect the error bars using a line?
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.scatter`.

        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The matplotlib axes object.
        """
        if 'ls' not in kwargs:
            kwargs['linestyle'] = linestyle
        return self._create_plot(method='errorbar', **kwargs)
