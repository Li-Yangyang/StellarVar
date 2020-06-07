#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module provides various helper functions."""
import pkg_resources

import pandas as pd
import numpy as np
import kplr

KOI_CATA = "/data/mcquillan_acf_kois.txt"
KOI_CATA = pkg_resources.resource_filename(__name__, KOI_CATA)

def koi2kid(koi_id):
    """
    Given KOI id, this function will find coordinated kepler id (KIC id) using the catalog provided by user
    param koi_id: KOI id num
    """
    client = kplr.API("./data/")
    koi = client.koi(koi_id + 0.01)
    kepid = koi.kepid
    return kepid

class Planets(object):
    """
    Initialize planets object with period, duration and epoch with corresponding errors.
    """
    def __init__(self, period, epoch, duration, name=''):
        if type(period) in (float, int):
            period = (period, np.nan, np.nan)
        if type(epoch) in (float, int):
            epoch = (epoch, np.nan, np.nan)
        if type(duration) in (float, int):
            duration = (duration, np.nan, np.nan)

        assert len(period) == 3
        assert len(epoch) == 3
        assert len(duration) == 3
        
        self._period = tuple(period)
        self._epoch = tuple(epoch)
        self._duration = tuple(duration)

        self.name = name

    @property
    def period(self):
        return self._period[0]

    @property
    def epoch(self):
        return self._epoch[0]

    @property
    def e1_period(self):
        return self._period[1]
    
    @property
    def e2_period(self):
        return self._period[2]

    @property
    def e1_epoch(self):
        return self._epoch[1]
    
    @property
    def e2_epoch(self):
        return self._epoch[2]
    
    @property
    def duration(self):
        if self._duration[0] == None:
            self._duration[0] = np.nan 
        return self._duration[0]/24.0
    
    @property
    def e1_duration(self):
        if type(self._duration[1]) == type(None):
            duration_e1 = 0
        else:
            duration_e1 = self._duration[1]
        return duration_e1/24.0
    
    @property
    def e2_duration(self):
        if type(self._duration[2]) == type(None):
            duration_e2 = 0
        else:
            duration_e2 = self._duration[2]
        return duration_e2/24.0
    
def kepler_planet(koi_id, i=None):
    """
    Find exist koi planets in the star and assemble parameters into Planet class
    """
    client = kplr.API("./data/")
    
    if type(i)==int:
        ilist = [i]
    elif i is None:
        client = kplr.API("./data/")
        koi = client.koi(koi_id + 0.01)
        count = koi.koi_count     
        clist = range(1, count+1)
    else:
        clist = i
    koi_planets_list = [koi_id + i*0.01 for i in clist]
    
    planets = []
    for k in koi_planets_list:
        p = client.koi(k)
        planets.append(Planets((p.koi_period, p.koi_period_err1, p.koi_period_err2),
                              (p.koi_time0bk, p.koi_time0bk_err1, p.koi_time0bk_err2), 
                              (p.koi_duration, p.koi_duration_err1, p.koi_duration_err2),
                               name=p.kepoi_name))
    return planets  

def window_rms(a, window_size):
    """
    Compute rms within given mean value window
    """
    a2 = np.power(a,2)
    window = np.ones(window_size)/float(window_size)
    return np.sqrt(np.convolve(a2, window, 'same'))
#def my_custom_corrector_func(lc):
#    corrected_lc = lc.normalize().flatten(window_length=401, return_trend=True)[1]
#    return corrected_lc
