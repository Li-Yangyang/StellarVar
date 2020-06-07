#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from distutils.core import setup

setup(name='StellarVar',
      version='0.9',
      description='exoplanet based python tool for analysing stellar activity',
      author='Yangyang Li',
      author_email='yangyangli@ufl.edu',
      packages=['StellarVar'],
      package_data={'StellarVar': ['data/lightcurve.mplstyle', 'data/mcquillan_acf_kois.txt', 'data/cumulative.csv']},
      install_requires=[
          'kplr', 'exoplanet', 'lightkurve'
      ],
      )
