"""
General repository of functions of all kinds
=====================================

.. module:: lib_Miguel
   :platform: macOS
   :synopsis: tool library thats serves as a temporary repository of all the functions created by me

.. moduleauthor:: Miguel Donderis <mdondv@gmail.com>

Description needed.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import circmean

import logging

import os 
os.chdir('/Users/miguel/Desktop/')
os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin'

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{cmbright}')

logging.getLogger('lib_NeuroDyn').addHandler(logging.NullHandler())

from lib_analysis import save_plot
from lib_pdb_analysis import compute_weights_estim

__author__ = 'Miguel Donderis'

"""
Methods
-------
"""

def plot_estimation_weights(df, save=False, **kwargs):
    avg_choice = np.sort(df[df.do_choice == 1].loc[:, 'average'].unique())
    avg_nochoice = np.sort(df[df.do_choice == 0].loc[:, 'average'].unique())

    est_choice = df[df.do_choice == 1].loc[:, ['average', 'estim']].groupby('average').apply(circmean, low=-180, high=180).values
    est_nochoice = df[df.do_choice == 0].loc[:, ['average', 'estim']].groupby('average').apply(circmean, low=-180, high=180).values

    w, _, _ = compute_weights_estim(df)
    w_nochoice = np.array([w[0][1], w[1][1]])
    w_choice = np.array([w[0][0], w[1][0]]) 

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1.75, 1]})

    ax1.plot(avg_nochoice, est_nochoice, c='orange', linestyle='dashed', alpha=1, lw=3.5)
    ax1.plot(avg_choice, est_choice, c='blue', alpha=1, lw=3.5)
    ax1.plot(np.linspace(-20, 20, 100), np.linspace(-20, 20, 100), '--', c='grey', lw=1)

    ax1.set_yticks(np.arange(-20, 21, 10))
    ax1.tick_params(axis='both', which='major', labelsize=15)

    ax1.set_title('\\textbf{Estimation curves}', fontsize=20)
    ax1.legend(['No choice', 'Choice'], fontsize=12.5)
    ax1.set_xlim([-21, 21])
    ax1.set_ylim([-21, 21])

    ax2.plot(np.array(['First\n stimulus', 'Second\n stimulus']), w_nochoice, lw=1.5, c='blue')
    ax2.plot(np.array(['First\n stimulus', 'Second\n stimulus']), w_choice, lw=1.5, c='orange')

    ax2.scatter(np.array(['First\n stimulus', 'Second\n stimulus']), w_nochoice, lw=3, c='blue')
    ax2.scatter(np.array(['First\n stimulus', 'Second\n stimulus']), w_choice, lw=3, c='orange')

    ax2.annotate(w_choice[0].round(3), (0, w_choice[0] + 0.01), fontsize=12) 
    ax2.annotate(w_nochoice[0].round(3), (0, w_nochoice[0] - 0.025), fontsize=12) 
    ax2.annotate(w_choice[1].round(3), (1, w_choice[1] + 0.01), fontsize=12) 
    ax2.annotate(w_nochoice[1].round(3), (1, w_nochoice[1] - 0.03), fontsize=12) 

    ax2.spines.top.set_visible(False)
    ax2.spines.bottom.set_visible(True)
    ax2.spines.left.set_visible(True)
    ax2.spines.right.set_visible(False)
    ax2.spines.bottom.set_bounds((0, 1))
    ax2.tick_params(axis='both', which='major', labelsize=15)

    ax2.set_title('\\textbf{Stimulus weights}', fontsize=20)
    ax2.legend(['No choice', 'Choice'], fontsize=12.5)

    axes = [ax1, ax2]

    if save:
        directory = kwargs.get('fig_dir', 'figs/')
        filename = directory + f"pdb_est_weights"
        save_plot(fig, filename, **kwargs)

    return fig, axes

