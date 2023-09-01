"""
Post-decisional bias models. Analysis library (:mod:`lib_pdb_analysis`)
=======================================================================

.. currentmodule:: lib_pdb_analysis
   :platform: Linux
   :synopsis: module for analyzing simulated data from the ring attractor network performing a
              confirmation bias (post-decisional bias) task.

.. moduleauthor:: Jose M. Esnaola-Acebes <josemesnaola@gmail.com>

This library contains functions to process and analyze the data obtained through the scripts:
 * :mod:`ring_simulation`. When performing the confirmation bias (branch: confirmation_bias).

Classes
-------

.. autosummary::
   :toctree: generated/

   FitModel                  Metaclass for fitting the data to a model by maximizing loglikelihood
                              using the subplex algorithm.
   ChoiceSelectiveModelFast  Class that fits the choice-selecteive model (fast version).

Plotting methods
----------------

.. autosummary::
   :toctree: generated/

   plot_pm             Psychometric function
   plot_estimation     Estimation function
   plot_ppk            Psychophysical kernel

Implementation
--------------

.. todo::

   Give a brief description about the data-frames that are use in the different functions in this library.
"""

import time
from stat import S_ISREG, ST_CTIME, ST_MODE

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import nlopt
import pandas as pd
import statsmodels.api as sm
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.stats import logistic
from scipy.stats import norm, sem
import re
# from statsmodels.stats.anova import AnovaRM

from lib_analysis import log_reg, save_plot, compute_estimation, compute_pm_data
from lib_plotting import *
from lib_ring import get_phases_and_amplitudes_auto
from lib_sconf import now, check_overwrite

logging.getLogger('lib_pdb_analysis').addHandler(logging.NullHandler())

__author__ = 'Jose M. Esnaola-Acebes'
__docformat__ = 'reStructuredText'

"""
Functions for the model-based analysis
"""


def stim_permutations(first_stim_samples, second_stim_samples=None, extreme=False):
    if np.any(second_stim_samples is None):
        second_stim_samples = first_stim_samples
    perm = []
    ext1 = np.abs(np.max(first_stim_samples)) + np.abs(np.min(second_stim_samples))
    ext2 = np.abs(np.min(first_stim_samples)) + np.abs(np.max(second_stim_samples))
    for x1 in np.sort(first_stim_samples):
        for x2 in np.sort(second_stim_samples):
            if not extreme and np.abs(x1) + np.abs(x2) >= ext1 and np.abs(x1) + np.abs(x2) >= ext2 and np.sign(
                    x2) != np.sign(x1):
                pass
            else:
                perm.append([x1, x2])
    return np.array(perm)


def cumnormal(p, x):
    if p[1] == 0:
        y = (x == 0) * 1.0
    else:
        y = norm.cdf(x, p[0], p[1])
    return y


def cumnormal_ll(p, inputs, responses):
    """p:         tuple (mu, sigma)
       inputs:    array, first stimulus x_1 \in {-20, -10, 0, 10, 20}
       responses: array of ones and zeros CW choices
    """

    w = cumnormal(p, inputs)
    negll = -np.sum(responses * np.log(w) + (1 - responses) * np.log(1 - w))
    return negll


def fit_cumnormal(x, y, **kwargs):
    niter = 1000
    bias_start = 40.0 * np.random.sample(niter) - 20.0
    slope_start = 50.0 * np.random.sample(niter)
    iter_params = np.ones((niter, 2)) * np.nan
    neglliter = np.ones((niter, 1)) * np.nan
    min_options = {'disp': kwargs.get('disp', False), 'maxiter': kwargs.get('maxiP', 100),
                   'ftol': kwargs.get('ftolP', 0.01), 'eps': kwargs.get('epsP', 0.1),
                   'maxls': kwargs.get('maxlsP', 20), 'maxfun': kwargs.get('maxfeP', 100)}
    for i in range(niter):
        res = minimize(cumnormal_ll, np.array([bias_start[i], slope_start[i]]), args=(x, y),
                       bounds=((-100.0, 100.0), (0.0001, 100.0)),
                       options=min_options)
        logging.debug(
            'Minimization process was %s.' % (u"\u001b[32msuccessful" if res.success else u"\u001b[31munsuccessful"))
        iter_params[i] = res['x']
        neglliter[i] = cumnormal_ll(iter_params[i], x, y)
    idx = np.argmin(neglliter)
    bias = iter_params[idx, 0]
    slope = iter_params[idx, 1]
    return bias, slope


def get_files_by_date(dirpath=r'.'):
    # get all entries in the directory w/ stats
    entries = (os.path.join(dirpath, fn) for fn in os.listdir(dirpath))
    entries = ((os.stat(path), path) for path in entries)

    # leave only regular files, insert creation date
    entries = ((stat[ST_CTIME], path) for stat, path in entries if S_ISREG(stat[ST_MODE]))
    # NOTE: on Windows `ST_CTIME` is a creation date
    #  but on Unix it could be something else
    # NOTE: use `ST_MTIME` to sort by a modification date
    return sorted(entries)


def get_recent_file(path, name, extension='.npy'):
    allfiles = get_files_by_date(path)[::-1]
    name = path + '/' + name
    for (t, rfile) in allfiles:
        if rfile.startswith(name) and rfile.endswith(extension):
            return rfile
    return None


def compute_psychometric(dataframe, **kwargs):
    filename = kwargs.get('name', '')
    # Select the data:
    logging.info(f"Obtaining psychometric curve for '{filename}'...")
    # Select the data:
    try:
        dataframe = dataframe[(np.abs(dataframe.do_choice) == 1) & (np.abs(dataframe.binchoice) == 1)].copy()
    except AttributeError:
        logging.error(f"Something went wrong with the database '{filename}'. Check format...")

    logistic_fits = []
    logistic_points = []
    logistic_points_errors = []
    # Check whether there previously generated data
    saved_file = kwargs.get('psy_data', 'data_talluri/psychometric_%s.npy' % filename)
    logging.debug('Trying to recover the data from %s...' % saved_file)
    try:
        # noinspection PyTypeChecker
        saved_data = dict(np.load(saved_file, allow_pickle=True, encoding='bytes').item())
        logistic_fits = saved_data['logisticFits']
        logistic_points = saved_data['logisticPoints']
        logistic_points_errors = saved_data['logisticPoints_errors']
        logging.debug('Data successfully loaded.')
    except (IOError, AttributeError, FileNotFoundError):
        logging.debug('Psychometric data was not found. Computing now ...')
        logging.info('')
        for k, sj in enumerate(dataframe.subj.unique()):
            # noinspection PyProtectedMember
            logging.info('Fitting logistic curve for Subject %2d.' % sj)
            dat = dataframe[dataframe.subj == sj].copy()
            with np.errstate(divide='ignore', invalid='ignore'):
                logistic_fits.append(fit_cumnormal(dat.x1, (dat.binchoice > 0) * 1, **kwargs))

            logistic_points.append([])
            logistic_points_errors.append([])
            for x in np.sort(dat.x1.unique()):
                logistic_points[-1].append(np.mean((dat[dat.x1 == x].binchoice + 1) / 2.0))
                try:
                    logistic_points_errors[-1].append(sem((dat[dat.x1 == x].binchoice + 1) / 2.0))
                except FloatingPointError:
                    pass
        saved_file = 'data_talluri/psychometric_%s' % filename
        logging.debug('Writing psychometric data to %s.npy ...' % saved_file)
        data_to_be_saved = {'logisticFits': logistic_fits, 'logisticPoints': logistic_points,
                            'logisticPoints_errors': logistic_points_errors}
        np.save(saved_file, data_to_be_saved, allow_pickle=True)
    except IndexError:
        logging.error(
            "Error when trying to load data from %s. IndexError occurred: data format is probably wrong." % saved_file)

    return logistic_fits, logistic_points, logistic_points_errors


class FitModel:
    def __init__(self, subjs=(0,), dim=0, global_type='subplex', local_type=None):
        self.logger = logging.getLogger('lib_pdb_analysis.FitModel')
        self.subjects = subjs
        self.model_name = "None"
        self.dim = dim

        # TODO: make this more general
        algorithms = {'subplex': nlopt.LN_SBPLX, 'subplex_fortran': None, 'minimize': minimize}
        # Set the optimization algorithm
        if isinstance(algorithms[global_type], int):
            self.g_opt = nlopt.opt(algorithms[global_type], self.dim)
            self.g_opt.set_min_objective(self.fun)  # Function to be minimized
        else:
            logging.warning("Algorithm '%s' not implemented." % global_type)
            exit(-1)
        self.local_type = local_type

        # Counters
        self.evals = 0
        self.min_evals = 0
        self.subplex_evals = 0
        self.elapsed_time = 0

    def _fit_model(self, start_pt, **kwargs):
        lbound = kwargs.get('lbound', (0, 0, 0))
        ubound = kwargs.get('ubound', (50, 10, 10))
        # OPTIONS for subplex
        # self.g_opt.set_xtol_rel(kwargs.get('sxrtol', 0.1))
        # self.g_opt.set_xtol_abs(kwargs.get('sxatol', 0.0001))
        # self.g_opt.set_ftol_rel(kwargs.get('sftolr', 0.01))
        self.g_opt.set_ftol_abs(kwargs.get('sftola', 0.000001))
        self.g_opt.set_maxeval(kwargs.get('smaxeval', 100))
        self.g_opt.set_lower_bounds(np.array(list(lbound)))
        self.g_opt.set_upper_bounds(np.array(list(ubound)))
        self.subplex_evals = 0
        self.elapsed_time = 0
        # OPTIONS for minimize
        # noinspection PyTypeChecker
        min_bounds = list(np.array([lbound, ubound]).transpose().tolist())
        # min_options = {'disp': kwargs.get('disp', False), 'maxiter': kwargs.get('maxiM', 100),
        #                'ftol': kwargs.get('ftol', 0.0000000001), 'maxfun': kwargs.get('maxevalm', 100),
        #                'gtol': kwargs.get('gtol', 0.0000000001), 'xtol': 0.1, 'eps': np.finfo(float).eps}
        min_options = {'disp': kwargs.get('disp', False), 'maxiter': kwargs.get('maxiM', 100),
                       'maxfun': kwargs.get('maxevalm', 100)}
        min_options = kwargs.get('min_options', min_options)
        self.logger.debug(min_options)
        self.min_evals = 0
        # Call subplex
        if kwargs.get('subplex', False):
            self.logger.debug("Performing 'subplex' optimization.")
            self.evals = 0
            fit_params = self.g_opt.optimize(start_pt)
            self.logger.debug("'Subplex' optimization done.")
            self.subplex_evals = self.evals * 1
        else:
            fit_params = start_pt
        # Call minimize
        self.logger.debug("Minimizing.")
        self.evals = 0
        min_results = minimize(self.fun, fit_params, method=kwargs.get('mmethod', 'L-BFGS-B'),
                               bounds=min_bounds, options=min_options)
        self.min_evals = self.evals * 1
        logging.info('Minimization process was %s.' %
                     (u"\u001b[32msuccessful" if min_results.success else u"\u001b[31munsuccessful"))
        return min_results

    def fun(self, p, *args, **kwargs):
        """ Function to be minimized. Prototype. To be replaced by the actual function."""
        if len(args):
            # First argument should be the gradient
            # grad = args[0]
            if len(args[0]) > 0:
                args[0][:] = np.sum(2.0 * p)
        return np.sum(p ** 2)

    def aicbic(self, logl, numparams, numobs=None):
        """ AICBIC Akaike and Bayesian information criteria
            Description:
                Given optimized log-likelihood function values logL obtained by fitting
                a model to data, compute the Akaike (AIC) and Bayesian (BIC)
                information criteria. Since information criteria penalize models with
                additional parameters, aic and bic select models based on both goodness
                of fit and parsimony. When using either AIC or BIC, models that
                minimize the criteria are preferred.

            Input arguments:
            ****************
            :param logl: Vector of optimized log-likelihood objective function values associated with
                         parameter estimates of various models.
            :param numparams: Number of estimated parameters associated with each value in logL.
                              numParam may be a scalar applied to all values in logL, or a vector the
                              same length as logL. All elements of numParam must be positive integers.
            :param numobs: (optional) Sample sizes of the observed data associated with each value of logL.
                           numObs is required for computing BIC, but not AIC. numObs may be a scalar
                           applied to all values in logL, or a vector the same length as logL.
                           All elements numObs must be positive integers.

            Input arguments:
            ****************
            aic: Vector of AIC statistics associated with each logL objective function value.
                 The AIC statistic is defined as:
                     aic = -2*logL + 2*numParam
            bic: Vector of BIC statistics associated with each logL objective function value.
                 The BIC statistic is defined as:
                     bic = -2*logL + numParam*log(numObs)

            Reference:
            **********
            [1] Box, G.E.P., Jenkins, G.M., Reinsel, G.C., "Time Series Analysis:
                Forecasting and Control", 3rd edition, Prentice Hall, 1994.

            Code adapted from MathWorks (Copyright 1999-2010 The MathWorks, Inc.)
        """

        # Check for a vector
        if isinstance(logl, (list, np.ndarray)):
            logl = np.array(logl)
        else:
            self.logger.error("The input parameter 'logl' must be a 1-d vector (list or ndarray).")
            raise TypeError
        # Ensure numParam is a scalar, or compatible vector, of positive integers:
        if isinstance(numparams, (list, np.ndarray, int)):
            # Must be integers
            if isinstance(numparams, (list, np.ndarray)):
                numparams = np.array(numparams, dtype=int)
                if np.any(numparams <= 0):
                    self.logger.error("Numparams must contain positive integer values.")
                    raise ValueError
                if len(numparams) != len(logl):
                    self.logger.error(
                        "Lenght of numparams (%d) does not match length of logl (%d)." % (len(numparams), len(logl)))
                    raise IndexError
            else:
                numparams = int(numparams)
                if numparams <= 0:
                    self.logger.error("Numparams must contain positive integer values.")
                    raise ValueError
                numparams = np.ones(len(logl), dtype=int) * numparams
        # Ensure numObs is a scalar, or compatible vector, of positive integers:
        if numobs != None:
            if isinstance(numobs, (list, np.ndarray)):
                numobs = np.array(numobs)
                if len(numobs) != len(logl):
                    self.logger.error(
                        "Lenght of numparams (%d) does not match length of logl (%d)." % (len(numobs), len(logl)))
                    raise IndexError
            elif isinstance(numobs, int):
                if numobs <= 0:
                    self.logger.error("Numobs must contain positive integer values.")
                    raise ValueError
                numobs = np.ones(len(logl), dtype=int) * numobs
            else:
                self.logger.error("Numobs must be a positive integer scalar or vector.")
                raise TypeError

        # Compute AIC
        aic = -2.0 * logl + 2.0 * numparams

        # Compute BIC if requested:
        if numobs != None:
            bic = -2.0 * logl + numparams * np.log(numobs)
        else:
            bic = []

        return aic, bic


class ChoiceSelectiveModelFast(FitModel):
    def __init__(self, dataframe, subjs, psycho_fits, num_parameters, name='Selective Gain', **kwargs):
        FitModel.__init__(self, subjs, dim=num_parameters)
        self.logger = logging.getLogger('pdb.ChoiceSelectiveModel')
        self.model_name = name
        self.data = dataframe.copy()
        self.noise_fits = psycho_fits[:, 1]
        self.bias_fits = -psycho_fits[:, 0]
        self.compare = kwargs.get('compare', False)
        trial_opts = ['all', 'error', 'zero']
        self.trials = kwargs.get('trials', 'all')
        if self.trials not in trial_opts:
            self.trials = trial_opts[0]

        # ###############################################
        # Preprocess the data (take only the useful data)
        self.data = dataframe[np.isin(dataframe.subj, subjs, invert=False)].copy()

        # Take only choice trials
        choice_trials = (self.data.do_choice == 1) & (np.abs(self.data.binchoice) == 1)
        if self.trials == 'error':
            choice_trials = choice_trials & (self.data.bincorrect == 0)
        elif self.trials == 'zero':
            choice_trials = choice_trials & (self.data.x1 == 0)
        self.data = self.data[choice_trials]

        # Consistency cannot be define for trials where x2 = 0. Remove them.
        if self.compare:
            trls2use = (self.data.x2 != 0) & (self.data.x1 != 0)
        else:
            trls2use = (self.data.x2 != 0)
        self.data = self.data[trls2use]
        # ###############################################

        # ##########################################################################
        # Create a dataframe with unique combinations of the experimental conditions
        combinations = stim_permutations(list(np.sort(self.data.x1.unique())), extreme=True)
        # combinations = stim_permutations([-20, -10, 0, 10, 20])
        # combinations = stim_permutations([-20, -15, -10, -5, 0, 5, 10, 15, 20])
        # combinations = stim_permutations(list(np.arange(-20, 21, 1)), extreme=True)
        combinations = combinations[~(combinations[:, 1] == 0)]
        self.dat_unique = pd.DataFrame(combinations, columns=['x1', 'x2'])
        self.dat_unique['b'] = 1
        self.dat_unique = self.dat_unique.append(pd.DataFrame(combinations, columns=['x1', 'x2']), ignore_index=True)
        self.dat_unique.loc[self.dat_unique['b'].isnull(), 'b'] = -1
        self.dat_unique['b'] = self.dat_unique.b.astype(int)
        self.dat_unique['counts'] = 0
        self.dat_unique['label'] = np.arange(len(self.dat_unique))

        # add a unique counter to each trial in the dataset indicating which combination is
        self.data['comb'] = np.nan
        for idx, row in self.dat_unique.iterrows():
            ma_comb = (self.data.x1 == row.x1) & (self.data.x2 == row.x2) & (self.data.binchoice == row.b)
            self.data.loc[ma_comb, 'comb'] = row.label
            self.dat_unique['counts'].iloc[idx] = np.array(ma_comb, dtype=int).sum()
        ma_consistent = (self.dat_unique.x2 * self.dat_unique.b > 0)
        self.dat_unique['trial'] = 'consistent'
        self.dat_unique.loc[~ma_consistent, 'trial'] = 'inconsistent'

        self.dat_uniq = {'consistent': self.dat_unique.loc[self.dat_unique['trial'] == 'consistent'].copy(),
                         'inconsistent': self.dat_unique.loc[self.dat_unique['trial'] == 'inconsistent'].copy()}

        self.num_comb = len(self.dat_uniq['consistent'])

        # ##########################################################################
        # Variables which are used in function fun (mutable), adjusted for each subject
        self.dat_subj = None
        self.dat = None
        self.psycho_noise = self.noise_fits[0]
        self.psycho_bias = self.bias_fits[0]
        self.num_trls = int(1000)
        self.x1 = np.array(self.dat_uniq['consistent'].x1)
        self.x2 = np.array(self.dat_uniq['consistent'].x2)
        self.realdecision = np.ones(self.num_trls, dtype=int)
        self.realevaluation = np.ones(self.num_trls) * 0.0
        self.levaluation = np.ones((self.num_trls, 1)) * np.nan

        self.trialtype = None

        # Variables which are used in function fun (non-mutable)
        step = kwargs.get('xstep', 0.05)  # This stepsize is VERY small!!
        self.x = np.arange(-180.0, 180, step)
        self.xt = self.x.reshape((len(self.x), 1))
        self.xt_m = (self.x * np.ones((self.num_comb, 1))).T
        self.oneid = np.identity(self.num_comb, dtype=bool)
        self.sign = {1: (self.x < 0), -1: (self.x >= 0)}
        self.ma_signs = None

        # Variables storing the results
        self.fparams = {'consistent': [], 'inconsistent': []}
        self.fnlogl = {'consistent': [], 'inconsistent': []}
        self.subj_params = []
        self.subj_params_o = {'consistent': [], 'inconsistent': []}
        self.subj_NlogL = []
        self.bic = []

        # ANOVA measures (different approach to Bharath)
        self.cols = ['subj_id', 't', 'o', 'w']
        self.dataframe = pd.DataFrame(columns=self.cols)
        self.res2way = None
        self.ar = {}

    def __call__(self, *args, **kwargs):
        lbound = kwargs.get('lbound', (0, 0, 0))
        ubound = kwargs.get('ubound', (50, 10, 10))
        kwargs['lbound'] = lbound
        kwargs['ubound'] = ubound

        # Check if previous data exists
        simulation = kwargs.get('simulation', '')
        filename = kwargs.get('name', '')
        saved_file = kwargs.get('csm_data', 'data_talluri/ChoiceSelectiveModel_%s_best.npy' % filename)
        self.logger.debug('Trying to recover the best data from %s...' % saved_file)
        try:
            # noinspection PyTypeChecker
            saved_data = dict(np.load(saved_file, allow_pickle=True).item())
            compute = kwargs.get('Force_compute')
        except IOError:
            # Check any file which starts wich ChoiceSelectiveModel_data
            compute = True
            try:
                saved_file = get_recent_file('data_talluri', 'ChoiceSelectiveModel_%s_' % filename)
                self.logger.debug('Trying to recover any data from %s...' % saved_file)
                # noinspection PyTypeChecker
                saved_data = dict(np.load(saved_file, allow_pickle=True).item())
            except (OSError, IOError, TypeError, AttributeError):
                saved_data = None

        if saved_data != None:
            self.subj_params_o = saved_data['subj_params_o']
            if not compute:
                self.subj_params = saved_data['subj_params']
                self.subj_NlogL = saved_data['subj_NlogL']
                self.bic = saved_data['bic']
                self.dataframe = saved_data['dataframe']
                self.res2way = saved_data['anova_table']
                self.ar = saved_data['anova']
                self.data_to_be_saved = {'subj_params': self.subj_params, 'subj_NlogL': self.subj_NlogL,
                                         'bic': self.bic, 'dataframe': self.dataframe, 'anova_table': self.res2way,
                                         'anova': self.ar, 'subj_params_o': self.subj_params_o}
            self.logger.debug('Data successfully loaded.')

        if compute:
            logging.debug('Computing now ...')
            # Loop of subjects
            for k, sj in enumerate(self.subjects):
                self.logger.info('Subject %2d: Selecting the data and initializing parameters.' % sj)
                # Variables storing the results
                self.fparams = {'consistent': [], 'inconsistent': []}
                self.fnlogl = {'consistent': [], 'inconsistent': []}

                self.psycho_bias = self.bias_fits[k]
                self.psycho_noise = self.noise_fits[k]
                self.dat_subj = self.data.loc[self.data.subj == sj]
                self.logger.debug('Shape of filtered data: %s' % str(np.shape(self.dat_subj)))
                for trialtypes in ['consistent', 'inconsistent']:
                    if trialtypes == 'consistent':
                        self.trialtype = (np.sign(self.dat_subj.binchoice) == np.sign(self.dat_subj.x2))
                    elif trialtypes == 'inconsistent':
                        self.trialtype = (np.sign(self.dat_subj.binchoice) != np.sign(self.dat_subj.x2))
                    self.dat = self.dat_subj.loc[self.trialtype]
                    self.logger.debug('Shape of refiltered data: %s' % str(np.shape(self.dat)))
                    self.x1, self.x2 = np.array(self.dat_uniq[trialtypes].x1 + self.psycho_bias), np.array(
                        self.dat_uniq[trialtypes].x2 + self.psycho_bias)
                    self.ones = np.ones(len(self.x1))
                    self.realdecision = np.array(self.dat.binchoice)
                    self.realevaluation = np.array(self.dat.estim)
                    self.num_trls = len(self.realevaluation)
                    self.levaluation = np.ones(self.num_trls) * np.nan

                    # Add estimates to the unique dataframe
                    self.dat_uniq[trialtypes]['estim'] = None
                    for i, (idx, row) in enumerate(self.dat_uniq[trialtypes].iterrows()):
                        self.dat_uniq[trialtypes]['estim'].iloc[i] = np.array(
                            self.dat.estim.loc[self.dat.comb == row.label]).tolist()
                    self.dat_uniqnow = self.dat_uniq[trialtypes].copy()
                    self.ma_signs = []
                    for idx, row in self.dat_uniq[trialtypes].iterrows():
                        self.ma_signs.append(self.sign[row.b])
                    self.ma_signs = np.array(self.ma_signs).T

                    self.logger.info("%14s Fitting model '%s' for %s trials (%d)." % (
                        '->', self.model_name, trialtypes, self.num_trls))
                    if saved_data != None:
                        starting_pt = np.array(self.subj_params_o[trialtypes][k])
                    else:
                        starting_pt = np.random.choice(np.arange(0.5, ubound[0], 0.5), 1).tolist()
                        starting_pt += np.random.choice(np.arange(0.05, ubound[1], 0.05), 2).tolist()
                        starting_pt = np.array(starting_pt)

                    self.logger.info("%14s Initial parameters are: %s." % ('->', starting_pt))
                    min_results = self._fit_model(starting_pt, **kwargs)
                    prms = min_results.x
                    self.dataframe = self.dataframe.append(pd.DataFrame([[sj, trialtypes, 1, prms[1]]],
                                                                        columns=self.cols,
                                                                        index=[len(self.dataframe.index)]))
                    self.dataframe = self.dataframe.append(pd.DataFrame([[sj, trialtypes, 2, prms[2]]],
                                                                        columns=self.cols,
                                                                        index=[len(self.dataframe.index)]))

                    self.fparams[trialtypes].append(min_results.x)
                    self.fnlogl[trialtypes].append(min_results.fun)
                    try:
                        self.subj_params_o[trialtypes][k] = min_results.x
                    except IndexError:
                        self.subj_params_o[trialtypes].append(min_results.x)

                # Gather the fitted parameters
                self.subj_params.append(np.concatenate(
                    [np.array(self.fparams['consistent']), np.array(self.fparams['inconsistent'])]).T.ravel())
                self.subj_NlogL.append(np.array(self.fnlogl['consistent']) + np.array(self.fnlogl['inconsistent']))
                self.logger.debug(f"Total score of fit: {self.subj_NlogL[-1]}")
                # Compute the AICBIC
                aic, bic = self.aicbic(-self.subj_NlogL[-1], 6, len(self.dat_subj))
                self.bic.append(bic)

            # Once finished, print the gain
            self.logger.info(f"Gain '{filename}' finished. Saving now ...")
            # Compute 2-way ANOVA measure
            # TODO: fix AttributeError
            self.dataframe.subj_id = self.dataframe.subj_id.astype('int16')
            # Overide the Anova analysis
            # try:
            #     aovrm2way = AnovaRM(self.dataframe, 'w', 'subj_id', within=['t', 'o'])
            #     self.res2way = aovrm2way.fit().anova_table
            #     self.ar['f'] = self.res2way.get_value('t:o', 'F Value')
            #     self.ar['p'] = self.res2way.get_value('t:o', 'Pr > F')
            #     self.ar['num'] = self.res2way.get_value('t:o', 'Num DF')
            #     self.ar['den'] = self.res2way.get_value('t:o', 'Den DF')
            # except (UnboundLocalError, AttributeError):
            #     self.logger.error('Anova analysis failed.')

            # Save the data
            if filename != '':
                saved_file = 'data_talluri/ChoiceSelectiveModel_%s_%s' % (filename, ('-'.join(now('_', '_'))))
            else:
                saved_file = 'data_talluri/ChoiceSelectiveModel_%sdata_%s' % (simulation, ('-'.join(now('_', '_'))))
            self.logger.debug('Writing data to %s.npy ...' % saved_file)
            self.data_to_be_saved = {'subj_params': self.subj_params, 'subj_NlogL': self.subj_NlogL,
                                     'bic': self.bic, 'dataframe': self.dataframe, 'anova_table': self.res2way,
                                     'anova': self.ar, 'subj_params_o': self.subj_params_o}
            np.save(saved_file, self.data_to_be_saved, allow_pickle=True)

    def fun(self, p, *args, **kwargs):
        time1 = time.time()
        self.evals += 1
        self.logger.debug('Function evaluation: %6d.' % self.evals)
        self.logger.debug('Parameters: %s' % p)
        # Data and variables from outside (initialization outside the functions
        # will speed-up the fitting process).
        #    (for a specific subject): "data" , "psycho_bias", "psycho_noise"
        #    ("x1", "x2") (stimulus arrays), "realdecision", "realevaluation"
        #    "number of trials" (length of x1), "orientation vector" with a fixed "stepsize"

        # The function builds the probability distribution functions (pdf) of the model:
        #   Y = w1*X1 + w2*X2 + noise

        # Equivalent vectorial representation
        y1cc = norm.pdf(self.xt, self.x1 * p[1], self.ones * self.psycho_noise * np.abs(p[1]))
        y1cc[self.ma_signs] = 0
        areas = np.trapz(y1cc.T, self.xt_m.T)
        y1cc = y1cc / areas
        y2cc = norm.pdf(self.xt, self.x2 * p[2],
                        self.ones * np.sqrt((self.psycho_noise * np.abs(p[2])) ** 2 + p[0] ** 2))

        n1 = fftconvolve(y1cc, y2cc, mode='same', axes=0)
        # The fft is not perfect: negative (but of the order of e-15) values are zero
        n1[n1 < 0] = 0
        areas = np.trapz(n1.T, self.xt_m.T)
        n1 = n1 / areas

        # TODO: find a better way of obtaining the evaluations of the estimations
        self.levaluation = []
        for k, (idx, row) in enumerate(self.dat_uniqnow.iterrows()):
            f = interp1d(self.x, n1[:, k])
            self.levaluation.append(f(np.array(row.estim)))
        self.levaluation = np.concatenate(np.array(self.levaluation))
        result = -np.sum(np.log(self.levaluation))

        self.logger.debug('Evaluation result: %s\t Time: %f.' % (result, time.time() - time1))
        self.elapsed_time += (time.time() - time1)
        if np.isnan(result):
            return 1E100
        elif np.isinf(np.abs(result)):
            return 1E100
        else:
            return result

    def generate_surrogate_data(self, **kwargs):
        dataframe = kwargs.get('dataframe', self.data).copy()
        self.logger.debug("Generating surrogate data from 'gain' model.")
        prmts = kwargs.get('prmts', {'consistent': [5.0, 0.29, 0.5], 'inconsistent': [4.0, 0.31, 0.15]})
        bias = kwargs.get('psy_bias', self.bias_fits)
        slope = kwargs.get('psy_slope', self.noise_fits)

        # compute the mean (or median) value of the bias and slope of the psychometric function
        self.logger.debug('Obtaining psychometric curve.')
        if kwargs.get('median', False):
            m_bias = np.median(bias)
            m_slope = np.median(slope)
        else:
            m_bias = np.mean(bias)
            m_slope = np.mean(slope)
        psychometric_f = np.array(cumnormal([m_bias, m_slope], np.sort(dataframe.x1.unique())))
        # Generate binary choices from the psychometric function
        draw = np.random.rand(len(dataframe))
        self.logger.debug('Generating binary choices.')
        for k, x in enumerate(np.sort(dataframe.x1.unique())):
            dataframe.loc[(dataframe.x1 == x) & (draw <= psychometric_f[k]), 'binchoice'] = 1.0
            dataframe.loc[(dataframe.x1 == x) & (draw > psychometric_f[k]), 'binchoice'] = -1.0
        dataframe.loc[np.sign(dataframe.x1) == np.sign(dataframe.binchoice), 'bincorrect'] = 1.0
        dataframe.loc[np.sign(dataframe.x1) != np.sign(dataframe.binchoice), 'bincorrect'] = 0.0

        for trialtypes in ['consistent', 'inconsistent']:
            if trialtypes == 'consistent':
                trialtype = (np.sign(dataframe.binchoice) == np.sign(dataframe.x2))
            else:
                trialtype = (np.sign(dataframe.binchoice) != np.sign(dataframe.x2))
            dat = dataframe[trialtype]
            x1 = np.array(dat.x1 + m_slope * np.random.randn(len(dat.x1)) + m_bias)
            x2 = np.array(dat.x2 + m_slope * np.random.randn(len(dat.x2)) + m_bias)

            # New estimation
            p = prmts[trialtypes]
            estim = p[1] * x1 + p[2] * x2 + p[0] * np.random.randn(len(x1))
            dataframe.loc[trialtype, 'estim'] = estim

        return dataframe


"""
Methods to plot the data from the post-decision bias simulations 
"""

def compute_weights_estim(data):
    """Compute stimulus weights 
    
    :param pd.DataFrame data: dataframe containing the simulation results.
    :return: the weights of the different stimulus, their standard errors and their differences for choice and no-choice trials, respectively
    :rtype: (list of float, list of float, list of float)
    """
    choice, no_choice = data.loc[data.do_choice == 1], data.loc[data.do_choice == 0]
    labels = re.findall(r'x[0-9]+\b', '|'.join(data.columns))
    w, wse, dw = [], [], []

    for data in [choice, no_choice]:
        x, y = np.array(data[labels].astype(float)), np.array(data['estim'].astype(float))
        
        # Ordinary Least Squares
        res = sm.OLS(y, x).fit()
        w.append(res.params)    
        wse.append(res.bse) # standard error of the parameters estimates
        dw.append(np.diff(res.params))  # difference between weights of stimulus

    return w, wse, dw


def plot_pdb_trial(sdata, rdata, ddata, opts, save=False, **kwargs):
    (tsteps, ntrials, n) = rdata.shape
    theta = np.arange(n) / n * 180 * 2 - 180

    dts = opts.get('save_interval', 0.01)
    tpoints = np.arange(0, tsteps) * dts
    tmax = tpoints[-1]

    cue = opts.get('cue_duration', 0.750)
    dcue = opts.get('dcue_duration', 2.0)

    # Data post-processing:
    r_aligned, phases, amps, amps_m = get_phases_and_amplitudes_auto(rdata, aligned_profiles=True)
    trials = sdata.loc[sdata.chosen != -1, 'index'].to_numpy()

    #### Example trial plot (colormap, amplitude, phase, stimuli, profiles)

    trial_index = kwargs.pop('selected_trial', np.random.choice(trials, 1)[0])
    trial = np.argwhere(trials == trial_index)[0][0]

    data_trial = sdata.iloc[trial_index]
    r = rdata[:, trial, :]
    phase = np.rad2deg(phases[:, trial])
    amp, ampm = (amps[:, trial], amps_m[:, trial])
    dleft = ddata['d12'][:, 0, trial]
    dright = ddata['d12'][:, 1, trial]

    # Figure and Axes
    custom_plot_params(1.2, latex=False)
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.autolayout'] = False

    # Grid(s)
    gs1 = gridspec.GridSpec(4, 1)
    gs_bar = gridspec.GridSpec(4, 1)

    # Plot margins and grid margins
    left = 0.11
    right_1 = 0.89
    hspace = 0.15
    wspace = 0.2

    # G1
    (top1, bottom1) = (0.95, 0.1)
    gs1.update(left=left, right=right_1, top=top1, bottom=bottom1, hspace=hspace, wspace=wspace)
    # Gbar
    (top_bar, bottom_bar, left_bar, right_bar) = (top1, bottom1, right_1 + 0.01, right_1 + 0.03)
    gs_bar.update(left=left_bar, right=right_bar, top=top_bar, bottom=bottom_bar, hspace=hspace, wspace=wspace)

    # Figure and Axes
    fig = plt.figure(figsize=np.array([5.0, 5.1]) * 1.0)
    fig.set_tight_layout(False)

    ax_ph = fig.add_subplot(gs1[0, 0])
    ax_c1 = fig.add_subplot(gs1[1, 0], sharex=ax_ph)
    ax_amp = fig.add_subplot(gs1[2, 0], sharex=ax_ph)
    ax_dec = fig.add_subplot(gs1[3, 0])

    axes = [ax_ph, ax_c1, ax_amp, ax_dec]

    mod_axes([ax_amp, ax_ph, ax_dec])
    ax_ph.spines['bottom'].set_position('center')

    ax_bar = create_colorbar_ax(fig, gs_bar[1, 0])  # Ax for the color-bar of the color-plot

    # Actual plotting starts here
    # Amplitude of the bump
    ax_amp.plot(tpoints[:-1], amp[:-1], label='Amplitude')
    ax_amp.legend(frameon=False, fontsize=8)
    ax_amp.set_xlim(0, tmax)
    ax_amp.set_ylim(0, np.max(r) * 1.01)
    ax_amp.set_ylabel('Firing Rate (Hz)', labelpad=10)
    plt.setp(ax_amp.get_xticklabels(), visible=False)

    # Decision network
    ax_dec.plot(tpoints[:-1], dleft[:-1], label=r"$r_L$")
    ax_dec.plot(tpoints[:-1], dright[:-1], label=r"$r_R$")
    ax_dec.plot(tpoints[:-1], np.ones_like(dright[:-1]) * 50, '--', color='k', lw=0.5)
    ax_dec.set_xlim(cue - 0.2, cue + dcue + 0.4)
    ax_dec.set_ylabel(r"$r_L$, $r_R$ (Hz)", labelpad=10)
    ax_dec.legend(frameon=False)
    ax_dec.set_xlabel('Time (s)')
    ax_ph.text(2.7, 0.95, r'Decision window', ha='right', va='top', transform=ax_ph.get_xaxis_transform(),
               fontsize=8)

    # Phase evolution and stimuli
    nframes = kwargs.get('nframes', 2)
    labels = ['x%d' % k for k in range(1, nframes + 1)]
    stim_phases = data_trial[labels].to_numpy(int)
    stim_times = np.array([0, cue + dcue])
    logging.debug('Orientations of the frames are: %s' % stim_phases)

    stim_color = mcolors.to_rgba('cornflowerblue', 0.3)
    ax_ph.bar(stim_times, stim_phases, align='edge', width=cue * 1.0, lw=1.0, ec='cornflowerblue',
              fc=stim_color, label=r'$\theta_i^{\text{stim}}$')
    ax_ph.legend(frameon=False, fontsize=8, ncol=2)
    ax_ph.set_ylim(-30, 30)
    ax_ph.set_yticks([-20, -10, 0, 10, 20])
    ax_ph.set_ylabel(r'$\theta\ (^\circ)$')
    ax_ph.set_xticks([cue, cue + dcue, cue + dcue + cue])
    ax_ph.get_xaxis().set_tick_params(which='both', direction='in', pad=-10)
    # ax_ph.get_xaxis().set_tick_params(direction='in', pad=-15)
    # plt.setp(ax_ph.get_xticklabels(), pad=-15)

    # Colorplot of the bump
    cmap = plt.get_cmap('hot')
    bump1 = ax_c1.pcolormesh(tpoints[0::1], theta, r[0::1].T, vmin=0, cmap=cmap)
    bump1.set_edgecolor('face')
    ax_c1.plot(tpoints[:-1], phase[:-1], color=cmap(0.0), linewidth=1.5)
    ax_c1.plot(tpoints[:-1], np.ones_like(phase[:-1]) * 0.0, '--', color='k', lw=0.5)
    # ax_c1.set_xlabel('Time (s)')
    ax_c1.set_ylabel(r'$\theta\ (^\circ)$', labelpad=-1)
    ax_c1.set_ylim(-180, 180)
    ax_c1.set_yticks([-180, -90, 0, 90, 180])
    plt.setp(ax_c1.get_xticklabels(), visible=False)

    # Colorbar
    cbar = plt.colorbar(bump1, cax=ax_bar, fraction=1.0, orientation='vertical', aspect=15)
    cbar.set_label('Firing Rate (Hz)', fontsize=8)
    cbar.solids.set_edgecolor("face")

    # Change ticklabels format (from latex to plain text)
    for ax in [ax_amp, ax_ph, ax_c1, ax_dec]:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    ax_dec.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax_ph.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    rectspan(cue, cue + dcue, [ax_ph, ax_amp, ax_dec], color="gray", linewidth=0.5, alpha=0.2, alpha_lines=0.5)

    fig.text(right_1, 1, r'Trial %d.' % (trial_index + 1), ha='right', va='top')
    fig.set_tight_layout(False)

    if save:
        directory = kwargs.get('fig_dir', 'figs/')
        filename = directory + f"pdb_single_trial_{trial}"
        save_plot(fig, filename, **kwargs)

    return fig, axes


def format_df(sdata):
    sdata['estim'] = sdata['estim'].astype(float)
    sdata['ph1'] = sdata['ph1'].astype(float)
    sdata['ph2'] = sdata['ph2'].astype(float)
    sdata['bincorrect'] = sdata['bincorrect'].astype(int)
    sdata['do_choice'] = sdata['do_choice'].astype(int)
    if -1 in sdata.binchoice.unique():
        sdata['binchoice'] = sdata['binchoice'].apply(lambda x: (x + 1) / 2)  # (-1, 1) -> (0, 1)
    return sdata


def plot_pdb_summ(sdata, save=False, **kwargs):
    sdata = format_df(sdata)

    choice_data = sdata.loc[sdata.do_choice == 1]
    nochoice_data = sdata.loc[sdata.do_choice == 0]

    correct_data = choice_data.loc[choice_data.binchoice == choice_data.bincorrect]
    incorrect_data = choice_data.loc[choice_data.binchoice != choice_data.bincorrect]

    # Figure and Axes
    custom_plot_params(1.2, latex=False)
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.autolayout'] = False

    # Grid(s)
    gs1 = gridspec.GridSpec(1, 3)
    gs2 = gridspec.GridSpec(1, 3)
    gs3 = gridspec.GridSpec(1, 2)

    # Plot margins and grid margins
    left = 0.12
    right_1 = 0.95
    hspace = 0.15
    wspace = 0.2

    # G1
    (top1, bottom1) = (0.95, 0.7)
    gs1.update(left=left, right=right_1, top=top1, bottom=bottom1, hspace=hspace, wspace=wspace)
    # G2
    (top2, bottom2) = (bottom1 - 0.05, 0.5)
    gs2.update(left=left, right=right_1, top=top2, bottom=bottom2, hspace=hspace, wspace=wspace)
    # G3
    (top3, bottom3) = (bottom2 - 0.15, 0.1)
    gs3.update(left=left, right=right_1, top=top3, bottom=bottom3, hspace=hspace, wspace=wspace)

    # Figure and Axes
    fig = plt.figure(figsize=np.array([5.0, 5.1]) * 1.0)
    fig.set_tight_layout(False)

    axs_estim = []
    axs_distr = []
    for k in range(3):
        if k == 0:
            ax_top = fig.add_subplot(gs1[0, k])
            ax_bottom = fig.add_subplot(gs2[0, k], sharex=ax_top)
        else:
            ax_top = fig.add_subplot(gs1[0, k], sharex=axs_estim[0], sharey=axs_estim[0])
            ax_bottom = fig.add_subplot(gs2[0, k], sharex=axs_estim[0], sharey=axs_distr[0])

        axs_estim.append(ax_top)
        axs_distr.append(ax_bottom)

    ax_ppk = fig.add_subplot(gs3[0, 0])
    ax_pm = fig.add_subplot(gs3[0, 1])

    mod_axes([axs_estim])
    mod_axes([axs_distr])
    mod_axes([ax_ppk, ax_pm])

    red = "#E24A33"
    blue = "#348ABD"
    green = 'green'
    # (0, (3, 10, 1, 15)) means (3pt line, 10pt space, 1pt line, 15pt space)
    dashed = (0, (10, 3))
    bins = np.linspace(-20, 20, 9)
    # Estimation plots
    color_pairs = [('k', blue), ('k', green), ('k', red)]
    label_pairs = [('No-choice', 'Choice'), (None, 'Correct'), (None, 'Incorrect')]
    data_pairs = [(nochoice_data, choice_data), (nochoice_data, correct_data), (nochoice_data, incorrect_data)]
    first = True
    for ax1, ax2, data_pair, color_pair, label_pair in zip(axs_estim, axs_distr, data_pairs, color_pairs, label_pairs):
        ax1.plot([-20, 20], [-20, 20], linestyle=dashed, lw=0.5, color='k')
        for data, color, label in zip(data_pair, color_pair, label_pair):
            cats, avg, semm = compute_estimation(data, xlabel='average', ylabel='estim',
                                                       bins=kwargs.get('nbins', 21), lim=kwargs.get('eslim', 20))

            # group = data.groupby('average')
            # avg = group.estim.mean()
            # semm = group.estim.sem()
            # cats = group.mean().index.to_numpy()
            ax1, p, er = plot_error_filled(cats, avg, semm, ax=ax1, color=color)
            p.set_marker('o')
            p.set_label(label)


            # Histograms
            if first:
                hist1, bins1 = np.histogram(sdata.average, bins=9)
                ax2.bar(bins, hist1, color='k', width=3)
                hist2, bins1 = np.histogram(data.average, bins=9)
                ax2.bar(bins, hist2, color=color, width=3)
            else:
                if color != 'k':
                    hist1, bins1 = np.histogram(data.average, bins=9)
                    ax2.bar(bins, hist1, color=color, width=3)
        ax1.set_xlim(-22, 22)
        ax1.set_ylim(-21, 21)
        ax2.set_xlabel('Mean direction')
        ax1.legend(frameon=False, fontsize=8, loc='upper left')
        plt.setp(ax1.get_xticklabels(), visible=False)

        if not first:
            plt.setp(ax1.get_yticklabels(), visible=False)
            plt.setp(ax2.get_yticklabels(), visible=False)
        else:
            ax1.set_ylabel('Estimation')
            ax2.set_ylabel(r'\# trials')
            first = False
            ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    # PPK
    w, ws = compute_weights_estim(sdata)
    ax_ppk.errorbar(np.arange(2), w[1], ws[1], marker='o', ms=5, label='No-choice trials')
    ax_ppk.errorbar(np.arange(2), w[0], ws[0], marker='o', ms=5, label='Choice trials')
    ax_ppk.legend(frameon=False)
    ax_ppk.set_ylim(0.0, 0.6)
    ax_ppk.set_xlim(-0.5, 1.5)
    ax_ppk.set_xticks([0, 1])
    ax_ppk.set_ylabel('Weights')
    ax_ppk.set_xticklabels(['Interval 1', 'Interval 2'])

    ax_ppk.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # PM
    choice_data['binchoice'] = choice_data['binchoice'].apply(lambda x: (x * 2) - 1)
    model, mframe, choice_data = log_reg(choice_data, ['x1'])
    beta = mframe.beta.to_numpy()[0]
    error = mframe.errors.to_numpy()[0]

    # avg = 50 * (choice_data.groupby('x1').binchoice.mean().to_numpy() + 1)
    # std = 50 * choice_data.groupby('x1').binchoice.std().to_numpy()
    # x = choice_data.groupby('x1').mean().index.to_numpy()
    #
    # ax_pm.errorbar(x, avg, std, marker='o')
    xi = np.linspace(-20, 20, 1000)
    ax_pm.plot(xi, np.ones_like(xi) * 50, ls=dashed, lw=0.3, color='k')
    ax_pm.plot([0, 0], [0, 100], ls=dashed, lw=0.3, color='k')
    ax_pm.fill_between(xi, 100 * logistic.cdf(xi * (beta + error)), 100 * logistic.cdf(xi * (beta - error)), alpha=0.4)
    ax_pm.plot(xi, 100.0 * logistic.cdf(xi * beta))
    ax_pm.set_xlim(-25, 25)
    ax_pm.set_xlabel('Stimulus 1 orientation')
    ax_pm.set_ylabel(r'Performance \%')
    try:
        ax_ppk.spines['left'].set_bounds((0.0, 0.6))
        ax_ppk.spines['bottom'].set_bounds((0, 1))
        ax_pm.spines['left'].set_bounds((0.0, 100))
        ax_pm.spines['bottom'].set_bounds((-20, 20))
    except TypeError:
        ax_ppk.spines['left'].set_bounds(0.0, 0.6)
        ax_ppk.spines['bottom'].set_bounds(0, 1)
        ax_pm.spines['left'].set_bounds(0.0, 100)
        ax_pm.spines['bottom'].set_bounds(-20, 20)
    ax_pm.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax_pm.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    fig.set_tight_layout(False)

    if save:
        directory = kwargs.get('fig_dir', 'figs/')
        filename = directory + f"pdb_statistics"
        save_plot(fig, filename, **kwargs)

    return fig, [axs_estim, axs_distr, ax_ppk, ax_pm]


def plot_basic_measures(sdata, fit_results, logfits, save=False, **kwargs):
    names = kwargs.pop('names', [])
    ymax = 0
    for fr in fit_results:
        nc, w1c, w2c = fr['subj_params_o']['consistent'][0]
        ni, w1i, w2i = fr['subj_params_o']['inconsistent'][0]
        newmax = 1.1 * np.max([w1c, w1i, w2c, w2i])
        newmax = np.ceil(newmax * 10) / 10
        if ymax < newmax:
            ymax = newmax

    # Figure and Axes
    custom_plot_params(1.2, latex=False)
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.autolayout'] = False
    figsize = (2.5 + (len(sdata) - 1) * 2.0, 6)

    blue = "#348ABD"
    red = "#E24A33"

    dashed = (0, (10, 3))

    # Figure and Axes
    figkwargs = {}
    fig, axs, figkwargs = set_plot_environment(nrows=3, ncols=len(sdata), figsize=figsize, **figkwargs)
    mod_axes(axs)
    if len(sdata) == 1:
        axs = np.reshape(axs, (3, 1))

    for k, (df, fr, lf, daxs) in enumerate(zip(sdata, fit_results, logfits, axs.T)):
        ax1, ax2, ax3 = daxs

        df = format_df(df)

        # Estimation plot
        ax2.plot([-20, 20], [-20, 20], linestyle=dashed, lw=0.5, color='k')
        group = df.groupby('average')
        avg = group.estim.mean()
        semm = group.estim.sem()
        cats = group.mean().index.to_numpy()
        ax2, p, er = plot_error_filled(cats, avg, semm, ax=ax2, color=blue)
        p.set_marker('o')
        ax2.set_xlim(-22, 22)
        ax2.set_ylim(-21, 21)
        ax2.set_xlabel('Mean direction')
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))

        # Weights
        nc, w1c, w2c = fr['subj_params_o']['consistent'][0]
        ni, w1i, w2i = fr['subj_params_o']['inconsistent'][0]
        ax3.plot(np.arange(2), [w1c, w2c], marker='o', ms=5, label='Consistent trials')
        ax3.plot(np.arange(2), [w1i, w2i], marker='o', ms=5, label='Inconsistent trials')

        ax3.set_ylim(0.0, ymax)
        ax3.set_xlim(-0.5, 1.5)
        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(['Interval 1', 'Interval 2'])

        ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # Psychometric plot
        xpoints = np.sort(df.x1.unique())
        ax1.errorbar(xpoints, 100 * np.array(lf['points'][0]), 100 * np.array(lf['errors'][0]), ls='None', marker='o',
                     label='Data', color=red)
        xi = np.linspace(-20, 20, 1000)
        ax1.plot(xi, 100.0 * cumnormal(lf['fits'][0], xi), label='Cumnormal', color=red, alpha=0.5)

        df['binchoice'] = df['binchoice'].apply(lambda x: (x * 2) - 1)
        model, mframe, df = log_reg(df, ['x1'])
        beta = mframe.beta.to_numpy()[0]
        error = mframe.errors.to_numpy()[0]

        # avg = 50 * (choice_data.groupby('x1').binchoice.mean().to_numpy() + 1)
        # std = 50 * choice_data.groupby('x1').binchoice.std().to_numpy()
        # x = choice_data.groupby('x1').mean().index.to_numpy()
        #
        # ax_pm.errorbar(x, avg, std, marker='o')
        ax1.plot(xi, np.ones_like(xi) * 50, ls=dashed, lw=0.3, color='k')
        ax1.plot([0, 0], [0, 100], ls=dashed, lw=0.3, color='k')
        ax1.fill_between(xi, 100 * logistic.cdf(xi * (beta + error)), 100 * logistic.cdf(xi * (beta - error)),
                         alpha=0.4, color=blue)
        ax1.plot(xi, 100.0 * logistic.cdf(xi * beta), label='Log Reg.', color=blue)

        ax1.set_xlim(-25, 25)
        ax1.set_xlabel('Stimulus 1 orientation')

        if names[k].startswith('gain_'):
            gain_value = names[k].strip('gain_')
            if len(gain_value) == 2:
                try:
                    gain = int(gain_value) / 10.0
                except ValueError:
                    gain = 0
                ax1.set_title(f"Gain: {gain}")
            else:
                kn, ks = gain_value[0:2], gain_value[2:]
                try:
                    kn, ks = int(kn) / 10.0, int(ks) / 10.0
                except ValueError:
                    kn, ks = 0, 0
                ax1.set_title(f"kN: {kn}, kS: {ks}")

        try:
            ax3.spines['left'].set_bounds((0.0, ymax))
            ax3.spines['bottom'].set_bounds((0, 1))
            ax1.spines['left'].set_bounds((0.0, 100))
            ax1.spines['bottom'].set_bounds((-20, 20))
            ax2.spines['left'].set_bounds((-20, 20))
            ax2.spines['bottom'].set_bounds((-20, 20))
        except TypeError:
            ax3.spines['left'].set_bounds(0.0, ymax)
            ax3.spines['bottom'].set_bounds(0, 1)
            ax1.spines['left'].set_bounds(0.0, 100)
            ax1.spines['bottom'].set_bounds(-20, 20)
            ax2.spines['left'].set_bounds(-20, 20)
            ax2.spines['bottom'].set_bounds(-20, 20)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))

        if k == 0:
            ax1.set_ylabel(r'P(right) \%')
            ax2.set_ylabel('Estimation')
            ax3.set_ylabel('Weights')
            ax3.legend(frameon=False, fontsize=8)
            ax1.legend(frameon=False, fontsize=8, loc='upper left')
        else:
            plt.setp(ax1.get_yticklabels(), visible=False)
            plt.setp(ax2.get_yticklabels(), visible=False)
            plt.setp(ax3.get_yticklabels(), visible=False)

    fig.tight_layout()

    if save:
        directory = kwargs.get('fig_dir', 'figs/')
        filename = directory + f"pdb_weights_new"
        save_plot(fig, filename, **kwargs)

    return fig, axs


def plot_weights_gains(data_dir='data_talluri', save=False, **kwargs):
    name = 'ChoiceSelectiveModel_gain_'
    # Load data
    logging.info('Getting file list...')
    data = []
    gains = []
    file_list = os.listdir(data_dir)
    knks = False
    for file_path in file_list:
        file_name, extension = os.path.splitext(file_path)
        props = file_name.split('_')
        print(props)
        if file_name.startswith(name) and extension == '.npy':
            if 'gain' in props:
                try:
                    gain_arg = np.argwhere(np.array(props) == 'gain')[0][0] + 1
                    t_gain = props[gain_arg]
                except IndexError:
                    t_gain = '00'

                if t_gain not in gains:
                    gains.append(t_gain)
                    file_name = get_recent_file(data_dir, f"{name}{t_gain}")
                    data.append(dict(np.load(file_name, allow_pickle=True).item()))
                    logging.debug(f"Loaded file '{file_name}'.")

    # Sort results by gain
    sorted_args = np.argsort(np.array(gains, dtype=int))
    data = list(np.array(data)[sorted_args])
    gains = np.array(gains)
    gains = gains[sorted_args]

    # Extract weights
    wc = []
    wi = []
    for dat in data:
        wc.append(dat['subj_params_o']['consistent'][0])
        wi.append(dat['subj_params_o']['inconsistent'][0])

    if kwargs.get('save_weights', False):
        dict_weights = dict(gain=[], kn=[], ks=[], trialtype=[], interval=[], weight=[], noise=[])
        for k, gain in enumerate(gains):
            for trialtype, weights in zip(['consistent', 'inconsistent'], [wc[k], wi[k]]):
                noise = weights[0]
                weights = weights[1:]
                for interval, weight in zip([1, 2], weights):
                    dict_weights['gain'].append(int(gain))
                    if len(gain) > 2:
                        dict_weights['kn'].append(int(gain[0:2])/10.0)
                        dict_weights['ks'].append(int(gain[2:])/10.0)
                    dict_weights['trialtype'].append(trialtype)
                    dict_weights['interval'].append(interval)
                    dict_weights['weight'].append(weight)
                    dict_weights['noise'].append(noise)
        df_weights = pd.DataFrame.from_dict(dict_weights)
        weights_filename = check_overwrite(kwargs.get('weights_file', 'all_weights.csv'), auto=True)
        df_weights.to_csv(weights_filename)

    gains = np.array(gains, dtype=float) / 10.0

    # Figure
    custom_plot_params(1.2, latex=False)
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.autolayout'] = False
    figsize = kwargs.pop('figsize', (5, 2.5))
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey='row', sharex='row', figsize=figsize)
    mod_axes(axs)
    ax1, ax2 = axs

    # Weights in interval 1
    ax1.plot(gains, np.array(wc)[:, 1], '-o', ms=5, label='Consistent')
    ax1.plot(gains, np.array(wi)[:, 1], '--o', ms=5, label='Inconsistent')
    ax1.legend(frameon=False, fontsize=8, loc='lower left')
    ax1.set_title('Weights in interval 1')

    # Weights in interval 2
    ax2.plot(gains, np.array(wc)[:, 2], '-o', ms=5, label='Consistent')
    ax2.plot(gains, np.array(wi)[:, 2], '--o', ms=5, label='Inconsistent')
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_title('Weights in interval 2')

    ymax = np.max((np.array(wc)[:, 1:], np.array(wi)[:, 1:]))
    ax1.set_ylim(0 - 0.05, ymax + 0.05)
    ax1.set_xlim(gains[0] - 0.1, gains[-1] + 0.1)
    ax1.set_xlabel('Gain')
    ax2.set_xlabel('Gain')
    ax1.set_ylabel('Weight')
    try:
        ax1.spines['left'].set_bounds((0.0, ymax))
        ax1.spines['bottom'].set_bounds((gains[0], gains[-1]))
        ax2.spines['left'].set_bounds((0.0, ymax))
        ax2.spines['bottom'].set_bounds((gains[0], gains[-1]))
    except TypeError:
        ax1.spines['left'].set_bounds(0.0, ymax)
        ax1.spines['bottom'].set_bounds(gains[0], gains[-1])
        ax2.spines['left'].set_bounds(0.0, ymax)
        ax2.spines['bottom'].set_bounds(gains[0], gains[-1])

    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    fig.tight_layout()

    plt.show()
    if save:
        directory = kwargs.get('fig_dir', 'figs/')
        filename = directory + f"pdb_weights_all_new"
        save_plot(fig, filename, **kwargs)

    return fig, axs


def plot_pdb_weights(sdata, save=True, **kwargs):
    # Choice-selective model (compute weights)
    init_opts = dict(data='none', db='DEBUG', preprocess=False, data_dir='./data_talluri',
                     lbound=(0.0, 0.0, 0.0), ubound=(10.0, 2.0, 2.0), sftola=0.000001, smaxeval=1000, maxiM=1000,
                     maxevalm=1000, subplex=True, cpus=1,  maxiP=2000, ftolP=0.0000001, maxfeP=2000, maxlsP=1000,
                     epsP=0.00001)

    slogfits, slogpoints, slogpoints_errors = compute_psychometric(sdata, name='noname', **init_opts)
    slogfits = np.array(slogfits)
    logfits = dict(fits=slogfits, points=slogpoints, errors=slogpoints_errors)
    
    # Fit data to model
    model = ChoiceSelectiveModelFast(sdata, [1], slogfits, 3)
    #logging.info('Fitting model...')
    model(Force_compute=True, name='noname', **init_opts)
    # Weights
    nc, w1c, w2c = model.data_to_be_saved['subj_params_o']['consistent'][0]
    ni, w1i, w2i = model.data_to_be_saved['subj_params_o']['inconsistent'][0]

    sdata = format_df(sdata)  # This function is not compatible with ``compute_psychometric

    choice_data = sdata.loc[sdata.do_choice == 1].copy()
    nochoice_data = sdata.loc[sdata.do_choice == 0].copy()

    # Select trials with first stimulus equal to 0
    zero_trials = choice_data.loc[choice_data.x1 == 0].copy()
    cw_consistent = zero_trials.loc[(zero_trials.binchoice == 1) & (zero_trials.x2 > 0)].copy()
    cw_inconsistent = zero_trials.loc[(zero_trials.binchoice == 1) & (zero_trials.x2 < 0)].copy()
    ccw_consistent = zero_trials.loc[(zero_trials.binchoice == 0) & (zero_trials.x2 < 0)].copy()
    ccw_inconsistent = zero_trials.loc[(zero_trials.binchoice == 0) & (zero_trials.x2 > 0)].copy()

    # Figure and Axes
    custom_plot_params(1.2, latex=False)
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.autolayout'] = False

    # Grid(s)
    gs1 = gridspec.GridSpec(1, 5)

    # Plot margins and grid margins
    left = 0.06
    right_1 = 0.99
    hspace = 1.0
    wspace = 0.35

    # G1
    (top1, bottom1) = (0.95, 0.18)
    gs1.update(left=left, right=right_1, top=top1, bottom=bottom1, hspace=hspace, wspace=wspace)

    # Figure and Axes
    fig = plt.figure(figsize=np.array([10.0, 1.8]) * 1.2)
    fig.set_tight_layout(False)

    ax_pm = fig.add_subplot(gs1[0, 0])
    ax_st = fig.add_subplot(gs1[0, 1])
    ax_cb = fig.add_subplot(gs1[0, 2])
    ax_w1 = fig.add_subplot(gs1[0, 3])
    ax_w2 = fig.add_subplot(gs1[0, 4], sharex=ax_w1)
    axs = [ax_pm, ax_st, ax_cb, ax_w1, ax_w2]

    mod_axes(axs)

    red = "#E24A33"
    blue = "#348ABD"

    # (0, (3, 10, 1, 15)) means (3pt line, 10pt space, 1pt line, 15pt space)
    dashed = (0, (10, 3))

    # Estimation plot
    ax_st.plot([-20, 20], [-20, 20], linestyle=dashed, lw=0.5, color='k')
    for data, color, label in zip([nochoice_data, choice_data], [red, blue], ['No-choice', 'Choice']):
        cats, avg, std = compute_estimation(data, xlabel='average', ylabel='estim',
                                                       bins=kwargs.get('nbins', 11), lim=kwargs.get('eslim', 20))
        # group = data.groupby('average')
        # avg = group.estim.mean()
        # semm = group.estim.sem()  # Standard error of the mean or ...
        # std = group.estim.std()   # ... standard error?
        # cats = group.mean().index.to_numpy()
        ax_st, p, er = plot_error_filled(cats, avg, std, ax=ax_st, color=color)
        if color == blue:  # Plot choice data also in the confirmation bias plot
            ax_cb.plot(cats, avg, linestyle='dashed', linewidth=0.5, color=blue)
        p.set_marker('o')
        p.set_label(label)
    ax_st.set_xlim(-22, 22)
    ax_st.set_ylim(-21, 21)
    ax_st.set_xlabel('Mean direction')
    ax_st.set_ylabel('Estimation')
    ax_st.legend(frameon=False, fontsize=8, loc='upper left')
    ax_st.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax_st.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    # Confirmation bias plot
    ax_cb.plot([-10, 10], [-10, 10], linestyle=dashed, lw=0.5, color='k')
    linestyles = ['solid', 'dashed', 'solid', 'dashed']
    colors = ['black', 'black', 'gray', 'gray']
    label1 = ['CW choice, ', 'CW choice, ', 'CCW choice, ', 'CCW choice, ']
    label2 = ['consistent', 'inconsistent', 'consistent', 'inconsistent']
    for k, data in enumerate([cw_consistent, cw_inconsistent, ccw_consistent, ccw_inconsistent]):
        # cats, avg, std = compute_estimation(data, xlabel='average', ylabel='estim',
        #                                        bins=kwargs.get('nbins', 11), lim=kwargs.get('eslim', 10))
        group = data.groupby('average')
        avg = group.estim.mean()
        semm = group.estim.sem()  # Standard error of the mean or ...
        std = group.estim.std()   # ... standard error?
        cats = group.mean().index.to_numpy()
        ax_cb, p, er = plot_error_filled(cats, avg, std, ax=ax_cb, color=colors[k], linestyle=linestyles[k])
        p.set_marker('o')
        p.set_label(f"{label1[k]}{label2[k]}")
    ax_cb.set_xlim(-12, 12)
    ax_cb.set_ylim(-15, 15)
    ax_cb.set_xlabel('Mean direction')
    ax_cb.legend(frameon=False, fontsize=6, loc='upper left')
    ax_cb.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax_cb.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    # PPK 1 (choice, no-choice)
    w, ws, dw = compute_weights_estim(sdata)
    ax_w1.errorbar(np.arange(2), w[1], ws[1], marker='o', ms=5, label='No-choice trials')
    ax_w1.errorbar(np.arange(2), w[0], ws[0], marker='o', ms=5, label='Choice trials')
    ax_w1.legend(frameon=False, fontsize=8)
    ymax1 = 1.1 * np.max([np.max(w), np.max(ws)])
    ax_w1.set_ylim(0.0, ymax1)
    ax_w1.set_xlim(-0.5, 1.5)
    ax_w1.set_xticks([0, 1])
    ax_w1.set_ylabel('Weights')
    ax_w1.set_xticklabels(['Interval 1', 'Interval 2'])
    ax_w1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # PPK 2 (choice: consistent, inconsistent)
    ax_w2.plot(np.arange(2), [w1c, w2c], marker='o', ms=5, label='Consistent trials', color='black')
    ax_w2.plot(np.arange(2), [w1i, w2i], marker='o', ms=5, label='Inconsistent trials', color='black',
               linestyle='dashed')
    ax_w2.legend(frameon=False, fontsize=8)
    ymax2 = 1.1 * np.max([w1c, w1i, w2c, w2i])
    ax_w2.set_ylim(0.0, ymax2)
    ax_w2.set_xlim(-0.5, 1.5)
    ax_w2.set_xticks([0, 1])
    ax_w2.set_ylabel('Weights')
    ax_w2.set_xticklabels(['Interval 1', 'Interval 2'])
    ax_w2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # PM
    # xpoints = np.sort(sdata.x1.unique())
    # ax_pm.errorbar(xpoints, 100 * np.array(logfits['points'][0]), 100 * np.array(logfits['errors'][0]), ls='None',
    #                marker='o', label='Data', color='black')
    xpoints, ypoints, errors, _ = compute_pm_data(sdata, bins=7, lim=20, xlabel='x1', ylabel='phd', compute_errorbars=True)
    ax_pm.errorbar(xpoints, 100 * ypoints, ls='None', marker='o', label='Data', color='black')
    xi = np.linspace(-20, 20, 1000)
    ax_pm.plot(xi, 100.0 * cumnormal(logfits['fits'][0], xi), label='Cumnormal', color='black', lw=0.5)

    # choice_data['binchoice'] = choice_data['binchoice'].apply(lambda x: (x * 2) - 1)
    # model, mframe, choice_data = log_reg(choice_data, ['x1'])
    # beta = mframe.beta.to_numpy()[0]
    # error = mframe.errors.to_numpy()[0]

    ax_pm.plot(xi, np.ones_like(xi) * 50, ls=dashed, lw=0.3, color='k')
    ax_pm.plot([0, 0], [0, 100], ls=dashed, lw=0.3, color='k')
    # ax_pm.fill_between(xi, 100 * logistic.cdf(xi * (beta + error)), 100 * logistic.cdf(xi * (beta - error)), alpha=0.4)
    # ax_pm.plot(xi, 100.0 * logistic.cdf(xi * beta))
    ax_pm.set_xlim(-25, 25)
    ax_pm.set_xlabel('Stimulus 1 orientation')
    ax_pm.set_ylabel(r'Proportion CW \%')

    try:
        ax_w1.spines['left'].set_bounds((0.0, ymax1))
        ax_w1.spines['bottom'].set_bounds((0, 1))
        ax_w2.spines['left'].set_bounds((0.0, ymax2))
        ax_w2.spines['bottom'].set_bounds((0, 1))
        ax_pm.spines['left'].set_bounds((0.0, 100))
        ax_pm.spines['bottom'].set_bounds((-20, 20))
    except TypeError:
        ax_w1.spines['left'].set_bounds(0.0, ymax1)
        ax_w1.spines['bottom'].set_bounds(0, 1)
        ax_w2.spines['left'].set_bounds(0.0, ymax2)
        ax_w2.spines['bottom'].set_bounds(0, 1)
        ax_pm.spines['left'].set_bounds(0.0, 100)
        ax_pm.spines['bottom'].set_bounds(-20, 20)
    ax_pm.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax_pm.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    fig.set_tight_layout(False)

    if save:
        directory = kwargs.get('fig_dir', 'figs/')
        filename = directory + f"pdb_statistics"
        save_plot(fig, filename, **kwargs)

    return fig, axs


from scipy.stats import circmean, circstd

def plot_estimation_weights(data, save=False, **kwargs):
    avg_nochoice = np.sort(data[data.do_choice == 0].loc[:, 'average'].unique())
    est_nochoice = data[data.do_choice == 0].loc[:, ['average', 'estim']].groupby('average').apply(circmean, low=-180, high=180).values
    std_nochoice = data[data.do_choice == 1].loc[:, ['average', 'estim']].groupby('average').apply(circstd, low=-180, high=180).values

    avg_choice = np.sort(data[data.do_choice == 1].loc[:, 'average'].unique())
    est_choice = data[data.do_choice == 1].loc[:, ['average', 'estim']].groupby('average').apply(circmean, low=-180, high=180).values
    std_choice = data[data.do_choice == 1].loc[:, ['average', 'estim']].groupby('average').apply(circstd, low=-180, high=180).values

    w, _, _ = compute_weights_estim(data)
    w_choice, w_nochoice = np.array([w[0][0], w[1][0]]), np.array([w[0][1], w[1][1]])    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1.75, 1]})

    ax1.plot(avg_nochoice, est_nochoice, c='orange', linestyle='dashed', alpha=1, lw=3.5)
    ax1.plot(avg_choice, est_choice, c='blue', alpha=1, lw=3.5)
    ax1.plot(np.linspace(-20, 20, 100), np.linspace(-20, 20, 100), '--', c='grey', lw=1)

    ax1.set_yticks(np.arange(-20, 21, 10))
    ax1.tick_params(axis='both', which='major', labelsize=15)

    ax1.set_title('\\textbf{Estimation curves}', fontsize=20)
    ax1.legend(['No choice', 'Choice'], fontsize=12.5)
    ax1.set_xlim([-21, 21])
    ax1.set_ylim([-25, 25])

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

    fig

    return fig, axes