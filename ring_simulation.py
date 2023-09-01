"""
Simulation of a bump attractor network for stimulus integration
===============================================================

.. module:: ring_simulation
   :platform: Linux
   :synopsis: simulation of the ring attractor network for stimulus integration tasks.

.. moduleauthor:: Jose M. Esnaola-Acebes <josemesnaola@gmail.com>

This script runs a simulation of a bump attractor network (**cite**) to model stimulus integration in visually
guided perceptual tasks.


Methods included in :mod:'ring_simulation'
------------------------------------------

.. autosummary::
   :toctree: generated/

   set_up_simulation       Set up the simulated task environment.
   process_results         Pre-Process the results of the simulation.
   simulate_task_parallel  Prepare the simulation variables to be run in parallel.
   run_simulation          Simulation's main body: run the simulation.
   do_bifurcation          Automatically guided bifurcation generator.

Default parameters
------------------

Table.

Implementation
--------------

Description needed.
"""

import logging
import os
import sys
import timeit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, './lib/') 
from lib_sconf import Parser, check_overwrite, create_dir, get_paths, log_conf, path_exists, save_obj

file_paths = get_paths(__file__)
sys.path.insert(0, file_paths['s_dir'] + '/lib/')
import gc
import multiprocessing as mp

from lib_analysis import normalize_dataframe, plot_decision_circ, plot_response_time, plot_stats_summ, plot_trial
from lib_NeuroDyn import sigmoid, ssn_f
from lib_parallel import ParallelSimulation
from lib_pdb_analysis import plot_pdb_weights, plot_estimation_weights
from lib_plotting import plt
from lib_ring import circ_dist, compute_phase, connectivity, gaussian_connectivity, icritical, load_ic, ou_process, save_ic, sigmoid_pw_v

from trial_generator import pdb_new_stimuli

logging.getLogger('ring_simulation').addHandler(logging.NullHandler())

__author__ = 'Miguel Donderis, Jose M. Esnaola-Acebes'

init_options = {'tau': 0.02, 'dt': 2E-4, 'tmax': 2.05, 'n': 200, 'm': [-2.0, 1.0, 0.5], 'dcue': [0.75, 2.75],
                'tauOU': 1E-3, 'sigmaOU': 0.0, 'nframes': 8, 'cue_duration': 0.750, 'save_interval': 0.01}

logger = None


def kernel(theta, choice='left', maxv=1.0, shape='triangle'):
    k = np.ones_like(theta) * 0.0
    n = len(k)

    if shape == 'linear':
        k = np.ones_like(theta) * maxv
        if choice == 'left':
            k[n // 2:] = 0
        else:
            k[:n // 2] = 0
    elif shape == 'triangle':
        if choice == 'left':
            k = maxv * (-theta / np.pi)
        elif choice == 'right':
            k = maxv * (theta / np.pi)
    return k


def run_simulation(samples, init_ph, choice, inputs, opts, **kwargs):
    """Simulation's main function. Runs the integration time-loop.

    :param np.ndarray samples: Matrix that includes the information of the stimuli.
    :param np.ndarray init_ph: Array with the initial positions of the bump.
    :param np.ndarray choice: Vector that indicates whether an intermittent choice is taken or not.
    :param np.ndarray inputs: A matrix containing all possible stimuli inputs.
    :param dict opts: Dictionary that includes all necessary parameters and options.
    :param kwargs: Additional keyword arguments passed by the :mod:'ParallelSimulation'.
    :return: Simulation's results.
    """
    
    save = opts.get('save_fr', False)
    is_bump = not opts.get('no_bump', False)    # Default is True
    tmax = opts.get('tmax', 2.05)               # Maximum time
    dt = opts.get('dt', 2E-4)                   # Time step
    tau = opts.get('tau', 20E-3)                # Neurons' time constant
    n = opts['n']                           

    # Number of trials is determined by the samples table
    ntrials = len(samples[:, 0])

    # Stimulus time-related parameters
    n_frames = opts.get('nframes', 2)
    cue_duration = opts.get('cue_duration', 0.750)      # Stimulus duration
    tcue = [0.0, cue_duration]
    tcues = np.array(np.array(tcue) / dt, dtype=int)
    lencue = np.diff(tcues).item()
    frame = 0                                           # Frame counter
    total_stimulus_duration = n_frames * cue_duration   # Total stimulus duration
    if tmax < total_stimulus_duration: 
        tmax = total_stimulus_duration + 0.05

    # Decision period
    decision_duration = opts.get('dcue_duration', 2.0)
    dcue = [cue_duration, cue_duration + decision_duration]
    dcues = np.array(np.array(dcue) / dt, dtype=int)
    urgency_duration = opts.get('urgency_duration', decision_duration)

    # Variables (time, space and firing rate)
    tpoints = np.arange(0, tmax, dt)
    theta = np.arange(n) / n * (2 * np.pi) - np.pi
    nsteps = len(tpoints)

    rsteps = nsteps if save == 'all' else 2
    block = kwargs.get('block', 0)
    logging.debug(f'Simulating block {block}...')

    # Saving the firing rate profile
    saving_counter = 0
    if save == 'partial':
        # 2.05 seconds are simulated => with a 50 ms interval 41 points are saved
        saving_interval = opts.get('save_interval', 0.025)
        tpoints_save = np.arange(0, tmax - dt, saving_interval)
        save_steps = len(tpoints_save)
        save_steps = save_steps if save_steps % 2 == 0 else save_steps - 1
        r_profiles = np.ones((save_steps, ntrials, n)) * .0
        rssn_profiles = np.ones((save_steps, ntrials, 2 * n)) * .0
        saving_sampling = nsteps // save_steps
        logging.info('Saving firing rate every %d ms. Number of points: %d. Sampling period: %d' % (saving_interval * 1E3, save_steps, saving_sampling))
        
        # Saving decision network activity
        d12 = np.ones((save_steps, 2, ntrials)) * 0.0
        lri = np.ones((save_steps, 2, ntrials)) * 0.0
        us = np.ones((save_steps, ntrials)) * 0.0
        
        # Saving the perfectly integrated stimulus
        total_stim = np.zeros((ntrials, n))
        perfect_phase = np.ones((save_steps, ntrials))

    else:   # Dummy variables to avoid annoying warnings
        r_profiles = np.array([[[0]]])   # 3-dim array for compatibility reasons
        saving_sampling = 0
        d12 = np.array([[[0]]])
        lri = np.array([[[0]]])
        us = np.array([[0]])
        total_stim = np.array([[0]])
        perfect_phase = np.array([[0]])

    # Firing rate matrix
    r = np.ones((rsteps, ntrials, n)) * 0.01 
    phases = np.zeros((n_frames, ntrials))

    # Integration times
    response_times = np.ones(ntrials) * -1.0

    # Connectivity
    modes = opts.get('m', [-1.0, 2.0, 0.2])
    cnt = connectivity(n, modes)

    # Parameters for simulating the Ornstein–Uhlenbeck process
    sigmaou = opts.get('sigmaOU', 0.6)
    tau_ou = opts.get('tauOU', 1E-3)
    worker = ParallelSimulation.worker(real=True)
    seed = np.random.seed((int(timeit.default_timer() * 10000 + worker) // worker) % (2 ** 32 - 1))
    ou = ou_process(dt, lencue * n_frames + 1, 0.0, sigmaou, tau_ou, ntrials, n, seed=seed)

    # Sensory inputs to the integration circuit
    icrit = icritical(modes[0], modes[1], tau=tau)
    i0_over = opts.get('i0', 0.01)
    i1 = opts.get('i1', 0.01)

    # Phase computation
    cosine = np.cos(theta)
    sine = np.sin(theta)
    flat = np.ones_like(theta)  # auxiliary vector for computing the top-down non-selective feedback

    """
    DECISION CIRCUIT (READOUT NETWORK) SETUP
    """
    dtaue = opts.get('dtaue', 20E-3)    # 20 ms
    dtaui = opts.get('dtaui', 20E-3)    # 20 ms
    rec_exc = opts.get('js', 1.9)
    cross_inh = opts.get('jc', 1.0)
    g = opts.get('g', 1.0)
    i_i = opts.get('i_i', 0.2)
    d1, d2 = np.ones((rsteps, ntrials)) * 0.0, np.ones((rsteps, ntrials)) * 0.0
    ri = np.ones((rsteps, ntrials)) * 0.0

    # Urgency signal (sub-critical pitchfork)
    i_rest, i_urge = opts.get('i_rest', 0.33), opts.get('i_urge', 0.50)   # Bistable and pure Winner-Take-All (WTA) regimes
    go_signal, go_interval = cue_duration, urgency_duration    # Urgency signal time interval (seconds)
    urgency_signal = np.ones((nsteps, ntrials), dtype=float) * i_rest
    urgency_signal[int(go_signal / dt):(int(go_signal / dt) + int(go_interval / dt)), choice] = i_urge

    # Left and right inputs to the decision network
    max_signal = opts.get('max', 1.0)
    kernel_l = kernel(theta, 'left', maxv=max_signal, shape='linear')
    kernel_r = kernel(theta, 'right', maxv=max_signal, shape='linear')
    left_input, right_input = 0.0, 0.0

    # Decision making vectors
    decided = np.ones(ntrials, dtype=bool) * 0
    decision_threshold = opts.get('threshold', 50)
    decision_phase = np.ones(ntrials) * 0.0

    """
    Initial condition of the bump (load the profile from dictionary)
    """
    logging.debug('Bump: %d' % is_bump)
    if opts.get('init_load', False) or is_bump:
        effe_i0_over = opts.get('i0_init', i0_over - 2.0 * sigmaou ** 2)
        r0 = load_ic('./obj/r0_initial_conditions.npy', critical=('n',), n=n, i0_over=effe_i0_over,
                     w0=modes[0], w1=modes[1], w2=modes[2])
        if r0 is not False:
            r[-1] = np.repeat([r0], ntrials, axis=0)  # (ntrials x n)

        # Set the initial phase of the bump to the phase of the nobump simulation after the first stimulus frame
        if opts.get('nobias', False):
            #logger.debug("Simulating an unbiased system.")
            for k, ph in enumerate(init_ph):
                r[-1, k] = np.roll(r[-1, k], ph)

    """
    SENSORY CIRCUIT (STABILIZED SUPRALINEAR NETWORK) SETUP
    """
    # default neuron and network parameters
    ne = opts.get('ne', opts.get('n', 200))  # number of excitatory neurons
    ni = opts.get('ni', opts.get('n', 200))  # number of inhibitory neurons
    ssn_taue = opts.get('ssn_taue', 20e-3)   # [s] time constant of rate equation for excitatory neurons
    ssn_taui = opts.get('ssn_taui', 10e-3)   # [s] time constant of rate equation for inhibitory neurons

    i0e = 0.0  # external bias current to excitatory neurons
    i0i = 0.0  # external bias current to inhibitory neurons

    # default stimulus parameters
    i0_stim_e = opts.get('i0_stim_e', 0.0)   # strength of external stimulus (no modulation)
    i1_stim_e = opts.get('i1_stim_e', 20.0)  # strength of external stimulus (modulated)
    i0_stim_i = opts.get('i0_stim_i', 0.0)   # strength of external stimulus (no modulation)
    i1_stim_i = opts.get('i1_stim_i', 20.0)  # strength of external stimulus (modulated)

    # Stimulus inputs to the sensory circuit (precomputed)
    input1_e = i1_stim_e * np.take(inputs[0], samples[:, frame], axis=0)  # (ntrials x n/2)
    input1_i = i1_stim_i * np.take(inputs[1], samples[:, frame], axis=0)  # (ntrials x n/2)

    # Set up sensory network. Depending on above parameters.
    w_ssn = opts.get('w_ssn', None)
    if w_ssn is None:
        raise AttributeError('The connectivity matrix must be precomputed!')

    # Set up external bias currents, noise, and time constants
    i0_ssn = np.concatenate((i0e * np.ones(ne), i0i * np.ones(ni)))
    tau_ssn = np.concatenate((ssn_taue * np.ones(ne), ssn_taui * np.ones(ni)))

    # Firing rate of the sensory circuit 
    r_ssn = np.ones((rsteps, ntrials, ne + ni)) * 0.0

    """
    SELECTIVE MODULATION (ATTENTION): Modulation from the decision circuit to the sensory circuit
    """
    att = opts.get('attention', True)
    th_att = np.deg2rad(opts.get('th_att', 20))
    s_att_e, s_att_i = np.deg2rad(opts.get('s_att_e', 10)), np.deg2rad(opts.get('s_att_i', 10))
    i_att_e, i_att_i = opts.get('i_att_e', 2.0), opts.get('i_att_i', 2.5)
    sg_exc, sg_inh = opts.get('sg_exc', 1.0), opts.get('sg_inh', 1.0)

    gauss_mod_exc_1 = np.exp(-circ_dist(theta, -th_att) ** 2 / (2.0 * s_att_e ** 2))  
    gauss_mod_inh_1 = np.exp(-circ_dist(theta, -th_att) ** 2 / (2.0 * s_att_i ** 2)) 
    gauss_mod_exc_2 = np.exp(-circ_dist(theta, th_att) ** 2 / (2.0 * s_att_e ** 2))   
    gauss_mod_inh_2 = np.exp(-circ_dist(theta, th_att) ** 2 / (2.0 * s_att_i ** 2))  

    # Global modulation, choice commitment
    base_fr = 11.59
    decided_fr = 24.47
    choice_mod = opts.get('choice', True)
    global_gain = opts.get('gg', 0.001)
    if choice_mod:
        base_i0 = base_fr / decided_fr * global_gain
    else:
        base_i0 = 0.0

    """
    Saving inputs to the integration circuit (only block 0)
    """
    logging.debug("New simulation launched.")

    time_init = timeit.default_timer()
    tstep = 0
    temps = 0.0

    np.seterr(all='raise')
    ttau = dt / tau
    stim_input = 0.0 * np.concatenate((i0_stim_e + input1_e, i0_stim_i + input1_i), axis=1)  # (ntrials, n)
    sensory_noise = 0.0
    s1 = 0
    
    while temps < (tmax - dt):
        # Time step variables
        kp = tstep % rsteps
        k = (tstep + rsteps - 1) % rsteps

        # 2. Presynaptic inputs
        s = 1.0 / n * np.dot(r[k], cnt)  # (ntrials, n) x (n, n) = (ntrials, n)
        # Monitor changes in i0
        i0 = icrit + i0_over - base_i0
        # 2. Presynaptic inputs (sensory circuit)
        i_ssn = np.dot(r_ssn[k], w_ssn.T) + i0_ssn

        # Bottom up signals: Sensory feedforward input (from SSN to integration circuit)
        sensory_input = i1 * (-2.0 + r_ssn[k, :, 0:ne]) / 3.9  # this number makes the 1st mode of the input to be ~ i1

        # DECISION CIRCUIT -> SENSORY CIRCUIT: Top down modulation from the decision circuit to the sensory circuit. 
        # TODO: add parameter to adjust the amplitude of the overall modulation (now is divided by 60 hz)
        if att == True:
            att_mod_e = (np.dot(np.array([d1[k]]).T, np.array([gauss_mod_exc_1])) +
                         np.dot(np.array([d2[k]]).T, np.array([gauss_mod_exc_2]))) / 60.0 * i_att_e * sg_exc
            att_mod_i = (np.dot(np.array([d1[k]]).T, np.array([gauss_mod_inh_2])) +
                         np.dot(np.array([d2[k]]).T, np.array([gauss_mod_inh_1]))) / 60.0 * i_att_i * sg_inh
            att_mod = np.concatenate((att_mod_e, att_mod_i), axis=1) 
        elif att == False:
            att_mod = 0.0

        """
        NON-SELECTIVE MODULATION: Modulation from the decision circuit to the integration circuit
        """
        if choice_mod == True:
            decision_feedback = np.dot(np.array([global_gain * (d2[k] + d1[k]) / 2.0 / decided_fr]).T, np.array([flat]))
        else:
            decision_feedback = np.zeros((ntrials, n))

        # Stimuli 1 or 2 are ON
        if tcues[0] <= tstep < tcues[1]:
            if tstep == tcues[0]:
                decision_phase = compute_phase(r[k], n, cosine, sine)
                input1_e = i1_stim_e * np.take(inputs[0], samples[:, frame], axis=0)                # (ntrials x n/2)
                input1_i = i1_stim_i * np.take(inputs[1], samples[:, frame], axis=0)                # (ntrials x n/2)
                stim_input = np.concatenate((i0_stim_e + input1_e, i0_stim_i + input1_i), axis=1)   # (ntrials, n)
            sensory_noise = ou[s1]                                                                  # (ntrials, n)                                                              
            s1 += 1
        elif tstep == tcues[1]:  # The stimulus has finished. 
            # We store the phase of the bump at the end of the stimulus frame
            phases[frame] = compute_phase(r[k], n, cosine, sine)
            frame += 1
            if frame == 1:  # If the previous stimulus was the first one, set next stimulus' period
                tcues = [dcues[1], dcues[1] + lencue] 
            else:   # The previous stimulus was the second: end the stimulus
                stim_input = 0.0
                sensory_noise = 0.0
        elif dcues[0] <= tstep < dcues[1]:              # The decision period
            stim_input = 0.0
            sensory_noise = 0.0
            if not np.alltrue(decided):                 # Check if the decision has been made
                decided_now = ((d1[k] >= decision_threshold) | (d2[k] >= decision_threshold))
                new_decided = (decided_now & ~decided)
                if len(np.argwhere(new_decided)) > 0:
                    response_times[np.argwhere(new_decided)] = temps - go_signal
                    decision_phase[np.argwhere(new_decided)] = compute_phase(r[k, np.argwhere(new_decided)], n, cosine, sine)
                    decided = (decided | new_decided)

        # 3. Integration
        """
        Simulation of the dynamics of the SENSORY CIRCUIT.
        - Feedforward input to the integration circuit (sensory information).
        - Feedback input from the decision circuit (selective modulation).
        """
        try:
            r_ssn[kp] = r_ssn[k] + dt / tau_ssn * (-r_ssn[k] + ssn_f(i_ssn + stim_input + att_mod))

        except FloatingPointError:  # Prevent UnderFlow errors by setting the firing rate to 0.0
            r_ssn[r_ssn < 1E-12] = 0.0
            logger.warning('Overflow or underflow detected, check the integration of the SSN circuit.')

        """
        Simulation of the dynamics of the INTEGRATION CIRCUIT.
        """
        try:
            r[kp] = r[k] + ttau * (-r[k] + sigmoid_pw_v(i0 + sensory_input + sensory_noise + decision_feedback
                                                        + tau * s, tau=tau))  # shape=(ntrials, n)
        except FloatingPointError:  # Prevent UnderFlow errors by setting the firing rate to 0.0
            r[r < 1E-12] = 0.0

        """
        Simulation of the dynamics of the DECISION CIRCUIT.
        
        Decision circuit: the decision readout is performed by a second neural network consisting on a pair of 
        excitatory neural populations, both recurrently connected to a pool of interneurons. 
        This network [Wong and Wang (2006), Roxin and Ledberg (2008)] can be described with two identical recurrently 
        coupled neural populations, with local recurrent excitation and cross-coupled inhibition.
        
        The decision readout process starts with "an urgency signal" consisting on a non-specific external input
        targeting both populations, which will set the readout network in a WTA dynamical regime, where the activity 
        of one of the two populations will dominate over the other. Once a firing-rate threshold is reached the 
        decision (or the motor planning) is taken.
        """

        left_input = 1 / n / 2.0 * np.dot(r[k], kernel_l)
        right_input = 1 / n / 2.0 * np.dot(r[k], kernel_r)
        try:
            input_1 = dtaue * (rec_exc * d1[k] - cross_inh * ri[k]) + left_input + urgency_signal[tstep]
            input_2 = dtaue * (rec_exc * d2[k] - cross_inh * ri[k]) + right_input + urgency_signal[tstep]
            input_i = dtaui * g * (d1[k] + d2[k]) + i_i

            d1[kp] = d1[k] + dt / dtaue * (-d1[k] + sigmoid(input_1) / dtaue)
            d2[kp] = d2[k] + dt / dtaue * (-d2[k] + sigmoid(input_2) / dtaue)
            ri[kp] = ri[k] + dt / dtaui * (-ri[k] + sigmoid(input_i) / dtaui) 
        except FloatingPointError:
            d1[d1 < 1E-12] = 0.0
            d2[d2 < 1E-12] = 0.0
            ri[ri < 1E-12] = 0.0
 
        if save == 'partial' and tstep % saving_sampling == 0 and tstep != 0:
            r_profiles[saving_counter] = r[kp]
            rssn_profiles[saving_counter] = r_ssn[kp]
            d12[saving_counter] = np.array([d1[kp], d2[kp]])
            lri[saving_counter] = np.array([left_input, right_input])
            us[saving_counter] = np.array([urgency_signal[kp]])
            total_stim += (sensory_input + sensory_noise)
            perfect_phase[saving_counter] = compute_phase(total_stim, n, cosine, sine)
            saving_counter += 1

        temps += dt
        tstep += 1

    del ou
    gc.collect()

    temps -= dt
    tstep -= 1
    frame -= 1

    # Stop the timer
    logging.debug('Total time: {}.'.format(timeit.default_timer() - time_init))

    if opts.get('init_save', False):
        effe_i0_over = opts.get('i0_init', i0_over - sigmaou ** 2)
        save_ic('./obj/r0_initial_conditions.npy', r[-1, 0], n=n, i0_over=effe_i0_over,
                w0=modes[0], w1=modes[1], w2=modes[2])

    logging.debug('Preprocessing block data...')
    # TODO: check if frame is 1 (should be)
    estimation = phases[frame]
    last_phase = compute_phase(r[(tstep + rsteps - 1) % rsteps], n, cosine, sine)
    bin_choice = np.array(((d12[-2, 1] - d12[-2, 0]) >= 0) * 2 - 1, dtype=int)
    bin_choice[bin_choice == 0] = 1

    if save == 'all':
        return estimation, bin_choice, response_times, phases, decision_phase, last_phase, r, np.array(
            [d1, d2]).swapaxes(0, 1), np.array([left_input, right_input]), perfect_phase
    else:
        return estimation, bin_choice, response_times, phases, decision_phase, last_phase, r_profiles,\
            rssn_profiles, d12, lri, us, perfect_phase


def simulate_task_parallel(opts, n_trials=10000, chunk=1000, **kwargs):
    n = opts.get('n', 200)  # Number of neurons (spatial dimension)
    theta = np.arange(n) / n * (2 * np.pi) - np.pi

    # Set-up simulation environment: stimuli.
    inputs, samples_idx, simu_data = set_up_simulation(theta, n_trials, opts, **kwargs)

    # Run simulated trials in parallel
    sys.stdout.flush()
    processes = opts.get('cpus', mp.cpu_count() // 2)  # Default: Take all cpus (assuming hyperthreading)
    
    # noinspection PyTypeChecker
    parallel_simu = ParallelSimulation(run_simulation, n_trials, chunk, processes,
                                       show_progress=not opts.get('quiet', False))
    results = parallel_simu((samples_idx, simu_data.init_ph_n.to_numpy(dtype=int),
                             simu_data.do_choice.to_numpy(dtype=bool),), (inputs, opts))

    # Collect results
    logging.info('Merging chunked data...')
    return process_results(results, parallel_simu.n_b, opts.get('nframes', 2), chunk, simu_data, **kwargs)

import math 
def set_up_simulation(theta, n_trials, opts, **kwargs):
    n_frames = 2

    # orient_categories = opts.get('stim', np.arange(opts.get('min_orient'), 
    #                                                opts.get('max_orient') + opts.get('step_orient'), 
    #                                                opts.get('step_orient')))  # Input categories  
    orient_categories = np.array([-20, -10, -2, 0, 2, 10, 20])
    # orient_categories = np.delete(orient_categories, math.floor(orient_categories.size / 2)) 
    if np.all(orient_categories == 0) and not opts.get('init_save'):
        orient_categories = np.linspace(-20, 20, 5)

    # Precompute all possible orientations
    logging.info('Setting possible spatial inputs...')
    theta_m, orientations_m = np.meshgrid(theta, orient_categories)

    # Stimulus related parameters and the actual stimuli
    sigma_stim_e = np.deg2rad(opts.get('sigma_stim_e', 15))
    sigma_stim_i = np.deg2rad(opts.get('sigma_stim_i', 15))

    stim_e = np.exp(-circ_dist(theta_m, np.deg2rad(orientations_m)) ** 2 / (2.0 * sigma_stim_e ** 2))
    stim_i = np.exp(-circ_dist(theta_m, np.deg2rad(orientations_m)) ** 2 / (2.0 * sigma_stim_i ** 2))

    inputs = (stim_e, stim_i)  # tuple of 2 arrays with sizes (num_stim x n)
    del theta_m, orientations_m
    gc.collect()

    # Set up connectivity using default neuron and network parameters
    ne = opts.get('ne', opts.get('n', 200))  # number of exc neurons
    ni = opts.get('ni', opts.get('n', 200))  # number of inh neurons
    sigma_ori = opts.get('sigma_ori', np.deg2rad(2 * 32))  # parameter defining width of Gaussian connection profiles
    gee = opts.get('gee', 7.92)  # strength of excitation to excitatory neurons
    gie = opts.get('gie', 7.56)  # strength of excitation to inhibitory neurons
    gei = opts.get('gei', 4.14)  # strength of inhibition to excitatory neurons
    gii = opts.get('gii', 3.24)  # strength of inhibition to inhibitory neurons
    w_ssn = gaussian_connectivity(gee, gie, gii, gei, sigma_ori, ne=ne, ni=ni)
    opts['w_ssn'] = w_ssn

    # Setup dataframe
    stim_labels = ['x%di' % i for i in range(1, n_frames + 1)]
    phase_labels = ['ph%d' % i for i in range(1, n_frames + 1)]
    cols = ['estim', 'binchoice', 'binrt'] + phase_labels

    # Create random stimuli sequences:  (n_trials x num_frames)
    logger.info('Setting up stimuli dataframe.')
    stim_file = opts.get('stim_data', 'Stim_pdb_df_%d.npy' % n_trials)
    if stim_file == 'default':
        stim_file = 'Stim_pdb_df_%d.npy' % n_trials
    if not path_exists('./obj/' + stim_file):
        logger.warning(f"Custom stim. file './obj/{stim_file}' not found.")

    if opts.get('custom', False) or not path_exists('./obj/' + stim_file):
        logger.info('Generating new stimuli.')
        simu_data = pdb_new_stimuli(n_trials, orient_categories, **opts)
        simu_data['init_ph_n'] = np.ones(len(simu_data), dtype=int)
    else:
        logger.debug('Loading stimuli data from %s' % ('./obj/' + stim_file))
        simu_data = pd.read_pickle('./obj/' + stim_file)    
        load_init_ph = True
        nobump_data = dict()
        try:
            nobump_data = pd.read_csv(f"./results/pdb_simu_10000_nobump_sigma-0.15_i0"
                                      f"-{opts.get('i0'):.2f}_frames-8_t-{opts.get('tmax'):.1f}.csv") # puede aqui haber un problema con el nuevo nombre que yo he definido abajo?
        except FileNotFoundError:
            try:
                nobump_data = pd.read_csv(f"./results/pdb_simu_10000_nobump_sigma-0.15_i0"
                                          f"-{opts.get('i0'):.2f}_frames-8_t-3.0.csv")
            except FileNotFoundError:
                simu_data['init_ph_n'] = np.ones(len(simu_data), dtype=int)
                load_init_ph = False
        if load_init_ph:
            simu_data['init_ph_n'] = ((nobump_data['ph1'] + 180) / 360 * len(theta) - len(theta) / 2).astype(int)

    simu_data = pd.concat([simu_data, pd.DataFrame(columns=cols)], sort=False).reset_index()
    simu_data[['estim', 'binrt'] + phase_labels] = simu_data[['estim', 'binrt'] + phase_labels].astype(float)
    simu_data[['binchoice', 'bincorrect']] = simu_data[['binchoice', 'bincorrect']].astype(float)
    simu_data.rename(columns=dict(level_0='Trial'))
    simu_data.index.name = 'Trial'
    logging.debug('Stimuli categories: %s' % simu_data.category.unique())
    samples_idx = np.array(simu_data[stim_labels].to_numpy(), dtype=int)  # type: np.ndarray
    mylog(0)

    return inputs, samples_idx, simu_data


def process_results(results, n_blocks, n_frames, chunksize, dataframe, **kwargs):
    """

    :param results: List of results from :func:`run_simulation`, arranged in blocks.
    :param int n_blocks: Number of blocks in which the results are distributed.
    :param int n_frames: Number of frames of the task stimulus.
    :param int chunksize: Size of each simulation block (trials per block).
    :param pd.DataFrame dataframe: Data-frame containing the design of the simulated task.
    :param kwargs: Additional keyword arguments.
    :return: Rearranged results.
    """

    phase_labels = ['ph%d' % i for i in range(1, n_frames + 1)]
    r_profiles = []
    rssn_profiles = []
    d12 = []
    lri = []
    us = []
    perfect_phase = []

    for k in range(n_blocks):
        res = results[k]
        i1, i2 = k * chunksize, (k + 1) * chunksize - 1
        dataframe.loc[i1:i2, 'estim'] = np.rad2deg(res[0])
        dataframe.loc[i1:i2, 'binchoice'] = res[1]
        dataframe.loc[i1:i2, 'binrt'] = res[2]
        dataframe.loc[i1:i2, phase_labels] = np.rad2deg(res[3].T)
        dataframe.loc[i1:i2, 'phd'] = np.rad2deg(res[4].T)
        dataframe.loc[i1:i2, 'ph_last'] = np.rad2deg(res[5].T)
        r_profiles.append(res[6])
        rssn_profiles.append(res[7])
        d12.append(res[8])
        lri.append(res[9])
        us.append(res[10])
        perfect_phase.append(res[11])

    # reshaping from (blocks, rsteps/saving_counter, block_trials, n) to (rsteps/saving_counter, total_trials, n):
    r_profiles = np.concatenate(tuple(r_profiles), axis=1)
    rssn_profiles = np.concatenate(tuple(rssn_profiles), axis=1)
    d12 = np.concatenate(tuple(d12), axis=2)
    lri = np.concatenate(tuple(lri), axis=2)
    us = np.concatenate(tuple(us), axis=1)
    perfect_phase = np.concatenate(tuple(perfect_phase), axis=1)
    mylog(0)
    return dataframe, r_profiles, rssn_profiles, dict(d12=d12, lri=lri, us=us), perfect_phase


if __name__ == '__main__':
    # -- Simulation configuration: parsing, debugging.
    pars = Parser(desc='Simulation of the bump attractor network.', conf='conf_simu_pdb.txt',
                  groups=('Parameters', 'Network', 'Stimulus', 'Decision_circuit'))
    conf_options = pars.opts
    logger, mylog = log_conf(pars.debug_level)
    logger.debug("Command line arguments are: %s" % sys.argv)

    try:
        conf_options.update(get_paths(__file__))
    except NameError:
        pass    
    init_options.update(conf_options)
    num_trials = init_options.get('ntrials', 100)
    overwrite = init_options.get('overwrite', False)
    results_dir = str(init_options.get('res_dir', './results/'))
    create_dir(results_dir)

    min_oc, max_oc, step_oc = init_options.get('min_orient', -20), init_options.get('max_orient', 20), init_options.get('step_orient', 5)
    
    data, rdata, ssndata, ddata, p_ph = simulate_task_parallel(init_options, num_trials, 
                                                                        chunk=init_options.get('chunk', 100))
    
    # Create a file name for data saving
    bump = not init_options.get('no_bump', False)   # Default is True
    file_ref = ('bump' if bump else 'nobump')
    if bump == True:
        file_ref += ('' if init_options.get('_nobias', False) else '_biased')
    file_ref += (f'_attention-{init_options["attention"]}')
    file_ref += (f'_ntrials-{num_trials}_oc-range({min_oc},{max_oc},{step_oc})')
    file_ref += (f'_sigma-{init_options["sigmaOU"]:.2f}_i0-{init_options["i0"]:.2f}')
    file_ref += (f'_frames-{init_options["nframes"]:d}_t-{init_options["tmax"]:.1f}')
    
    results_dir += f'{file_ref}/' 
    if not os.path.exists(results_dir): os.makedirs(results_dir)   

    # np.save(results_dir + f'/us_{file_ref}.npy', ddata['us'])     # urgency signal to the decision circuit
    np.save(results_dir + f'/lri_{file_ref}.npy', ddata['lri'])   # left and right input to the decision circuit
    np.save(results_dir + f'/r_dc_{file_ref}.npy', ddata['d12'])  # activity of the decision circuit
    np.save(results_dir + f'/r_ic_{file_ref}.npy', rdata)         # activity of the integration circuit
    # np.save(results_dir + f'/r_sc_{file_ref}.npy', ssndata)       # activity of the sensory circuit

    # Save data
    if init_options.get('save', True) == True:
        sample_size = init_options.get('sample', 500)
        if sample_size > len(data): sample_size = len(data)
        random_choice = np.sort(np.random.choice(len(data), sample_size, replace=False))
        data['chosen'] = -1
        data.loc[random_choice, 'chosen'] = random_choice
        sampled_rdata = rdata[:, random_choice]
        sampled_ddata = ddata.copy()
        sampled_ddata['d12'] = sampled_ddata['d12'][:, :, random_choice]
        sampled_ddata['lri'] = sampled_ddata['lri'][:, :, random_choice]
        sampled_p_ph = p_ph[:, random_choice]

        results_dict = dict(conf=init_options, rates=sampled_rdata, data=data, ddata=sampled_ddata, p_ph=sampled_p_ph)
        filename = check_overwrite(results_dir + f'pdb_{file_ref}' + '.npy', force=overwrite, auto=True)
        filename, extension = os.path.splitext(filename)
        logger.info(f'Changing directory to {os.path.dirname(filename)}')
        logger.info(f'Saving data to {os.path.basename(filename)}.npy ...')
        if save_obj(results_dict, filename, extension='.npy'):
            mylog(0)
        else:
            mylog.msg(1, up_lines=2)
        logger.info(f'Saving data-frame to {os.path.basename(filename)}.csv...')
        try:
            data.to_csv(filename + '.csv')  # Save the data-frame in csv format
            mylog(0)
        except FileNotFoundError:
            mylog(1)

    # Plotting
    if init_options.get('plot', False):
        logger.info('Plotting some preliminary results...')  # Summary results of the pdb task
        f_pdg, axs_pdb = plot_pdb_weights(data, save=True, **init_options)
        
    #     f, ax = plot_trial(rdata, data, save=init_options.get('save', True), **init_options)
    #     #norm_data, lbls = normalize_dataframe(data)
    #     #f2, axs2 = plot_stats_summ(norm_data, sigma=init_options['sigmaOU'], i0=init_options['i0'], auto=True, bump=bump, old_ppk=True, save=init_options.get('save', True))
    #     #f2, axs2 = plot_stats_summ(norm_data, sigma=init_options['sigmaOU'], i0=init_options['i0'], auto=True, bump=bump, old_ppk=True, save=True)
    #     #f3, axs3 = plot_decision_circ(data, rdata, ddata, init_options, save=init_options.get('save', False), **init_options)
    #     #f3, axs3 = plot_decision_circ(data, rdata, ddata, init_options, save=True, **init_options)
    #     #f4, axs4 = plot_response_time(data, init_options, save=init_options.get('save', True), **init_options)
    #     #f4, axs4 = plot_response_time(data, init_options, save=True, **init_options)
        
    #     if init_options.get('show_plots', True):
    #         #logger.info('Showing plot now...')
    #         plt.show()

        #f, axes = plot_estimation_weights(data)
