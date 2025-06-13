# the code below uses CE2 interferometer which is not implemented in bilby
# it can be created by copying CE2* files to the locaton of ther bilby detectors and psds

from tqdm import tqdm
import bilby
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import json
from jointRB.LikelihoodRB import RelBinning

injection_A = {
    "mass_ratio": 0.6157258746024716,
    "chirp_mass": 30.436780509375577,
    "luminosity_distance": 5344.605494083707,
    "dec": 0.3153509604743991,
    "ra": 5.247038620869544,
    "theta_jn": 1.5812888541529055,
    "psi": 1.072954254064659,
    "phase": 2.834000786332104,
    "chi_1": 0.10553045484638737,
    "chi_2": -0.185616711205764,
    "geocent_time": 0.0
}
injection_B ={
    "mass_ratio": 0.22291670532709276,
    "chirp_mass": 28.950984926824592,
    "luminosity_distance": 13045.442063337407,
    "dec": 0.5069247737606671,
    "ra": 4.895532749849941,
    "theta_jn": 2.711043886326933,
    "psi": 2.2522067062609503,
    "phase": 1.1843346929984002,
    "chi_1": 0.015906227405961714,
    "chi_2": 0.2321025781009781,
    "geocent_time": -0.00618093560585308
}
        
from jointRB import fisher_biases

from scipy.optimize import differential_evolution

def get_initial_maximimum_posterior_sample(beta):
    new_bound_widths = np.maximum(np.diag(covariance), np.diag(generator.covariance_A))**0.5*4
    new_bound_widths
    bounds = [[a, b] for a,b in zip(generator.lb, generator.ub)]
    for i, bound in enumerate(bounds):
        if initial_guess[i] + new_bound_widths[i] < bounds[i][1]:
            bounds[i][1] = initial_guess[i] + new_bound_widths[i]
        if initial_guess[i] - new_bound_widths[i] > bounds[i][0]:
            bounds[i][0] = initial_guess[i] - new_bound_widths[i]

    if bounds[r_idx][1] > 1.:
        bounds[r_idx][1] = 1
    #print(bounds)

    def neg_log_likelihood(x):
        params = {param: x[i] for i, param in enumerate(param_list)}
        likelihood.parameters = params
        return -beta * likelihood.log_likelihood_ratio()

    res = differential_evolution(neg_log_likelihood, bounds, popsize=15, init="sobol")
    return res



def neg_l(x):
    params = {param: x[i] for i, param in enumerate(param_list)}
    likelihood.parameters = params
    return -1* likelihood.log_likelihood_ratio()

from scipy.optimize import minimize
        
bilby.core.utils.log.setup_logger(log_level='WARNING')

param_list = list(injection_A.keys())

seed = 705625
np.random.seed(seed)
bilby.core.utils.random.seed(seed)

sampling_frequency = 2048
minimum_frequency =5.

duration = 40.0

waveform_arguments = dict(waveform_approximant = 'IMRPhenomXP',
                        reference_frequency = 50.,
                        minimum_frequency = minimum_frequency,
                        maximum_frequency = sampling_frequency/2)

waveform_generator = bilby.gw.WaveformGenerator(
    duration = duration, sampling_frequency = sampling_frequency,
    frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments = waveform_arguments)

ifos = bilby.gw.detector.InterferometerList(['CE', 'CE2', 'ET'])
ifos.set_strain_data_from_zero_noise(duration = duration,
                                                sampling_frequency = sampling_frequency,
                                                start_time = injection_A['geocent_time']-duration+4)

for ifo in ifos:
    ifo.minimum_frequency = minimum_frequency

from copy import copy
    
priors = bilby.gw.prior.BBHPriorDict(aligned_spin = True)
priors['mass_1'].maximum = 700
priors['mass_2'].maximum = 700
priors['luminosity_distance'].maximum=5*injection_A["luminosity_distance"]
priors['chirp_mass'].minimum = injection_A['chirp_mass']*0.5
priors['chirp_mass'].maximum = injection_A['chirp_mass']*1.5
priors['geocent_time'] = bilby.core.prior.Uniform(name = 'geocent_time',
                                                minimum = injection_A['geocent_time']-0.1, 
                                                maximum = injection_A['geocent_time']+0.1,
                                                latex_label = '$t_{c}$')

noise_array = np.zeros([len(ifos), len(ifos[0].frequency_array)], dtype = complex)
for l, ifo in enumerate(ifos):
    noise_array[l] = ifo.strain_data.frequency_domain_strain

param_list = list(injection_A.keys())

ifos.inject_signal(waveform_generator=waveform_generator, parameters=injection_A)
ifos.inject_signal(waveform_generator=waveform_generator, parameters=injection_B)

waveform_generator_2 = bilby.gw.WaveformGenerator(
    duration = duration, sampling_frequency = sampling_frequency,
    frequency_domain_source_model = bilby.gw.source.binary_black_hole_frequency_sequence,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments = waveform_arguments)

likelihood = RelBinning(
    ifos, waveform_generator = waveform_generator_2, delta=0.01,
    ref_injection=injection_A, priors=priors)

# initial estimation of maxL point

generator = fisher_biases.FisherGenerator(ifos, waveform_generator, priors, noise_array, param_list, injection_A, prior_limit_schema=1)
initial_guess, covariance = generator.approximate_likelihood(injection_B, fisher_at='bias', shift_by_noise=False)

# refine by 1st sampling step
r_idx = param_list.index('mass_ratio')
if initial_guess[r_idx] > 1.:
    initial_guess[r_idx] = 1. / initial_guess[r_idx]
for k, param in enumerate(param_list):
    if priors[param].boundary == 'periodic':
        initial_guess[k] = initial_guess[k] % priors[param].maximum

result = get_initial_maximimum_posterior_sample(1.)

# refine by 2nd sampling step
bounds = [[a, b] for a,b in zip(generator.lb, generator.ub)]
refined_guess = result.x
for k, param in enumerate(param_list):
    if priors[param].boundary == 'periodic':
        refined_guess[k] = refined_guess[k] % priors[param].maximum
result2 = minimize(neg_l, refined_guess, bounds=bounds, method = 'Nelder-Mead')

refined_guess2 = result2.x
for k, param in enumerate(param_list):
    if priors[param].boundary == 'periodic':
        refined_guess2[k] = refined_guess2[k] % priors[param].maximum
point = {key:refined_guess2[i] for i, key in enumerate(param_list)}

# get likelihood mean and covariance
generator = fisher_biases.FisherGenerator(ifos, waveform_generator, priors, noise_array, param_list, point, prior_limit_schema=1)
mean3, cov3 = generator.approximate_likelihood(params_B=None, fisher_at='injection', shift_by_noise=False)

mean3, cov3 = generator.approximate_likelihood(params_B=None, fisher_at='injection', shift_by_noise=False)
# get samples from the posterior
samples, weights = generator.draw_samples_from_fisher(mean3, cov3, 10000)

posterior = pd.DataFrame()
for i, key in enumerate(param_list):
    posterior[key]=samples[:,i]
posterior['weights'] = weights
posterior.to_csv('posterior.csv', index=False)