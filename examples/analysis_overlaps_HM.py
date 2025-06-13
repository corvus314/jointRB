import bilby
from jointRB import JointLikelihoodRBHoM
import numpy as np
import json
import argparse
from copy import copy

total_err = 0.03

outdir = 'outdir'
label = 'overlaps_HM'

parameters_eventA = {
    "mass_ratio": 0.2231017059328386,
    "chirp_mass": 22.353653661893304,
    "luminosity_distance": 7268.571745364448,
    "dec": 0.6557802485585996,
    "ra": 2.180684709384296,
    "theta_jn": 1.8552485990854504,
    "psi": 2.317315123447218,
    "phase": 0.736517637678661,
    "a_1": 0.5874657489189719,
    "a_2": 0.7740290285427935,
    "tilt_1": 1.8400317067401333,
    "tilt_2": 2.4431533314845897,
    "phi_12": 1.4830050470970393,
    "phi_jl": 3.5537095497820763,
    "geocent_time": 0.0
}
parameters_eventB = {
    "mass_ratio": 0.37172837551203913,
    "chirp_mass": 47.64517481623601,
    "luminosity_distance": 50094.34936920892,
    "dec": -0.25296571570747095,
    "ra": 4.80942360638223,
    "theta_jn": 0.7722659493489856,
    "psi": 0.3869169812264324,
    "phase": 1.0168760933964565,
    "a_1": 0.8251185871884739,
    "a_2": 0.03742190139192977,
    "tilt_1": 1.2114859809247314,
    "tilt_2": 1.731705006294819,
    "phi_12": 1.2192989717535039,
    "phi_jl": 4.538201491570135,
    "geocent_time": 0.013082053778854186
}

np.random.seed(5039232)
bilby.core.utils.random.seed(5039232)
minimum_frequency =5.
duration = 59.0
sampling_frequency = 4096.


injection=dict()
combined = dict()
for key in parameters_eventA.keys():
    combined['%s_A'%key] = parameters_eventA[key]
    combined['%s_B'%key] = parameters_eventB[key]
    injection['%s_A'%key] = parameters_eventA[key]
    injection['%s_B'%key] = parameters_eventB[key]

test_params = injection.copy()
fudge = 0.02
test_params['chirp_mass_A'] *= (1.+fudge)
test_params['chirp_mass_B'] *= (1.+fudge)
test_params['mass_ratio_A'] = min(test_params['mass_ratio_A']*(1.+fudge), 1.)
test_params['mass_ratio_B'] = min(test_params['mass_ratio_B']*(1.+fudge), 1.)


mode_array = [[2, 1], [2, 2], [3, 2], [3, 3], [4, 4]]

waveform_arguments = dict(waveform_approximant = 'IMRPhenomXPHM',
                            reference_frequency = 20.,
                            sampling_frequency = sampling_frequency,
                            minimum_frequency = minimum_frequency,
                            maximum_frequency = sampling_frequency/2,
                            mode_array = mode_array)

waveform_generator = bilby.gw.WaveformGenerator(
    duration = duration, sampling_frequency = sampling_frequency,
    frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments = waveform_arguments)
ifos = bilby.gw.detector.InterferometerList(['CE', 'ET'])
for ifo in ifos:
    ifo.minimum_frequency = minimum_frequency

ifos.set_strain_data_from_power_spectral_densities(duration = duration,
                                        sampling_frequency = sampling_frequency,
                                        start_time = parameters_eventA['geocent_time']-duration+2.)

ifos.inject_signal(waveform_generator=waveform_generator, parameters=parameters_eventA)
ifos.inject_signal(waveform_generator=waveform_generator, parameters=parameters_eventB)

print("setting up prior")
temp_priors = bilby.gw.prior.BBHPriorDict()

priors = bilby.core.prior.PriorDict()

for key in temp_priors.keys():
    priors['%s_A'%key] = copy(temp_priors[key])
    priors['%s_B'%key] = copy(temp_priors[key])

priors['geocent_time_A'] = bilby.core.prior.Uniform(name = 'geocent_time_A',
                                                    minimum = parameters_eventA['geocent_time']-0.1, 
                                                    maximum = parameters_eventA['geocent_time']+0.1,
                                                    latex_label = '$t_{c, A}$')

priors['geocent_time_B'] = bilby.core.prior.Uniform(name = 'geocent_time_B',
                                                    minimum = parameters_eventB['geocent_time']-0.1, 
                                                    maximum = parameters_eventB['geocent_time']+0.1,
                                                    latex_label = '$t_{c, B}$')
priors['chirp_mass_A'].minimum = parameters_eventA['chirp_mass']-5.
priors['chirp_mass_A'].maximum = parameters_eventA['chirp_mass']+5.
priors['luminosity_distance_A'].minimum=0.2*parameters_eventA["luminosity_distance"]
priors['luminosity_distance_A'].maximum=5*parameters_eventA["luminosity_distance"]
priors['chirp_mass_B'].minimum = parameters_eventB['chirp_mass']-5.
priors['chirp_mass_B'].maximum = parameters_eventB['chirp_mass']+5.
priors['luminosity_distance_B'].minimum=0.2*parameters_eventB["luminosity_distance"]
priors['luminosity_distance_B'].maximum=5*parameters_eventB["luminosity_distance"]

lal_max_f = JointLikelihoodRBHoM.lal_f_max(
    JointLikelihoodRBHoM.relby_conversion(combined, waveform_arguments, n_signals=2))

likelihood = JointLikelihoodRBHoM.OverlappingSignalsRBHOM(
    ifos, injection, test_params,
    priors, lal_max_f, mode_array, waveform_arguments,
    total_err, grid_choice = 'bisection', n_signals=2, delete_strains = True)

results = bilby.run_sampler(likelihood = likelihood, injection_parameters=injection,
                            priors = priors, sampler = 'dynesty', nlive = 1500,
                            naccept = 60, check_point_plot = True, check_point_delta_t = 1800, #1800
                            print_method = 'interval-60', sample = 'acceptance-walk',
                            npool = 16, outdir = outdir, label = label)
results.save_posterior_samples()

