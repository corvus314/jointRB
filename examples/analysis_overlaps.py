import bilby
from jointRB.JointLikelihoodRB import OverlappingSignalsRelBinning
import numpy as np


outdir = f'outdir'
label = f'overlaps'

parameters_eventA ={
    "mass_ratio": 0.836905750635001,
    "chirp_mass": 11.642410663571638,
    "luminosity_distance": 6545.638556527845,
    "dec": 0.6917991728087165,
    "ra": 5.465286497812088,
    "theta_jn": 1.556706780883036,
    "psi": 1.950153490070258,
    "phase": 3.496425059947941,
    "chi_1": -0.008838732800268959,
    "chi_2": 0.07391971403445817,
    "geocent_time": 0.0
}
parameters_eventB = {
    "mass_ratio": 0.8713643916251477,
    "chirp_mass": 14.438622404435074,
    "luminosity_distance": 6149.882257130636,
    "dec": 0.0816789756880578,
    "ra": 3.7729691857988414,
    "theta_jn": 1.719818193677204,
    "psi": 1.6325547617287894,
    "phase": 1.9746134246922862,
    "chi_1": 0.07596938787031247,
    "chi_2": 0.015143749249390836,
    "geocent_time": 0.0
}


np.random.seed(0)
bilby.core.utils.random.seed(0)
sampling_frequency = 2048
minimum_frequency =5.

duration = 171.0


injection=dict()
for key in parameters_eventA.keys():
    injection['%s_A'%key] = parameters_eventA[key]
    injection['%s_B'%key] = parameters_eventB[key]

waveform_arguments = dict(waveform_approximant = 'IMRPhenomXP',
                            reference_frequency = 50.,
                            minimum_frequency = minimum_frequency,
                            maximum_frequency = sampling_frequency/2)

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
                                        start_time = parameters_eventA['geocent_time']-duration+4)

ifos.inject_signal(waveform_generator=waveform_generator, parameters=parameters_eventA)
ifos.inject_signal(waveform_generator=waveform_generator, parameters=parameters_eventB)

from copy import copy

temp_priors = bilby.gw.prior.BBHPriorDict(aligned_spin = True)
temp_priors['mass_1'].maximum = 700
temp_priors['mass_2'].maximum = 700
temp_priors['luminosity_distance'].maximum=5*max(parameters_eventA["luminosity_distance"],
                                                    parameters_eventB["luminosity_distance"])
temp_priors['chirp_mass'].minimum = parameters_eventA['chirp_mass']*0.5
temp_priors['chirp_mass'].maximum = parameters_eventA['chirp_mass']*1.5
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

waveform_generator_2 = bilby.gw.WaveformGenerator(
    duration = duration, sampling_frequency = sampling_frequency,
    frequency_domain_source_model = bilby.gw.source.binary_black_hole_frequency_sequence,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments = waveform_arguments)

likelihood = OverlappingSignalsRelBinning(
    ifos, waveform_generator = waveform_generator_2, delta=0.01,
    ref_injection=injection, N_overlaps = 2, priors=priors)

results = bilby.run_sampler(likelihood = likelihood, injection_parameters=injection,
                            priors = priors, sampler = 'dynesty', naccept = 60,
                            npool = 16, nlive = 1500, maxmcmc = 10000,
                            outdir = outdir, label = label, print_method = 'interval-60',
                            check_point_plot = True, sample = 'acceptance-walk')
