import bilby
from jointRB.LikelihoodRB import RelBinning
import numpy as np


outdir = f'outdir'
label = f'single'

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

np.random.seed(0)
bilby.core.utils.random.seed(0)
sampling_frequency = 2048
minimum_frequency =5.

duration = 171.0

injection = parameters_eventA

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

from copy import copy

priors = bilby.gw.prior.BBHPriorDict(aligned_spin = True)
priors['mass_1'].maximum = 700
priors['mass_2'].maximum = 700
priors['luminosity_distance'].maximum=5*parameters_eventA["luminosity_distance"]
priors['chirp_mass'].minimum = parameters_eventA['chirp_mass']*0.5
priors['chirp_mass'].maximum = parameters_eventA['chirp_mass']*1.5
priors['geocent_time'] = bilby.core.prior.Uniform(
    name = 'geocent_time',
    minimum = parameters_eventA['geocent_time']-0.1, 
    maximum = parameters_eventA['geocent_time']+0.1,
    latex_label = '$t_{c}$')

waveform_generator_2 = bilby.gw.WaveformGenerator(
    duration = duration, sampling_frequency = sampling_frequency,
    frequency_domain_source_model = bilby.gw.source.binary_black_hole_frequency_sequence,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments = waveform_arguments)

likelihood = RelBinning(
    ifos, waveform_generator = waveform_generator_2, delta=0.01,
    ref_injection=injection, priors=priors)


results = bilby.run_sampler(likelihood = likelihood, injection_parameters=injection,
                            priors = priors, sampler = 'dynesty', naccept = 60,
                            npool = 16, nlive = 1200, maxmcmc = 10000,
                            outdir = outdir, label = label, print_method = 'interval-60',
                            check_point_plot = True, sample = 'acceptance-walk')
    
