from bilby.gw import GravitationalWaveTransient
import numpy as np
from bilby.core.utils import (
    logger, speed_of_light, solar_mass, radius_of_earth,
    gravitational_constant
)
from tqdm import tqdm




class OverlappingSignals(GravitationalWaveTransient):
    """
    Class encoding overlapping signals
    """

    def __init__(
        self, interferometers, waveform_generator, inj_example, N_overlaps = 2,
        distance_marginalization=False, phase_marginalization=False, priors=None,
        distance_marginalization_lookup_table=None, reference_frame="sky", time_reference="geocenter"
    ):
        super().__init__(
            interferometers=interferometers, waveform_generator=waveform_generator, priors=priors,
            distance_marginalization=distance_marginalization, phase_marginalization=phase_marginalization,
            time_marginalization=False, distance_marginalization_lookup_table=distance_marginalization_lookup_table,
            jitter_time=False, reference_frame=reference_frame, time_reference=time_reference
        )
        self.N_overlaps = N_overlaps
        self._get_parameter_sets(inj_example)
        
    def _get_parameter_sets(self, inj_example):
        from math import inf
        full_parameters_list = []
        parameters_list=[]
        for i in range(self.N_overlaps):
            full_parameters_list.append([key for key, value in inj_example.items() if chr(ord('A') + i) in key])
            parameters_list.append([name[:name.rfind('_')] for name in full_parameters_list[i]])

        self.full_parameters_list=full_parameters_list
        self.parameters_list=parameters_list
        
        
        
    def calculate_snrs(self, waveform_polarizations, parameters, interferometer):
        """
        Compute the snrs

        Parameters
        ==========
        waveform_polarizations: dict
            A dictionary of waveform polarizations and the corresponding array
        interferometer: bilby.gw.detector.Interferometer
            The bilby interferometer object

        """
        signal = np.zeros(len(interferometer.frequency_array), dtype=complex)
        for i in range(self.N_overlaps):
            signal += interferometer.get_detector_response(
                waveform_polarizations[i], parameters[i])
        _mask = interferometer.frequency_mask

        if 'recalib_index' in self.parameters:
            signal[_mask] *= self.calibration_draws[interferometer.name][int(self.parameters['recalib_index'])]

        d_inner_h = interferometer.inner_product(signal=signal)
        optimal_snr_squared = interferometer.optimal_snr_squared(signal=signal)
        complex_matched_filter_snr = d_inner_h / (optimal_snr_squared**0.5)

        d_inner_h_array = None
        optimal_snr_squared_array = None

        normalization = 4 / self.waveform_generator.duration

        if self.time_marginalization and self.calibration_marginalization:

            d_inner_h_integrand = np.tile(
                interferometer.frequency_domain_strain.conjugate() * signal /
                interferometer.power_spectral_density_array, (self.number_of_response_curves, 1)).T

            d_inner_h_integrand[_mask] *= self.calibration_draws[interferometer.name].T

            d_inner_h_array = 4 / self.waveform_generator.duration * np.fft.fft(
                d_inner_h_integrand[0:-1], axis=0
            ).T

            optimal_snr_squared_integrand = (
                normalization * np.abs(signal)**2 / interferometer.power_spectral_density_array
            )
            optimal_snr_squared_array = np.dot(
                optimal_snr_squared_integrand[_mask],
                self.calibration_abs_draws[interferometer.name].T
            )

        elif self.time_marginalization and not self.calibration_marginalization:
            d_inner_h_array = normalization * np.fft.fft(
                signal[0:-1]
                * interferometer.frequency_domain_strain.conjugate()[0:-1]
                / interferometer.power_spectral_density_array[0:-1]
            )

        elif self.calibration_marginalization and ('recalib_index' not in self.parameters):
            d_inner_h_integrand = (
                normalization *
                interferometer.frequency_domain_strain.conjugate() * signal
                / interferometer.power_spectral_density_array
            )
            d_inner_h_array = np.dot(d_inner_h_integrand[_mask], self.calibration_draws[interferometer.name].T)

            optimal_snr_squared_integrand = (
                normalization * np.abs(signal)**2 / interferometer.power_spectral_density_array
            )
            optimal_snr_squared_array = np.dot(
                optimal_snr_squared_integrand[_mask],
                self.calibration_abs_draws[interferometer.name].T
            )

        return self._CalculatedSNRs(
            d_inner_h=d_inner_h, optimal_snr_squared=optimal_snr_squared,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_array=d_inner_h_array,
            optimal_snr_squared_array=optimal_snr_squared_array)

    
    
    def log_likelihood_ratio(self):
        parameters = [{self.parameters_list[i][j]:self.parameters[self.full_parameters_list[i][j]]
                      for j in range(len(self.parameters_list[i]))} for i in range(self.N_overlaps)]
        waveform_polarizations = \
            [self.waveform_generator.frequency_domain_strain(parameters[i]) for i in range(self.N_overlaps)]

        for i in range(self.N_overlaps):
            parameters[i].update(self.get_sky_frame_parameters(parameters[i]))

        if waveform_polarizations is None:
            return np.nan_to_num(-np.inf)

        d_inner_h = 0.
        optimal_snr_squared = 0.
        complex_matched_filter_snr = 0.

        if self.time_marginalization and self.calibration_marginalization:
            if self.jitter_time:
                self.parameters['geocent_time'] += self.parameters['time_jitter']

            d_inner_h_array = np.zeros(
                (self.number_of_response_curves, len(self.interferometers.frequency_array[0:-1])),
                dtype=np.complex128)
            optimal_snr_squared_array = np.zeros(self.number_of_response_curves, dtype=np.complex128)

        elif self.time_marginalization:
            if self.jitter_time:
                self.parameters['geocent_time'] += self.parameters['time_jitter']
            d_inner_h_array = np.zeros(len(self._times), dtype=np.complex128)

        elif self.calibration_marginalization:
            d_inner_h_array = np.zeros(self.number_of_response_curves, dtype=np.complex128)
            optimal_snr_squared_array = np.zeros(self.number_of_response_curves, dtype=np.complex128)

        for interferometer in self.interferometers:
            per_detector_snr = self.calculate_snrs(
                waveform_polarizations=waveform_polarizations,
                parameters = parameters,
                interferometer=interferometer)

            d_inner_h += per_detector_snr.d_inner_h
            optimal_snr_squared += np.real(per_detector_snr.optimal_snr_squared)
            complex_matched_filter_snr += per_detector_snr.complex_matched_filter_snr

            if self.time_marginalization or self.calibration_marginalization:
                d_inner_h_array += per_detector_snr.d_inner_h_array

            if self.calibration_marginalization:
                optimal_snr_squared_array += per_detector_snr.optimal_snr_squared_array

        if self.calibration_marginalization and self.time_marginalization:
            log_l = self.time_and_calibration_marginalized_likelihood(
                d_inner_h_array=d_inner_h_array,
                h_inner_h=optimal_snr_squared_array)
            if self.jitter_time:
                self.parameters['geocent_time'] -= self.parameters['time_jitter']

        elif self.calibration_marginalization:
            log_l = self.calibration_marginalized_likelihood(
                d_inner_h_calibration_array=d_inner_h_array,
                h_inner_h=optimal_snr_squared_array)

        elif self.time_marginalization:
            log_l = self.time_marginalized_likelihood(
                d_inner_h_tc_array=d_inner_h_array,
                h_inner_h=optimal_snr_squared)
            if self.jitter_time:
                self.parameters['geocent_time'] -= self.parameters['time_jitter']

        elif self.distance_marginalization:
            log_l = self.distance_marginalized_likelihood(
                d_inner_h=d_inner_h, h_inner_h=optimal_snr_squared)

        elif self.phase_marginalization:
            log_l = self.phase_marginalized_likelihood(
                d_inner_h=d_inner_h, h_inner_h=optimal_snr_squared)

        else:
            log_l = np.real(d_inner_h) - optimal_snr_squared / 2

        return float(log_l.real)
    
    def get_sky_frame_parameters(self, params):
        time = params['{}_time'.format(self.time_reference)]
        if not self.reference_frame == "sky":
            ra, dec = zenith_azimuth_to_ra_dec(
                params['zenith'], params['azimuth'],
                time, self.reference_frame)
        else:
            ra = params["ra"]
            dec = params["dec"]
        if "geocent" not in self.time_reference:
            geocent_time = time - self.reference_ifo.time_delay_from_geocenter(
                ra=ra, dec=dec, time=time
            )
        else:
            geocent_time = params["geocent_time"]
        return dict(ra=ra, dec=dec, geocent_time=geocent_time)
    
    
    
    
class OverlappingSignalsRelBinning(GravitationalWaveTransient):
    """
    Class encoding overlapping signals
    """

    def __init__(
        self, interferometers, waveform_generator, ref_injection, N_overlaps = 2, priors=None,
        reference_frame="sky", time_reference="geocenter", delta = 0.03
    ):
        super().__init__(
            interferometers=interferometers, waveform_generator=waveform_generator, priors=priors,
            distance_marginalization=False, phase_marginalization=False,
            time_marginalization=False, distance_marginalization_lookup_table=None,
            jitter_time=False, reference_frame=reference_frame, time_reference=time_reference
        )
        if 'minimum_frequency' in self.waveform_generator.waveform_arguments.keys():
            self.f_min = self.waveform_generator.waveform_arguments['minimum_frequency']
        else:
            self.f_min = 20.
        if 'maximum_frequency' in self.waveform_generator.waveform_arguments.keys():
            self.f_max = self.waveform_generator.waveform_arguments['maximum_frequency']
        else:
            # take Nyquist frequency
            self.f_max = self.waveform_generator.sampling_frequency/2.
        self.duration = self.waveform_generator.duration
        self.N_overlaps = N_overlaps
        self._get_parameter_sets(ref_injection)
        self._get_orig_frequency()
        self.delta = delta
        self._get_reference_strains()
        self._get_frequency_bands(delta)
        self._get_summary_data()
        
        strains = np.zeros((self.N_overlaps, len(self.interferometers), self.N_bins+1),
                            dtype=complex)
        for i in range(self.N_overlaps):
            strains[i,:,:]+=self.get_network_response(self.parameters_ref[i])
        self.inverse_binned_ref_strains = (1+0j)/strains
        self.inverse_binned_ref_strains[np.isnan(self.inverse_binned_ref_strains)]=0j
        
        
    def _get_parameter_sets(self, ref_injection):
        """
        based on reference strain infers parameters for each signal
        """
        from math import inf
        full_parameters_list = []
        parameters_list=[]
        for i in range(self.N_overlaps):
            full_parameters_list.append([key for key, value in ref_injection.items() if chr(ord('A') + i) in key])
            parameters_list.append([name[:name.rfind('_')] for name in full_parameters_list[i]])

        self.full_parameters_list=full_parameters_list
        self.parameters_list=parameters_list
        self.parameters_ref=[{self.parameters_list[i][j]:ref_injection[self.full_parameters_list[i][j]]
                              for j in range(len(self.parameters_list[i]))} for i in range(self.N_overlaps)]
        
    def _get_orig_frequency(self):
        """
        sets banding frequency to original frequencies at the beginning
        """
        self.frequency_array = self.interferometers[0].frequency_array
        self.frequency_mask = self.interferometers[0].strain_data.frequency_mask
        self.frequency_mask[self.frequency_array<self.f_min]=False
        self.frequency_mask[self.frequency_array>self.f_max]=False
        
        self.band_indecies = np.arange(len(self.frequency_array))[self.frequency_mask].tolist()
        self.frequency_bands = self.frequency_array[self.band_indecies]
        self.waveform_generator.waveform_arguments['frequencies'] = self.frequency_bands
        self.N_bins=len(self.band_indecies)-1
        self.f_m = (self.frequency_bands[:-1]+self.frequency_bands[1:])/2.
        self.bandwiths = self.frequency_bands[1:]-self.frequency_bands[:-1]
        
    def _get_frequency_bands(self, delta):
        """
        sets banding frequencies
        """
        powers = np.array([-5./3, -2./3, 1., 5./3, 7./3])
        signs = np.sign(powers)
        f_ref = np.array([((self.f_min)*(power<0)+(self.f_max)*(power>0.)) for power in powers])
        alphas=(2*np.pi/f_ref**powers)*signs
        dphi = np.sum([alpha*self.frequency_bands**power for alpha, power in zip(alphas,powers)],axis=0)
        new_bands = [0]
        i_start = 0
        for i_end in self.last_nonzero_frequency_indecies:
            for i in range(i_start, i_end+1):
                if np.abs(dphi[new_bands[-1]]-dphi[i])>=delta:
                    new_bands.append(i)
            if new_bands[-1]!=i_end:
                new_bands.append(i_end)
            i_start=i_end+1
            
        self.band_indecies = np.array(self.band_indecies)[new_bands].tolist()
        self.frequency_bands = self.frequency_array[self.band_indecies]
        self.waveform_generator.waveform_arguments['frequencies'] = self.frequency_bands
        self.N_bins=len(self.band_indecies)-1
        self.f_m = (self.frequency_bands[:-1]+self.frequency_bands[1:])/2.
        self.bandwiths = self.frequency_bands[1:]-self.frequency_bands[:-1]
        
    def get_network_response(self, parameters):
        """
        returns strain given parameters (as one matrix combining all the detectors
        """
        waveform_polarizations = \
            self.waveform_generator.frequency_domain_strain(parameters)
                

        parameters.update(self.get_sky_frame_parameters(parameters))
                
        strain = np.zeros((len(self.interferometers), len(self.frequency_bands)), dtype=complex)
        for i, ifo in enumerate(self.interferometers):
            for mode in waveform_polarizations:
                response = ifo.antenna_response(
                    parameters['ra'], parameters['dec'],
                    parameters['geocent_time'], parameters['psi'],
                    mode
                )
                strain[i,:] += waveform_polarizations[mode] * response
            dt = ifo.time_delay_from_geocenter(
                parameters['ra'], parameters['dec'],
                parameters['geocent_time'])
            dt_geocent = parameters['geocent_time'] - ifo.strain_data.start_time
            ifo_time = dt_geocent + dt
            strain[i,:] *= np.exp(-1j * 2. * np.pi * self.frequency_bands * ifo_time)

            calib_factor = ifo.calibration_model.get_calibration_factor(
                self.frequency_bands, prefix='recalib_{}_'.format(ifo.name), **parameters)

            strain[i,:] *= calib_factor
                

        return strain
        
    def _get_reference_strains(self):
        """
        gets reference strain on full frequencies
        """
        strains = np.zeros((self.N_overlaps, len(self.interferometers), len(self.frequency_array)),
                            dtype=complex)
        for i in range(self.N_overlaps):
            
            strains[i][:,self.frequency_mask]+=self.get_network_response(self.parameters_ref[i])
        self.ref_strains = strains
        
        self.last_nonzero_frequency_indecies=np.ones(self.N_overlaps, dtype=int)*(len(self.band_indecies)-1)
        for i in range(self.N_overlaps):
            while((strains[i,:,self.band_indecies[self.last_nonzero_frequency_indecies[i]]]==0j).all()
                  and self.last_nonzero_frequency_indecies[i]>0):
                self.last_nonzero_frequency_indecies[i]-=1
        self.last_nonzero_frequency_indecies=np.unique(self.last_nonzero_frequency_indecies)
        self.f_max = self.frequency_bands[self.last_nonzero_frequency_indecies[-1]]
        
        data = np.zeros((len(self.interferometers), len(self.frequency_array)), dtype=complex)
        for i, ifo in enumerate(self.interferometers):
            data[i,:]=ifo.strain_data.frequency_domain_strain
        self.data = np.expand_dims(data,0)
        
        psd = np.zeros((len(self.interferometers), len(self.frequency_array)), dtype=complex)
        for i, ifo in enumerate(self.interferometers):
            psd[i,:]=ifo.power_spectral_density_array
        self.inverse_psd=np.expand_dims(1./psd, 0)
        self.inverse_psd[np.isnan(self.inverse_psd)]=0.
        
        self.noise=-np.sum(np.real(self.data*np.conj(self.data)*self.inverse_psd))*2/self.duration
        
    def _get_summary_data(self):
        """
        sets up banding information
        """
        self.A_0=np.zeros((self.N_overlaps, len(self.interferometers), self.N_bins), dtype=complex)
        self.A_1=np.zeros((self.N_overlaps, len(self.interferometers), self.N_bins), dtype=complex)
        self.B_0=np.zeros((self.N_overlaps, self.N_overlaps, len(self.interferometers), self.N_bins), dtype=complex)
        self.B_1=np.zeros((self.N_overlaps, self.N_overlaps, len(self.interferometers), self.N_bins), dtype=complex)
        #loop = tqdm(range(self.N_bins))
        #loop.set_description_str(f"Building summary statistics")
        for band in range(self.N_bins):
            self.A_0[:,:,band]=np.sum(4./self.duration*self.data[:,:,self.band_indecies[band]:self.band_indecies[band+1]] *
                                      np.conj(self.ref_strains[:,:,self.band_indecies[band]:self.band_indecies[band+1]]) *
                                      self.inverse_psd[:,:,self.band_indecies[band]:self.band_indecies[band+1]], axis=2)
            self.A_1[:,:,band]=np.sum(4./self.duration*self.data[:,:,self.band_indecies[band]:self.band_indecies[band+1]] *
                                      np.conj(self.ref_strains[:,:,self.band_indecies[band]:self.band_indecies[band+1]]) *
                                      self.inverse_psd[:,:,self.band_indecies[band]:self.band_indecies[band+1]] *
                                      (self.frequency_array[self.band_indecies[band]:self.band_indecies[band+1]]-self.f_m[band]), axis = 2)
            """
            for i in range(self.N_overlaps):
                for j in range(i+1):
                    self.B_0[i,j,:,band]=np.sum(4./self.duration*self.ref_strains[i,:,self.band_indecies[band]:self.band_indecies[band+1]] *
                                              np.conj(self.ref_strains[j,:,self.band_indecies[band]:self.band_indecies[band+1]]) *
                                              self.inverse_psd[0,:,self.band_indecies[band]:self.band_indecies[band+1]], axis=1)
                    self.B_1[i,j,:,band]=np.sum(4./self.duration*self.ref_strains[i,:,self.band_indecies[band]:self.band_indecies[band+1]] *
                                              np.conj(self.ref_strains[j,:,self.band_indecies[band]:self.band_indecies[band+1]]) *
                                              self.inverse_psd[0,:,self.band_indecies[band]:self.band_indecies[band+1]] *
                                              (self.frequency_array[self.band_indecies[band]:self.band_indecies[band+1]]-self.f_m[band]), axis = 1)
                    if i!=j:
                        self.B_0[j,i,:,band]=np.conj(self.B_0[i,j,:,band])
                        self.B_1[j,i,:,band]=np.conj(self.B_1[i,j,:,band])
            """
            for i in range(self.N_overlaps):
                for j in range(self.N_overlaps):
                    self.B_0[i,j,:,band]=np.sum(4./self.duration*self.ref_strains[i,:,self.band_indecies[band]:self.band_indecies[band+1]] *
                                              np.conj(self.ref_strains[j,:,self.band_indecies[band]:self.band_indecies[band+1]]) *
                                              self.inverse_psd[0,:,self.band_indecies[band]:self.band_indecies[band+1]], axis=1)
                    self.B_1[i,j,:,band]=np.sum(4./self.duration*self.ref_strains[i,:,self.band_indecies[band]:self.band_indecies[band+1]] *
                                              np.conj(self.ref_strains[j,:,self.band_indecies[band]:self.band_indecies[band+1]]) *
                                              self.inverse_psd[0,:,self.band_indecies[band]:self.band_indecies[band+1]] *
                                              (self.frequency_array[self.band_indecies[band]:self.band_indecies[band+1]]-self.f_m[band]), axis = 1)
        # no need for full data anymore
        del self.data
        del self.inverse_psd
        del self.ref_strains

    
    
    def log_likelihood_ratio(self):
        """
        calculates approximated log_likelihood
        """
        parameters = [{self.parameters_list[i][j]:self.parameters[self.full_parameters_list[i][j]]
                      for j in range(len(self.parameters_list[i]))} for i in range(self.N_overlaps)]
        signals = np.zeros((self.N_overlaps, len(self.interferometers), self.N_bins+1),
                            dtype=complex)
        for i in range(self.N_overlaps):
            signals[i,:,:]+=self.get_network_response(parameters[i])

        ratios = signals*self.inverse_binned_ref_strains
        r_0 = (ratios[:,:,1:]+ratios[:,:,:-1])/2
        r_1 = (ratios[:,:,1:]-ratios[:,:,:-1])/self.bandwiths
        
        d_inner_h = np.sum(np.real(self.A_0*np.conj(r_0)+self.A_1*np.conj(r_1)))
        h_inner_h = 0.
        
        """
        for i in range(self.N_overlaps):
            for j in range(i+1):
                partial_sum = np.sum(np.real(self.B_0[i,j,:,:]*r_0[i,:,:]*np.conj(r_0[j,:,:])+self.B_1[i,j,:,:] * 
                                     (r_0[i,:,:]*np.conj(r_1[j,:,:])+np.conj(r_0[j,:,:])*r_1[i,:,:])))
                if i==j:
                    h_inner_h += partial_sum
                else:
                    h_inner_h += 2*partial_sum
        """
        for i in range(self.N_overlaps):
            for j in range(self.N_overlaps):
                h_inner_h += np.sum(np.real(self.B_0[i,j,:,:]*r_0[i,:,:]*np.conj(r_0[j,:,:])+self.B_1[i,j,:,:] *
                                            (r_0[i,:,:]*np.conj(r_1[j,:,:])+np.conj(r_0[j,:,:])*r_1[i,:,:])))
        
        
        return d_inner_h-0.5*h_inner_h
    
    def get_sky_frame_parameters(self, params):
        time = params['{}_time'.format(self.time_reference)]
        if not self.reference_frame == "sky":
            ra, dec = zenith_azimuth_to_ra_dec(
                params['zenith'], params['azimuth'],
                time, self.reference_frame)
        else:
            ra = params["ra"]
            dec = params["dec"]
        if "geocent" not in self.time_reference:
            geocent_time = time - self.reference_ifo.time_delay_from_geocenter(
                ra=ra, dec=dec, time=time
            )
        else:
            geocent_time = params["geocent_time"]
        return dict(ra=ra, dec=dec, geocent_time=geocent_time)
    
    def noise_log_likelihood(self):
        return self.noise