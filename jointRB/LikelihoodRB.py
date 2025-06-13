from bilby.gw import GravitationalWaveTransient
import numpy as np
from bilby.core.utils import (
    logger, speed_of_light, solar_mass, radius_of_earth,
    gravitational_constant
)
from tqdm import tqdm





class RelBinning(GravitationalWaveTransient):
    """
    Class encoding overlapping signals
    """

    def __init__(
        self, interferometers, waveform_generator, ref_injection, priors=None,
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
        self.parameters_ref = ref_injection
        self.duration = self.waveform_generator.duration
        self.start_times = np.zeros(len(self.interferometers))
        for i, ifo in enumerate(self.interferometers):
            self.start_times[i] = ifo.strain_data.start_time
        self._get_orig_frequency()
        self.delta = delta
        self._get_reference_strains()
        self._get_frequency_bands(delta)
        self._get_summary_data()
        
        strains = self.get_network_response(self.parameters_ref)
        self.inverse_binned_ref_strains = (1+0j)/strains
        self.inverse_binned_ref_strains[np.isnan(self.inverse_binned_ref_strains)]=0j

        
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
        i_end = self.last_nonzero_frequency_index
        for i in range(i_start, i_end+1):
            if np.abs(dphi[new_bands[-1]]-dphi[i])>=delta:
                new_bands.append(i)
        if new_bands[-1]!=i_end:
            new_bands.append(i_end)
            
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
            dt_geocent = parameters['geocent_time'] - self.start_times[i]
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
        strains = np.zeros((len(self.interferometers), len(self.frequency_array)),
                            dtype=complex)
        strains[:,self.frequency_mask]+=self.get_network_response(self.parameters_ref)
        self.ref_strains = strains
        
        self.last_nonzero_frequency_index=(len(self.band_indecies)-1)
        while((strains[:,self.band_indecies[self.last_nonzero_frequency_index]]==0j).all()
              and self.last_nonzero_frequency_index>0):
            self.last_nonzero_frequency_index-=1
        self.f_max = self.frequency_bands[self.last_nonzero_frequency_index]
        
        self.data = np.zeros((len(self.interferometers), len(self.frequency_array)), dtype=complex)
        for i, ifo in enumerate(self.interferometers):
            self.data[i,:]=ifo.strain_data.frequency_domain_strain
        
        psd = np.zeros((len(self.interferometers), len(self.frequency_array)), dtype=complex)
        for i, ifo in enumerate(self.interferometers):
            psd[i,:]=ifo.power_spectral_density_array
        self.inverse_psd=1./psd
        self.inverse_psd[np.isnan(self.inverse_psd)]=0.
        
        self.noise=-np.sum(np.real(self.data*np.conj(self.data)*self.inverse_psd))*2/self.duration
        
    def _get_summary_data(self):
        """
        sets up banding information
        """
        self.A_0=np.zeros((len(self.interferometers), self.N_bins), dtype=complex)
        self.A_1=np.zeros((len(self.interferometers), self.N_bins), dtype=complex)
        self.B_0=np.zeros((len(self.interferometers), self.N_bins), dtype=complex)
        self.B_1=np.zeros((len(self.interferometers), self.N_bins), dtype=complex)
        #loop = tqdm(range(self.N_bins))
        #loop.set_description_str(f"Building summary statistics")
        for band in range(self.N_bins):
            self.A_0[:,band]=np.sum(4./self.duration*self.data[:,self.band_indecies[band]:self.band_indecies[band+1]] *
                                      np.conj(self.ref_strains[:,self.band_indecies[band]:self.band_indecies[band+1]]) *
                                      self.inverse_psd[:,self.band_indecies[band]:self.band_indecies[band+1]], axis=1)
            self.A_1[:,band]=np.sum(4./self.duration*self.data[:,self.band_indecies[band]:self.band_indecies[band+1]] *
                                      np.conj(self.ref_strains[:,self.band_indecies[band]:self.band_indecies[band+1]]) *
                                      self.inverse_psd[:,self.band_indecies[band]:self.band_indecies[band+1]] *
                                      (self.frequency_array[self.band_indecies[band]:self.band_indecies[band+1]]-self.f_m[band]), axis = 1)

            self.B_0[:,band]=np.sum(4./self.duration*self.ref_strains[:,self.band_indecies[band]:self.band_indecies[band+1]] *
                                      np.conj(self.ref_strains[:,self.band_indecies[band]:self.band_indecies[band+1]]) *
                                      self.inverse_psd[:,self.band_indecies[band]:self.band_indecies[band+1]], axis=1)
            self.B_1[:,band]=np.sum(4./self.duration*self.ref_strains[:,self.band_indecies[band]:self.band_indecies[band+1]] *
                                      np.conj(self.ref_strains[:,self.band_indecies[band]:self.band_indecies[band+1]]) *
                                      self.inverse_psd[:,self.band_indecies[band]:self.band_indecies[band+1]] *
                                      (self.frequency_array[self.band_indecies[band]:self.band_indecies[band+1]]-self.f_m[band]), axis = 1)
        # no need for full data anymore
        del self.data
        del self.inverse_psd
        del self.ref_strains

    
    
    def log_likelihood_ratio(self):
        """
        calculates approximated log_likelihood
        """
        signals = self.get_network_response(self.parameters)

        ratios = signals*self.inverse_binned_ref_strains
        r_0 = (ratios[:,1:]+ratios[:,:-1])/2
        r_1 = (ratios[:,1:]-ratios[:,:-1])/self.bandwiths
        
        d_inner_h = np.sum(np.real(self.A_0*np.conj(r_0)+self.A_1*np.conj(r_1)))
        h_inner_h = np.sum(np.real(self.B_0[:,:]*r_0[:,:]*np.conj(r_0[:,:])+self.B_1[:,:] *
                                   (r_0[:,:]*np.conj(r_1[:,:])+np.conj(r_0[:,:])*r_1[:,:])))

        
        return d_inner_h-0.5*h_inner_h

    
    def noise_log_likelihood(self):
        return self.noise