import numpy as np
from copy import deepcopy

EPSILON = 2.0**(-53 / 3) # for num derivation, step should be cube root of preciision 
#EPSILON = 1e-08

def weighted_quantiles(samples, weights, qs):
    idxs = samples.argsort()
    sorted_weights = weights[idxs]
    cdf = np.cumsum(sorted_weights)
    cdf /= cdf[-1]
    return samples[idxs[[np.searchsorted(cdf, qs)]]]
    
class MultivariateGaussiaLikelihood(object):

    def __init__(self, mean, precision, param_list, parameters=None):
        """Bilby-compatible multivariate likelihood from precision matrix

        Parameters
        ==========
        mean: array_like
            Array with the mean values of distribution
        precision: array_like
            The ndim*ndim covariance matrix
        param_list: array_like
            List of parameters used in Fisher matrix
        parameters: dict
            A dictionary of the parameter names and associated values
        """
        self.parameters = parameters
        self.mean = mean
        self.precision = precision
        self.param_list = param_list
        self._meta_data = None
        self._marginalized_parameters = []

    def __repr__(self):
        return self.__class__.__name__ + '(parameters={})'.format(self.parameters)

    def log_likelihood(self):
        """

        Returns
        =======
        float
        """
        param_vector = np.array([self.parameters[parameter] for parameter in self.param_list])
        residual = param_vector - self.mean
        log_l = -0.5 * residual.dot(self.precision.dot(residual))
        return log_l

    def noise_log_likelihood(self):
        """

        Returns
        =======
        float
        """
        return 0.0

    def log_likelihood_ratio(self):
        """Difference between log likelihood and noise log likelihood

        Returns
        =======
        float
        """
        return self.log_likelihood()

    @property
    def meta_data(self):
        return getattr(self, '_meta_data', None)

    @meta_data.setter
    def meta_data(self, meta_data):
        if isinstance(meta_data, dict):
            self._meta_data = meta_data
        else:
            raise ValueError("The meta_data must be an instance of dict")

    @property
    def marginalized_parameters(self):
        return self._marginalized_parameters
    

def sample_likelihood(mu, precision, priors, param_list, N=1000, include_prior_informaton=False):
    outdir = 'fisher_bias_temp_sampling_dir'
    print('Matrix inversion failed. Estimating covariance by sampling.')
    import bilby
    import shutil
    starting_point = {key:mu[i] for i, key in enumerate(param_list)}
    likelihood = MultivariateGaussiaLikelihood(mu, precision, param_list, parameters=starting_point.copy())
    if include_prior_informaton:
        prior = priors
    else:
        prior = bilby.core.prior.PriorDict()
        #
        for param in param_list:
            # no prior information means that we effectively should not have prior boundaries
            if param in ['ra','dec', 'mass_ratio', 'symmetric_mass_ratio', 'chi_1', 'chi_2', 'theta_jn',
                         'psi', 'phase', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl']:
                prior[param] = bilby.core.prior.Uniform(
                    minimum = starting_point[param]-10, maximum = starting_point[param]+10)
            else:
                prior[param] = bilby.core.prior.Uniform(minimum = priors[param].minimum, maximum = priors[param].maximum)
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=prior,
        sampler="bilby_mcmc",
        nsamples=N,
        ntemps=1,
        nensemble=1,
        L1steps=1,
        print_dt=60,
        outdir=outdir,
        check_point_plot=False,
        resume=False,
        verbose=True,
        initial_sample_dict=starting_point,
    )
    shutil.rmtree(outdir)
    return np.cov(result.posterior[param_list].T)


class FisherGenerator:
    def __init__(self, ifos, waveform_generator, priors, noise_vector, parameter_list, signal_A, prior_limit_schema=1):
        self.prior_limit_schema = prior_limit_schema
        self.ifos = ifos
        self.N_dets = len(self.ifos)
        self.N_f = len(self.ifos[0].frequency_array)
        self.waveform_generator = waveform_generator
        self.noise_vector = noise_vector
        self.parameter_list = parameter_list
        self.N_params = len(self.parameter_list)
        self.signal_A = signal_A
        self.priors = deepcopy(priors)
        self.ipsd_array = np.zeros([len(self.ifos), len(self.ifos[0].frequency_array)])
        for i, ifo in enumerate(self.ifos):
            self.ipsd_array[i] = 1./ifo.power_spectral_density_array
        self.ipsd_array *= (4./ifos[0].duration)

        self.lb, self.ub = self.extract_boundaries()

        self.matrix_A, self.strain_derivatives_A, self.strain_A = self.gen_fisher_matrix(self.signal_A)
        self.injection_vector_A = np.array([self.signal_A[parameter] for parameter in self.parameter_list])
        self.injection_vector_A = self.limit_vector_to_hypercube(np.zeros(self.N_params), self.injection_vector_A)
        try:
            self.covariance_A = np.linalg.inv(self.matrix_A)
            self.check_if_singular(self.matrix_A, self.covariance_A)
        except ValueError as err:
            self.covariance_A = sample_likelihood(self.injection_vector_A, self.matrix_A, self.priors, self.parameter_list, N=1000)

    def check_if_singular(self, a, a_inverse):
        max_error = np.max(np.abs((a @ a_inverse) - np.eye(self.N_params)))
        if max_error > 0.1:
            raise ValueError('Singular Fisher matrix')
    def inner_product(self, aa, bb):
        return np.sum(np.real(np.conj(aa)*bb)*self.ipsd_array)

    def get_strains(self, injection):
        strain = {}
        polarizations = self.waveform_generator.frequency_domain_strain(injection)
        strain = np.array([ifo.get_detector_response(polarizations, injection) for ifo in self.ifos])
        return strain

    def derivative_vec_single(self, parameter_vector, whichparam="chirp_mass"):

        parvec_p = parameter_vector.copy()
        parvec_m = parameter_vector.copy()
        
        parvec_p[whichparam]+=EPSILON
        parvec_m[whichparam]-=EPSILON
        strain_p = self.get_strains(parvec_p)
        strain_m = self.get_strains(parvec_m)
        deriv_signal = (strain_p-strain_m)/(2*EPSILON)
        return deriv_signal


    def gen_fisher_matrix(self, params):
        p_vec = np.array([params[key] for key in self.parameter_list])
        p_vec = self.limit_vector_to_hypercube(np.zeros(self.N_params), p_vec)
        parameters = {key:p_vec[i] for i, key in enumerate(self.parameter_list)}
        strain = self.get_strains(parameters)
        matrix = np.zeros((self.N_params, self.N_params))
        strain_derivatives = np.zeros([self.N_params, self.N_dets, self.N_f], dtype=complex)
        for i in range(self.N_params):
            if self.parameter_list[i] == 'luminosity_distance': # can quickly calculate derivative
                strain_derivatives[i] = (-1 / parameters['luminosity_distance'])*strain
            else:
                strain_derivatives[i] = self.derivative_vec_single(
                    parameters, self.parameter_list[i])
        
        for i in range(self.N_params):
            for j in range(i+1):
                matrix[i,j] = self.inner_product(
                    strain_derivatives[i], strain_derivatives[j])
                matrix[j,i] = matrix[i,j]
        return matrix, strain_derivatives, strain

    def get_noise_deviation(self, covariance_A=None, derivs_A=None):
        if covariance_A is None and derivs_A is None:
            covariance_A = self.covariance_A
            derivs_A = self.strain_derivatives_A
        bias_vector = np.zeros(self.N_params)
        for i in range(self.N_params):
            bias_vector[i] = self.inner_product(derivs_A[i], self.noise_vector)
        return np.matmul(covariance_A, bias_vector)

    def get_bias(self, params_B):
        strain_B = self.get_strains(params_B)
        bias_vector = np.zeros(self.N_params)
        for i in range(self.N_params):
            bias_vector[i] = self.inner_product(self.strain_derivatives_A[i], strain_B)
        return np.matmul(self.covariance_A, bias_vector)

    def limit_vector_to_hypercube(self, origin_vec, delta_v):
        dv = delta_v.copy()
        if self.prior_limit_schema == 1:  # rescaling the vector
            for i in range(len(dv)):
                v = origin_vec+dv
                adjustment = v[i] - self.ub[i]
                if adjustment > 0.:
                    dv*=(1-adjustment/dv[i])
                adjustment = v[i] - self.lb[i]
                if adjustment < 0.:
                    dv*=(1-adjustment/dv[i])
        elif self.prior_limit_schema == 2:
            for i in range(len(dv)):
                v = origin_vec+dv
                adjustment = v[i] - self.ub[i]
                if adjustment > 0.:
                    dv[i] -= adjustment
                adjustment = v[i] - self.lb[i]
                if adjustment < 0.:
                    dv[i] -= adjustment
        # we dont want to have it exactely at the boundary (causes problems in computing fisher)
        for i in range(len(dv)):
            if dv[i] >= self.ub[i]-origin_vec[i]-EPSILON:
                dv[i] = self.ub[i]-origin_vec[i]-EPSILON
            if dv[i] <= self.lb[i]-origin_vec[i]+EPSILON:
                dv[i] = self.lb[i]-origin_vec[i]+EPSILON
        return dv
        #return delta_v

    def extract_boundaries(self):
        ub = np.ones(self.N_params) * np.inf
        lb = -1* ub

        for i, param in enumerate(self.parameter_list):
            if self.priors[param].boundary != 'periodic':
                ub[i] = self.priors[param].maximum
                lb[i] = self.priors[param].minimum
            else:
                #ub[i] = 6*np.pi
                #lb[i] = -6*np.pi
                self.priors[param].maximum = 6*np.pi
                self.priors[param].minimum = -6*np.pi
            if param in ['chi_1', 'chi_2']:
                ub[i] = self.priors[param].a_prior.maximum * self.priors[param].z_prior.maximum
                lb[i] = self.priors[param].a_prior.maximum * self.priors[param].z_prior.minimum
        ub[self.parameter_list.index('mass_ratio')] = np.inf
        return lb, ub
    
    def approximate_likelihood(self, params_B=None, fisher_at='noise', shift_by_noise=True):
        mean = self.injection_vector_A.copy()
        if params_B is not None:
            bias = self.get_bias(params_B)
            bias = self.limit_vector_to_hypercube(mean, bias)
            mean += bias
            if fisher_at in ['noise', 'bias']:
                params_shifted = {param:mean[i] for i, param in enumerate(self.parameter_list)}
                matrix, derivs, _ = self.gen_fisher_matrix(params_shifted)
                try:
                    covariance = np.linalg.inv(matrix)
                    self.check_if_singular(matrix, covariance)
                except ValueError as err:
                    covariance = sample_likelihood(mean, matrix, self.priors, self.parameter_list, N=1000)
            else:
                covariance = self.covariance_A
                derivs = self.strain_derivatives_A
        else:
            covariance = self.covariance_A
            derivs = self.strain_derivatives_A
        if shift_by_noise:
            noise_vector = self.get_noise_deviation(covariance, derivs)
            noise_vector = self.limit_vector_to_hypercube(mean, noise_vector)
            mean += noise_vector
            if fisher_at == 'noise':
                params_shifted = {param:mean[i] for i, param in enumerate(self.parameter_list)}
                matrix, _, _ = self.gen_fisher_matrix(params_shifted)
                covariance = np.linalg.inv(matrix)
                self.check_if_singular(matrix, covariance)
        covariance = (covariance+covariance.T)/2 # due to numerical solution, cov will not be exactely symmetric
        return mean, covariance
    
    def draw_samples_from_fisher(self, mean, covariance, target_sample_number):
        r_idx = self.parameter_list.index('mass_ratio')
        tmvn = TruncatedMVN(mean, covariance, self.lb, self.ub)
        max_iter = 5
        n_eff = 0
        iter = 0
        samples = np.zeros([0, self.N_params], dtype = float)
        weights = np.zeros([0], dtype = float)
        efficiency = 1
        while (n_eff < target_sample_number and iter < max_iter):
            samples_to_draw = int(1.1 / efficiency * (target_sample_number-n_eff) + 10)
            new_samples = tmvn.sample(samples_to_draw).T
            mask = new_samples[:, r_idx] > 1.
            new_samples[mask, r_idx] = 1 / new_samples[mask, r_idx] 
            new_weights = np.array([self.priors[param].ln_prob(new_samples[:,i]) for i, param in enumerate(self.parameter_list)])
            new_weights = np.sum(new_weights, axis=0)
            samples = np.concatenate((samples, new_samples), axis=0)
            weights = np.concatenate((weights, new_weights))
            w = np.exp(weights - np.nanmax(weights))
            n_eff = np.sum(w)**2/np.sum(w**2)
            efficiency = n_eff / len(weights)
            iter += 1
        return samples, w
    
    def get_quantiles(self, parameter, samples, weights, quantiles):
        idx = self.parameter_list.index(parameter)
        param_samples = samples[:,idx]
        return weighted_quantiles(param_samples, weights, quantiles)
    
    def generate_shifted_quantiles(self, params_B=None, quantiles=None, parameter=None,
                                   n_samples=5000, fisher_at='noise', shift_by_noise=True):
        mean, covariance = self.approximate_likelihood(params_B, fisher_at, shift_by_noise)
        samples, weights = self.draw_samples_from_fisher(mean, covariance, n_samples)
        if quantiles is None:
            qs = np.array([0.05, 0.5, 0.95])
        else:
            qs = quantiles
        if parameter is None:
            return {param: self.get_quantiles(param, samples, weights, qs) for param in self.parameter_list}
        else:
            return self.get_quantiles(parameter, samples, weights, qs)


import math
from scipy import special
from scipy import optimize

EPS = 10e-15


class TruncatedMVN:
    """
    Implementation copied from https://github.com/brunzema/truncated-mvn-sampler
    Create a normal distribution :math:`X  \sim N ({\mu}, {\Sigma})` subject to linear inequality constraints
    :math:`lb < X < ub` and sample from it using minimax tilting. Based on the MATLAB implemention by the authors
    (reference below).

    :param np.ndarray mu: (size D) mean of the normal distribution :math:`\mathbf {\mu}`.
    :param np.ndarray cov: (size D x D) covariance of the normal distribution :math:`\mathbf {\Sigma}`.
    :param np.ndarray lb: (size D) lower bound constrain of the multivariate normal distribution :math:`\mathbf lb`.
    :param np.ndarray ub: (size D) upper bound constrain of the multivariate normal distribution :math:`\mathbf ub`.
    :param Union[int, None] seed: a random seed.

    Note that the algorithm may not work if 'cov' is close to being rank deficient.

    Reference:
    Botev, Z. I., (2016), The normal law under linear restrictions: simulation and estimation via minimax tilting,
    Journal of the Royal Statistical Society Series B, 79, issue 1, p. 125-148,

    Example:
        >>> d = 10  # dimensions
        >>>
        >>> # random mu and cov
        >>> mu = np.random.rand(d)
        >>> cov = 0.5 - np.random.rand(d ** 2).reshape((d, d))
        >>> cov = np.triu(cov)
        >>> cov += cov.T - np.diag(cov.diagonal())
        >>> cov = np.dot(cov, cov)
        >>>
        >>> # constraints
        >>> lb = np.zeros_like(mu) - 2
        >>> ub = np.ones_like(mu) * np.inf
        >>>
        >>> # create truncated normal and sample from it
        >>> n_samples = 100000
        >>> samples = TruncatedMVN(mu, cov, lb, ub).sample(n_samples)

    Reimplementation by Paul Brunzema
    """

    def __init__(self, mu, cov, lb, ub, seed=None):
        self.dim = len(mu)
        if not cov.shape[0] == cov.shape[1]:
            raise RuntimeError("Covariance matrix must be of shape DxD!")
        if not (self.dim == cov.shape[0] and self.dim == len(lb) and self.dim == len(ub)):
            raise RuntimeError("Dimensions D of mean (mu), covariance matric (cov), lower bound (lb) "
                               "and upper bound (ub) must be the same!")

        self.cov = cov
        self.orig_mu = mu
        self.orig_lb = lb
        self.orig_ub = ub
        
        # permutated
        self.lb = lb - mu  # move distr./bounds to have zero mean
        self.ub = ub - mu  # move distr./bounds to have zero mean
        if np.any(self.ub <= self.lb):
            raise RuntimeError("Upper bound (ub) must be strictly greater than lower bound (lb) for all D dimensions!")

        # scaled Cholesky with zero diagonal, permutated
        self.L = np.empty_like(cov)
        self.unscaled_L = np.empty_like(cov)

        # placeholder for optimization
        self.perm = None
        self.x = None
        self.mu = None
        self.psistar = None

        # for numerics
        self.eps = EPS

        # a random state
        self.random_state = np.random.RandomState(seed)

    def sample(self, n):
        """
        Create n samples from the truncated normal distribution.

        :param int n: Number of samples to create.
        :return: D x n array with the samples.
        :rtype: np.ndarray
        """
        if not isinstance(n, int):
            raise RuntimeError("Number of samples must be an integer!")

        # factors (Cholesky, etc.) only need to be computed once!
        if self.psistar is None:
            self.compute_factors()

        # start acceptance rejection sampling
        rv = np.array([], dtype=np.float64).reshape(self.dim, 0)
        accept, iteration = 0, 0
        while accept < n:
            logpr, Z = self.mvnrnd(n, self.mu)  # simulate n proposals
            idx = -np.log(self.random_state.rand(n)) > (self.psistar - logpr)  # acceptance tests
            rv = np.concatenate((rv, Z[:, idx]), axis=1)  # accumulate accepted
            accept = rv.shape[1]  # keep track of # of accepted
            iteration += 1
            if iteration == 10 ** 3:
                print('Warning: Acceptance prob. smaller than 0.001.')
            elif iteration > 10 ** 4:
                accept = n
                rv = np.concatenate((rv, Z), axis=1)
                print('Warning: Sample is only approximately distributed.')

        # finish sampling and postprocess the samples!
        order = self.perm.argsort(axis=0)
        rv = rv[:, :n]
        rv = self.unscaled_L @ rv
        rv = rv[order, :]

        # retransfer to original mean
        rv += np.tile(self.orig_mu.reshape(self.dim, 1), (1, rv.shape[-1]))  # Z = X + mu
        return rv
    
    def compute_factors(self):
        # compute permutated Cholesky factor and solve optimization

        # Cholesky decomposition of matrix with permuation
        self.unscaled_L, self.perm = self.colperm()
        D = np.diag(self.unscaled_L)
        if np.any(D < self.eps):
            print('Warning: Method might fail as covariance matrix is singular!')

        # rescale
        scaled_L = self.unscaled_L / np.tile(D.reshape(self.dim, 1), (1, self.dim))
        self.lb = self.lb / D
        self.ub = self.ub / D

        # remove diagonal
        self.L = scaled_L - np.eye(self.dim)

        # get gradient/Jacobian function
        gradpsi = self.get_gradient_function()
        x0 = np.zeros(2 * (self.dim - 1))

        # find optimal tilting parameter non-linear equation solver
        sol = optimize.root(gradpsi, x0, args=(self.L, self.lb, self.ub), method='hybr', jac=True)
        if not sol.success:
            print('Warning: Method may fail as covariance matrix is close to singular!')
        self.x = sol.x[:self.dim - 1]
        self.mu = sol.x[self.dim - 1:]

        # compute psi star
        self.psistar = self.psy(self.x, self.mu)
        
    def reset(self):
        # reset factors -> when sampling, optimization for optimal tilting parameters is performed again

        # permutated
        self.lb = self.orig_lb - self.orig_mu  # move distr./bounds to have zero mean
        self.ub = self.orig_ub - self.orig_mu

        # scaled Cholesky with zero diagonal, permutated
        self.L = np.empty_like(self.cov)
        self.unscaled_L = np.empty_like(self.cov)

        # placeholder for optimization
        self.perm = None
        self.x = None
        self.mu = None
        self.psistar = None

    def mvnrnd(self, n, mu):
        # generates the proposals from the exponentially tilted sequential importance sampling pdf
        # output:     logpr, log-likelihood of sample
        #             Z, random sample
        mu = np.append(mu, [0.])
        Z = np.zeros((self.dim, n))
        logpr = 0
        for k in range(self.dim):
            # compute matrix multiplication L @ Z
            col = self.L[k, :k] @ Z[:k, :]
            # compute limits of truncation
            tl = self.lb[k] - mu[k] - col
            tu = self.ub[k] - mu[k] - col
            # simulate N(mu,1) conditional on [tl,tu]
            Z[k, :] = mu[k] + self.trandn(tl, tu)
            # update likelihood ratio
            logpr += lnNormalProb(tl, tu) + .5 * mu[k] ** 2 - mu[k] * Z[k, :]
        return logpr, Z

    def trandn(self, lb, ub):
        """
        Sample generator for the truncated standard multivariate normal distribution :math:`X \sim N(0,I)` s.t.
        :math:`lb<X<ub`.

        If you wish to simulate a random variable 'Z' from the non-standard Gaussian :math:`N(m,s^2)`
        conditional on :math:`lb<Z<ub`, then first simulate x=TruncatedMVNSampler.trandn((l-m)/s,(u-m)/s) and set
        Z=m+s*x.
        Infinite values for 'ub' and 'lb' are accepted.

        :param np.ndarray lb: (size D) lower bound constrain of the normal distribution :math:`\mathbf lb`.
        :param np.ndarray ub: (size D) upper bound constrain of the normal distribution :math:`\mathbf lb`.

        :return: D samples if the truncated normal distribition x ~ N(0, I) subject to lb < x < ub.
        :rtype: np.ndarray
        """
        if not len(lb) == len(ub):
            raise RuntimeError("Lower bound (lb) and upper bound (ub) must be of the same length!")

        x = np.empty_like(lb)
        a = 0.66  # threshold used in MATLAB implementation
        # three cases to consider
        # case 1: a<lb<ub
        I = lb > a
        if np.any(I):
            tl = lb[I]
            tu = ub[I]
            x[I] = self.ntail(tl, tu)
        # case 2: lb<ub<-a
        J = ub < -a
        if np.any(J):
            tl = -ub[J]
            tu = -lb[J]
            x[J] = - self.ntail(tl, tu)
        # case 3: otherwise use inverse transform or accept-reject
        I = ~(I | J)
        if np.any(I):
            tl = lb[I]
            tu = ub[I]
            x[I] = self.tn(tl, tu)
        return x

    def tn(self, lb, ub, tol=2):
        # samples a column vector of length=len(lb)=len(ub) from the standard multivariate normal distribution
        # truncated over the region [lb,ub], where -a<lb<ub<a for some 'a' and lb and ub are column vectors
        # uses acceptance rejection and inverse-transform method

        sw = tol  # controls switch between methods, threshold can be tuned for maximum speed for each platform
        x = np.empty_like(lb)
        # case 1: abs(ub-lb)>tol, uses accept-reject from randn
        I = abs(ub - lb) > sw
        if np.any(I):
            tl = lb[I]
            tu = ub[I]
            x[I] = self.trnd(tl, tu)

        # case 2: abs(u-l)<tol, uses inverse-transform
        I = ~I
        if np.any(I):
            tl = lb[I]
            tu = ub[I]
            pl = special.erfc(tl / np.sqrt(2)) / 2
            pu = special.erfc(tu / np.sqrt(2)) / 2
            x[I] = np.sqrt(2) * special.erfcinv(2 * (pl - (pl - pu) * self.random_state.rand(len(tl))))
        return x

    def trnd(self, lb, ub):
        # uses acceptance rejection to simulate from truncated normal
        x = self.random_state.randn(len(lb))  # sample normal
        test = (x < lb) | (x > ub)
        I = np.where(test)[0]
        d = len(I)
        while d > 0:  # while there are rejections
            ly = lb[I]
            uy = ub[I]
            y = self.random_state.randn(len(uy))  # resample
            idx = (y > ly) & (y < uy)  # accepted
            x[I[idx]] = y[idx]
            I = I[~idx]
            d = len(I)
        return x

    def ntail(self, lb, ub):
        # samples a column vector of length=len(lb)=len(ub) from the standard multivariate normal distribution
        # truncated over the region [lb,ub], where lb>0 and lb and ub are column vectors
        # uses acceptance-rejection from Rayleigh distr. similar to Marsaglia (1964)
        if not len(lb) == len(ub):
            raise RuntimeError("Lower bound (lb) and upper bound (ub) must be of the same length!")
        c = (lb ** 2) / 2
        n = len(lb)
        f = np.expm1(c - ub ** 2 / 2)
        x = c - np.log(1 + self.random_state.rand(n) * f)  # sample using Rayleigh
        # keep list of rejected
        I = np.where(self.random_state.rand(n) ** 2 * x > c)[0]
        d = len(I)
        while d > 0:  # while there are rejections
            cy = c[I]
            y = cy - np.log(1 + self.random_state.rand(d) * f[I])
            idx = (self.random_state.rand(d) ** 2 * y) < cy  # accepted
            x[I[idx]] = y[idx]  # store the accepted
            I = I[~idx]  # remove accepted from the list
            d = len(I)
        return np.sqrt(2 * x)  # this Rayleigh transform can be delayed till the end

    def psy(self, x, mu):
        # implements psi(x,mu); assumes scaled 'L' without diagonal
        x = np.append(x, [0.])
        mu = np.append(mu, [0.])
        c = self.L @ x
        lt = self.lb - mu - c
        ut = self.ub - mu - c
        p = np.sum(lnNormalProb(lt, ut) + 0.5 * mu ** 2 - x * mu)
        return p

    def get_gradient_function(self):
        # wrapper to avoid dependancy on self

        def gradpsi(y, L, l, u):
            # implements gradient of psi(x) to find optimal exponential twisting, returns also the Jacobian
            # NOTE: assumes scaled 'L' with zero diagonal
            d = len(u)
            c = np.zeros(d)
            mu, x = c.copy(), c.copy()
            x[0:d - 1] = y[0:d - 1]
            mu[0:d - 1] = y[d - 1:]

            # compute now ~l and ~u
            c[1:d] = L[1:d, :] @ x
            lt = l - mu - c
            ut = u - mu - c

            # compute gradients avoiding catastrophic cancellation
            w = lnNormalProb(lt, ut)
            pl = np.exp(-0.5 * lt ** 2 - w) / np.sqrt(2 * math.pi)
            pu = np.exp(-0.5 * ut ** 2 - w) / np.sqrt(2 * math.pi)
            P = pl - pu

            # output the gradient
            dfdx = - mu[0:d - 1] + (P.T @ L[:, 0:d - 1]).T
            dfdm = mu - x + P
            grad = np.concatenate((dfdx, dfdm[:-1]), axis=0)

            # construct jacobian
            lt[np.isinf(lt)] = 0
            ut[np.isinf(ut)] = 0

            dP = - P ** 2 + lt * pl - ut * pu
            DL = np.tile(dP.reshape(d, 1), (1, d)) * L
            mx = DL - np.eye(d)
            xx = L.T @ DL
            mx = mx[:-1, :-1]
            xx = xx[:-1, :-1]
            J = np.block([[xx, mx.T],
                          [mx, np.diag(1 + dP[:-1])]])
            return (grad, J)

        return gradpsi

    def colperm(self):
        perm = np.arange(self.dim)
        L = np.zeros_like(self.cov)
        z = np.zeros_like(self.orig_mu)

        for j in perm.copy():
            pr = np.ones_like(z) * np.inf  # compute marginal prob.
            I = np.arange(j, self.dim)  # search remaining dimensions
            D = np.diag(self.cov)
            s = D[I] - np.sum(L[I, 0:j] ** 2, axis=1)
            s[s < 0] = self.eps
            s = np.sqrt(s)
            tl = (self.lb[I] - L[I, 0:j] @ z[0:j]) / s
            tu = (self.ub[I] - L[I, 0:j] @ z[0:j]) / s
            pr[I] = lnNormalProb(tl, tu)
            # find smallest marginal dimension
            k = np.argmin(pr)

            # flip dimensions k-->j
            jk = [j, k]
            kj = [k, j]
            self.cov[jk, :] = self.cov[kj, :]  # update rows of cov
            self.cov[:, jk] = self.cov[:, kj]  # update cols of cov
            L[jk, :] = L[kj, :]  # update only rows of L
            self.lb[jk] = self.lb[kj]  # update integration limits
            self.ub[jk] = self.ub[kj]  # update integration limits
            perm[jk] = perm[kj]  # keep track of permutation

            # construct L sequentially via Cholesky computation
            s = self.cov[j, j] - np.sum(L[j, 0:j] ** 2, axis=0)
            if s < -0.01:
                raise RuntimeError("Sigma is not positive semi-definite")
            elif s < 0:
                s = self.eps
            L[j, j] = np.sqrt(s)
            new_L = self.cov[j + 1:self.dim, j] - L[j + 1:self.dim, 0:j] @ L[j, 0:j].T
            L[j + 1:self.dim, j] = new_L / L[j, j]

            # find mean value, z(j), of truncated normal
            tl = (self.lb[j] - L[j, 0:j - 1] @ z[0:j - 1]) / L[j, j]
            tu = (self.ub[j] - L[j, 0:j - 1] @ z[0:j - 1]) / L[j, j]
            w = lnNormalProb(tl, tu)  # aids in computing expected value of trunc. normal
            z[j] = (np.exp(-.5 * tl ** 2 - w) - np.exp(-.5 * tu ** 2 - w)) / np.sqrt(2 * math.pi)
        return L, perm


def lnNormalProb(a, b):
    # computes ln(P(a<Z<b)) where Z~N(0,1) very accurately for any 'a', 'b'
    p = np.zeros_like(a)
    # case b>a>0
    I = a > 0
    if np.any(I):
        pa = lnPhi(a[I])
        pb = lnPhi(b[I])
        p[I] = pa + np.log1p(-np.exp(pb - pa))
    # case a<b<0
    idx = b < 0
    if np.any(idx):
        pa = lnPhi(-a[idx])  # log of lower tail
        pb = lnPhi(-b[idx])
        p[idx] = pb + np.log1p(-np.exp(pa - pb))
    # case a < 0 < b
    I = (~I) & (~idx)
    if np.any(I):
        pa = special.erfc(-a[I] / np.sqrt(2)) / 2  # lower tail
        pb = special.erfc(b[I] / np.sqrt(2)) / 2  # upper tail
        p[I] = np.log1p(-pa - pb)
    return p


def lnPhi(x):
    # computes logarithm of  tail of Z~N(0,1) mitigating numerical roundoff errors
    out = -0.5 * x ** 2 - np.log(2) + np.log(special.erfcx(x / np.sqrt(2)) + EPS)  # divide by zeros error -> add eps
    return out
