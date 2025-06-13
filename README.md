# jointRB

This code is created for joint parameter estimation of multiple gravitational wave signals. It is meant to be used with [`bilby`](https://github.com/bilby-dev/bilby) package. It extends the defeault behaviour of bilby to handle simultaneous estimation of parameters of multiple signals in the data, provided they are of the same type and use the same waveform model. The code include two implementations of joint parameter estimation with relative binning, enabling PE to be performed on simulated ET and CE data. `jointRB.JointLikelihoodRB.OverlappingSignalsRelBinning` extends the standard RB implementation and as such should be used only on the waveforms without higer modes content and without precession. `jointRB.JointLikelihoodRBHoM.OverlappingSignalsRBHOM` includes those effects but is only compatible with `IMRPhenomXPHM` waveform model. We additionaly provide code for estimating the posteriors of overlapping signals with prior-informed Fisher matrices.

Relative binning is a method to perform bayesian parameter inference for gravitational waves in a faster way. It has been developped in [Zackay et al (2018)](https://arxiv.org/pdf/1806.08792.pdf), [Dai et al (2018)](https://arxiv.org/pdf/1806.08793.pdf), and [Dai et al, repo](https://bitbucket.org/dailiang8/gwbinning/src/master/). Additionally, attempts have been made to develop relative binning for higher-order modes waveforms (see [Leslie et al (2021)](https://arxiv.org/pdf/2109.09872.pdf)). For the higher-order mode version of our code, we adapt the [code](https://github.com/lemnis12/relativebilbying) implemetntation from [Narola et al (2023)](https://arxiv.org/pdf/2308.12140.pdf).

If you use this code for your work, please give due credits by citating the methodology papers:
```
@article{Zackay:2018qdy,
    author = "Zackay, Barak and Dai, Liang and Venumadhav, Tejaswi",
    title = "{Relative Binning and Fast Likelihood Evaluation for Gravitational Wave Parameter Estimation}",
    eprint = "1806.08792",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    month = "6",
    year = "2018"
}

@article{Dai:2018dca,
    author = "Dai, Liang and Venumadhav, Tejaswi and Zackay, Barak",
    title = "{Parameter Estimation for GW170817 using Relative Binning}",
    eprint = "1806.08793",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "6",
    year = "2018"
}
```

for higher order modes:
```
@article{Leslie:2021ssu,
    author = "Leslie, Nathaniel and Dai, Liang and Pratten, Geraint",
    title = "{Mode-by-mode relative binning: Fast likelihood estimation for gravitational waveforms with spin-orbit precession and multiple harmonics}",
    eprint = "2109.09872",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.1103/PhysRevD.104.123030",
    journal = "Phys. Rev. D",
    volume = "104",
    number = "12",
    pages = "123030",
    year = "2021"
}

@article{Narola:2023men,
    author = "Narola, Harsh and Janquart, Justin and Meijer, Quirijn and Haris, K. and Van Den Broeck, Chris",
    title = "{Relative binning for complete gravitational-wave parameter estimation with higher-order modes and precession, and applications to lensing and third-generation detectors}",
    eprint = "2308.12140",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "8",
    year = "2023"
}
```
For the prior-informed Fisher code, we use the following implementation of the truncated MVN sampler: https://github.com/brunzema/truncated-mvn-sampler


### Installation
- clone the repository
- run `pip install .`



### Structure of the repo
- `jointRB`: likelihoods required for joint and single parameter estimation with relative binning
- `examples`: folder with example runs for injection on how to use these scripts.
  - `analysis_single.py`: PE run on a single BBH signal without higher-order modes nor precession
  - `analysis_overlaps.py`: PE run on 2 overlapping BBH signals without higher-order modes nor precession
  - `analysis_overlaps_HM.py`: PE run on 2 overlapping BBH signals with higher-order modes and precession
  - `create_fisher_posterior.py`: posterior estimation with prior-informed Fisher matrix method
- `CE2`: data needed to create 2nd Cosmc Explorer interferometer in `bilby` 
