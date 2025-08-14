# Spectra-Tool
This program creates and modifies IR/Raman spectra from quantum chemical calculations.

Spectra can be created using Gaussian, Lorentzian or Pseudo-Voigt line broadening from a simple peak-intensity input file. Moreover, preexisting spectra can be read by the program and be further modified.

The generated/read spectra can be normalized (L1, L2, Inf-Max), linearily interpolated, inverted, or the maxima can be determined. Furthermore, an ALS or ARPLS baseline correction can be applied.

If two spectra are provided to the program, the correlation between both can be determined using the Pearson, Cauchy-Schwarz, Spearman or Euclidian Norm metric. 
