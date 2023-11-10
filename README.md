# Evaluation of beamformer algorithms for estimating orientations of neural sources

This repository contains the scripts and data to reproduce the results presented in our forthcoming paper on the performance of beamformer algorithms in estimating orientations of neural sources based on EEG and MEG data (preprint available [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4523138)).

We will quickly go over the scripts.

1. random\_orientations\_evaluation.py: In this script, random orientations are generated, which are then estimated using different beamformer algorithms. The differences between the true orientations and the reconstructed orientations are then evaluated.
2. influence\_of\_noise\_levels\_evaluation.py: In this script, the MEG noise level is kept fixed, while the EEG noise level varies. Then, for each value of the EEG noise level, random orientations are generated and then estimated using different beamformer algorithms. For each EEG noise level, the mean error between the true and the reconstruction orientations is then evaluated.
3. fixed\_orientations\_evaluation.py: In this script, the set of possible orientations is scanned with a certain resolution. Each of the orientations is then estimated multiple times and the mean estimation error is evaluated.

For a detailed description of these evaluation methods and the reasoning for choosing them, we refer to the preprint cited above.

Furthermore, the following scripts in this repository are used in the above evaluations.

1. orientation\_reconstruction\_utilities.py: This file implements the actual beamformer algorithms used in the reconstruction.
2. random\_orientation\_utilities.py: This file implements functions that generate random orientations on the unit sphere
3. visualization\_utilities.py: This file implements various functions used for the visualization of the results.

Furthermore, the folder 'data' contains the data used in the simulations.

- V5\_eeg\_lf.npy and FEF\_eeg\_lf.npy : 53 x 3 matrices containing the EEG lead fields
- V5\_meg\_lf.npy and FEF\_meg\_lf.npy : 272 x 3 matrices containing the MEG lead fields

We again refer to the preprint above for details on how these lead fields were computed.

Finally, the folder 'output' is supposed to contain the results of the evaluation scripts.
