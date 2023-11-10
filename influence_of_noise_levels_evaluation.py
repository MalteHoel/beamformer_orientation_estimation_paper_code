import numpy as np
import matplotlib.pyplot as plt
from random_orientation_utilities import *
from orientation_reconstruction_utilities import *
from multiprocessing import Pool
import statistics

#################################
# input data
#################################

eeg_lf = np.load('data/V5_eeg_lf.npy')
meg_lf = np.load('data/V5_meg_lf.npy')

#################################
# parameters
#################################

fq = 20
fs = 600
nr_samples = 36000
eeg_noise_levels = np.concatenate((np.linspace(0.5, 2, 4), np.linspace(3, 10, 8)))
meg_noise_level = 4.0
nr_orientations = 1000
regularization = 0.05

optimize_orientations = False
visualize_orientations = False

ug_color = 'b'
ag_color = 'y'
ung_color = 'r'

eeg_linestyle = 'solid'
emeg_linestyle = 'dashed'

eeg_markerstyle = 'x'
emeg_markerstyle = 'o'

fontsize = 20

#################################
# code
#################################

def single_emeg_reconstruction_experiment_wrapper(count, true_orientation, eeg_lf, meg_lf, eeg_noise_level, meg_noise_level, fq, fs, nr_samples, regularization):
  print(f"Simulating orientation {count}")
  return emeg_reconstruction_experiment(true_orientation, eeg_lf, meg_lf, eeg_noise_level, meg_noise_level, fq, fs, nr_samples, regularization)


if __name__ == '__main__':
  
  # perform simulation
  eeg_ug_means = []
  eeg_ag_means = []
  eeg_ung_means = []
  emeg_ug_means = []
  emeg_ag_means = []
  emeg_ung_means = []
  print('Performing simulation')
  for eeg_noise_level in eeg_noise_levels:
    # draw new orientations for each noise level
    random_orientations = random_points_on_unit_sphere_unif_distribution(nr_orientations, optimize_orientations, visualize_orientations)
    
    # perform reconstruction experiment for all orientations
    print(f'Currently simulating EEG noise level {eeg_noise_level} and MEG noise level {meg_noise_level}')
    
    job_argument_list = []
    for i, orientation in enumerate(random_orientations):
      job_argument_list.append((i, orientation, eeg_lf, meg_lf, eeg_noise_level, meg_noise_level, fq, fs, nr_samples, regularization))
    
    with Pool() as pool:
      emeg_reconstruction_results = pool.starmap(single_emeg_reconstruction_experiment_wrapper, job_argument_list)
    print(f'Simulation for EEG noise level {eeg_noise_level} and MEG noise level {meg_noise_level} finished')
    
    # extract means
    eeg_ug_means.append(statistics.mean([emeg_reconstruction_results[i]['eeg_ug_angle'] for i in range(nr_orientations)]))
    eeg_ag_means.append(statistics.mean([emeg_reconstruction_results[i]['eeg_ag_angle'] for i in range(nr_orientations)]))
    eeg_ung_means.append(statistics.mean([emeg_reconstruction_results[i]['eeg_ung_angle'] for i in range(nr_orientations)]))
    
    emeg_ug_means.append(statistics.mean([emeg_reconstruction_results[i]['emeg_ug_angle'] for i in range(nr_orientations)]))
    emeg_ag_means.append(statistics.mean([emeg_reconstruction_results[i]['emeg_ag_angle'] for i in range(nr_orientations)]))
    emeg_ung_means.append(statistics.mean([emeg_reconstruction_results[i]['emeg_ung_angle'] for i in range(nr_orientations)]))
  print('Simulation finished')

  fig, axes = plt.subplots(figsize = (14, 9))
  fig.suptitle('Mean angle difference for different EEG noise levels and fixed MEG noise level', fontsize = fontsize, weight = 'bold')
  fig.canvas.manager.set_window_title('Influence of noise levels results')
  
  axes.plot(eeg_noise_levels, eeg_ug_means, color = ug_color, marker = eeg_markerstyle, linestyle = eeg_linestyle, label = 'EEG : UG')
  axes.plot(eeg_noise_levels, eeg_ag_means, color = ag_color, marker = eeg_markerstyle, linestyle = eeg_linestyle, label = 'EEG : AG')
  axes.plot(eeg_noise_levels, eeg_ung_means, color = ung_color, marker = eeg_markerstyle, linestyle = eeg_linestyle, label = 'EEG : UNG')
  
  axes.plot(eeg_noise_levels, emeg_ug_means, color = ug_color, marker = emeg_markerstyle, markerfacecolor = 'None', linestyle = emeg_linestyle, label = 'EMEG : UG')
  axes.plot(eeg_noise_levels, emeg_ag_means, color = ag_color, marker = emeg_markerstyle, markerfacecolor = 'None', linestyle = emeg_linestyle, label = 'EMEG : AG')
  axes.plot(eeg_noise_levels, emeg_ung_means, color = ung_color, marker = emeg_markerstyle, markerfacecolor = 'None', linestyle = emeg_linestyle, label = 'EMEG : UNG')
  
  axes.set_xlabel('Noise $\sigma_{E}$', fontsize = fontsize)
  axes.set_ylabel('mean angle error (degree)', fontsize = fontsize)
  
  axes.set_xlim([0, upper_lim + 1])
  axes.set_xticks(list(range(0, upper_lim + 1, 1)))
  
  axes.set_ylim([0, 90])
  axes.set_yticks(list(range(0, 91, 10)))
  
  axes.axvline(meg_noise_level, 0, 70, color = 'k', linestyle = 'dashed')
  
  axes.text(0.95 * meg_noise_level, 75, '$\\sigma_{E} = \\sigma_{M}$', rotation = 90, horizontalalignment = 'center', verticalalignment= 'center', fontsize = fontsize)
  
  axes.text(meg_noise_level / 2, 75, '$\\sigma_{E} < \\sigma_{M}$', horizontalalignment = 'center', verticalalignment = 'center', fontsize = fontsize)
  axes.text(3 * meg_noise_level / 2, 75, '$\\sigma_{E} > \\sigma_{M}$', horizontalalignment = 'center', verticalalignment = 'center', fontsize = fontsize)
  
  axes.legend(loc = 'lower right')
  axes.grid(True)
  
  plt.show()
  fig.savefig('output/influence_of_noise_level_evaluation_results.pdf', format = 'pdf', bbox_inches='tight', pad_inches = 0)

