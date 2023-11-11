import numpy as np
import pandas as pd
import seaborn as sns
from multiprocessing import Pool
from orientation_reconstruction_utilities import *
from random_orientation_utilities import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import time

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
noise_levels = [0.5, 2.0, 4.0]
nr_orientations = 1000
regularization = 0.05

visualize_results = True

optimize_orientations = False
visualize_orientations = False

true_orientation_color = 'b'
reconstructed_orientation_color = 'r'

smallest_singular_color = 'y'
middle_singular_color = 'b'
largest_singular_color = 'k'

fontsize = 20

#################################
# code
#################################

def single_emeg_reconstruction_experiment_wrapper(count, true_orientation, eeg_lf, meg_lf, eeg_noise_level, meg_noise_level, fq, fs, nr_samples, regularization):
  print(f"Simulating orientation {count}")
  return emeg_reconstruction_experiment(true_orientation, eeg_lf, meg_lf, eeg_noise_level, meg_noise_level, fq, fs, nr_samples, regularization)


if __name__ == '__main__':
  
  mpl.rcParams["font.family"] = 'Arial'
  mpl.rcParams['mathtext.fontset'] = 'custom'
  mpl.rcParams['mathtext.it'] = 'Arial:italic'
  mpl.rcParams['mathtext.rm'] = 'Arial'
  
  # perform simulation
  df_list = []
  eeg_ug_orientations = []
  meg_ug_orientations = []
  print('Performing simulation')
  time_start = time.time()
  for noise_level in noise_levels:
    # draw new orientations for each noise level
    random_orientations = random_points_on_unit_sphere_unif_distribution(nr_orientations, optimize_orientations, visualize_orientations)
  
    # perform reconstruction experiment for all orientations
    print(f'Currently simulating noise level {noise_level}')
    
    job_argument_list = []
    for i, orientation in enumerate(random_orientations):
      job_argument_list.append((i, orientation, eeg_lf, meg_lf, noise_level, noise_level, fq, fs, nr_samples, regularization))
    
    with Pool() as pool:
      emeg_reconstruction_results = pool.starmap(single_emeg_reconstruction_experiment_wrapper, job_argument_list)
    print(f'Noise level {noise_level} finished')
    
    # save_results
    eeg_ug_angle_diffs = [emeg_reconstruction_results[i]['eeg_ug_angle'] for i in range(nr_orientations)]
    eeg_ag_angle_diffs = [emeg_reconstruction_results[i]['eeg_ag_angle'] for i in range(nr_orientations)]
    eeg_ung_angle_diffs = [emeg_reconstruction_results[i]['eeg_ung_angle'] for i in range(nr_orientations)]
    
    meg_ug_angle_diffs = [emeg_reconstruction_results[i]['meg_ug_angle'] for i in range(nr_orientations)]
    meg_ag_angle_diffs = [emeg_reconstruction_results[i]['meg_ag_angle'] for i in range(nr_orientations)]
    meg_ung_angle_diffs = [emeg_reconstruction_results[i]['meg_ung_angle'] for i in range(nr_orientations)]
    
    emeg_ug_angle_diffs = [emeg_reconstruction_results[i]['emeg_ug_angle'] for i in range(nr_orientations)]
    emeg_ag_angle_diffs = [emeg_reconstruction_results[i]['emeg_ag_angle'] for i in range(nr_orientations)]
    emeg_ung_angle_diffs = [emeg_reconstruction_results[i]['emeg_ung_angle'] for i in range(nr_orientations)]
    
    eeg_ug_orientations.append(np.array([emeg_reconstruction_results[i]['eeg_ug_ori'] for i in range(nr_orientations)]))
    meg_ug_orientations.append(np.array([emeg_reconstruction_results[i]['meg_ug_ori'] for i in range(nr_orientations)]))
    
    # EEG
    df_eeg_ug = pd.DataFrame(data={'beamformer'  : 'UG',
                                   'angle_diff'  : eeg_ug_angle_diffs,
                                   'noise_level' : noise_level,
                                   'modality'    : 'EEG'})
    df_eeg_ag = pd.DataFrame(data={'beamformer'  : 'AG',
                                   'angle_diff'  : eeg_ag_angle_diffs,
                                   'noise_level' : noise_level,
                                   'modality'    : 'EEG'})
    df_eeg_ung = pd.DataFrame(data={'beamformer' : 'UNG',
                                    'angle_diff' : eeg_ung_angle_diffs,
                                    'noise_level': noise_level,
                                    'modality'   : 'EEG'})
    
    # MEG
    df_meg_ug = pd.DataFrame(data={'beamformer'  : 'UG',
                                   'angle_diff'  : meg_ug_angle_diffs,
                                   'noise_level' : noise_level,
                                   'modality'    : 'MEG'})
    df_meg_ag = pd.DataFrame(data={'beamformer'  : 'AG',
                                   'angle_diff'  : meg_ag_angle_diffs,
                                   'noise_level' : noise_level,
                                   'modality'    : 'MEG'})
    df_meg_ung = pd.DataFrame(data={'beamformer' : 'UNG',
                                    'angle_diff' : meg_ung_angle_diffs,
                                    'noise_level': noise_level,
                                    'modality'   : 'MEG'})
    
    # EMEG                           
    df_emeg_ug = pd.DataFrame(data={'beamformer'  : 'UG',
                                   'angle_diff'  : emeg_ug_angle_diffs,
                                   'noise_level' : noise_level,
                                   'modality'    : 'EMEG'})
    df_emeg_ag = pd.DataFrame(data={'beamformer'  : 'AG',
                                   'angle_diff'  : emeg_ag_angle_diffs,
                                   'noise_level' : noise_level,
                                   'modality'    : 'EMEG'})
    df_emeg_ung = pd.DataFrame(data={'beamformer' : 'UNG',
                                    'angle_diff' : emeg_ung_angle_diffs,
                                    'noise_level': noise_level,
                                    'modality'   : 'EMEG'})
      

    combined_df = pd.concat([df_eeg_ug, df_eeg_ag, df_eeg_ung, df_meg_ug, df_meg_ag, df_meg_ung, df_emeg_ug, df_emeg_ag, df_emeg_ung])
    df_list.append(combined_df)

  print(f'Simulation finished')
  time_finish = time.time()
  print(f"Time simulations : {time_finish - time_start:.2f} sec")
  total_df = pd.concat(df_list)
  total_df.to_csv('output/random_orientation_evaluation_results.csv', index=False)

  if visualize_results:
    ################################
    # boxplot of angle differences
    ################################
    # plot results on a 3 x n grid (3 : nr of methodologies, n : nr of noise levels)
    nr_noise_levels = len(noise_levels)
    fig, axes_grid = plt.subplots(3, nr_noise_levels, layout = 'constrained', figsize = (15, 9))
    fig.canvas.manager.set_window_title('Random orientation results')
    fig.suptitle(f'Angle reconstruction errors (deg)', fontsize = fontsize, weight = 'bold')
    
    for i, modality in enumerate(['EEG', 'MEG', 'EMEG']):
      for j, noise_level in enumerate(noise_levels):
        axes_grid[i, j].set_yticks([0, 22.5, 45, 67.5, 90], ['0', '', '45', '', '90'])
        axes_grid[i, j].set_axisbelow(True)
        axes_grid[i, j].tick_params(direction = 'in')
        axes_grid[i, j].grid(True, 'major', 'y')
        
        sns.boxplot(data = total_df[(total_df['modality'] == modality) & (total_df['noise_level'] == noise_level)], x = 'beamformer', y = 'angle_diff', ax = axes_grid[i, j])
        
        axes_grid[i, j].set_ylim([0, 90])
        
        if i == 0:
          axes_grid[i, j].set_title(f"$\\sigma = {noise_level}$", fontsize = fontsize)
        if j == 0:
          axes_grid[i, j].set_ylabel(f"{modality}", rotation=90, fontsize = fontsize)
        else:
          axes_grid[i, j].set_ylabel("")
    
    ################################
    # reconstruction distribution on sphere/circle for UG beamformer
    ################################
    
    fig_reconstructions = plt.figure(figsize = (15, 9), layout = 'constrained')
    fig_reconstructions.suptitle('UG beamformer orientation bias visualization', fontsize = fontsize, weight = 'bold')
    fig_reconstructions.canvas.manager.set_window_title('UG bias visualization')
    
    nr_rows = 2
    nr_cols = nr_noise_levels
    
    # EEG
    U_EEG, S_EEG, V_EEG_transposed = np.linalg.svd(eeg_lf, full_matrices = False)
    
    axes_eeg_ug_noisy_reconstructions = []
    for i in range(nr_noise_levels):
      axes_eeg_reconstructions = fig_reconstructions.add_subplot(nr_rows, nr_cols, i + 1, projection = '3d')
      plot_sphere(axes_eeg_reconstructions, np.array([0.0, 0.0, 0.0]), 1, 50, False)
      axes_eeg_reconstructions.scatter(eeg_ug_orientations[i][:, 0], eeg_ug_orientations[i][:, 1], eeg_ug_orientations[i][:, 2], color = reconstructed_orientation_color)
      axes_eeg_reconstructions.set_title(f'$\\sigma = {noise_levels[i]}$', fontsize = fontsize)
      axes_eeg_ug_noisy_reconstructions.append(axes_eeg_reconstructions)
    
      # add singular vectors
      smallest_singular_vector_handle = axes_eeg_reconstructions.quiver([0], [0], [0], V_EEG_transposed[2, 0], V_EEG_transposed[2, 1], V_EEG_transposed[2, 2], color = smallest_singular_color)
      middle_singular_vector_handle = axes_eeg_reconstructions.quiver([0], [0], [0], V_EEG_transposed[1, 0], V_EEG_transposed[1, 1], V_EEG_transposed[1, 2], color = middle_singular_color)
      largest_singular_vector_handle = axes_eeg_reconstructions.quiver([0], [0], [0], V_EEG_transposed[0, 0], V_EEG_transposed[0, 1], V_EEG_transposed[0, 2], color = largest_singular_color)
    
    # MEG
    # first compute tangential projections of true orientations
    U, S, V_transposed = np.linalg.svd(meg_lf, full_matrices = False)
    random_orientations_tangential = np.array([V_transposed[:2, :] @ orientation for orientation in random_orientations])
    random_orientations_tangential = random_orientations_tangential / np.linalg.norm(random_orientations_tangential, axis=1, keepdims=True)
    
    linestyle_unit_circle = (0, (5, 5))
    limit = 1.1
    
    axes_meg_ug_noisy_reconstructions = []
    for i in range(nr_noise_levels):
      axes_meg_reconstructions = fig_reconstructions.add_subplot(nr_rows, nr_cols, nr_cols + i + 1)
      plot_circle(axes_meg_reconstructions, np.array([0.0, 0.0]), 1, 100, linestyle_unit_circle)
      axes_meg_reconstructions.scatter(meg_ug_orientations[i][:, 0], meg_ug_orientations[i][:, 1], color = reconstructed_orientation_color)
      axes_meg_reconstructions.set_title(f'$\\sigma = {noise_levels[i]}$', fontsize = fontsize)
      axes_meg_reconstructions.set_xlim([-limit, limit])
      axes_meg_reconstructions.set_ylim([-limit, limit])
      axes_meg_reconstructions.set_aspect('equal')
      axes_meg_ug_noisy_reconstructions.append(axes_meg_reconstructions)
      
      # add singular vectors
      axes_meg_reconstructions.arrow(0, 0, 0, 1, color = smallest_singular_color)
      axes_meg_reconstructions.arrow(0, 0, 1, 0, color = largest_singular_color)
      
      # add space above and below figure
      axes_meg_reconstructions.text(0.5, 1.3, 'PLACEHOLER', horizontalalignment = 'center', verticalalignment = 'center', fontsize = fontsize, transform=axes_meg_reconstructions.transAxes, alpha = 0.0)
      axes_meg_reconstructions.text(0.5, -0.3, 'PLACEHOLER', horizontalalignment = 'center', verticalalignment = 'center', fontsize = fontsize, transform=axes_meg_reconstructions.transAxes, alpha = 0.0)
      
    # add legend
    legend_handles = [Line2D([], [], color = 'white', marker = 'o', markerfacecolor = reconstructed_orientation_color, label = 'Reconstructed orientations'),
                      Line2D([0], [0], color = smallest_singular_color, label = 'Smallest singular vector'),
                      Line2D([0], [0], color = middle_singular_color, label = 'Middle singular vector'),
                      Line2D([0], [0], color = largest_singular_color, label = 'Largest singular vector')]
    
    fig_reconstructions.legend(handles = legend_handles, loc = 'lower center', title = 'Legend', ncol = 5)
    
    # show and save plot
    plt.show()
    fig.savefig('output/random_orientation_evaluation_results.pdf', format = 'pdf', bbox_inches = 'tight', pad_inches = 0)
    fig_reconstructions.savefig('output/ug_orientation_bias.pdf', format = 'pdf', bbox_inches = 'tight', pad_inches = 0)

