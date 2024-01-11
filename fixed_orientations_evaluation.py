import numpy as np
from orientation_reconstruction_utilities import *
import statistics
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
from matplotlib.lines import Line2D

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
modality = 'EEG'
regularization = 0.05
bootstrapping_repetitions = 100
use_nonuniform_noise = False

azimuth_min = 0
azimuth_max = np.pi
azimuth_resolution = 21
# for elevation = np.pi/2 (resp. -np.pi/2) all points, independent of azimuth angle,
# are the same. To avoid sampling the same point multiple times, we start a little below np.pi/2
# (resp. a little above -np.pi/2)
elevation_eps = np.pi / 100
elevation_min = -np.pi / 2 + elevation_eps
elevation_max = np.pi / 2 - elevation_eps
elevation_resolution = 21

header_fontsize = 20
text_fontsize = 12
small_fontsize = 10
label_fontsize = 7
colormap = 'viridis'

marker_smallest_singular = 'o'
marker_middle_singular = '>'
marker_largest_singular = '^'
singular_vector_color = 'r'

images_computed = False

#################################
# code
#################################

# map a tuple of spherical coordinates (r, theta, phi) to the corresponding cartesian coordinates, 
# with an arbitrary direction for the polar axis and the meridian plane
#   params:
#       - onb     :     3 x 3 array, where the columns are supposed to form an orthonormal basis of R^3. The last column is interpreted as the polar axis
#       - r       :     scalar, radius
#       - theta   :     scalar, elevation angle
#       - phi     :     scalar, azimuth angle
#   returns:
#       - pos     :     3-entry array
def generalized_spherical_to_cartesian(onb, r, theta, phi):
  return onb @ np.array([r * np.cos(theta) * np.cos(phi),
                         r * np.cos(theta) * np.sin(phi),
                         r * np.sin(theta)])

# map a tuple of cartesian coordinates (x, y, z) to the corresponding generalized spherical coordinates,
# with an arbitrary direction for the polar axis and the meridian plane. 
# This function is inverse to the function generalized_spherical_to_cartesian defined above
def generalized_cartesian_to_spherical(onb, x, y, z):
  standard_spherical_coords = onb.T @ np.array([x, y, z])
  r = np.linalg.norm(standard_spherical_coords)
  theta = np.arcsin(standard_spherical_coords[2] / r)
  if (theta + np.pi/2 < 100 * np.finfo(np.float32).eps) or (np.pi/2 - theta < 100 * np.finfo(np.float32).eps):
    phi = 0
  else:
    phi = np.arctan2(standard_spherical_coords[1], standard_spherical_coords[0])
  return np.array([r, theta, phi])

if __name__ == '__main__':
  
  mpl.rcParams["font.family"] = 'Arial'
  mpl.rcParams['mathtext.fontset'] = 'custom'
  mpl.rcParams['mathtext.rm'] = 'Arial'
  mpl.rcParams['mathtext.it'] = 'Arial:italic'
  
  # compute svd-based coordinates system
  U, S, V_transposed = np.linalg.svd(meg_lf, full_matrices = False)
  onb = V_transposed.T
  
  nr_pixels_per_image = azimuth_resolution * elevation_resolution

  if not images_computed: 
    eeg_ug_images = []
    eeg_ag_images = []
    eeg_ung_images = []
    
    meg_ug_images = []
    meg_ag_images = []
    meg_ung_images = []
    
    emeg_ug_images = []
    emeg_ag_images = []
    emeg_ung_images = []
    
    print(f'Computing images ({nr_pixels_per_image} pixels per image)')
    for noise_level in noise_levels:
      print(f'Computing images for noise level {noise_level}')
      # scan sphere
      eeg_ug_image = np.empty((azimuth_resolution, elevation_resolution))
      eeg_ag_image = np.empty((azimuth_resolution, elevation_resolution))
      eeg_ung_image = np.empty((azimuth_resolution, elevation_resolution))
      
      meg_ug_image = np.empty((azimuth_resolution, elevation_resolution))
      meg_ag_image = np.empty((azimuth_resolution, elevation_resolution))
      meg_ung_image = np.empty((azimuth_resolution, elevation_resolution))
      
      emeg_ug_image = np.empty((azimuth_resolution, elevation_resolution))
      emeg_ag_image = np.empty((azimuth_resolution, elevation_resolution))
      emeg_ung_image = np.empty((azimuth_resolution, elevation_resolution))
      
      for i, phi in enumerate(np.linspace(azimuth_min, azimuth_max, azimuth_resolution)):
        for j, theta in enumerate(np.linspace(elevation_min, elevation_max, elevation_resolution)):
          print(f'Pixel {j + azimuth_resolution * i}')
          
          orientation = generalized_spherical_to_cartesian(onb, 1, theta, phi)
          
          emeg_reconstruction_results = bootstrapped_emeg_reconstruction_experiment(orientation, eeg_lf, meg_lf, noise_level, noise_level, fq, fs, nr_samples, regularization, bootstrapping_repetitions)
          
          eeg_ug_image[i, j] = emeg_reconstruction_results['median_eeg_ug_angle']
          eeg_ag_image[i, j] = emeg_reconstruction_results['median_eeg_ag_angle']
          eeg_ung_image[i, j] = emeg_reconstruction_results['median_eeg_ung_angle']
          
          meg_ug_image[i, j] = emeg_reconstruction_results['median_meg_ug_angle']
          meg_ag_image[i, j] = emeg_reconstruction_results['median_meg_ag_angle']
          meg_ung_image[i, j] = emeg_reconstruction_results['median_meg_ung_angle']
          
          emeg_ug_image[i, j] = emeg_reconstruction_results['median_emeg_ug_angle']
          emeg_ag_image[i, j] = emeg_reconstruction_results['median_emeg_ag_angle']
          emeg_ung_image[i, j] = emeg_reconstruction_results['median_emeg_ung_angle']
          
      eeg_ug_images.append(eeg_ug_image)
      eeg_ag_images.append(eeg_ag_image)
      eeg_ung_images.append(eeg_ung_image)
      
      meg_ug_images.append(meg_ug_image)
      meg_ag_images.append(meg_ag_image)
      meg_ung_images.append(meg_ung_image)
      
      emeg_ug_images.append(emeg_ug_image)
      emeg_ag_images.append(emeg_ag_image)
      emeg_ung_images.append(emeg_ung_image)
    
    print('Images computed')
    
    # save results
    image_dict = {}
    image_dict['noise_levels'] = noise_levels
    
    image_dict['eeg_ug_images'] = eeg_ug_images
    image_dict['eeg_ag_images'] = eeg_ag_images
    image_dict['eeg_ung_images'] = eeg_ung_images
    
    image_dict['meg_ug_images'] = meg_ug_images
    image_dict['meg_ag_images'] = meg_ag_images
    image_dict['meg_ung_images'] = meg_ung_images
    
    image_dict['emeg_ug_images'] = emeg_ug_images
    image_dict['emeg_ag_images'] = emeg_ag_images
    image_dict['emeg_ung_images'] = emeg_ung_images
    
    with open('output/image_dictionary.pickle', 'wb') as file_handle:
      pickle.dump(image_dict, file_handle)
  else:
    print('Loading precomputed images (potentially overwriting noise_levels)')
    with open('output/image_dictionary.pickle', 'rb') as file_handle:
      image_dict = pickle.load(file_handle)
    
    noise_levels = image_dict['noise_levels']
    
    eeg_ug_images = image_dict['eeg_ug_images']
    eeg_ag_images = image_dict['eeg_ag_images']
    eeg_ung_images = image_dict['eeg_ung_images']
    
    meg_ug_images = image_dict['meg_ug_images']
    meg_ag_images = image_dict['meg_ag_images']
    meg_ung_images = image_dict['meg_ung_images']
    
    emeg_ug_images = image_dict['emeg_ug_images']
    emeg_ag_images = image_dict['emeg_ag_images']
    emeg_ung_images = image_dict['emeg_ung_images']
  
  #############################
  # visualize results
  #############################
  nr_beamformers = 3
  top_level_figure = plt.figure(layout = 'constrained', figsize = (13.5, 5))
  top_level_figure.canvas.manager.set_window_title(f'Fixed orientation evaluation')
  top_level_figure.suptitle(f'Fixed orientation angle differences', fontsize = header_fontsize, weight = 'bold')
  subfigures = top_level_figure.subfigures(1, nr_beamformers + 1, wspace = 0.03, width_ratios=[1, 1, 1, 0.15])
  subfigures[0].suptitle(12 * ' ' + f'MEG (tang.)', fontsize = header_fontsize - 1, weight = 'bold')
  subfigures[1].suptitle(11 * ' ' + f'EEG', fontsize = header_fontsize - 1, weight = 'bold')
  subfigures[2].suptitle(11 * ' ' + f'EMEG', fontsize = header_fontsize - 1, weight = 'bold')
  
  azimuth_scale = (azimuth_max - azimuth_min) / (azimuth_resolution - 1)
  elevation_scale = (elevation_max - elevation_min) / (elevation_resolution - 1)
  def azimuth_to_pixel_coords(phi):
    return (phi - azimuth_min) / azimuth_scale
  def elevation_to_pixel_coords(theta):
    return (theta - elevation_min) / elevation_scale
  
  azimuth_ticks = [azimuth_to_pixel_coords(phi) for phi in [0, np.pi / 2, np.pi]]
  elevation_ticks = [elevation_to_pixel_coords(theta) for theta in [-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2]]
  
  azimuth_labels = ['$0°$', '$90°$', '$180°$']
  elevation_labels = ['',  '$-45°$', '$0°$', '$45°$', '']
  
  # get (generalized) spherical coordinates of singular vectors of EEG leadfield
  U_EEG, S_EEG, V_EEG_transposed = np.linalg.svd(eeg_lf)
  eeg_singular_vectors = V_EEG_transposed.T
  
  # flip vectors so that they are aligned with the middle tangential singular vector
  if np.inner(eeg_singular_vectors[:, 0], onb[:, 1]) < 0:
    eeg_singular_vectors[:, 0] *= -1
  if np.inner(eeg_singular_vectors[:, 1], onb[:, 1]) < 0:
    eeg_singular_vectors[:, 1] *= -1 
  if np.inner(eeg_singular_vectors[:, 2], onb[:, 1]) < 0:
    eeg_singular_vectors[:, 2] *= -1  
  
  smallest_singular_vector_eeg_spherical = generalized_cartesian_to_spherical(onb, eeg_singular_vectors[0, 2], eeg_singular_vectors[1, 2], eeg_singular_vectors[2, 2])
  middle_singular_vector_eeg_spherical = generalized_cartesian_to_spherical(onb, eeg_singular_vectors[0, 1], eeg_singular_vectors[1, 1], eeg_singular_vectors[2, 1])
  largest_singular_vector_eeg_spherical = generalized_cartesian_to_spherical(onb, eeg_singular_vectors[0, 0], eeg_singular_vectors[1, 0], eeg_singular_vectors[2, 0])
  
  smallest_singular_vector_meg_spherical = np.array([1.0, np.pi / 2, 0.0])
  middle_singular_vector_meg_spherical = np.array([1.0, 0.0, np.pi / 2])
  largest_singular_vector_meg_spherical = np.array([1.0, 0.0, 0.0])
  
  def plot_images_on_subfig(subfig, noise_levels, ug_images, ag_images, ung_images, top_left_tag = '', largest_singular = None, middle_singular = None, smallest_singular = None,):
    nr_noise_levels = len(noise_levels)
    axes_grid = subfig.subplots(nr_noise_levels, nr_beamformers)
    
    subfig.text(0.08, 0.92, top_left_tag, fontsize = header_fontsize, weight = 'bold')
    
    # plot elevation and azimuth labels
    axes_grid[nr_noise_levels - 1, 1].text(0.5, -0.4, 'Radial', horizontalalignment = 'center', verticalalignment = 'center', fontsize = text_fontsize, transform = axes_grid[nr_noise_levels - 1, 1].transAxes)
    axes_grid[nr_noise_levels // 2, 0].text(-0.75, 0.5, 'Tangential', horizontalalignment = 'center', verticalalignment = 'center', fontsize = text_fontsize, rotation = 90, transform = axes_grid[nr_noise_levels // 2, 0].transAxes)
    
    padding = 10
    axes_grid[0, 0].set_title('UG', fontsize = small_fontsize)
    axes_grid[0, 1].set_title('AG', fontsize = small_fontsize)
    axes_grid[0, 2].set_title('UNG', fontsize = small_fontsize)
    
    for i in range(nr_noise_levels):
      axes_grid[i, 0].imshow(ug_images[i], colormap, vmin = 0, vmax = 90, origin = 'lower')
      axes_grid[i, 1].imshow(ag_images[i], colormap, vmin = 0, vmax = 90, origin = 'lower')
      axes_grid[i, 2].imshow(ung_images[i], colormap, vmin = 0, vmax = 90, origin = 'lower')
      
      # plot singualar vectors
      if largest_singular is not None:
        for j in range(nr_beamformers):
          axes_grid[i, j].scatter(elevation_to_pixel_coords(largest_singular[1]), azimuth_to_pixel_coords(largest_singular[2]), color = singular_vector_color, marker = marker_largest_singular, facecolors = 'None')
      if middle_singular is not None:
        for j in range(nr_beamformers):
          axes_grid[i, j].scatter(elevation_to_pixel_coords(middle_singular[1]), azimuth_to_pixel_coords(middle_singular[2]), color = singular_vector_color, marker = marker_middle_singular, facecolors = 'None')
      if smallest_singular is not None:
        for j in range(nr_beamformers):
          axes_grid[i, j].scatter(elevation_to_pixel_coords(smallest_singular[1]), azimuth_to_pixel_coords(smallest_singular[2]), color = singular_vector_color, marker = marker_smallest_singular, facecolors = 'None')
      
      
      axes_grid[i, 0].set_ylabel(f'$\\sigma ={noise_levels[i]}$', fontsize = small_fontsize, rotation = 90)
      axes_grid[i, 1].set_ylabel('')
      axes_grid[i, 2].set_ylabel('')
      
      for j in range(nr_beamformers):
        axes_grid[i, j].tick_params(axis = 'both', labelsize = label_fontsize)
      
      axes_grid[i, 0].set_yticks(azimuth_ticks, azimuth_labels)
      axes_grid[i, 1].set_yticks([], [])
      axes_grid[i, 2].set_yticks([], [])
      
      if i != nr_noise_levels - 1:
        axes_grid[i, 0].set_xticks([], [])
        axes_grid[i, 1].set_xticks([], [])
        axes_grid[i, 2].set_xticks([], [])
      else:
        axes_grid[i, 0].set_xticks(elevation_ticks, elevation_labels)
        axes_grid[i, 1].set_xticks(elevation_ticks, elevation_labels)
        axes_grid[i, 2].set_xticks(elevation_ticks, elevation_labels)
      
    # legend
    if largest_singular is not None:
      if smallest_singular is not None:
        ncol = 3
        legend_handles = [Line2D([], [], color = 'white', marker = marker_largest_singular, markerfacecolor = 'white', markeredgecolor = singular_vector_color, label = '$v_{E1}$'),
                          Line2D([], [], color = 'white', marker = marker_middle_singular, markerfacecolor = 'white', markeredgecolor = singular_vector_color, label = '$v_{E2}$'),
                          Line2D([], [], color = 'white', marker = marker_smallest_singular, markerfacecolor = 'white', markeredgecolor = singular_vector_color, label = '$v_{E3}$')
                         ]
        legend_bbox = (0.51, 0.0, 0.4, 0.08)
      else:
        ncol = 2
        legend_handles = [Line2D([], [], color = 'white', marker = marker_largest_singular, markerfacecolor = 'white', markeredgecolor = singular_vector_color, label = '$v_{t1}$'),
                          Line2D([], [], color = 'white', marker = marker_middle_singular, markerfacecolor = 'white', markeredgecolor = singular_vector_color, label = '$v_{t2}$')
                         ]
        legend_bbox = (0.39, 0.0, 0.4, 0.08)
        
      subfig.legend(handles = legend_handles, ncol = ncol, bbox_to_anchor = legend_bbox)
      
    axes_grid[nr_noise_levels - 1, 1].text(0.5, -0.8, 'PLACEHOLDER', horizontalalignment = 'center', verticalalignment = 'center', fontsize = text_fontsize, transform = axes_grid[nr_noise_levels - 1, 1].transAxes, alpha = 0.0)
      
  plot_images_on_subfig(subfigures[0], noise_levels, meg_ug_images, meg_ag_images, meg_ung_images, 'a', largest_singular_vector_meg_spherical, middle_singular_vector_meg_spherical)
  plot_images_on_subfig(subfigures[1], noise_levels, eeg_ug_images, eeg_ag_images, eeg_ung_images, 'b', largest_singular_vector_eeg_spherical, middle_singular_vector_eeg_spherical, smallest_singular_vector_eeg_spherical)
  plot_images_on_subfig(subfigures[2], noise_levels, emeg_ug_images, emeg_ag_images, emeg_ung_images, 'c')
  
  
  # add colorbar
  cmap = plt.get_cmap(colormap)
  ax_colormap = subfigures[3].add_subplot(10, 1, (3, 8))
  normalizer = mpl.colors.Normalize(vmin = 0, vmax = 90)
  colorbar_handle = subfigures[3].colorbar(mpl.cm.ScalarMappable(norm = normalizer, cmap = cmap), orientation = 'vertical', cax=ax_colormap, ticks = [10 * i for i in range(10)])
  colorbar_handle.ax.set_yticklabels([f'{10 * i}°' for i in range(10)], fontsize = label_fontsize)
  colorbar_handle.set_label(label = 'Angle', fontsize = small_fontsize)
  
  
  plt.show()
  top_level_figure.savefig(f'output/fixed_orientation_evaluation.pdf', format = 'pdf', bbox_inches = 'tight', pad_inches = 0)

