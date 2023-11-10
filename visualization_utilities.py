import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button
from matplotlib.collections import LineCollection
from matplotlib import cm, colors 

##############################################################
# Beamformer functionals
##############################################################

# functional whose maximum defines the unit gain orientation
#   params  : 
#       - covariance_matrix         :     n x n numpy array
#       - leadfield                 :     n x 3 or n x 2 numpy array
#   returns :
#       - functional                :     lambda describing the functional
def unit_gain_functional(covariance_matrix, lead_field_matrix):
  denominator_matrix = lead_field_matrix.T @ np.linalg.pinv(covariance_matrix) @ lead_field_matrix
  if lead_field_matrix.shape[1] == 3:
    ug_functional = lambda x, y, z : 1.0 / np.inner(np.array([x, y, z]), denominator_matrix @ np.array([x, y, z]))
  elif lead_field_matrix.shape[1] == 2:
    ug_functional = lambda x, y : 1.0 / np.inner(np.array([x, y]), denominator_matrix @ np.array([x, y]))
  else:
    raise ValueError('leadfield needs to consist of either 2 or 3 columns')
  return ug_functional

# functional whose maximum defines the array gain orientation
#   params  : 
#       - covariance_matrix         :     n x n numpy array
#       - leadfield                 :     n x 3 or n x 2 numpy array
#   returns :
#       - functional                :     lambda describing the functional
def array_gain_functional(covariance_matrix, lead_field_matrix):
  numerator_matrix = lead_field_matrix.T @ lead_field_matrix
  denominator_matrix = lead_field_matrix.T @ np.linalg.pinv(covariance_matrix) @ lead_field_matrix
  if lead_field_matrix.shape[1] == 3:
    ag_functional = lambda x, y, z : np.inner(np.array([x, y, z]), numerator_matrix @ np.array([x, y, z])) / np.inner(np.array([x, y, z]), denominator_matrix @ np.array([x, y, z]))
  elif lead_field_matrix.shape[1] == 2:
    ag_functional = lambda x, y : np.inner(np.array([x, y]), numerator_matrix @ np.array([x, y])) / np.inner(np.array([x, y]), denominator_matrix @ np.array([x, y]))
  else:
    raise ValueError('leadfield needs to consist of either 2 or 3 columns')
  return ag_functional

# functional whose maximum defines the unit noise gain orientation
#   params  : 
#       - covariance_matrix         :     n x n numpy array
#       - leadfield                 :     n x 3 or n x 2 numpy array
#   returns :
#       - functional                :     lambda describing the functional
def unit_noise_gain_functional(covariance_matrix, lead_field_matrix):
  numerator_matrix = lead_field_matrix.T @ np.linalg.pinv(covariance_matrix) @ lead_field_matrix
  denominator_matrix = lead_field_matrix.T @ np.linalg.pinv(covariance_matrix) @ np.linalg.pinv(covariance_matrix) @ lead_field_matrix
  if lead_field_matrix.shape[1] == 3:
    ung_functional = lambda x, y, z : np.inner(np.array([x, y, z]), numerator_matrix @ np.array([x, y, z])) / np.inner(np.array([x, y, z]), denominator_matrix @ np.array([x, y, z]))
  elif lead_field_matrix.shape[1] == 2:
    ung_functional = lambda x, y : np.inner(np.array([x, y]), numerator_matrix @ np.array([x, y])) / np.inner(np.array([x, y]), denominator_matrix @ np.array([x, y]))
  else:
    raise ValueError('leadfield needs to consist of either 2 or 3 columns')
  return ung_functional


##############################################################
# Sphere and circle plotting
##############################################################

# plot a sphere onto an axes object
#   params :
#     - axes              :     matplotlib axes object to plot onto
#     - center            :     3-entry array, center of the sphere to plot
#     - radius            :     scalar, radius of the sphere
#     - sphere_resolution :     scalar, step width in the parameter grid used for plotting
#     - surface           :     True performs a surface plot, False performs a wireframe plot
def plot_sphere(axes, center = np.array([0.0, 0.0, 0.0]), radius=1, sphere_resolution=20, surface=False):
  stride = 1
  theta_range = np.linspace(0, 1, sphere_resolution)
  phi_range = np.linspace(0, 1, sphere_resolution)
  theta_vals, phi_vals = np.meshgrid(theta_range, phi_range)
  x_coords_sphere = radius * np.sin(np.arccos(1 - 2 * theta_vals)) * np.cos(2 * np.pi * phi_vals) + center[0]
  y_coords_sphere = radius * np.sin(np.arccos(1 - 2 * theta_vals)) * np.sin(2 * np.pi * phi_vals) + center[1]
  z_coords_sphere = radius * np.cos(np.arccos(1 - 2 * theta_vals)) + center[2]
  
  if surface:
    sphere_plot = axes.plot_surface(x_coords_sphere, y_coords_sphere, z_coords_sphere, linewidth=0.0, cstride=stride, rstride=stride, color = 'b')
  else:
    sphere_plot = axes.plot_wireframe(x_coords_sphere, y_coords_sphere, z_coords_sphere, linewidth=0.5, cstride=stride, rstride=stride, color = 'c', alpha=0.5)
  
  return sphere_plot

# plot a circle onto an axes object
#   params :
#     - axes              :     matplotlib axes object to plot onto
#     - center            :     3-entry array, center of the circle to plot
#     - radius            :     scalar, radius of the circle
#     - circle_resolution :     scalar, number of sample points on the circle for plotting
#     - dashed            :     bool, if True plot the circle using a dashed line, else as a solid line
def plot_circle(axes, center = np.array([0.0, 0.0]), radius = 1, circle_resolution = 100, linestyle = 'dashed'):
  phi_range = np.linspace(0, 1, circle_resolution)
  circle_x = radius * np.cos(2 * np.pi * phi_range) + center[0]
  circle_y = radius * np.sin(2 * np.pi * phi_range) + center[1]
  
  circle_plot = axes.plot(circle_x, circle_y, linestyle = linestyle, color = 'k')
  return circle_plot

# plot a function on the unit sphere
#   params :
#     - function            :       function of the form f : R^3 -> R, taking arguments of the form f(x, y, z), where x, y, z are floating point numbers
#     - sphere_resolution   :       resolution used to sample the sphere surface
#     - alpha               :       scalar between 0 and 1, describing the transparancy of the surface
#     - backend             :       backend used for rendering. choose either 'matplotlib' or 'mayavi'
#     - axes                :       axes object to plot sphere onto, only used for 'matplotlib' backend
#     - use_colormap        :       color surfaces according to function
#     - plot_radial_extent  :       use absolute value of function for radial extent of surface
#     - colormap            :       colormap to be used if use_colormap is true
#     - normalize_colormap  :       transform function values to the interval [0, 1] before applying colormap
#     - color               :       surface color to be used if use_colormap is false
def plot_function_on_unit_sphere(function, sphere_resolution, alpha = 0.2, backend = 'matplotlib', axes = None, use_colormap = True, plot_radial_extent = False, colormap = 'inferno', normalize_colormap = True, color = 'c'):
  import numpy as np
  
  # get parametrization of sphere surface
  theta_range = np.linspace(0, 1, sphere_resolution)
  phi_range = np.linspace(0, 1, sphere_resolution)
  
  theta_vals, phi_vals = np.meshgrid(theta_range, phi_range)
  
  x_coords_sphere = np.sin(np.pi * theta_vals) * np.cos(2 * np.pi * phi_vals)
  y_coords_sphere = np.sin(np.pi * theta_vals) * np.sin(2 * np.pi * phi_vals)
  z_coords_sphere = np.cos(np.pi * theta_vals)
  
  # evalute function on surface points
  function_vectorized = np.vectorize(function)
  function_values = function_vectorized(x_coords_sphere, y_coords_sphere, z_coords_sphere)
  
  if plot_radial_extent:
    # scale values to that maximum value is 1
    absolute_values = np.absolute(function_values)
    abs_max = absolute_values.max()
    scaled_values = (1.0 / abs_max) * absolute_values
    
    x_coords_sphere = scaled_values * x_coords_sphere
    y_coords_sphere = scaled_values * y_coords_sphere
    z_coords_sphere = scaled_values * z_coords_sphere
  
  if backend == 'matplotlib':
    if use_colormap:
      if normalize_colormap:
        # transform values so that new_min = 0 and new_max = 1
        max_val = function_values.max()
        min_val = function_values.min()
        function_values_normalized = (function_values - min_val) / (max_val - min_val)
        color_distribution = plt.get_cmap(colormap)(function_values_normalized)
      else:
        # scale values so that they are contained in the interval [0, 1]
        # if the function attains negative values, this is the same as "normalized_colormap",
        # but if the function is nonnegative, the values are only scaled, and hence small
        # relative differences do not result in large colorchanges
        min_val = function_values.min()
        if min_val < 0:
          scaled_values = function_values - min_val
        else:
          scaled_values = function_values
        max_val = scaled_values.max()
        scaled_values = (1.0 / max_val) * scaled_values
        color_distribution = plt.get_cmap(colormap)(scaled_values)
      linewidths = 0.0
    else:
      rgba_color = colors.to_rgba(color)
      color_distribution = np.tile(rgba_color, (function_values.shape[0], function_values.shape[1], 1))
      linewidths = 0.25
    
    if axes is None:
      fig = plt.figure(figsize=plt.figaspect(1.0))
      axes = fig.add_subplot(111, projection='3d')
      surface_handle = axes.plot_surface(x_coords_sphere, y_coords_sphere, z_coords_sphere, rstride=1, cstride=1, facecolors=color_distribution, alpha = alpha, linewidths = linewidths, shade = False)
      axes.view_init(elev = 0, azim= 0)
      axes.set_axis_off()
      plt.show()
    else:
      surface_handle = axes.plot_surface(x_coords_sphere, y_coords_sphere, z_coords_sphere, rstride=1, cstride=1, facecolors=color_distribution, alpha = alpha, linewidths = linewidths, shade = False)
      axes.view_init(elev = 0, azim= 0)
      axes.set_axis_off()
  elif backend == 'mayavi':
    from mayavi import mlab
    mlab.figure()
    mlab.mesh(x_coords_sphere, y_coords_sphere, z_coords_sphere, scalars=function_vals_normalized, colormap='RdBu')
    mlab.view(azimuth=0, elevation = 0, distance=10, focalpoint = np.array([0, 0, 0]))
    mlab.show()
    surface_handle = None
  else:
    raise RuntimeError('unknown rendering backend')
  
  return surface_handle


# plot a function defined on the unit sphere
#   params:
#       - function          :       function of the form f : R^2 -> R taking arguments of the form f(x, y), where x and y are floating point numbers
#       - resolution        :       number of points used to sample the function
#       - axes              :       matplotlib axes to plot the function onto
#       - visualization     :       either 'radial_extent' or 'parametrization'
#       - use_colormap      :       bool, sets if line segments are colored
#       - colormap          :       colormap to use for coloring of line segments
#       - color             :       color to be used if no function based color mapping of line segments is performed
def plot_function_on_unit_circle(function, resolution = 50, axes = None, visualization = 'radial_extent', use_colormap = True, colormap = 'inferno', color = 'k', plot_unit_circle = True):
  
  # check if axes is given. Otherwise, create new axes
  if axes is None:
    fig, axes = plt.subplots()
  
  phi_range = np.linspace(0, 1, resolution)
  x_coords_circle = np.cos(2 * np.pi * phi_range)
  y_coords_circle = np.sin(2 * np.pi * phi_range)
  
  function_vectorized = np.vectorize(function)
  function_values = function_vectorized(x_coords_circle, y_coords_circle)
  
  if visualization == 'radial_extent':
    # scale radius by absolute value of function, scaled so that maximum value is 1
    abs_values = np.absolute(function_values)
    abs_max = abs_values.max()
    abs_values = (1.0 / abs_max) * abs_values
    
    x_coords_circle = abs_values * x_coords_circle
    y_coords_circle = abs_values * y_coords_circle
    
    # wrap points in segments 
    nr_segments = len(x_coords_circle) - 1
    segments = np.empty((nr_segments, 2, 2))
    
    # perform colormapping
    cmap = plt.get_cmap(colormap)
    color_array = np.empty((nr_segments, 4))
    if use_colormap:
      for i in range(nr_segments):
        color_array[i] = cmap((1.0 / abs_max) * function_values[i])
    else:
      rgba_color = colors.to_rgba(color)
      for i in range(nr_segments):
        color_array[i] = rgba_color
    
    for i in range(nr_segments):
      segments[i, 0, 0] = x_coords_circle[i]
      segments[i, 0, 1] = y_coords_circle[i]
      segments[i, 1, 0] = x_coords_circle[i+1]
      segments[i, 1, 1] = y_coords_circle[i+1]
    lines = LineCollection(segments, colors = color_array)
    
    curve_handle = axes.add_collection(lines)
    
    if plot_unit_circle:
      unit_circle_x = np.cos(2 * np.pi * phi_range)
      unit_circle_y = np.sin(2 * np.pi * phi_range)
      axes.plot(unit_circle_x, unit_circle_y, linestyle = (0, (5, 5)), color = 'k')
    
    axes.set_axis_off()
    
    limit = 1.5
    axes.set_xlim([-limit, limit])
    axes.set_ylim([-limit, limit])
  else:
    raise ValueError(f'unknown visualization method "{visualization}", please choose either "radial_extent" or "parametrization"')
  
  return curve_handle


