import numpy as np
import matplotlib.pyplot as plt
from visualization_utilities import *
from itertools import combinations

# use Coloumb's law to compute the forces on the individual point charges
def compute_force_interactions(points):
	
	number_of_points = len(points)
	forces = np.zeros(points.shape)
	
	for i, j in combinations(range(0, number_of_points), 2):
		from_j_on_i = points[i] - points[j]
		from_j_on_i = from_j_on_i / (np.linalg.norm(from_j_on_i) ** 3)
		
		forces[i] += from_j_on_i
		forces[j] -= from_j_on_i
	return forces

# simulate the movement of point charges constrained on a sphere
# params : 
#	- points 	: n x 3 array of points. Each row is interpreted as the cartesian coordiantes of a point
#	- nr_steps 	: We use Euler's method to simulate the movement of the point particles according to Newton's second law of motion
#			  This parameter sets the number of steps after which the simulation is terminated
#	- time_step	: Time step size for the simulation
def simulate_particle_movement(points, nr_steps=50, time_step=0.1):
  
  old_points = points
  velocities = np.zeros(points.shape)
  has_converged = False
  
  for i in range(0, nr_steps):
    print(f"Iteration {i}")
    # compute forces
    forces = compute_force_interactions(old_points)
    # update velocities
    velocities = velocities + time_step * forces
    # update positions
    new_points = old_points + time_step * velocities
    # scale points back to sphere
    new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
    old_points = new_points
  
  return new_points

# generate random points on the unit sphere. The points are generated according to a uniform distribution on the sphere
# params :
#	- nr_points         : number of points to generate
# - optimize          : optimize positions of random point to achieve a more uniform distribution
# - visualize         : show distribution of generated points on the unit sphere 
def random_points_on_unit_sphere_unif_distribution(nr_points, optimize = True, visualize = True):
  
  # initialize rng
  rng = np.random.default_rng()
  
  uniform_samples_theta = rng.uniform(0.0, 1.0, nr_points)
  uniform_samples_phi = rng.uniform(0.0, 1.0, nr_points)
  
  x_vals = [np.sin(np.arccos(1 - 2 * theta)) * np.cos(2 * np.pi * phi) for theta, phi in zip(uniform_samples_theta, uniform_samples_phi)]
  y_vals = [np.sin(np.arccos(1 - 2 * theta)) * np.sin(2 * np.pi * phi) for theta, phi in zip(uniform_samples_theta, uniform_samples_phi)]
  z_vals = [np.cos(np.arccos(1 - 2 * theta))                           for theta, phi in zip(uniform_samples_theta, uniform_samples_phi)]
  
  random_points = np.empty((nr_points, 3))
  random_points[:, 0] = x_vals
  random_points[:, 1] = y_vals
  random_points[:, 2] = z_vals
  
  if optimize:
    nr_iterations = 20
    print(f"Optimizing points (Iterations : {nr_iterations})")
    random_points = simulate_particle_movement(random_points, nr_iterations)
    print(f"Points optimized")
  
  if visualize:
    fig = plt.figure()
    fig.suptitle('Random orientations used in simulation')
    axes = fig.add_subplot(111, projection='3d')
    plot_sphere(axes, np.array([0.0, 0.0, 0.0]), 1, 50, False)
    axes.scatter(random_points[:, 0], random_points[:, 1], random_points[:, 2])
    plt.show()
    
  
  return random_points
