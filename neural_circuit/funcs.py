import brainpy.math as bm
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Apply Gaussian smoothing to heatmaps
def gauss_filter(heatmaps):
  """
  Apply Gaussian smoothing to the heatmaps.

  Parameters:
  heatmaps (numpy.ndarray): 3D array of heatmaps with shape (M, K, N).

  Returns:
  numpy.ndarray: Smoothed heatmaps.
  """
  sigma = 1.0  # Standard deviation for Gaussian smoothing
  N = heatmaps.shape[-1]
  for k in range(N):
    map_k = heatmaps[:, :, k]
    filtered_map = gaussian_filter(map_k, sigma=sigma)
    filtered_map = np.where(map_k == 0, 0, filtered_map)
    heatmaps[:, :, k] = filtered_map
  return heatmaps

# Keep the top N elements in a matrix and set others to zero
def keep_top_n(mat, n):
  """
  Keep the top N elements in the matrix and set others to zero.

  Parameters:
  mat (numpy.ndarray): Input matrix.
  n (int): Number of top elements to keep.

  Returns:
  numpy.ndarray: Matrix with only top N elements retained.
  """
  n = int(n)
  mat = mat.flatten()
  sorted_indices = bm.argsort(mat)[::-1]  # Sort indices in descending order
  top_n_indices = sorted_indices[:n]  # Get indices of top N elements
  result = bm.zeros_like(mat)  # Create a zero matrix
  result[top_n_indices] = mat[top_n_indices]
  return result

# Normalize the rows of a matrix
def normalize_rows(matrix):
  """
  Normalize the rows of a matrix.

  Parameters:
  matrix (numpy.ndarray): Input matrix.

  Returns:
  numpy.ndarray: Row-normalized matrix.
  """
  column_norms = np.linalg.norm(matrix, axis=0)
  standardized_matrix = matrix / column_norms[np.newaxis, :]
  standardized_matrix[:, np.where(column_norms == 0)[0]] = 0  # Handle zero norms
  return standardized_matrix

# Map angles to the range [-pi, pi]
def period_bound(A):
  """
  Map angles to the range [-pi, pi].

  Parameters:
  A (numpy.ndarray): Input array of angles.

  Returns:
  numpy.ndarray: Array with angles mapped to [-pi, pi].
  """
  B = bm.where(A > bm.pi, A - 2 * bm.pi, A)
  C = bm.where(B < -bm.pi, B + 2 * bm.pi, B)
  return C

# Generate a comparison plot for field scores
def draw_field_score(score_1, score_2, figname):
  """
  Generate a comparison plot for field scores.

  Parameters:
  score_1 (numpy.ndarray): First set of scores.
  score_2 (numpy.ndarray): Second set of scores.
  figname (str): Filename to save the plot.
  """
  plt.figure()
  plt.plot(score_1, '.')
  plt.plot(score_2, '.')
  plt.legend(['Before', 'After'])
  plt.xlabel('Cell index')
  plt.ylabel('Max firing rate')
  plt.tight_layout()
  plt.savefig(figname)

# Draw the distribution of place fields
def draw_field_distribution(fr, place_index, loc, figname):
  """
  Draw the distribution of place fields.

  Parameters:
  fr (numpy.ndarray): Firing rate matrix.
  place_index (numpy.ndarray): Indices of place cells.
  loc (numpy.ndarray): Locations of cells.
  figname (str): Filename to save the plot.
  """
  x1, x2, y1, y2 = -2.05, 2.05, -2.05, 2.05
  loc_dis = -1
  loc_land = bm.array([[loc_dis, loc_dis], [-loc_dis, loc_dis], [loc_dis, -loc_dis], [-loc_dis, -loc_dis]])
  x_bound = np.array([x1, x2, x2, x1, x1])
  y_bound = np.array([y1, y1, y2, y2, y1])
  plt.figure()
  Center_exc = place_center(fr, place_index, loc)
  plt.scatter(Center_exc[:, 0], Center_exc[:, 1])
  plt.plot(x_bound, y_bound)
  plt.plot(loc_land[:, 0], loc_land[:, 1], 'r.', markersize=10)
  plt.tight_layout()
  plt.savefig(figname)

# Compute softmax with optional scaling factor
def softmax(x, beta=1.):
  """
  Compute the softmax of an array.

  Parameters:
  x (numpy.ndarray): Input array.
  beta (float): Scaling factor.

  Returns:
  numpy.ndarray: Softmax-transformed array.
  """
  exp_x = bm.exp((x - bm.max(x)) * beta) - bm.exp(-bm.max(x) * beta)
  return exp_x / bm.sum(exp_x)

# Normalize rows of a matrix
def normalize_rows(matrix):
  """
  Normalize the rows of a matrix.

  Parameters:
  matrix (numpy.ndarray): Input matrix.

  Returns:
  numpy.ndarray: Row-normalized matrix.
  """
  column_sums = bm.sqrt(bm.sum(matrix**2, axis=1))
  normalized_matrix = matrix / column_sums[:, np.newaxis]
  return normalized_matrix

# Keep the top N elements in a matrix and return their indices
def keep_top_n(mat, n):
  """
  Keep the top N elements in the matrix and return their indices.

  Parameters:
  mat (numpy.ndarray): Input matrix.
  n (int): Number of top elements to keep.

  Returns:
  tuple: Matrix with only top N elements retained and their indices.
  """
  n = int(n)
  mat = mat.flatten()
  sorted_indices = bm.argsort(mat)[::-1]
  top_n_indices = sorted_indices[:n]
  result = bm.zeros_like(mat)
  result[top_n_indices] = mat[top_n_indices]
  return result, top_n_indices

# Select place cells based on their scores
def place_cell_select_score(HPC_fr, thres, thres_num=1):
  """
  Select place cells based on their scores.

  Parameters:
  HPC_fr (numpy.ndarray): Firing rate matrix.
  thres (float or list): Threshold(s) for selection.
  thres_num (int): Number of thresholds.

  Returns:
  tuple: Place field scores, indices of selected cells, and their count.
  """
  HPC_fr = bm.as_numpy(HPC_fr)
  res = HPC_fr / np.mean(HPC_fr, axis=0)
  place_field_score = np.sum(np.multiply(HPC_fr / np.sum(HPC_fr, axis=0), 
                       (np.log2(res, out=np.zeros_like(res), where=(res != 0)))), axis=0)
  place_index = []
  place_num = np.zeros(thres_num)
  if thres_num > 1:
    for i in range(thres_num):
      if i == 0:
        place_index.append(np.argwhere(place_field_score > thres[i]))
      else:
        place_index.append(np.argwhere((place_field_score > thres[i]) & (place_field_score < thres[i - 1])))
      place_num[i] = place_index[i].shape[0]
  else:
    place_index = np.argwhere(place_field_score > thres)
    place_num = place_index.shape[0]
  return place_field_score, place_index, place_num

# Select place cells based on firing rate
def place_cell_select_fr(HPC_fr, pos, thres=0.002):
  """
  Select place cells based on firing rate.

  Parameters:
  HPC_fr (numpy.ndarray): Firing rate matrix.
  pos (numpy.ndarray): Position matrix.
  thres (float): Threshold for selection.

  Returns:
  tuple: Place scores, indices of selected cells, and their count.
  """
  place_score, sigma = compute_place_score(HPC_fr, pos)
  place_index = np.argwhere(place_score > thres)
  place_num = place_index.shape[0]
  return place_score, place_index, place_num

# Compute circular distance between two angles
def circular_dist(a, b):
  """
  Compute the circular distance between two angles.

  Parameters:
  a (float or numpy.ndarray): First angle(s).
  b (float or numpy.ndarray): Second angle(s).

  Returns:
  numpy.ndarray: Circular distance(s).
  """
  dis = a - b
  dis = np.where(dis > np.pi, dis - 2 * np.pi, dis)
  dis = np.where(dis < -np.pi, dis + 2 * np.pi, dis)
  return dis

# Map angles to the range [-pi, pi]
def map2pi(a):
  """
  Map angles to the range [-pi, pi].

  Parameters:
  a (numpy.ndarray): Input array of angles.

  Returns:
  numpy.ndarray: Array with angles mapped to [-pi, pi].
  """
  b = bm.where(a > np.pi, a - np.pi * 2, a)
  c = bm.where(b < -np.pi, b + np.pi * 2, b)
  return c

# Compute the angle between two vectors
def angle_between_vectors(v1, v2):
  """
  Compute the angle between two vectors.

  Parameters:
  v1 (numpy.ndarray): First vector.
  v2 (numpy.ndarray): Second vector.

  Returns:
  float: Angle between the vectors in radians.
  """
  v1_norm = bm.linalg.norm(v1)
  v2_norm = bm.linalg.norm(v2)
  dot_product = bm.dot(v1, v2)
  norm_mul = v1_norm * v2_norm
  cos_theta = bm.where(norm_mul == 0., 0, dot_product / norm_mul)
  angle = bm.arccos(bm.clip(cos_theta, -1.0, 1.0))
  angle_cross = bm.cross(v1, v2)
  angle = bm.where(angle_cross > 0, -angle, angle)
  return angle


def conn_center(W, axis):
  """
  Compute the center of connectivity along a specified axis.

  Parameters:
  W (numpy.ndarray): Weight matrix.
  axis (int): Axis along which to compute the center.

  Returns:
  numpy.ndarray: Computed center of connectivity.
  """
  if axis == 0:
    num_mec = W.shape[0]
    phi = np.linspace(-np.pi, np.pi, num_mec + 1)
    phi = phi[:-1]
    expphi = np.exp(1j * phi)
    sum_center = np.sum(expphi.reshape(-1, 1) * W, axis=axis)
    center = np.angle(sum_center)
  else:
    num_mec = W.shape[1]
    phi = np.linspace(-np.pi, np.pi, num_mec + 1)
    phi = phi[:-1]
    expphi = np.exp(1j * phi)
    sum_center = np.sum(np.transpose(expphi.reshape(-1, 1)) * W, axis=axis)
    center = np.angle(sum_center)
  return center

def place_field_center(HPC_fr, place_index, loc):
  """
  Compute the center of place fields for given place cells.

  Parameters:
  HPC_fr (numpy.ndarray): Firing rate matrix.
  place_index (numpy.ndarray): Indices of place cells.
  loc (numpy.ndarray): Locations of cells.

  Returns:
  tuple: Sorted indices of place cells and their centers.
  """
  exppos = np.exp(1j * loc)
  place_num = int(place_index.shape[0])
  center = np.zeros(place_num, )
  for i in range(place_num):
    fr_probe = HPC_fr[:, place_index[i]]
    fr_probe = fr_probe.reshape(-1, )
    center[i] = np.angle(np.sum(exppos * fr_probe))
  Center_place = np.sort(center)
  index = np.argsort(center)
  center_index = place_index[index]
  return center_index, Center_place

def bump_center(fr, pos):
  """
  Compute the center of a bump in the activity map.

  Parameters:
  fr (numpy.ndarray): Firing rate array.
  pos (numpy.ndarray): Positions array.

  Returns:
  float: Computed center of the bump.
  """
  exppos = np.exp(1j * pos)
  center = np.angle(np.sum(exppos * fr.reshape(-1,)))
  return center

def place_center(HPC_fr, place_index, loc):
  """
  Compute the spatial center of place cells.

  Parameters:
  HPC_fr (numpy.ndarray): Firing rate matrix.
  place_index (numpy.ndarray): Indices of place cells.
  loc (numpy.ndarray): Locations of cells.

  Returns:
  numpy.ndarray: Centers of place cells.
  """
  place_num = int(place_index.shape[0])
  Center = np.zeros([place_num, 2])
  fr_probe = HPC_fr[:, place_index.reshape(-1,)]
  for i in range(place_num):
    max_time = np.argmax(fr_probe[:, i], axis=0)
    Center[i, :] = loc[max_time, :]
  return Center

def get_center_grid(r):
  """
  Compute the center of activity in a grid.

  Parameters:
  r (numpy.ndarray): Activity matrix.

  Returns:
  numpy.ndarray: Centers of activity in the grid.
  """
  num_x = int(np.sqrt(r.shape[1]))
  phi_x = np.linspace(-np.pi, np.pi, num_x + 1)
  x = phi_x[0:-1]
  x_grid, y_grid = np.meshgrid(x, x)
  x_grid = x_grid.flatten()
  y_grid = y_grid.flatten()
  exppos_x = np.exp(1j * x_grid).reshape(1, -1)
  exppos_y = np.exp(1j * y_grid).reshape(1, -1)
  r = np.where(r > np.max(r) * 0.1, r, 0)
  center = np.zeros([r.shape[0], 2])

  center[:, 0] = np.angle(np.sum(exppos_x * r, axis=1))
  center[:, 1] = np.angle(np.sum(exppos_y * r, axis=1))
  return center

def get_center_band(r):
  """
  Compute the center of activity in a band.

  Parameters:
  r (numpy.ndarray): Activity matrix.

  Returns:
  numpy.ndarray: Centers of activity in the band.
  """
  num = r.shape[1]
  x = bm.linspace(-np.pi, np.pi, num, endpoint=False)
  exppos_x = bm.exp(1j * x)
  center = bm.angle(bm.sum(exppos_x[np.newaxis, :] * r, axis=1))
  return center

def get_center_hpc(fr, center_x, center_y, thres=0.2):
  """
  Compute the center of activity for HPC cells.

  Parameters:
  fr (numpy.ndarray): Firing rate matrix.
  center_x (numpy.ndarray): X-coordinates of centers.
  center_y (numpy.ndarray): Y-coordinates of centers.
  thres (float): Threshold for activity.

  Returns:
  tuple: X and Y coordinates of centers.
  """
  max_values = fr.max(axis=1).reshape(-1, 1)
  fr = np.where(fr < max_values * thres, 0, fr)
  sum_fr = np.sum(fr, axis=1).reshape(-1, 1)
  Cx = np.matmul(fr, center_x.reshape([-1, 1])) / sum_fr
  Cy = np.matmul(fr, center_y.reshape([-1, 1])) / sum_fr
  return Cx, Cy

def create_directory_if_not_exists(file_path):
  """
  Create a directory if it does not exist.

  Parameters:
  file_path (str): Path to the file or directory.
  """
  directory = os.path.dirname(file_path)
  if not os.path.exists(directory):
    os.makedirs(directory)

def calculate_place_field_scores(heatmap):
  """
  Calculate Place Field Scores for each place cell in the heatmap.

  Parameters:
  heatmap (numpy.ndarray): 2D array of shape (T, N), where T is the number of spatial positions and N is the number of place cells.

  Returns:
  numpy.ndarray: Vector of shape (N,) containing Place Field Scores for each place cell.
  """
  if len(heatmap.shape) != 2:
    raise ValueError("Input heatmap must be a 2D array.")
  
  T, N = heatmap.shape
  place_field_scores = np.zeros(N)
  
  for i in range(N):
    tuning_map = heatmap[:, i]
    tuning_map_normalized = (tuning_map - tuning_map.min()) / (tuning_map.max() - tuning_map.min())
    P = tuning_map_normalized / tuning_map_normalized.sum()
    P = np.clip(P, 1e-10, 1.0)
    log_P = -np.log2(P)
    entropy = np.dot(P, log_P)
    place_field_scores[i] = -entropy
  
  return place_field_scores

def compute_place_score(activity_matrix, position_matrix):
  """
  Compute place scores and sigma values for neurons based on activity and position matrices.

  Parameters:
  activity_matrix (numpy.ndarray): Activity matrix of shape (T, N).
  position_matrix (numpy.ndarray): Position matrix of shape (T, 2).

  Returns:
  tuple: Place scores and sigma values for each neuron.
  """
  T, N = activity_matrix.shape
  sigma = np.zeros(N)
  place_scores = np.zeros(N)
  x = position_matrix[:, 0]
  y = position_matrix[:, 1]
  for neuron in range(N):
    hpc_fr = activity_matrix[:, neuron]
    hpc_fr = np.where(hpc_fr > 1, hpc_fr, 0)
    max_time = np.argmax(hpc_fr)
    mean_x = position_matrix[max_time, 0]
    mean_y = position_matrix[max_time, 1]
    fr_sum = np.sum(hpc_fr)

    if fr_sum > 1:
      sigma_x = np.sqrt(np.sum(hpc_fr * (position_matrix[:, 0] - mean_x) ** 2) / fr_sum)
      sigma_y = np.sqrt(np.sum(hpc_fr * (position_matrix[:, 1] - mean_y) ** 2) / fr_sum)
      sigma_mean = (sigma_x + sigma_y) / 2
      sigma_mean = np.clip(sigma_mean, 0.05, 0.2)
      sigma[neuron] = sigma_mean

      gaussian = np.exp(-((x - mean_x) ** 2 / (2 * sigma[neuron] ** 2) + (y - mean_y) ** 2 / (2 * sigma[neuron] ** 2)))
      heatmap_normalized = hpc_fr.flatten() / bm.sum(hpc_fr)
      gaussian_normalized = gaussian.flatten() / np.sum(gaussian)
      place_scores[neuron] = np.dot(heatmap_normalized, gaussian_normalized) / (np.linalg.norm(heatmap_normalized) * np.linalg.norm(gaussian_normalized))
    else:
      place_scores[neuron] = 0
  
  return place_scores, sigma

def draw_square(x0, y0, x1, y1):
  """
  Draw a square on a plot.

  Parameters:
  x0 (float): X-coordinate of the bottom-left corner.
  y0 (float): Y-coordinate of the bottom-left corner.
  x1 (float): X-coordinate of the top-right corner.
  y1 (float): Y-coordinate of the top-right corner.
  """
  if x0 >= x1 or y0 >= y1:
    raise ValueError("Bottom-left corner must be smaller than top-right corner.")
  
  square_points = [
    (x0, y0),
    (x1, y0),
    (x1, y1),
    (x0, y1),
    (x0, y0)
  ]
  x_coords, y_coords = zip(*square_points)
  plt.plot(x_coords, y_coords, 'k', linewidth=2, marker='o')
  plt.xlim(x0 - 0.01, x1 + 0.01)
  plt.ylim(y0 - 0.01, y1 + 0.01)
  plt.gca().set_aspect('equal', adjustable='box')

def compute_place_score_from_heatmap(heatmap, width, height):
  """
  Compute place scores for neurons based on the heatmap.

  Parameters:
  heatmap (numpy.ndarray): Firing rate heatmap of shape (M, K, N).
  width (float): Width of the environment.
  height (float): Height of the environment.

  Returns:
  tuple: Place scores and sigma values for each neuron.
  """
  M, K, N = heatmap.shape
  place_scores = np.zeros(N)
  sigma = np.zeros(N)

  x_bins = np.linspace(0, width, M, endpoint=False) + width / (2 * M)
  y_bins = np.linspace(0, height, K, endpoint=False) + height / (2 * K)
  x, y = np.meshgrid(x_bins, y_bins, indexing="ij")

  for neuron in range(N):
    hpc_fr = heatmap[:, :, neuron]
    max_bin = np.unravel_index(np.argmax(hpc_fr), hpc_fr.shape)
    mean_x = x[max_bin]
    mean_y = y[max_bin]

    fr_sum = np.sum(hpc_fr)
    sigma_x = np.sqrt(np.sum(hpc_fr * (x - mean_x) ** 2) / fr_sum)
    sigma_y = np.sqrt(np.sum(hpc_fr * (y - mean_y) ** 2) / fr_sum)
    sigma_mean = (sigma_x + sigma_y) / 2
    sigma_mean = np.clip(sigma_mean, 0.025, 0.5)
    sigma[neuron] = sigma_mean

    gaussian = np.exp(-((x - mean_x) ** 2 / (2 * sigma[neuron] ** 2) + (y - mean_y) ** 2 / (2 * sigma[neuron] ** 2)))
    heatmap_normalized = hpc_fr.flatten() / np.sum(hpc_fr)
    gaussian_normalized = gaussian.flatten() / np.sum(gaussian)
    place_scores[neuron] = np.max(hpc_fr) * np.dot(heatmap_normalized, gaussian_normalized) / (np.linalg.norm(heatmap_normalized) * np.linalg.norm(gaussian_normalized))

  return place_scores, sigma

def place_center_heatmap(heatmap, place_index, width, height):
  """
  Compute the spatial center of place cells from a heatmap.

  Parameters:
  heatmap (numpy.ndarray): Firing rate heatmap of shape (M, K, N).
  place_index (numpy.ndarray): Indices of place cells.
  width (float): Width of the environment.
  height (float): Height of the environment.

  Returns:
  numpy.ndarray: Centers of place cells.
  """
  M, K, N = heatmap.shape
  x_bins = np.linspace(0, width, M, endpoint=False) + width / (2 * M)
  y_bins = np.linspace(0, height, K, endpoint=False) + height / (2 * K)
  x, y = np.meshgrid(x_bins, y_bins, indexing="ij")

  place_num = int(place_index.shape[0])
  Center = np.zeros([place_num, 2])
  fr_probe = heatmap[:, :, place_index.reshape(-1,)]
  for i in range(place_num):
    matrix = fr_probe[:, :, i]
    flat_index = np.argmax(matrix)
    row_index, col_index = np.unravel_index(flat_index, matrix.shape)
    Center[i, 0] = x_bins[row_index]
    Center[i, 1] = y_bins[col_index]
  return Center

from numba import njit, prange

@njit(parallel=True)
def compute_firing_field(A, positions, width, height, M, K):
  """
  Compute the firing field heatmap for neurons.

  Parameters:
  A (numpy.ndarray): Activity matrix of shape (T, N).
  positions (numpy.ndarray): Position matrix of shape (T, 2).
  width (float): Width of the environment.
  height (float): Height of the environment.
  M (int): Number of bins along the width.
  K (int): Number of bins along the height.

  Returns:
  numpy.ndarray: Heatmap of shape (M, K, N).
  """
  T, N = A.shape
  heatmaps = np.zeros((M, K, N))
  bin_counts = np.zeros((M, K))

  bin_width = width / M
  bin_height = height / K
  x_bins = np.clip(((positions[:, 0]) // bin_width).astype(np.int32), 0, M - 1)
  y_bins = np.clip(((positions[:, 1]) // bin_height).astype(np.int32), 0, K - 1)

  for t in prange(T):
    x_bin = x_bins[t]
    y_bin = y_bins[t]
    heatmaps[x_bin, y_bin, :] += A[t, :]
    bin_counts[x_bin, y_bin] += 1

  for n in range(N):
    heatmaps[:, :, n] = np.where(bin_counts > 0, heatmaps[:, :, n] / bin_counts, 0)
  
  return heatmaps
