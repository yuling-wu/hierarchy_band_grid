# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.ndimage as ndimage

"""Grid score calculations.
"""

def circle_mask(size, radius, in_val=1.0, out_val=0.0):
  """Calculating the grid scores with different radius."""
  sz = [math.floor(size[0] / 2), math.floor(size[1] / 2)]
  x = np.linspace(-sz[0], sz[1], size[1])
  x = np.expand_dims(x, 0)
  x = x.repeat(size[0], 0)
  y = np.linspace(-sz[0], sz[1], size[1])
  y = np.expand_dims(y, 1)
  y = y.repeat(size[1], 1)
  z = np.sqrt(x**2 + y**2)
  z = np.less_equal(z, radius)
  vfunc = np.vectorize(lambda b: b and in_val or out_val)
  return vfunc(z)


class GridScorer(object):
  """Class for scoring ratemaps given trajectories."""

  def __init__(self, nbins, coords_range, mask_parameters, min_max=False):
    """Scoring ratemaps given trajectories.
    Args:
      nbins: Number of bins per dimension in the ratemap.
      coords_range: Environment coordinates range.
      mask_parameters: parameters for the masks that analyze the angular
        autocorrelation of the 2D autocorrelation.
      min_max: Correction.
    """
    self._nbins = nbins
    self._min_max = min_max
    self._coords_range = coords_range
    self._corr_angles = [30, 45, 60, 90, 120, 135, 150]
    # Create all masks
    self._masks = [(self._get_ring_mask(mask_min, mask_max), (mask_min,
                                                              mask_max))
                   for mask_min, mask_max in mask_parameters]
    # Mask for hiding the parts of the SAC that are never used
    self._plotting_sac_mask = circle_mask(
        [self._nbins * 2 - 1, self._nbins * 2 - 1],
        self._nbins,
        in_val=1.0,
        out_val=np.nan)

  def calculate_ratemap(self, xs, ys, activations, statistic='mean'):
    return scipy.stats.binned_statistic_2d(
        xs,
        ys,
        activations,
        bins=self._nbins,
        statistic=statistic,
        range=self._coords_range)[0]

  def _get_ring_mask(self, mask_min, mask_max):
    n_points = [self._nbins * 2 - 1, self._nbins * 2 - 1]
    return (circle_mask(n_points, mask_max * self._nbins) *
            (1 - circle_mask(n_points, mask_min * self._nbins)))

  def grid_score_60(self, corr):
    if self._min_max:
      return np.minimum(corr[60], corr[120]) - np.maximum(
          corr[30], np.maximum(corr[90], corr[150]))
    else:
      return (corr[60] + corr[120]) / 2 - (corr[30] + corr[90] + corr[150]) / 3

  def grid_score_90(self, corr):
    return corr[90] - (corr[45] + corr[135]) / 2

  def calculate_sac(self, seq1):
    """Calculating spatial autocorrelogram."""
    seq2 = seq1

    def filter2(b, x):
      stencil = np.rot90(b, 2)
      return scipy.signal.convolve2d(x, stencil, mode='full')

    seq1 = np.nan_to_num(seq1)
    seq2 = np.nan_to_num(seq2)

    ones_seq1 = np.ones(seq1.shape)
    ones_seq1[np.isnan(seq1)] = 0
    ones_seq2 = np.ones(seq2.shape)
    ones_seq2[np.isnan(seq2)] = 0

    seq1[np.isnan(seq1)] = 0
    seq2[np.isnan(seq2)] = 0

    seq1_sq = np.square(seq1)
    seq2_sq = np.square(seq2)

    seq1_x_seq2 = filter2(seq1, seq2)
    sum_seq1 = filter2(seq1, ones_seq2)
    sum_seq2 = filter2(ones_seq1, seq2)
    sum_seq1_sq = filter2(seq1_sq, ones_seq2)
    sum_seq2_sq = filter2(ones_seq1, seq2_sq)
    n_bins = filter2(ones_seq1, ones_seq2)
    n_bins_sq = np.square(n_bins)

    std_seq1 = np.power(
        np.subtract(
            np.divide(sum_seq1_sq, n_bins),
            (np.divide(np.square(sum_seq1), n_bins_sq))), 0.5)
    std_seq2 = np.power(
        np.subtract(
            np.divide(sum_seq2_sq, n_bins),
            (np.divide(np.square(sum_seq2), n_bins_sq))), 0.5)
    covar = np.subtract(
        np.divide(seq1_x_seq2, n_bins),
        np.divide(np.multiply(sum_seq1, sum_seq2), n_bins_sq))
    x_coef = np.divide(covar, np.multiply(std_seq1, std_seq2))
    x_coef = np.real(x_coef)
    x_coef = np.nan_to_num(x_coef)
    return x_coef

  def rotated_sacs(self, sac, angles):
    return [
        scipy.ndimage.interpolation.rotate(sac, angle, reshape=False)
        for angle in angles
    ]

  def get_grid_scores_for_mask(self, sac, rotated_sacs, mask):
    """Calculate Pearson correlations of area inside mask at corr_angles."""
    masked_sac = sac * mask
    ring_area = np.sum(mask)
    # Calculate dc on the ring area
    masked_sac_mean = np.sum(masked_sac) / ring_area
    # Center the sac values inside the ring
    masked_sac_centered = (masked_sac - masked_sac_mean) * mask
    variance = np.sum(masked_sac_centered**2) / ring_area + 1e-5
    corrs = dict()
    for angle, rotated_sac in zip(self._corr_angles, rotated_sacs):
      masked_rotated_sac = (rotated_sac - masked_sac_mean) * mask
      cross_prod = np.sum(masked_sac_centered * masked_rotated_sac) / ring_area
      corrs[angle] = cross_prod / variance
    return self.grid_score_60(corrs), self.grid_score_90(corrs), variance

  def get_scores(self, rate_map):
    """Get summary of scrores for grid cells."""
    sac = self.calculate_sac(rate_map)
    rotated_sacs = self.rotated_sacs(sac, self._corr_angles)

    scores = [
        self.get_grid_scores_for_mask(sac, rotated_sacs, mask)
        for mask, mask_params in self._masks  # pylint: disable=unused-variable
    ]
    scores_60, scores_90, variances = map(np.asarray, zip(*scores))  # pylint: disable=unused-variable
    max_60_ind = np.argmax(scores_60)
    max_90_ind = np.argmax(scores_90)

    return (scores_60[max_60_ind], scores_90[max_90_ind],
            self._masks[max_60_ind][1], self._masks[max_90_ind][1], sac, max_60_ind)

  def plot_ratemap(self, ratemap, ax=None, title=None, *args, **kwargs):  # pylint: disable=keyword-arg-before-vararg
    """Plot ratemaps."""
    if ax is None:
      ax = plt.gca()
    # Plot the ratemap
    ax.imshow(ratemap, interpolation='none', *args, **kwargs)
    # ax.pcolormesh(ratemap, *args, **kwargs)
    ax.axis('off')
    if title is not None:
      ax.set_title(title)

  def plot_sac(self,
               sac,
               mask_params=None,
               ax=None,
               title=None,
               *args,
               **kwargs):  # pylint: disable=keyword-arg-before-vararg
    """Plot spatial autocorrelogram."""
    if ax is None:
      ax = plt.gca()
    # Plot the sac
    useful_sac = sac * self._plotting_sac_mask
    ax.imshow(useful_sac, interpolation='none', *args, **kwargs)
    # ax.pcolormesh(useful_sac, *args, **kwargs)
    # Plot a ring for the adequate mask
    if mask_params is not None:
      center = self._nbins - 1
      ax.add_artist(
          plt.Circle(
              (center, center),
              mask_params[0] * self._nbins,
              # lw=bump_size,
              fill=False,
              edgecolor='k'))
      ax.add_artist(
          plt.Circle(
              (center, center),
              mask_params[1] * self._nbins,
              # lw=bump_size,
              fill=False,
              edgecolor='k'))
    ax.axis('off')
    if title is not None:
      ax.set_title(title)



"""Band score calculations.
"""
import numpy.fft as nf
from scipy.optimize import curve_fit, leastsq

class BandScorer(object):
  """Class for scoring band cells given rate maps."""

  def __init__(self, nbins, box_width, box_height):
    """Scoring band cells given rate maps.
    Args:
      nbins: Number of bins per dimension in the ratemap.
      box_width: Width of the environment box.
      box_height: Height of the environment box.
    """
    self._nbins = nbins
    self.box_width = box_width
    self.box_height = box_height
    self.dx = self.box_width/self._nbins
    self.dy = self.box_height/self._nbins

  def compute_band_score(self, rate):
    """
    Compute the band score for a given rate map.
    This function calculates the band score, which is a measure of the spatial 
    periodicity of a rate map. It uses a 2D Fourier transform to analyze the 
    frequency components of the rate map and fits a Gaussian function to the 
    spectrum to extract parameters such as the dominant frequency and orientation.
    Args:
      rate (numpy.ndarray): A 1D array representing the rate map, which is 
        reshaped into a 2D grid based on `self._nbins`.
    Returns:
      tuple: A tuple containing the following elements:
        - band_scores (float): The computed band score, which quantifies 
          the correlation between the Fourier spectrum and the fitted Gaussian.
        - ratemap (numpy.ndarray): The reshaped 2D rate map.
        - params (list): The parameters of the fitted Gaussian function 
          [amplitude, frequency, orientation, standard deviation].
        - fft_rate (numpy.ndarray): The 2D Fourier spectrum of the rate map.
        - gx (numpy.ndarray): The Gaussian function evaluated over the spectrum.
        - k (float): The dominant spatial frequency.
        - phi (float): The orientation of the dominant frequency in radians.
        - sigma (float): The standard deviation of the Gaussian fit.
    Raises:
      RuntimeError: If the curve fitting process fails to converge.
    """
    # Create coordinate grid in virtual space: scale to [-2, 2] in x, and [-2*dy/dx, 2*dy/dx] in y
    X, Y = np.meshgrid(np.linspace(-2, 2, self._nbins), np.linspace(-2*self.dy/self.dx, 2*self.dy/self.dx, self._nbins))
    x_flat, y_flat = X.flatten(), Y.flatten()
    loc = np.stack([x_flat, y_flat])

    # Compute spectrum
    def spectrum(ratemap):
        res = ratemap.shape[0]
        fft_rate = np.abs(nf.fftshift(nf.fft2(ratemap)))
        fft_rate[:int(res/2),:] = 0
        return fft_rate
    fft_rate = spectrum(rate)

    # Define the Gaussian function
    def gaussian(loc, A, k, phi, sigma):
        x = loc[0]
        y = loc[1]
        return A * np.exp(-((x - k*np.cos(phi))**2 / (2 * sigma**2) + (y - k*np.sin(phi))**2 / (2 * sigma**2)))

    initial_guess = [1.0, 0.2, 0.0, 0.1]

    try:
        params, _ = curve_fit(lambda xy, A, k, phi, sigma: gaussian(xy, A, k, phi, sigma), 
                                (loc), 
                                fft_rate.ravel(), 
                                p0=initial_guess, 
                                bounds=([0, 0.2, 0, 0.05], [np.inf, 1, np.pi, 0.5]),
                                maxfev=1000) 
    except RuntimeError as e:
        print("Warning:", e)
        params, _, _, _, _ = leastsq(
            lambda xy: fft_rate.ravel() - gaussian(loc, *xy), initial_guess or np.ones(len(initial_guess)), full_output=True, maxfev=1000
        )
  
    k0 = params[1]
    phi = params[2]
    sigma = params[3]
    kx = k0*np.cos(phi)/2*(1/self.dx/2)*np.pi*2  # scale kx, ky to the frequency in the real physical space
    ky = k0*np.sin(phi)/2*(1/self.dy/2)*np.pi*2
    k = np.sqrt(kx**2+ky**2)  # The maximum frequency is 1/dx, correspoding k = np.pi*2/dx
    gx = gaussian(loc, params[0], params[1], params[2], params[3])

    # Compute correlation
    score = np.dot(fft_rate.ravel(), gx.ravel()) / (1e-8+np.linalg.norm(fft_rate.ravel()) * np.linalg.norm(gx.ravel())) / sigma

    return score, params, fft_rate, gx, k, phi, sigma


  def compute_orientation(self, heatmap, plot=False):
    # Compute the Fourier transform
    fft_result = np.fft.fftshift(np.fft.fft2(heatmap))
    power_spectrum = np.abs(fft_result) ** 2

    # Obtain frequency coordinates
    freq_x = np.fft.fftfreq(self._nbins, d=self.dx)
    freq_shifted_x = np.fft.fftshift(freq_x)
    freq_y = np.fft.fftfreq(self._nbins, d=self.dy)
    freq_shifted_y = np.fft.fftshift(freq_y)
    X, Y = np.meshgrid(freq_shifted_x, freq_shifted_y)

    # Calculate the phase angle of each frequency component
    orientations = np.arctan2(Y, X)
    orientations = np.where(orientations >= 0, orientations, orientations + np.pi)  # Restrict to 0 to ?
    
    # Calculate the weighted average orientation angle (considering ? rotation symmetry)
    weighted_orientations = power_spectrum * np.exp(1j * 2 * orientations)
    avg_orientation = np.angle(np.sum(weighted_orientations))  # Average orientation angle
    avg_orientation = avg_orientation + np.pi  # Ensure it is in the range 0 to ?
    avg_orientation = avg_orientation / 2
    avg_orientation = np.where(avg_orientation >= 170 / 180 * np.pi, avg_orientation - np.pi, avg_orientation)

    if plot:
      # Plot the original heatmap and power spectrum
      plt.figure(figsize=(12, 6))
      plt.subplot(1, 2, 1)
      space_range = ((-self.box_width/2, self.box_width/2), (-self.box_height/2, self.box_height/2))
      plt.imshow(heatmap.T, extent=(space_range[0][0], space_range[0][1], space_range[1][0], space_range[1][1]), origin="lower")
      plt.title("Heatmap of Band Cell")
      plt.colorbar(label="Firing Rate")

      plt.subplot(1, 2, 2)
      plt.imshow(power_spectrum, extent=(freq_shifted_x[0], freq_shifted_x[-1], freq_shifted_y[0], freq_shifted_y[-1]), origin="lower")
      plt.title("Power Spectrum of Heatmap")
      plt.colorbar(label="Power")
      plt.xlabel("Frequency X")
      plt.ylabel("Frequency Y")

      plt.show()

    return 180 - avg_orientation * (180 / np.pi)
  

  def compute_phases(self, L, theta, rates):
      """
      Compute the phases of band cells given their parameters and rate maps.
      Args:
        L : spacing for the band cell.
        theta : orientation angle (in radians) for the band cell.
        rates : the rate map of a band cell.
      Returns:
        numpy.ndarray: computed phase for a band cell.
      """    
      # "loc" here is different from what we use in "compute_band_score", it's the real physical location
      X, Y = np.meshgrid(np.linspace(-self.box_width/2, self.box_width/2, self._nbins), np.linspace(-self.box_height/2, self.box_height/2, self._nbins))
      x_flat, y_flat = X.flatten(), Y.flatten()
      loc = np.stack([x_flat, y_flat])

      # Project rate map along theta to calculate phi
      direction_vec = np.array([np.cos(theta), np.sin(theta)])
      phi = np.dot(direction_vec, loc)
      
      # Compute phase by taking modulo with spacing L
      loc_phase = np.mod(phi, L) / L * 2 * np.pi - np.pi  # Normalize to [-¦Ð, ¦Ð]
      phase = np.angle(np.sum(np.exp(loc_phase * 1j) * rates) / np.sum(rates))

      return phase



def direction_score(activations_theta):
    Ng = activations_theta.shape[0]
    scores = []
    A_params = []
    mu_params = []
    sigma_params = []
    theta = np.linspace(-np.pi, np.pi, activations_theta.shape[1])
    
    def gaussian(x, A, mu, sigma):
      dis = np.where(np.abs(x-mu) < np.pi, np.abs(x-mu), np.pi*2-np.abs(x-mu))
      return A * np.exp(-dis**2 / (2 * sigma**2))

    for i in range(Ng):
        y_data = activations_theta[i]
        
        # Fit the Gaussian function to the data
        try:
            popt, _ = curve_fit(gaussian, theta, y_data, p0=[1, 0, 1])
            fitted_curve = gaussian(theta, *popt)

            # Calculate cosine similarity
            cos_sim = np.dot(y_data, fitted_curve) / (np.linalg.norm(y_data) * np.linalg.norm(fitted_curve))
            scores.append(cos_sim)
            
            # Save the parameters
            A_params.append(popt[0])
            mu_params.append(popt[1])
            sigma_params.append(popt[2])
        except RuntimeError:
            scores.append(0)
            A_params.append(0)
            mu_params.append(0)
            sigma_params.append(0)
    
    return np.array(scores), np.array(A_params), np.array(mu_params), np.array(sigma_params)
