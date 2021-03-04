#!/usr/bin/env python3
# coding: utf-8

# # Registration of Tungsten fibres on XCT images
#
# This demo aims to demonstrate the use of [gVirtualXRay](http://gvirtualxray.sourceforge.net/) and mathematical optimisation to register polygon meshes onto X-ray microtomography (micro-CT) scans of a tungsten fibre. Below is an example of CT slice.
#
# ![The fibre.](scanned_object.png)
#
# Our simulations include beam-hardening due to polychromatism and they take into account the response of the detector.
#
# We use SimpleGVXR's Python wrapper and Python packages commonly used in tomography reconstruction ([Tomopy](https://tomopy.readthedocs.io/en/latest/)), image processing ([scikit-image](https://scikit-image.org/) and [SimpleITK](https://simpleitk.org/)), computer vision ([OpenCV](https://www.opencv.org/)), and non-linear numerical optimization ([CMA-ES, Covariance Matrix Adaptation Evolution Strategy](https://github.com/CMA-ES/pycma)).
#
# ## Import packages
#
# We need to import a few libraries (called packages in Python). We use:
#
# - `copy`: duplicating images using deepcopies;
# - `math`: the `floor` function;
# - `os`: creating a new directory
# - `glob`: retrieving file names in a directory;
# - `numpy`: who doesn't use numpy?
# - `imageio`: creating GIF files;
# - `skimage`: comparing the reference CT slice and the simulated one, computing the Radon transform of an image, and perform a CT reconstruction using FBP and SART;
# - `tomopy`: another package for CT reconstruction;
# - `SimpleITK`: image processing and saving volume data;
# - OpenCV (`cv2`): Hough transform and bilateral filter (an edge-preserving smoothing filter);
# - `matplotlib`: plotting data;
# - `cma`: non-linear numerical optimization;
# - `lsf`: the line spread function to filter the X-ray images; and
# - `gvxrPython3`: simulation of X-ray images using the Beer-Lambert law on GPU.

# In[12]:

import time
import argparse  # Process the cmd line
import copy, math, os, glob, sys

import numpy as np

import matplotlib
matplotlib.use('AGG')   # generate postscript output by default

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm

from skimage.transform import iradon
from skimage.util import compare_images
from scipy import ndimage
import skimage.io as io

# import tomopy

import SimpleITK as sitk


import cv2

import cma

#from FlyAlgorithm import *
from lsf import *

import gvxrPython3 as gvxr
import Simulation

# In[13]:

def processCmdLine():
    parser = argparse.ArgumentParser(description='Compute the path length using gVirtualXRay (http://gvirtualxray.sourceforge.net/).')
    parser.add_argument('--input', help='Sinogram file', nargs=1, type=str, required=True);
    parser.add_argument('--output', help='Output dir', nargs=1, type=str, required=True);
    return parser.parse_args()

args = processCmdLine();

output_directory = args.output[0];
if not os.path.exists(output_directory):
    os.makedirs(output_directory);



# ## Global variables
#
# We need some global variables.
#
# - `Simulation.reference_CT`: The reference XCT slice;
# - `Simulation.reference_sinogram`: The Radon transform of the reference XCT slice;
# - `Simulation.pixel_spacing_in_micrometre` and `Simulation.pixel_spacing_in_mm`: The physical distance between the centre of two successive pixel;
# - `Simulation.number_of_projections`: The total number of angles in the sinogram;
# - `Simulation.angular_span_in_degrees`: The angular span covered by the sinogram;
# - `Simulation.angular_step`: the angular step; and
# - `Simulation.theta`: The rotation angles in degrees (vertical axis of the sinogram).
#

# In[14]:




# ## Load the image data
#
# Load and display the reference projections from a raw binary file, i.e. the target of the registration.

# In[15]:



# Target of the registration
start_time = time.time();
reference_sinogram, reference_CT, normalised_reference_sinogram, normalised_reference_CT = Simulation.createTargetFromRawSinogram(args.input[0]);
elapsed_time = time.time() - start_time
print("RECONSTRUCTION:", elapsed_time);

# In[ ]:




# ## Set the X-ray simulation environment

# First we create an OpenGL context, here using EGL, i.e. no window.

Simulation.initGVXR();
# In[ ]:




# In[ ]:


# ## Registration of a cube

# ## Normalise the image data
#
# Zero-mean unit-variance normalisation is applied to use the reference images in objective functions and perform the registration. Note that it is called standardisation (Z-score Normalisation) in machine learning. It is computed as follows:
#
# $$I' = \frac{I - \bar{I}}{\sigma}$$
#
# Where $I'$ is the image after the original image $I$ has been normalised, $\bar{I}$ is the average pixel value of $I$, and $\sigma$ is its standard deviation.

# In[ ]:




# In[ ]:


# In[ ]:



# In[ ]:


Simulation.use_fibres = False;


# In[29]:
# The registration has already been performed. Load the results.
if os.path.isfile(output_directory + "/cube1.dat"):
    current_best = np.loadtxt(output_directory + "/cube1.dat");
# Perform the registration using CMA-ES
else:
    start_time = time.time()

    best_fitness = sys.float_info.max;
    matrix_id = 0;

    opts = cma.CMAOptions()
    opts.set('tolfun', 1e-3);
    opts['tolx'] = 1e-3;
    opts['bounds'] = [5*[-0.5], 5*[0.5]];
    #opts['seed'] = 123456789;
    # opts['maxiter'] = 5;

    es = cma.CMAEvolutionStrategy(5 * [0], 0.5, opts);
    es.optimize(Simulation.fitnessFunctionCube);

    current_best = copy.deepcopy(es.result.xbest); # [-0.12174177  0.07941929 -0.3949529  -0.18708068 -0.23998638]
    np.savetxt(output_directory + "/cube1.dat", current_best, header='x,y,rotation_angle,w,h');
    elapsed_time = time.time() - start_time
    print("CUBE1", elapsed_time);













# ### Apply the result of the registration

# In[30]:


# Save the result
Simulation.setMatrix(current_best);


simulated_sinogram, normalised_projections, raw_projections_in_keV = Simulation.simulateSinogram();

simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
reconstruction_CT_matrix = iradon(simulated_sinogram.T, theta=Simulation.theta, circle=True);

volume = sitk.GetImageFromArray(reconstruction_CT_matrix);
volume.SetSpacing([Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/reconstruction_CT_matrix1.mha", useCompression=True);


print("Matrix1 params:", current_best);
normalised_reconstruction_CT_matrix = (reconstruction_CT_matrix - reconstruction_CT_matrix.mean()) / reconstruction_CT_matrix.std();
ZNCC_CT = np.mean(np.multiply(normalised_reconstruction_CT_matrix.flatten(), normalised_reference_CT.flatten()));
print("Matrix1 CT ZNCC:", ZNCC_CT);


comp_equalized = compare_images(Simulation.reference_CT, reconstruction_CT_matrix, method='checkerboard');
volume = sitk.GetImageFromArray(comp_equalized);
volume.SetSpacing([Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/compare_reconstruction_CT_matrix1.mha", useCompression=True);

comp_equalized = compare_images(normalised_reference_CT, normalised_reconstruction_CT_matrix, method='checkerboard');
comp_equalized -= np.min(comp_equalized);
comp_equalized /= np.max(comp_equalized);
comp_equalized *= 255;
comp_equalized = np.array(comp_equalized, dtype=np.uint8);
io.imsave(output_directory + "/compare_reconstruction_CT_matrix1.png", comp_equalized)




# ## Find circles
#
# We can use the Hoguh transform to detect where circles are in the image. However, the input image in OpenCV's function must be in UINT8. We blur it using a bilateral filter (an edge-preserving smoothing filter).

# ### Convert the image to UINT8

# We first create a function to convert images in floating point numbers into UINT8.

# In[34]:




# We blur the CT scan using a bilateral filter. It preserves edges.

# In[35]:


start_time = time.time()

Simulation.centroid_set = Simulation.findCircles(Simulation.reference_CT);


# In[49]:



# In[50]:

#
# ### Optimise fibre radius

# In[56]:




# In[57]:

    # An individual is made of two floating point numbers:
    # - the radius of the SiC fibre
    # - the ratio    radius of the W core / radius of the SiC fibre


# The registration has already been performed. Load the results.
if os.path.isfile(output_directory + "/fibre_radius1.dat"):
    temp = np.loadtxt(output_directory + "/fibre_radius1.dat");
    Simulation.core_radius = temp[0];
    Simulation.fibre_radius = temp[1];
# Perform the registration using CMA-ES
else:
    ratio = Simulation.core_radius / Simulation.fibre_radius;

    x0 = [Simulation.fibre_radius, ratio];
    bounds = [[5, 0.01], [1.5 * Simulation.fibre_radius, 0.95]];

    best_fitness = sys.float_info.max;
    radius_fibre_id = 0;

    opts = cma.CMAOptions()
    opts.set('tolfun', 1e-3);
    opts['tolx'] = 1e-3;
    opts['bounds'] = bounds;
    #opts['seed'] = 987654321;
    # opts['maxiter'] = 5;

    es = cma.CMAEvolutionStrategy(x0, 0.9, opts);
    es.optimize(Simulation.fitnessFunctionFibres);
    elapsed_time = time.time() - start_time
    print("FIBRES1",elapsed_time);
    Simulation.fibre_radius = es.result.xbest[0];
    Simulation.core_radius = Simulation.fibre_radius * es.result.xbest[1];

    np.savetxt(output_directory + "/fibre_radius1.dat", [Simulation.core_radius, Simulation.fibre_radius], header='core_radius_in_um,fibre_radius_in_um');



# Load the matrix
Simulation.setMatrix(current_best);

# Load the cores and fibres
Simulation.setFibres(Simulation.centroid_set);





simulated_sinogram, normalised_projections, raw_projections_in_keV = Simulation.simulateSinogram();

simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
reconstruction_CT_fibres = iradon(simulated_sinogram.T, theta=Simulation.theta, circle=True);

volume = sitk.GetImageFromArray(reconstruction_CT_fibres);
volume.SetSpacing([Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/reconstruction_CT_fibres1.mha", useCompression=True);



print("Radii1:", Simulation.core_radius, Simulation.fibre_radius);
normalised_reconstruction_CT_fibres = (reconstruction_CT_fibres - reconstruction_CT_fibres.mean()) / reconstruction_CT_fibres.std();
ZNCC_CT = np.mean(np.multiply(normalised_reconstruction_CT_fibres.flatten(), normalised_reference_CT.flatten()));
print("Fibres1 CT ZNCC:", ZNCC_CT);

comp_equalized = compare_images(Simulation.reference_CT, reconstruction_CT_fibres, method='checkerboard');
volume = sitk.GetImageFromArray(comp_equalized);
volume.SetSpacing([Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/compare_reconstruction_CT_fibres1.mha", useCompression=True);

comp_equalized = compare_images(normalised_reference_CT, normalised_reconstruction_CT_fibres, method='checkerboard');
comp_equalized -= np.min(comp_equalized);
comp_equalized /= np.max(comp_equalized);
comp_equalized *= 255;
comp_equalized = np.array(comp_equalized, dtype=np.uint8);
io.imsave(output_directory + "/compare_reconstruction_CT_fibres1.png", comp_equalized)








Simulation.use_fibres = True;
# The registration has already been performed. Load the results.
if os.path.isfile(output_directory + "/cube2.dat"):
    current_best = np.loadtxt(output_directory + "/cube2.dat");
# Perform the registration using CMA-ES
else:
    start_time = time.time()

    best_fitness = sys.float_info.max;
    matrix_id = 0;

    opts = cma.CMAOptions()
    opts.set('tolfun', 1e-3);
    opts['tolx'] = 1e-3;
    opts['bounds'] = [5*[-0.5], 5*[0.5]];
    #opts['seed'] = 123456789;
    # opts['maxiter'] = 5;

    es = cma.CMAEvolutionStrategy(current_best, 0.125, opts);
    es.optimize(Simulation.fitnessFunctionCube);

    current_best = copy.deepcopy(es.result.xbest); # [-0.12174177  0.07941929 -0.3949529  -0.18708068 -0.23998638]
    np.savetxt(output_directory + "/cube2.dat", current_best, header='x,y,rotation_angle,w,h');
    elapsed_time = time.time() - start_time
    print("CUBE2", elapsed_time);





# Load the matrix
Simulation.setMatrix(current_best);

# Load the cores and fibres
Simulation.setFibres(Simulation.centroid_set);


simulated_sinogram, normalised_projections, raw_projections_in_keV = Simulation.simulateSinogram();

simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
reconstruction_CT_matrix = iradon(simulated_sinogram.T, theta=Simulation.theta, circle=True);

volume = sitk.GetImageFromArray(reconstruction_CT_matrix);
volume.SetSpacing([Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/reconstruction_CT_matrix2.mha", useCompression=True);


print("Matrix2 params:", current_best);
normalised_reconstruction_CT_matrix = (reconstruction_CT_matrix - reconstruction_CT_matrix.mean()) / reconstruction_CT_matrix.std();
ZNCC_CT = np.mean(np.multiply(normalised_reconstruction_CT_matrix.flatten(), normalised_reference_CT.flatten()));
print("Matrix2 CT ZNCC:", ZNCC_CT);

comp_equalized = compare_images(Simulation.reference_CT, reconstruction_CT_matrix, method='checkerboard');
volume = sitk.GetImageFromArray(comp_equalized);
volume.SetSpacing([Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/compare_reconstruction_CT_matrix2.mha", useCompression=True);

comp_equalized = compare_images(normalised_reference_CT, normalised_reconstruction_CT_matrix, method='checkerboard');
comp_equalized -= np.min(comp_equalized);
comp_equalized /= np.max(comp_equalized);
comp_equalized *= 255;
comp_equalized = np.array(comp_equalized, dtype=np.uint8);
io.imsave(output_directory + "/compare_reconstruction_CT_matrix2.png", comp_equalized)








Simulation.centroid_set = Simulation.refineCentrePositions(Simulation.centroid_set, reconstruction_CT_fibres);






# Load the matrix
Simulation.setMatrix(current_best);

# Load the cores and fibres
Simulation.setFibres(Simulation.centroid_set);





simulated_sinogram, normalised_projections, raw_projections_in_keV = Simulation.simulateSinogram();

simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
reconstruction_CT_fibres = iradon(simulated_sinogram.T, theta=Simulation.theta, circle=True);

volume = sitk.GetImageFromArray(reconstruction_CT_fibres);
volume.SetSpacing([Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/reconstruction_CT_fibres2.mha", useCompression=True);



print("Radii2:", Simulation.core_radius, Simulation.fibre_radius);
normalised_reconstruction_CT_fibres = (reconstruction_CT_fibres - reconstruction_CT_fibres.mean()) / reconstruction_CT_fibres.std();
ZNCC_CT = np.mean(np.multiply(normalised_reconstruction_CT_fibres.flatten(), normalised_reference_CT.flatten()));
print("Fibres2 CT ZNCC:", ZNCC_CT);

comp_equalized = compare_images(Simulation.reference_CT, reconstruction_CT_fibres, method='checkerboard');
volume = sitk.GetImageFromArray(comp_equalized);
volume.SetSpacing([Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/compare_reconstruction_CT_fibres2.mha", useCompression=True);

comp_equalized = compare_images(normalised_reference_CT, normalised_reconstruction_CT_fibres, method='checkerboard');
comp_equalized -= np.min(comp_equalized);
comp_equalized /= np.max(comp_equalized);
comp_equalized *= 255;
comp_equalized = np.array(comp_equalized, dtype=np.uint8);
io.imsave(output_directory + "/compare_reconstruction_CT_fibres2.png", comp_equalized)



















# The registration has already been performed. Load the results.
if os.path.isfile(output_directory + "/fibre_radius2.dat"):
    temp = np.loadtxt(output_directory + "/fibre_radius2.dat");
    core_radius = temp[0];
    Simulation.fibre_radius = temp[1];
# Perform the registration using CMA-ES
else:
    ratio = core_radius / Simulation.fibre_radius;

    x0 = [Simulation.fibre_radius, ratio];
    bounds = [[5, 0.01], [1.5 * Simulation.fibre_radius, 0.95]];

    best_fitness = sys.float_info.max;
    radius_fibre_id = 0;

    opts = cma.CMAOptions()
    opts.set('tolfun', 1e-3);
    opts['tolx'] = 1e-3;
    opts['bounds'] = bounds;
    #opts['seed'] = 987654321;
    # opts['maxiter'] = 5;

    es = cma.CMAEvolutionStrategy(x0, 0.25, opts);
    es.optimize(Simulation.fitnessFunctionFibres);
    elapsed_time = time.time() - start_time
    print("FIBRES2",elapsed_time);
    Simulation.fibre_radius = es.result.xbest[0];
    core_radius = Simulation.fibre_radius * es.result.xbest[1];

    np.savetxt(output_directory + "/fibre_radius2.dat", [core_radius, Simulation.fibre_radius], header='core_radius_in_um,fibre_radius_in_um');



# Load the matrix
Simulation.setMatrix(current_best);

# Load the cores and fibres
Simulation.setFibres(Simulation.centroid_set);





simulated_sinogram, normalised_projections, raw_projections_in_keV = Simulation.simulateSinogram();

simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
reconstruction_CT_fibres = iradon(simulated_sinogram.T, theta=Simulation.theta, circle=True);

volume = sitk.GetImageFromArray(reconstruction_CT_fibres);
volume.SetSpacing([Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/reconstruction_CT_fibres3.mha", useCompression=True);



print("Radii3:", core_radius, Simulation.fibre_radius);
normalised_reconstruction_CT_fibres = (reconstruction_CT_fibres - reconstruction_CT_fibres.mean()) / reconstruction_CT_fibres.std();
ZNCC_CT = np.mean(np.multiply(normalised_reconstruction_CT_fibres.flatten(), normalised_reference_CT.flatten()));
print("Fibres3 CT ZNCC:", ZNCC_CT);

comp_equalized = compare_images(Simulation.reference_CT, reconstruction_CT_fibres, method='checkerboard');
volume = sitk.GetImageFromArray(comp_equalized);
volume.SetSpacing([Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/compare_reconstruction_CT_fibres3.mha", useCompression=True);

comp_equalized = compare_images(normalised_reference_CT, normalised_reconstruction_CT_fibres, method='checkerboard');
comp_equalized -= np.min(comp_equalized);
comp_equalized /= np.max(comp_equalized);
comp_equalized *= 255;
comp_equalized = np.array(comp_equalized, dtype=np.uint8);
io.imsave(output_directory + "/compare_reconstruction_CT_fibres3.png", comp_equalized)


















# Find the cylinder in the centre of the image
best_centre = None;
best_distance = sys.float_info.max;

for centre in Simulation.centroid_set:
    distance = math.pow(centre[0] - Simulation.reference_CT.shape[1] / 2,2 ) + math.pow(centre[1] - Simulation.reference_CT.shape[0] / 2, 2);

    if best_distance > distance:
        best_distance = distance;
        best_centre = copy.deepcopy(centre);


# The registration has already been performed. Load the results.
if os.path.isfile(output_directory + "/laplacian.dat"):
    temp = np.loadtxt(output_directory + "/laplacian.dat");
    sigma_core = temp[0];
    sigma_fibre = temp[1];
    sigma_matrix = temp[2];
    k_core = temp[3];
    k_fibre = temp[4];
    k_matrix = temp[5];
    Simulation.fibre_radius = temp[6];
# Perform the registration using CMA-ES
else:

    sigma_core = 0.15;
    sigma_fibre = 0.35;
    sigma_matrix = 0.2;

    k_core = 5;
    k_fibre = 3;
    k_matrix = 2.0;

    x0 = [sigma_core, sigma_fibre, sigma_matrix, k_core, k_fibre, k_matrix, Simulation.fibre_radius];
    bounds = [[0.005, 0.05, 0.005, 0.0, 0.0, 0.0, 0.95 * Simulation.fibre_radius], [0.6, 0.6, 0.6, 15, 5, 5, 1.05 * Simulation.fibre_radius]];

    best_fitness = sys.float_info.max;
    laplacian_id = 0;

    opts = cma.CMAOptions()
    opts.set('tolfun', 1e-3);
    opts['tolx'] = 1e-3;
    opts['bounds'] = bounds;
    #opts['seed'] = 987654321;
    # opts['maxiter'] = 5;
    opts['CMA_stds'] = [0.25, 0.25, 0.25, 2.0, 2.0, 2.0, Simulation.fibre_radius * 0.025];


    es = cma.CMAEvolutionStrategy(x0, 0.25, opts);
    es.optimize(Simulation.fitnessFunctionLaplacian);
    elapsed_time = time.time() - start_time
    print("LAPLACIAN",elapsed_time);

    sigma_core = es.result.xbest[0];
    sigma_fibre = es.result.xbest[1];
    sigma_matrix = es.result.xbest[2];
    k_core = es.result.xbest[3];
    k_fibre = es.result.xbest[4];
    k_matrix = es.result.xbest[5];
    Simulation.fibre_radius = es.result.xbest[6];

    np.savetxt(output_directory + "/laplacian.dat", [sigma_core, sigma_fibre, sigma_matrix, k_core, k_fibre, k_matrix, Simulation.fibre_radius], header='sigma_core,sigma_fibre,sigma_matrix,k_core,k_fibre,k_matrix,fibre_radius_in_um');



# Load the matrix
Simulation.setMatrix(current_best);

# Load the cores and fibres
Simulation.setFibres(Simulation.centroid_set);


simulated_sinogram, normalised_projections, raw_projections_in_keV = Simulation.simulateSinogram([sigma_core, sigma_fibre, sigma_matrix], [k_core, k_fibre, k_matrix]);

simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
reconstruction_CT_laplacian = iradon(simulated_sinogram.T, theta=Simulation.theta, circle=True);

volume = sitk.GetImageFromArray(reconstruction_CT_laplacian);
volume.SetSpacing([Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/reconstruction_CT_laplacian.mha", useCompression=True);



print("Laplacian:", sigma_core, sigma_fibre, sigma_matrix, k_core, k_fibre, k_matrix, Simulation.fibre_radius);
normalised_reconstruction_CT_laplacian = (reconstruction_CT_laplacian - reconstruction_CT_laplacian.mean()) / reconstruction_CT_laplacian.std();
ZNCC_CT = np.mean(np.multiply(normalised_reconstruction_CT_laplacian.flatten(), normalised_reference_CT.flatten()));
print("Laplacian CT ZNCC:", ZNCC_CT);

comp_equalized = compare_images(Simulation.reference_CT, reconstruction_CT_laplacian, method='checkerboard');
volume = sitk.GetImageFromArray(comp_equalized);
volume.SetSpacing([Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/compare_reconstruction_CT_laplacian.mha", useCompression=True);

comp_equalized = compare_images(normalised_reference_CT, normalised_reconstruction_CT_laplacian, method='checkerboard');
comp_equalized -= np.min(comp_equalized);
comp_equalized /= np.max(comp_equalized);
comp_equalized *= 255;
comp_equalized = np.array(comp_equalized, dtype=np.uint8);
io.imsave(output_directory + "/compare_reconstruction_CT_laplacian.png", comp_equalized)



















roi_length = 40;
fibre_radius_in_px = Simulation.fibre_radius / Simulation.pixel_spacing_in_micrometre
core_radius_in_px = core_radius / Simulation.pixel_spacing_in_micrometre

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return np.array(mask, dtype=bool);




pixel_range = np.linspace(-Simulation.value_range, Simulation.value_range, num=int(Simulation.num_samples), endpoint=True)

for label, sigma, k in zip(["core", "fibre", "matrix"], [sigma_core, sigma_fibre, sigma_matrix], [k_core, k_fibre, k_matrix]):
    np.savetxt(output_directory + "/laplacian_" + label + ".txt", Simulation.laplacian(pixel_range, sigma) * k);





test_fibre_in_centre = np.array(copy.deepcopy(reconstruction_CT_laplacian[best_centre[1] - roi_length:best_centre[1] + roi_length, best_centre[0] - roi_length:best_centre[0] + roi_length]));

reference_fibre_in_centre = np.array(copy.deepcopy(Simulation.reference_CT[best_centre[1] - roi_length:best_centre[1] + roi_length, best_centre[0] - roi_length:best_centre[0] + roi_length]));






volume = sitk.GetImageFromArray(test_fibre_in_centre);
volume.SetSpacing([Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/simulated_fibre_in_centre.mha", useCompression=True);


volume = sitk.GetImageFromArray(reference_fibre_in_centre);
volume.SetSpacing([Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/reference_fibre_in_centre.mha", useCompression=True);

np.savetxt(output_directory + "/profile_reference_fibre_in_centre.txt", np.diag(reference_fibre_in_centre));
np.savetxt(output_directory + "/profile_simulated_fibre_in_centre.txt", np.diag(test_fibre_in_centre));


mask_shape = reference_fibre_in_centre.shape;

core_mask = create_circular_mask(mask_shape[1], mask_shape[0], None, core_radius_in_px);

fibre_mask = create_circular_mask(mask_shape[1], mask_shape[0], None, fibre_radius_in_px);
matrix_mask = np.logical_not(fibre_mask);

#fibre_mask = np.subtract(fibre_mask, core_mask);
fibre_mask = np.bitwise_xor(fibre_mask, core_mask);

#TypeError: numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, the `^` operator, or the logical_xor function instead.


core_mask = ndimage.binary_erosion(core_mask).astype(core_mask.dtype);
np.savetxt(output_directory + "/core_mask.txt", core_mask);

fibre_mask = ndimage.binary_erosion(fibre_mask).astype(core_mask.dtype);
np.savetxt(output_directory + "/fibre_mask.txt", fibre_mask);

matrix_mask = ndimage.binary_erosion(matrix_mask).astype(core_mask.dtype);
np.savetxt(output_directory + "/matrix_mask.txt", matrix_mask);




index = np.nonzero(core_mask);
print("CORE REF (MIN, MEDIAN, MAX, MEAN, STDDEV):",
        np.min(reference_fibre_in_centre[index]),
        np.median(reference_fibre_in_centre[index]),
        np.max(reference_fibre_in_centre[index]),
        np.mean(reference_fibre_in_centre[index]),
        np.std(reference_fibre_in_centre[index]));

print("CORE SIMULATED (MIN, MEDIAN, MAX, MEAN, STDDEV):",
        np.min(test_fibre_in_centre[index]),
        np.median(test_fibre_in_centre[index]),
        np.max(test_fibre_in_centre[index]),
        np.mean(test_fibre_in_centre[index]),
        np.std(test_fibre_in_centre[index]));

index = np.nonzero(fibre_mask);
print("FIBRE REF (MIN, MEDIAN, MAX, MEAN, STDDEV):",
        np.min(reference_fibre_in_centre[index]),
        np.median(reference_fibre_in_centre[index]),
        np.max(reference_fibre_in_centre[index]),
        np.mean(reference_fibre_in_centre[index]),
        np.std(reference_fibre_in_centre[index]));

print("FIBRE SIMULATED (MIN, MEDIAN, MAX, MEAN, STDDEV):",
        np.min(test_fibre_in_centre[index]),
        np.median(test_fibre_in_centre[index]),
        np.max(test_fibre_in_centre[index]),
        np.mean(test_fibre_in_centre[index]),
        np.std(test_fibre_in_centre[index]));

index = np.nonzero(matrix_mask);
print("MATRIX REF (MIN, MEDIAN, MAX, MEAN, STDDEV):",
        np.min(reference_fibre_in_centre[index]),
        np.median(reference_fibre_in_centre[index]),
        np.max(reference_fibre_in_centre[index]),
        np.mean(reference_fibre_in_centre[index]),
        np.std(reference_fibre_in_centre[index]));

print("MATRIX SIMULATED (MIN, MEDIAN, MAX, MEAN, STDDEV):",
        np.min(test_fibre_in_centre[index]),
        np.median(test_fibre_in_centre[index]),
        np.max(test_fibre_in_centre[index]),
        np.mean(test_fibre_in_centre[index]),
        np.std(test_fibre_in_centre[index]));
