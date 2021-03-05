#!/usr/bin/env python3
# coding: utf-8


import time
import argparse  # Process the cmd line
import copy, math, os, glob, sys

import numpy as np

import matplotlib
matplotlib.use('AGG')   # generate postscript output by default

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm

from scipy import ndimage

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


# ## Load the image data
#
# Load and display the reference projections from a raw binary file, i.e. the target of the registration.

# In[15]:



# Target of the registration
start_time = time.time();
reference_sinogram, reference_CT, normalised_reference_sinogram, normalised_reference_CT = Simulation.createTargetFromRawSinogram(args.input[0]);
elapsed_time = time.time() - start_time
print("RECONSTRUCTION:", elapsed_time);



# Set the X-ray simulation environment
Simulation.initGVXR();


################################################################################
##### OPTIMISE THE CUBE FROM SCRATCH
################################################################################

Simulation.use_fibres = False;


# The registration has already been performed. Load the results.
if os.path.isfile(output_directory + "/cube1.dat"):
    Simulation.matrix_geometry_parameters = np.loadtxt(output_directory + "/cube1.dat");
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

    Simulation.matrix_geometry_parameters = copy.deepcopy(es.result.xbest); # [-0.12174177  0.07941929 -0.3949529  -0.18708068 -0.23998638]
    np.savetxt(output_directory + "/cube1.dat", Simulation.matrix_geometry_parameters, header='x,y,rotation_angle,w,h');
    elapsed_time = time.time() - start_time
    print("CUBE1", elapsed_time);

# Apply the result of the registration
Simulation.setMatrix(Simulation.matrix_geometry_parameters);

# Simulate the corresponding CT aquisition
simulated_sinogram, normalised_projections, raw_projections_in_keV = Simulation.simulateSinogram();

# Store the corresponding results on the disk
ZNCC_CT, CT_slice_from_simulated_sinogram = Simulation.reconstructAndStoreResults(simulated_sinogram, output_directory + "/matrix1");
print("Matrix1 params:", Simulation.matrix_geometry_parameters);
print("Matrix1 CT ZNCC:", ZNCC_CT);


################################################################################
##### FIND CIRCLES
################################################################################
start_time = time.time()

Simulation.centroid_set = Simulation.findCircles(Simulation.reference_CT);


################################################################################
##### OPTIMISE THE RADII AFTER OPTIMISING THE CUBE FROM SCRATCH
################################################################################

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

# Apply the result of the registration
Simulation.setMatrix(Simulation.matrix_geometry_parameters);
Simulation.setFibres(Simulation.centroid_set);

# Simulate the corresponding CT aquisition
simulated_sinogram, normalised_projections, raw_projections_in_keV = Simulation.simulateSinogram();

# Store the corresponding results on the disk
ZNCC_CT, CT_slice_from_simulated_sinogram = Simulation.reconstructAndStoreResults(simulated_sinogram, output_directory + "/fibres1");
print("Fibres1 params:", Simulation.core_radius, Simulation.fibre_radius);
print("Fibres1 CT ZNCC:", ZNCC_CT);


################################################################################
##### OPTIMISE THE CUBE AGAIN, BUT THI TIME TURNING ON THE CYLINDERS
################################################################################

Simulation.use_fibres = True;
# The registration has already been performed. Load the results.
if os.path.isfile(output_directory + "/cube2.dat"):
    Simulation.matrix_geometry_parameters = np.loadtxt(output_directory + "/cube2.dat");
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

    es = cma.CMAEvolutionStrategy(Simulation.matrix_geometry_parameters, 0.125, opts);
    es.optimize(Simulation.fitnessFunctionCube);

    Simulation.matrix_geometry_parameters = copy.deepcopy(es.result.xbest); # [-0.12174177  0.07941929 -0.3949529  -0.18708068 -0.23998638]
    np.savetxt(output_directory + "/cube2.dat", Simulation.matrix_geometry_parameters, header='x,y,rotation_angle,w,h');
    elapsed_time = time.time() - start_time
    print("CUBE2", elapsed_time);


# Apply the result of the registration
Simulation.setMatrix(Simulation.matrix_geometry_parameters);
Simulation.setFibres(Simulation.centroid_set);

# Simulate the corresponding CT aquisition
simulated_sinogram, normalised_projections, raw_projections_in_keV = Simulation.simulateSinogram();

# Store the corresponding results on the disk
ZNCC_CT, CT_slice_from_simulated_sinogram = Simulation.reconstructAndStoreResults(simulated_sinogram, output_directory + "/matrix2");
print("matrix2 params:", Simulation.core_radius, Simulation.fibre_radius);
print("matrix2 CT ZNCC:", ZNCC_CT);


################################################################################
##### RECENTRE THE CYLINDERS
################################################################################

Simulation.centroid_set = Simulation.refineCentrePositions(Simulation.centroid_set, CT_slice_from_simulated_sinogram);

# Find the position of the fibre that in the most in the centre of the CT slice
Simulation.findFibreInCentreOfCtSlice();

# Apply the result of the registration
Simulation.setMatrix(Simulation.matrix_geometry_parameters);
Simulation.setFibres(Simulation.centroid_set);

# Simulate the corresponding CT aquisition
simulated_sinogram, normalised_projections, raw_projections_in_keV = Simulation.simulateSinogram();

# Store the corresponding results on the disk
ZNCC_CT, CT_slice_from_simulated_sinogram = Simulation.reconstructAndStoreResults(simulated_sinogram, output_directory + "/fibres2");
print("Fibres2 CT ZNCC:", ZNCC_CT);


################################################################################
##### OPTIMISE THE RADII AFTER RECENTRING THE CYLINDERS
################################################################################

# The registration has already been performed. Load the results.
if os.path.isfile(output_directory + "/fibre_radius3.dat"):
    temp = np.loadtxt(output_directory + "/fibre_radius3.dat");
    core_radius = temp[0];
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

    es = cma.CMAEvolutionStrategy(x0, 0.25, opts);
    es.optimize(Simulation.fitnessFunctionFibres);
    elapsed_time = time.time() - start_time
    print("FIBRES2",elapsed_time);
    Simulation.fibre_radius = es.result.xbest[0];
    core_radius = Simulation.fibre_radius * es.result.xbest[1];

    np.savetxt(output_directory + "/fibre_radius3.dat", [Simulation.core_radius, Simulation.fibre_radius], header='core_radius_in_um,fibre_radius_in_um');

# Apply the result of the registration
Simulation.setMatrix(Simulation.matrix_geometry_parameters);
Simulation.setFibres(Simulation.centroid_set);

# Simulate the corresponding CT aquisition
simulated_sinogram, normalised_projections, raw_projections_in_keV = Simulation.simulateSinogram();

# Store the corresponding results on the disk
ZNCC_CT, CT_slice_from_simulated_sinogram = Simulation.reconstructAndStoreResults(simulated_sinogram, output_directory + "/fibres3");
print("Fibres3 params:", Simulation.core_radius, Simulation.fibre_radius);
print("Fibres3 CT ZNCC:", ZNCC_CT);





################################################################################
##### OPTIMISE THE LAPLACIAN OF THE FIBRE
################################################################################

# The registration has already been performed. Load the results.
if os.path.isfile(output_directory + "/laplacian1.dat"):
    temp = np.loadtxt(output_directory + "/laplacian1.dat");
    sigma_core = temp[0];
    k_core = temp[1];
    Simulation.fibre_radius = temp[2];
# Perform the registration using CMA-ES
else:

    sigma_core = 0.15;
    sigma_fibre = 0.35;
    sigma_matrix = 0.2;

    k_core = 5;
    k_fibre = 3;
    k_matrix = 2.0;

    x0 = [sigma_fibre, k_fibre, Simulation.fibre_radius];
    bounds = [[0.005, 0.05, 0.95 * Simulation.fibre_radius], [0.6, 15, 1.05 * Simulation.fibre_radius]];

    best_fitness = sys.float_info.max;
    laplacian_id = 0;

    opts = cma.CMAOptions()
    opts.set('tolfun', 1e-3);
    opts['tolx'] = 1e-3;
    opts['bounds'] = bounds;
    #opts['seed'] = 987654321;
    # opts['maxiter'] = 5;
    opts['CMA_stds'] = [0.25, 0.25, Simulation.fibre_radius * 0.025];


    es = cma.CMAEvolutionStrategy(x0, 0.25, opts);
    es.optimize(Simulation.fitnessFunctionLaplacian);
    elapsed_time = time.time() - start_time
    print("LAPLACIAN1",elapsed_time);

    sigma_fibre = es.result.xbest[0];
    k_fibre = es.result.xbest[1];
    Simulation.fibre_radius = es.result.xbest[2];

    np.savetxt(output_directory + "/laplacian1.dat", [sigma_fibre, k_fibre,Simulation.fibre_radius], header='sigma_fibre,k_fibre,fibre_radius_in_um');

# Apply the result of the registration
Simulation.setMatrix(Simulation.matrix_geometry_parameters);
Simulation.setFibres(Simulation.centroid_set);

# Simulate the corresponding CT aquisition
simulated_sinogram, normalised_projections, raw_projections_in_keV = Simulation.simulateSinogram([sigma_fibre], [k_fibre], ["fibre"]);

# Store the corresponding results on the disk
ZNCC_CT, CT_slice_from_simulated_sinogram = Simulation.reconstructAndStoreResults(simulated_sinogram, output_directory + "/laplacian1");
print("Laplacian1_fibre params:", Simulation.core_radius, Simulation.fibre_radius);
print("Laplacian1_fibre CT ZNCC:", ZNCC_CT);

#
#
# fibre_radius_in_px = Simulation.fibre_radius / Simulation.pixel_spacing_in_micrometre
# core_radius_in_px = core_radius / Simulation.pixel_spacing_in_micrometre
#
# def create_circular_mask(h, w, center=None, radius=None):
#
#     if center is None: # use the middle of the image
#         center = (int(w/2), int(h/2))
#     if radius is None: # use the smallest distance between the center and image walls
#         radius = min(center[0], center[1], w-center[0], h-center[1])
#
#     Y, X = np.ogrid[:h, :w]
#     dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
#
#     mask = dist_from_center <= radius
#     return np.array(mask, dtype=bool);
#
#
#
#
# pixel_range = np.linspace(-Simulation.value_range, Simulation.value_range, num=int(Simulation.num_samples), endpoint=True)
#
# for label, sigma, k in zip(["core", "fibre", "matrix"], [sigma_core, sigma_fibre, sigma_matrix], [k_core, k_fibre, k_matrix]):
#     np.savetxt(output_directory + "/laplacian_" + label + ".txt", Simulation.laplacian(pixel_range, sigma) * k);
#
#
#
#
#
# test_fibre_in_centre = np.array(copy.deepcopy(CT_slice_from_simulated_sinogram[cylinder_position_in_centre_of_slice[1] - Simulation.roi_length:cylinder_position_in_centre_of_slice[1] + Simulation.roi_length, cylinder_position_in_centre_of_slice[0] - Simulation.roi_length:cylinder_position_in_centre_of_slice[0] + Simulation.roi_length]));
#
# reference_fibre_in_centre = np.array(copy.deepcopy(Simulation.reference_CT[cylinder_position_in_centre_of_slice[1] - Simulation.roi_length:cylinder_position_in_centre_of_slice[1] + Simulation.roi_length, cylinder_position_in_centre_of_slice[0] - Simulation.roi_length:cylinder_position_in_centre_of_slice[0] + Simulation.roi_length]));
#
#
#
#
#
#
# volume = sitk.GetImageFromArray(test_fibre_in_centre);
# volume.SetSpacing([Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm]);
# sitk.WriteImage(volume, output_directory + "/simulated_fibre_in_centre.mha", useCompression=True);
#
#
# volume = sitk.GetImageFromArray(reference_fibre_in_centre);
# volume.SetSpacing([Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm]);
# sitk.WriteImage(volume, output_directory + "/reference_fibre_in_centre.mha", useCompression=True);
#
# np.savetxt(output_directory + "/profile_reference_fibre_in_centre.txt", np.diag(reference_fibre_in_centre));
# np.savetxt(output_directory + "/profile_simulated_fibre_in_centre.txt", np.diag(test_fibre_in_centre));
#
#
# mask_shape = reference_fibre_in_centre.shape;
#
# core_mask = create_circular_mask(mask_shape[1], mask_shape[0], None, core_radius_in_px);
#
# fibre_mask = create_circular_mask(mask_shape[1], mask_shape[0], None, fibre_radius_in_px);
# matrix_mask = np.logical_not(fibre_mask);
#
# #fibre_mask = np.subtract(fibre_mask, core_mask);
# fibre_mask = np.bitwise_xor(fibre_mask, core_mask);
#
# #TypeError: numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, the `^` operator, or the logical_xor function instead.
#
#
# core_mask = ndimage.binary_erosion(core_mask).astype(core_mask.dtype);
# np.savetxt(output_directory + "/core_mask.txt", core_mask);
#
# fibre_mask = ndimage.binary_erosion(fibre_mask).astype(core_mask.dtype);
# np.savetxt(output_directory + "/fibre_mask.txt", fibre_mask);
#
# matrix_mask = ndimage.binary_erosion(matrix_mask).astype(core_mask.dtype);
# np.savetxt(output_directory + "/matrix_mask.txt", matrix_mask);
#
#
#
#
# index = np.nonzero(core_mask);
# print("CORE REF (MIN, MEDIAN, MAX, MEAN, STDDEV):",
#         np.min(reference_fibre_in_centre[index]),
#         np.median(reference_fibre_in_centre[index]),
#         np.max(reference_fibre_in_centre[index]),
#         np.mean(reference_fibre_in_centre[index]),
#         np.std(reference_fibre_in_centre[index]));
#
# print("CORE SIMULATED (MIN, MEDIAN, MAX, MEAN, STDDEV):",
#         np.min(test_fibre_in_centre[index]),
#         np.median(test_fibre_in_centre[index]),
#         np.max(test_fibre_in_centre[index]),
#         np.mean(test_fibre_in_centre[index]),
#         np.std(test_fibre_in_centre[index]));
#
# index = np.nonzero(fibre_mask);
# print("FIBRE REF (MIN, MEDIAN, MAX, MEAN, STDDEV):",
#         np.min(reference_fibre_in_centre[index]),
#         np.median(reference_fibre_in_centre[index]),
#         np.max(reference_fibre_in_centre[index]),
#         np.mean(reference_fibre_in_centre[index]),
#         np.std(reference_fibre_in_centre[index]));
#
# print("FIBRE SIMULATED (MIN, MEDIAN, MAX, MEAN, STDDEV):",
#         np.min(test_fibre_in_centre[index]),
#         np.median(test_fibre_in_centre[index]),
#         np.max(test_fibre_in_centre[index]),
#         np.mean(test_fibre_in_centre[index]),
#         np.std(test_fibre_in_centre[index]));
#
# index = np.nonzero(matrix_mask);
# print("MATRIX REF (MIN, MEDIAN, MAX, MEAN, STDDEV):",
#         np.min(reference_fibre_in_centre[index]),
#         np.median(reference_fibre_in_centre[index]),
#         np.max(reference_fibre_in_centre[index]),
#         np.mean(reference_fibre_in_centre[index]),
#         np.std(reference_fibre_in_centre[index]));
#
# print("MATRIX SIMULATED (MIN, MEDIAN, MAX, MEAN, STDDEV):",
#         np.min(test_fibre_in_centre[index]),
#         np.median(test_fibre_in_centre[index]),
#         np.max(test_fibre_in_centre[index]),
#         np.mean(test_fibre_in_centre[index]),
#         np.std(test_fibre_in_centre[index]));
