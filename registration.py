#!/usr/bin/env python3
# coding: utf-8


import time
import argparse  # Process the cmd line
import copy, math, os, glob, sys

import numpy as np

from skimage.transform import iradon

import matplotlib
matplotlib.use('TkAGG')   # generate postscript output by default

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from matplotlib.widgets import Slider, Button, RadioButtons

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
    parser.add_argument('--metrics', help='MAE, RMSE, or MRE', nargs=1, type=str, required=True);

    parser.add_argument('--normalisation', dest='normalisation', action='store_true')
    parser.add_argument('--no-normalisation', dest='normalisation', action='store_false')
    parser.set_defaults(normalisation=True)

    parser.add_argument('--sinogram', dest='sinogram', action='store_true')
    parser.add_argument('--projections', dest='sinogram', action='store_false')
    parser.set_defaults(sinogram=True)

    return parser.parse_args()




DEBUG_FLAG=False;

args = processCmdLine();

output_directory = args.output[0];
if not os.path.exists(output_directory):
    os.makedirs(output_directory);

Simulation.output_directory = output_directory;


Simulation.metrics_function = args.metrics[0];
Simulation.use_normalisation = args.normalisation;
Simulation.use_sinogram = args.sinogram;



print("METRICS: ", Simulation.metrics_function);
if Simulation.use_normalisation:
    print("WITH ZERO-MEAN, UNIT-VARIANCE NORMALISATION");
else:
    print("WITHOUT ZERO-MEAN, UNIT-VARIANCE NORMALISATION");

if Simulation.use_sinogram:
    print("COMPUTED ON SINOGRAMS");
else:
    print("COMPUTED ON PROJECTIONS");



# ## Load the image data
#
# Load and display the reference projections from a raw binary file, i.e. the target of the registration.

# In[15]:



# Target of the registration
start_time = time.time();
reference_normalised_projections, reference_sinogram, reference_CT, normalised_reference_sinogram, normalised_reference_CT = Simulation.createTargetFromRawSinogram(args.input[0]);
elapsed_time = time.time() - start_time
print("RECONSTRUCTION:", elapsed_time);

volume = sitk.GetImageFromArray(reference_CT);
volume.SetSpacing([Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/reference_CT.mha", useCompression=True);

volume = sitk.GetImageFromArray(normalised_reference_CT);
volume.SetSpacing([Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/normalised_reference_CT.mha", useCompression=True);

# Set the X-ray simulation environment
Simulation.initGVXR();

np.savetxt(output_directory + "/LSF_original.txt", Simulation.lsf_kernel);





# sigma_core = 0.15;
# sigma_fibre = 1.5;
# sigma_matrix = 0.2;
#
# k_core = 5;
# k_fibre = 1;
# k_matrix = 2.0;
#
#
# x = np.linspace(-Simulation.value_range, Simulation.value_range, num=int(Simulation.num_samples), endpoint=True)
# laplacian_kernel = Simulation.laplacian(x, sigma_fibre);
#
# print(np.sum(laplacian_kernel));
#
# fig, ax = plt.subplots()
# plt.subplots_adjust(left=0.25, bottom=0.25)
# l, = plt.plot(x, laplacian_kernel * k_fibre, lw=2);
#
# axcolor = 'lightgoldenrodyellow'
# axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
# axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
#
# delta_f = 0.05
# sfreq = Slider(axfreq, 'Sigma', 0.0001, 3.5, valinit=sigma_fibre, valstep=delta_f)
# samp = Slider(axamp, 'k', 0.0, 5000.0, valinit=k_fibre)
#
# resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
# button = Button(resetax, 'Apply', color=axcolor, hovercolor='0.975')
#
#
#
#
# fibre_L_buffer  = sitk.GetArrayFromImage(sitk.ReadImage(output_directory + "/l_buffer_fibre.mha"));
# core_L_buffer   = sitk.GetArrayFromImage(sitk.ReadImage(output_directory + "/l_buffer_core.mha"));
# matrix_L_buffer = sitk.GetArrayFromImage(sitk.ReadImage(output_directory + "/l_buffer_matrix.mha"));
#
#
# def simulateAndReconstruct():
#     global sigma_fibre, k_fibre, samp, sfreq, laplacian_kernel, x;
#     global fibre_L_buffer, core_L_buffer, matrix_L_buffer;
#
#     k_fibre = samp.val
#     sigma_fibre = sfreq.val
#     laplacian_kernel = Simulation.laplacian(x, sigma_fibre);
#
#     projection = np.zeros(fibre_L_buffer.shape);
#     phase_contrast_image = np.zeros(fibre_L_buffer.shape);
#
#     total_energy = 0.0;
#
#     attenuation =  core_L_buffer * 348.9097883430308 + matrix_L_buffer * 16.53631368289138;
#     projection = 32.999999821186066 * 0.9700000286102295 * np.exp(-attenuation);
#
#     attenuation =  core_L_buffer * 56.46307119094464 + matrix_L_buffer * 2.708343134657077;
#     projection += 65.99999964237213 * 0.019999999552965164 * np.exp(-attenuation);
#     #
#     attenuation =  core_L_buffer * 87.90873756872163 + matrix_L_buffer * 1.2023011012123404;
#     projection += 98.9999994635582 * 0.009999999776482582 * np.exp(-attenuation);
#
#
#     for y in range(fibre_L_buffer.shape[0]):
#         phase_contrast_image[y] += np.convolve((fibre_L_buffer)[y], laplacian_kernel, mode='same');
#         # phase_contrast_image[y] += np.convolve((core_L_buffer)[y], laplacian_kernel, mode='same');
#         # phase_contrast_image[y] += np.convolve((matrix_L_buffer)[y], laplacian_kernel, mode='same');
#
#     phase_contrast_image *= k_fibre;
#
#     volume = sitk.GetImageFromArray(phase_contrast_image);
#     # volume.SetSpacing([pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);
#     sitk.WriteImage(volume, output_directory + "/phase_contrast_image.mha", useCompression=True);
#
#
#     projection -= phase_contrast_image;
#
#     projection /= 32.999999821186066 * 0.9700000286102295 #+ 65.99999964237213 * 0.019999999552965164 + 98.9999994635582 * 0.009999999776482582;
#
#     threshold = 1e-6
#     projection[projection < threshold] = threshold;
#
#     simulated_sinogram = -np.log(projection);
#     simulated_sinogram /= Simulation.pixel_spacing_in_micrometre * gvxr.getUnitOfLength("um") / gvxr.getUnitOfLength("cm");
#
#
#
#     volume = sitk.GetImageFromArray(simulated_sinogram);
#     sitk.WriteImage(volume, output_directory + "/sinogram_with_laplacian.mha", useCompression=True);
#
#     CT_laplacian = iradon(simulated_sinogram.T, theta=Simulation.theta, circle=True);
#
#     volume = sitk.GetImageFromArray(CT_laplacian);
#     # volume.SetSpacing([pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);
#     sitk.WriteImage(volume, output_directory + "/CT_with_laplacian.mha", useCompression=True);
#
#     return CT_laplacian;
#
#
# CT_laplacian = simulateAndReconstruct();
# fig2, ax2 = plt.subplots()
# fig_img = plt.imshow(CT_laplacian)
# fig_img.set_cmap('gray')
#
#
# def update(val):
#     global sigma_fibre, k_fibre, x, laplacian_kernel;
#
#     k_fibre = samp.val
#     sigma_fibre = sfreq.val
#     laplacian_kernel = Simulation.laplacian(x, sigma_fibre);
#
#     l.set_ydata(laplacian_kernel * k_fibre)
#     ax.set_ylim(np.min(laplacian_kernel * k_fibre), np.max(laplacian_kernel * k_fibre))
#
#     fig.canvas.draw_idle()
#
#
# sfreq.on_changed(update)
# samp.on_changed(update)
#
#
#
# def reset(event):
#
#     global sigma_fibre, k_fibre;
#
#     fig_img.set_data(simulateAndReconstruct());
#     fig2.canvas.draw_idle()
#
#
#
#
#
#
# button.on_clicked(reset)



# plt.show()

#







































################################################################################
##### OPTIMISE THE CUBE FROM SCRATCH
################################################################################

Simulation.use_fibres = False;

# The registration has already been performed. Load the results.
if os.path.isfile(output_directory + "/cube1.dat"):
    Simulation.matrix_geometry_parameters = np.loadtxt(output_directory + "/cube1.dat");
# Perform the registration using CMA-ES
else:
    ZNCC_CT=0.0;

    # Restart if the ZNCC is too low
    while ZNCC_CT < 0.40:
        print("OPTIMISE THE CUBE FROM SCRATCH")
        start_time = time.time()

        Simulation.best_fitness = sys.float_info.max;
        matrix_id = 0;

        opts = cma.CMAOptions()
        opts.set('tolfun', 1e-4);
        opts['tolx'] = 1e-4;
        opts['bounds'] = [5*[-0.5], 5*[0.5]];
        #opts['seed'] = 123456789;
        # opts['maxiter'] = 5;

        es = cma.CMAEvolutionStrategy([0.0, 0.0, 0.0, 0.256835938, 0.232903226], 0.5, opts);
        es.optimize(Simulation.fitnessFunctionCube);

        Simulation.matrix_geometry_parameters = copy.deepcopy(es.result.xbest); # [-0.12174177  0.07941929 -0.3949529  -0.18708068 -0.23998638]
        np.savetxt(output_directory + "/cube1.dat", Simulation.matrix_geometry_parameters, header='x,y,rotation_angle,w,h');
        elapsed_time = time.time() - start_time

        # Apply the result of the registration
        Simulation.setMatrix(Simulation.matrix_geometry_parameters);

        # Simulate the corresponding CT aquisition
        simulated_sinogram, normalised_projections, raw_projections_in_keV = Simulation.simulateSinogram();

        # Store the corresponding results on the disk
        ZNCC_CT, CT_slice_from_simulated_sinogram = Simulation.reconstructAndStoreResults(simulated_sinogram, output_directory + "/matrix1");
    print("CUBE1", elapsed_time);

# Apply the result of the registration
Simulation.setMatrix(Simulation.matrix_geometry_parameters);

if not DEBUG_FLAG:
    # Simulate the corresponding CT aquisition
    simulated_sinogram, normalised_projections, raw_projections_in_keV = Simulation.simulateSinogram();

    temp = copy.deepcopy(normalised_projections);
    temp.shape = reference_normalised_projections.shape

    volume = sitk.GetImageFromArray(temp);
    sitk.WriteImage(volume, output_directory + "/Matrix1-normalised_projections.mha", useCompression=True);

    # Store the corresponding results on the disk
    ZNCC_CT, CT_slice_from_simulated_sinogram = Simulation.reconstructAndStoreResults(simulated_sinogram, output_directory + "/matrix1");
    print("Matrix1 params:", Simulation.matrix_geometry_parameters);
    print("Matrix1 CT ZNCC:", ZNCC_CT);




################################################################################
##### FIND CIRCLES
################################################################################
start_time = time.time()

Simulation.centroid_set = Simulation.findCircles(Simulation.reference_CT);

# Find the position of the fibre that in the most in the centre of the CT slice
Simulation.findFibreInCentreOfCtSlice();

# Create the binary masks
reference_fibre_in_centre = np.array(copy.deepcopy(Simulation.reference_CT[Simulation.cylinder_position_in_centre_of_slice[1] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[1] + Simulation.roi_length, Simulation.cylinder_position_in_centre_of_slice[0] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[0] + Simulation.roi_length]));
mask_shape = reference_fibre_in_centre.shape;
core_mask, fibre_mask, matrix_mask = Simulation.createMasks(mask_shape);

# Save the binary masks
core_mask = ndimage.binary_erosion(core_mask).astype(core_mask.dtype);
np.savetxt(output_directory + "/core_mask1.txt", core_mask);

fibre_mask = ndimage.binary_erosion(fibre_mask).astype(core_mask.dtype);
np.savetxt(output_directory + "/fibre_mask1.txt", fibre_mask);

matrix_mask = ndimage.binary_erosion(matrix_mask).astype(core_mask.dtype);
np.savetxt(output_directory + "/matrix_mask1.txt", matrix_mask);


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
    print("OPTIMISE THE RADII AFTER OPTIMISING THE CUBE FROM SCRATCH");
    ratio = Simulation.core_radius / Simulation.fibre_radius;

    x0 = [Simulation.fibre_radius, ratio];
    bounds = [[5, 0.01], [1.5 * Simulation.fibre_radius, 0.95]];

    Simulation.best_fitness = sys.float_info.max;
    radius_fibre_id = 0;

    opts = cma.CMAOptions()
    opts.set('tolfun', 1e-4);
    opts['tolx'] = 1e-4;
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
ZNCC_CT, CT_slice_from_simulated_sinogram = Simulation.reconstructAndStoreResults(simulated_sinogram, output_directory + "/fibres1");

if not DEBUG_FLAG:

    temp = copy.deepcopy(normalised_projections);
    temp.shape = reference_normalised_projections.shape

    volume = sitk.GetImageFromArray(temp);
    sitk.WriteImage(volume, output_directory + "/Fibres1-normalised_projections.mha", useCompression=True);

    # Store the corresponding results on the disk

    print("Fibres1 params:", Simulation.core_radius, Simulation.fibre_radius);
    print("Fibres1 CT ZNCC:", ZNCC_CT);

# Find the fibre in the centre of the reference and simulated slices
test_fibre_in_centre = np.array(copy.deepcopy(CT_slice_from_simulated_sinogram[Simulation.cylinder_position_in_centre_of_slice[1] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[1] + Simulation.roi_length, Simulation.cylinder_position_in_centre_of_slice[0] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[0] + Simulation.roi_length]));
reference_fibre_in_centre = np.array(copy.deepcopy(Simulation.reference_CT[Simulation.cylinder_position_in_centre_of_slice[1] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[1] + Simulation.roi_length, Simulation.cylinder_position_in_centre_of_slice[0] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[0] + Simulation.roi_length]));
Simulation.printMuStatistics("Fibres1", reference_fibre_in_centre, test_fibre_in_centre, core_mask, fibre_mask, matrix_mask);





################################################################################
##### RECENTRE THE CYLINDERS
################################################################################

Simulation.centroid_set = Simulation.refineCentrePositions(Simulation.centroid_set, CT_slice_from_simulated_sinogram);

# Find the position of the fibre that in the most in the centre of the CT slice
Simulation.findFibreInCentreOfCtSlice();

# Create the binary masks
reference_fibre_in_centre = np.array(copy.deepcopy(Simulation.reference_CT[Simulation.cylinder_position_in_centre_of_slice[1] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[1] + Simulation.roi_length, Simulation.cylinder_position_in_centre_of_slice[0] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[0] + Simulation.roi_length]));
mask_shape = reference_fibre_in_centre.shape;
core_mask, fibre_mask, matrix_mask = Simulation.createMasks(mask_shape);

# Save the binary masks
core_mask = ndimage.binary_erosion(core_mask).astype(core_mask.dtype);
np.savetxt(output_directory + "/core_mask2.txt", core_mask);

fibre_mask = ndimage.binary_erosion(fibre_mask).astype(core_mask.dtype);
np.savetxt(output_directory + "/fibre_mask2.txt", fibre_mask);

matrix_mask = ndimage.binary_erosion(matrix_mask).astype(core_mask.dtype);
np.savetxt(output_directory + "/matrix_mask2.txt", matrix_mask);

# Apply the result of the registration
Simulation.setMatrix(Simulation.matrix_geometry_parameters);
Simulation.setFibres(Simulation.centroid_set);

if not DEBUG_FLAG:
    # Simulate the corresponding CT aquisition
    simulated_sinogram, normalised_projections, raw_projections_in_keV = Simulation.simulateSinogram();

    temp = copy.deepcopy(normalised_projections);
    temp.shape = reference_normalised_projections.shape

    volume = sitk.GetImageFromArray(temp);
    sitk.WriteImage(volume, output_directory + "/Fibres2-normalised_projections.mha", useCompression=True);

    # Store the corresponding results on the disk
    ZNCC_CT, CT_slice_from_simulated_sinogram = Simulation.reconstructAndStoreResults(simulated_sinogram, output_directory + "/fibres2");
    print("Fibres2 CT ZNCC:", ZNCC_CT);

    # Find the fibre in the centre of the reference and simulated slices
    test_fibre_in_centre = np.array(copy.deepcopy(CT_slice_from_simulated_sinogram[Simulation.cylinder_position_in_centre_of_slice[1] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[1] + Simulation.roi_length, Simulation.cylinder_position_in_centre_of_slice[0] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[0] + Simulation.roi_length]));
    reference_fibre_in_centre = np.array(copy.deepcopy(Simulation.reference_CT[Simulation.cylinder_position_in_centre_of_slice[1] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[1] + Simulation.roi_length, Simulation.cylinder_position_in_centre_of_slice[0] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[0] + Simulation.roi_length]));
    Simulation.printMuStatistics("Fibres2", reference_fibre_in_centre, test_fibre_in_centre, core_mask, fibre_mask, matrix_mask);


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

    Simulation.best_fitness = sys.float_info.max;
    radius_fibre_id = 0;

    opts = cma.CMAOptions()
    opts.set('tolfun', 1e-4);
    opts['tolx'] = 1e-4;
    opts['bounds'] = bounds;
    #opts['seed'] = 987654321;
    # opts['maxiter'] = 5;

    es = cma.CMAEvolutionStrategy(x0, 0.25, opts);
    es.optimize(Simulation.fitnessFunctionFibres);
    elapsed_time = time.time() - start_time
    print("FIBRES3",elapsed_time);
    Simulation.fibre_radius = es.result.xbest[0];
    core_radius = Simulation.fibre_radius * es.result.xbest[1];

    np.savetxt(output_directory + "/fibre_radius3.dat", [Simulation.core_radius, Simulation.fibre_radius], header='core_radius_in_um,fibre_radius_in_um');

# Apply the result of the registration
Simulation.setMatrix(Simulation.matrix_geometry_parameters);
Simulation.setFibres(Simulation.centroid_set);

# Simulate the corresponding CT aquisition
simulated_sinogram, normalised_projections, raw_projections_in_keV = Simulation.simulateSinogram();

if not DEBUG_FLAG:

    # Store the corresponding results on the disk
    ZNCC_CT, CT_slice_from_simulated_sinogram = Simulation.reconstructAndStoreResults(simulated_sinogram, output_directory + "/fibres3");
    print("Fibres3 params:", Simulation.core_radius, Simulation.fibre_radius);
    print("Fibres3 CT ZNCC:", ZNCC_CT);

    temp = copy.deepcopy(normalised_projections);
    temp.shape = reference_normalised_projections.shape

    volume = sitk.GetImageFromArray(temp);
    sitk.WriteImage(volume, output_directory + "/Fibres3-normalised_projections.mha", useCompression=True);

    # Find the fibre in the centre of the reference and simulated slices
    test_fibre_in_centre = np.array(copy.deepcopy(CT_slice_from_simulated_sinogram[Simulation.cylinder_position_in_centre_of_slice[1] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[1] + Simulation.roi_length, Simulation.cylinder_position_in_centre_of_slice[0] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[0] + Simulation.roi_length]));
    reference_fibre_in_centre = np.array(copy.deepcopy(Simulation.reference_CT[Simulation.cylinder_position_in_centre_of_slice[1] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[1] + Simulation.roi_length, Simulation.cylinder_position_in_centre_of_slice[0] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[0] + Simulation.roi_length]));
    Simulation.printMuStatistics("Fibres3", reference_fibre_in_centre, test_fibre_in_centre, core_mask, fibre_mask, matrix_mask);



################################################################################
##### BEAM SPECTRUM
################################################################################

# The registration has already been performed. Load the results.
if os.path.isfile(output_directory + "/spectrum.dat"):
    temp = np.loadtxt(output_directory + "/spectrum.dat");

    # The beam specturm. Here we have a polychromatic beam.
    Simulation.energy_spectrum = [(33, temp[0], "keV"), (66, temp[1], "keV"), (99, temp[2], "keV")];

# Perform the registration using CMA-ES
else:
    ratio = Simulation.core_radius / Simulation.fibre_radius;

    x0 = [0.97, 0.2, 0.1];
    bounds = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];

    Simulation.best_fitness = sys.float_info.max;

    opts = cma.CMAOptions()
    opts.set('tolfun', 1e-4);
    opts['tolx'] = 1e-4;
    opts['bounds'] = bounds;

    es = cma.CMAEvolutionStrategy(x0, 0.25, opts);
    es.optimize(Simulation.fitnessHarmonics);
    elapsed_time = time.time() - start_time
    print("HARMONICS",elapsed_time);

    total = es.result.xbest[0] + es.result.xbest[1] + es.result.xbest[2];
    Simulation.energy_spectrum = [(33, es.result.xbest[0] / total, "keV"), (66, es.result.xbest[1] / total, "keV"), (99, es.result.xbest[2] / total, "keV")];

    np.savetxt(output_directory + "/spectrum.dat", [es.result.xbest[0] / total, es.result.xbest[1] / total, es.result.xbest[2] / total], header='weight of main energy,weight of first order harmonics,weight of second order harmonics');

# Apply the result of the registration
gvxr.resetBeamSpectrum();
for energy, percentage, unit in Simulation.energy_spectrum:
    gvxr.addEnergyBinToSpectrum(energy, unit, percentage);

# Simulate the corresponding CT aquisition
simulated_sinogram, normalised_projections, raw_projections_in_keV = Simulation.simulateSinogram();

if not DEBUG_FLAG:

    # Store the corresponding results on the disk
    ZNCC_CT, CT_slice_from_simulated_sinogram = Simulation.reconstructAndStoreResults(simulated_sinogram, output_directory + "/fibres3");
    print("Harmonics params:", es.result.xbest[0] / total, es.result.xbest[1] / total, es.result.xbest[2] / total);
    print("Harmonics CT ZNCC:", ZNCC_CT);

    temp = copy.deepcopy(normalised_projections);
    temp.shape = reference_normalised_projections.shape

    volume = sitk.GetImageFromArray(temp);
    sitk.WriteImage(volume, output_directory + "/harmonics-normalised_projections.mha", useCompression=True);

    # Find the fibre in the centre of the reference and simulated slices
    test_fibre_in_centre = np.array(copy.deepcopy(CT_slice_from_simulated_sinogram[Simulation.cylinder_position_in_centre_of_slice[1] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[1] + Simulation.roi_length, Simulation.cylinder_position_in_centre_of_slice[0] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[0] + Simulation.roi_length]));
    reference_fibre_in_centre = np.array(copy.deepcopy(Simulation.reference_CT[Simulation.cylinder_position_in_centre_of_slice[1] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[1] + Simulation.roi_length, Simulation.cylinder_position_in_centre_of_slice[0] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[0] + Simulation.roi_length]));
    Simulation.printMuStatistics("Harmonics", reference_fibre_in_centre, test_fibre_in_centre, core_mask, fibre_mask, matrix_mask);




################################################################################
##### OPTIMISE THE NOISE LEVEL
################################################################################

# The registration has already been performed. Load the results.
if os.path.isfile(output_directory + "/poisson-noise.dat") and os.path.isfile(output_directory + "/poisson-noise.dat"):
    temp = np.loadtxt(output_directory + "/poisson-noise.dat");
    Simulation.bias = temp[0];
    Simulation.gain = temp[1];
    Simulation.scale = temp[2];

# Perform the registration using CMA-ES
else:

    Simulation.reference_noise_ROI = copy.deepcopy(reference_normalised_projections[450:550,0:140]);

    temp = copy.deepcopy(normalised_projections);
    temp.shape = reference_normalised_projections.shape

    volume = sitk.GetImageFromArray(temp);
    sitk.WriteImage(volume, output_directory + "/poisson-noise-normalised_projections.mha", useCompression=True);

    Simulation.normalised_projections_ROI = temp[450:550,0:140];

    Simulation.reference_noise_ROI_stddev = 0;
    for y in range(Simulation.reference_noise_ROI.shape[0]):
        Simulation.reference_noise_ROI_stddev += Simulation.reference_noise_ROI[y].std();
    Simulation.reference_noise_ROI_stddev /= Simulation.reference_noise_ROI.shape[0];

    volume = sitk.GetImageFromArray(Simulation.reference_noise_ROI);
    sitk.WriteImage(volume, output_directory + "/reference_noise_ROI.mha", useCompression=True);

    volume = sitk.GetImageFromArray(Simulation.normalised_projections_ROI);
    sitk.WriteImage(volume, output_directory + "/normalised_projections_ROI.mha", useCompression=True);

    Simulation.bias = 0.0;
    Simulation.gain = 1.0;
    Simulation.scale = 1.0;

    x0 = [Simulation.bias, Simulation.gain, Simulation.scale];

    bounds = [
        [ 0.0, 0.0, 0.0],
        [ 5.0, 10.0, 5.0]
    ];

    Simulation.best_fitness = sys.float_info.max;

    opts = cma.CMAOptions()
    opts.set('tolfun', 1e-8);
    opts['tolx'] = 1e-8;
    opts['bounds'] = bounds;
    #opts['seed'] = 987654321;
    # opts['maxiter'] = 5;
    opts['CMA_stds'] = [1, 1, 1];


    # IND 2.19746627320312	1.16136683253601	0.763740221230013	249.875214879601	0.503314643902767	38.2390121454358	53.2213098006193	    0.084343959685696	0.992778341956513
    # IND 2.07828517522359	8.86453347432242E-05	0.755930952404442	249.992034434311	0.483822894369194	23.3017161885803	53.4611294410208	0.08427650560267	0.992771013383053
    # IND 0.22475860581346976 7.3330722215476225 0.6901736846902663 997.9090436639837 0.629128141527589 807.8482231256484 53.63342291025315 0.07658148986147018 0.9941233938771149
    es = cma.CMAEvolutionStrategy(x0, 0.25, opts);
    es.optimize(Simulation.fitnessFunctionNoise);
    elapsed_time = time.time() - start_time
    print("NOISE", elapsed_time);

    Simulation.bias = es.result.xbest[0];
    Simulation.gain = es.result.xbest[1];
    Simulation.scale = es.result.xbest[2];

    np.savetxt(output_directory + "/poisson-noise.dat", [Simulation.bias, Simulation.gain, Simulation.scale], header='bias, gain, scale');

# Apply the result of the registration
# Poisson noise
map = (normalised_projections + Simulation.bias) * Simulation.gain;
map[map < 0] = 0;
noise_map = np.random.poisson(map);

# Add the noise
normalised_projections += Simulation.scale * noise_map;

simulated_sinogram = Simulation.computeSinogramFromFlatField(normalised_projections);
ZNCC_CT, CT_slice_from_simulated_sinogram = Simulation.reconstructAndStoreResults(simulated_sinogram, output_directory + "/noise");
print("Noise params:", Simulation.bias, Simulation.gain, Simulation.scale);
print("Noise CT ZNCC:", ZNCC_CT);

# Find the fibre in the centre of the reference and simulated slices
test_fibre_in_centre = np.array(copy.deepcopy(CT_slice_from_simulated_sinogram[Simulation.cylinder_position_in_centre_of_slice[1] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[1] + Simulation.roi_length, Simulation.cylinder_position_in_centre_of_slice[0] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[0] + Simulation.roi_length]));
reference_fibre_in_centre = np.array(copy.deepcopy(Simulation.reference_CT[Simulation.cylinder_position_in_centre_of_slice[1] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[1] + Simulation.roi_length, Simulation.cylinder_position_in_centre_of_slice[0] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[0] + Simulation.roi_length]));
Simulation.printMuStatistics("Noise", reference_fibre_in_centre, test_fibre_in_centre, core_mask, fibre_mask, matrix_mask);


temp = copy.deepcopy(normalised_projections);
temp.shape = reference_normalised_projections.shape
volume = sitk.GetImageFromArray(temp);
sitk.WriteImage(volume, output_directory + "/normalised_projections-noisy.mha", useCompression=True);

# if not DEBUG_FLAG:
#
#     # Store the corresponding results on the disk
#     ZNCC_CT, CT_slice_from_simulated_sinogram = Simulation.reconstructAndStoreResults(simulated_sinogram, output_directory + "/laplacian-LSF");
#     print("Laplacian-LSF params:", Simulation.k_core, Simulation.k_fibre, Simulation.k_matrix);
#     print("lsf params:", Simulation.a2, Simulation.b2, Simulation.c2, Simulation.d2, Simulation.e2, Simulation.f2);
#     print("Laplacian-LSF CT ZNCC:", ZNCC_CT);
#
#     pixel_range = np.linspace(-Simulation.value_range, Simulation.value_range, num=int(Simulation.num_samples), endpoint=True);
#
#     for label, sigma, k in zip(["core", "fibre", "matrix"], [Simulation.sigma_core, Simulation.sigma_fibre, Simulation.sigma_matrix], [Simulation.k_core, Simulation.k_fibre, Simulation.k_matrix]):
#         np.savetxt(output_directory + "/laplacian3_kernel_" + label + ".dat", Simulation.laplacian(pixel_range, sigma) * k);
#
#     # Find the fibre in the centre of the reference and simulated slices
#     test_fibre_in_centre = np.array(copy.deepcopy(CT_slice_from_simulated_sinogram[Simulation.cylinder_position_in_centre_of_slice[1] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[1] + Simulation.roi_length, Simulation.cylinder_position_in_centre_of_slice[0] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[0] + Simulation.roi_length]));
#     reference_fibre_in_centre = np.array(copy.deepcopy(Simulation.reference_CT[Simulation.cylinder_position_in_centre_of_slice[1] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[1] + Simulation.roi_length, Simulation.cylinder_position_in_centre_of_slice[0] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[0] + Simulation.roi_length]));
#
#     #  Save the corresponding fibres
#     volume = sitk.GetImageFromArray(test_fibre_in_centre);
#     volume.SetSpacing([Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm]);
#     sitk.WriteImage(volume, output_directory + "/laplacian-LSF_simulated_fibre_in_centre.mha", useCompression=True);
#
#     volume = sitk.GetImageFromArray(reference_fibre_in_centre);
#     volume.SetSpacing([Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm]);
#     sitk.WriteImage(volume, output_directory + "/laplacian-LSF_reference_fibre_in_centre.mha", useCompression=True);
#
#     # Save the corresponding diagonal profiles
#     np.savetxt(output_directory + "/laplacian-LSF_profile_reference_fibre_in_centre.txt", np.diag(reference_fibre_in_centre));
#     np.savetxt(output_directory + "/laplacian-LSF_profile_simulated_fibre_in_centre.txt", np.diag(test_fibre_in_centre));
#
#
#     Simulation.printMuStatistics("Laplacian-LSF", reference_fibre_in_centre, test_fibre_in_centre, core_mask, fibre_mask, matrix_mask);





################################################################################
##### OPTIMISE THE LAPLACIAN OF THE CORE, FIBRE, and MATRIX, as well as the radius of the fibre
################################################################################

# The registration has already been performed. Load the results.
if os.path.isfile(output_directory + "/laplacian1.dat"):
    temp = np.loadtxt(output_directory + "/laplacian1.dat");
    Simulation.sigma_core = temp[0];
    Simulation.k_core = temp[1];
    Simulation.sigma_fibre = temp[2];
    Simulation.k_fibre = temp[3];
    Simulation.sigma_matrix = temp[4];
    Simulation.k_matrix = temp[5];
    Simulation.core_radius = temp[6];
    Simulation.fibre_radius = temp[7];

# Perform the registration using CMA-ES
else:

    Simulation.sigma_core = 5.;
    Simulation.sigma_fibre = 0.75;
    Simulation.sigma_matrix = 0.6;

    Simulation.k_core = 1000;
    Simulation.k_fibre = 1000;
    Simulation.k_matrix = 1000.0;

    x0 = [Simulation.sigma_core, Simulation.k_core, Simulation.sigma_fibre, Simulation.k_fibre, Simulation.sigma_matrix, Simulation.k_matrix, Simulation.core_radius, Simulation.fibre_radius];
    bounds = [[0.005, 0.0, 0.005, 0.0, 0.005, 0.0, 0.95 * Simulation.core_radius, 0.95 * Simulation.fibre_radius],
              [10.0, 2000, 2.5, 2000, 2.5, 2000, 1.15 * Simulation.core_radius, 1.15 * Simulation.fibre_radius]];

    Simulation.best_fitness = sys.float_info.max;
    laplacian_id = 0;

    opts = cma.CMAOptions()
    opts.set('tolfun', 1e-4);
    opts['tolx'] = 1e-4;
    opts['bounds'] = bounds;
    #opts['seed'] = 987654321;
    # opts['maxiter'] = 5;
    opts['CMA_stds'] = [0.25, 20.25, 0.25, 20.25, 0.25, 20.25, Simulation.core_radius * 0.1, Simulation.fibre_radius * 0.1];


    # IND 2.19746627320312	1.16136683253601	0.763740221230013	249.875214879601	0.503314643902767	38.2390121454358	53.2213098006193	    0.084343959685696	0.992778341956513
    # IND 2.07828517522359	8.86453347432242E-05	0.755930952404442	249.992034434311	0.483822894369194	23.3017161885803	53.4611294410208	0.08427650560267	0.992771013383053
    # IND 0.22475860581346976 7.3330722215476225 0.6901736846902663 997.9090436639837 0.629128141527589 807.8482231256484 53.63342291025315 0.07658148986147018 0.9941233938771149
    es = cma.CMAEvolutionStrategy(x0, 0.25, opts);
    es.optimize(Simulation.fitnessFunctionLaplacian);
    elapsed_time = time.time() - start_time
    print("LAPLACIAN1",elapsed_time);

    Simulation.sigma_core = es.result.xbest[0];
    Simulation.k_core = es.result.xbest[1];
    Simulation.sigma_fibre = es.result.xbest[2];
    Simulation.k_fibre = es.result.xbest[3];
    Simulation.sigma_matrix = es.result.xbest[4];
    Simulation.k_matrix = es.result.xbest[5];
    Simulation.core_radius = es.result.xbest[6];
    Simulation.fibre_radius = es.result.xbest[7];

    np.savetxt(output_directory + "/laplacian1.dat", [Simulation.sigma_core, Simulation.k_core, Simulation.sigma_fibre, Simulation.k_fibre, Simulation.sigma_matrix, Simulation.k_matrix, Simulation.core_radius, Simulation.fibre_radius], header='sigma_core, k_core, sigma_fibre, k_fibre, sigma_matrix, k_matrix, core_radius_in_um, fibre_radius_in_um');

# Apply the result of the registration
Simulation.setMatrix(Simulation.matrix_geometry_parameters);
Simulation.setFibres(Simulation.centroid_set);


# Create the binary masks
reference_fibre_in_centre = np.array(copy.deepcopy(Simulation.reference_CT[Simulation.cylinder_position_in_centre_of_slice[1] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[1] + Simulation.roi_length, Simulation.cylinder_position_in_centre_of_slice[0] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[0] + Simulation.roi_length]));
mask_shape = reference_fibre_in_centre.shape;
core_mask, fibre_mask, matrix_mask = Simulation.createMasks(mask_shape);

# Save the binary masks
core_mask = ndimage.binary_erosion(core_mask).astype(core_mask.dtype);
np.savetxt(output_directory + "/core_mask.txt", core_mask);

fibre_mask = ndimage.binary_erosion(fibre_mask).astype(core_mask.dtype);
np.savetxt(output_directory + "/fibre_mask.txt", fibre_mask);

matrix_mask = ndimage.binary_erosion(matrix_mask).astype(core_mask.dtype);
np.savetxt(output_directory + "/matrix_mask.txt", matrix_mask);


if not DEBUG_FLAG:
    # Simulate the corresponding CT aquisition
    simulated_sinogram, normalised_projections, raw_projections_in_keV = Simulation.simulateSinogram([Simulation.sigma_core, Simulation.sigma_fibre, Simulation.sigma_matrix], [Simulation.k_core, Simulation.k_fibre, Simulation.k_matrix], ["core", "fibre", "matrix"]);

    map = (normalised_projections + Simulation.bias) * Simulation.gain;
    map[map < 0] = 0;
    noise_map = np.random.poisson(map);
    normalised_projections += Simulation.scale * noise_map;
    simulated_sinogram = Simulation.computeSinogramFromFlatField(normalised_projections);

    temp = copy.deepcopy(normalised_projections);
    temp.shape = reference_normalised_projections.shape

    volume = sitk.GetImageFromArray(temp);
    sitk.WriteImage(volume, output_directory + "/laplacian1-normalised_projections.mha", useCompression=True);

    # Store the corresponding results on the disk
    ZNCC_CT, CT_slice_from_simulated_sinogram = Simulation.reconstructAndStoreResults(simulated_sinogram, output_directory + "/laplacian1");
    print("Laplacian1 params:", Simulation.sigma_core, Simulation.sigma_fibre, Simulation.sigma_matrix, Simulation.k_core, Simulation.k_fibre, Simulation.k_matrix, Simulation.core_radius, Simulation.fibre_radius);
    print("Laplacian1 CT ZNCC:", ZNCC_CT);

    pixel_range = np.linspace(-Simulation.value_range, Simulation.value_range, num=int(Simulation.num_samples), endpoint=True);

    for label, sigma, k in zip(["core", "fibre", "matrix"], [Simulation.sigma_core, Simulation.sigma_fibre, Simulation.sigma_matrix], [Simulation.k_core, Simulation.k_fibre, Simulation.k_matrix]):
        np.savetxt(output_directory + "/laplacian_kernel_1_" + label + ".dat", Simulation.laplacian(pixel_range, sigma) * k);

    # Find the fibre in the centre of the reference and simulated slices
    test_fibre_in_centre = np.array(copy.deepcopy(CT_slice_from_simulated_sinogram[Simulation.cylinder_position_in_centre_of_slice[1] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[1] + Simulation.roi_length, Simulation.cylinder_position_in_centre_of_slice[0] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[0] + Simulation.roi_length]));
    reference_fibre_in_centre = np.array(copy.deepcopy(Simulation.reference_CT[Simulation.cylinder_position_in_centre_of_slice[1] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[1] + Simulation.roi_length, Simulation.cylinder_position_in_centre_of_slice[0] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[0] + Simulation.roi_length]));
    Simulation.printMuStatistics("Laplacian1", reference_fibre_in_centre, test_fibre_in_centre, core_mask, fibre_mask, matrix_mask);

    #  Save the corresponding fibres
    volume = sitk.GetImageFromArray(test_fibre_in_centre);
    volume.SetSpacing([Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm]);
    sitk.WriteImage(volume, output_directory + "/laplacian1_simulated_fibre_in_centre.mha", useCompression=True);

    volume = sitk.GetImageFromArray(reference_fibre_in_centre);
    volume.SetSpacing([Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm]);
    sitk.WriteImage(volume, output_directory + "/laplacian1_reference_fibre_in_centre.mha", useCompression=True);

    # Save the corresponding diagonal profiles
    np.savetxt(output_directory + "/laplacian1_profile_reference_fibre_in_centre.txt", np.diag(reference_fibre_in_centre));
    np.savetxt(output_directory + "/laplacian1_profile_simulated_fibre_in_centre.txt", np.diag(test_fibre_in_centre));






################################################################################
##### OPTIMISE THE LAPLACIAN OF THE CORE, FIBRE, and MATRIX, as well as the LSF
################################################################################

# The registration has already been performed. Load the results.
if os.path.isfile(output_directory + "/laplacian3.dat") and os.path.isfile(output_directory + "/lsf2.dat"):
    temp = np.loadtxt(output_directory + "/laplacian3.dat");
    Simulation.k_core = temp[0];
    Simulation.k_fibre = temp[1];
    Simulation.k_matrix = temp[2];

    temp = np.loadtxt(output_directory + "/lsf2.dat");
    Simulation.a2 = temp[0];
    Simulation.b2 = temp[1];
    Simulation.c2 = temp[2];
    Simulation.d2 = temp[3];
    Simulation.e2 = temp[4];
    Simulation.f2 = temp[5];

# Perform the registration using CMA-ES
else:

    a2 = 601.873;
    b2 = 54.9359;
    c2 = -3.58452;
    d2 = 0.469614;
    e2 = 6.32561e+09;
    f2 = 1.0;

    Simulation.best_fitness = sys.float_info.max;

    x0 = [
        Simulation.k_core,
        Simulation.k_fibre,
        Simulation.k_matrix,
        a2, b2, c2, d2, e2, f2
    ];

    bounds = [
        [
            Simulation.k_core-500,
            Simulation.k_fibre-500,
            Simulation.k_matrix-500,
            a2 - a2 / 4.,
            b2 - b2 / 4.,
            c2 + c2 / 4.,
            d2 - d2 / 4.,
            e2 - e2 / 4.,
            f2 - f2/ 4.
        ],
        [
            Simulation.k_core+500,
            Simulation.k_fibre+500,
            Simulation.k_matrix+500,
            a2 + a2 / 4.,
            b2 + b2 / 4.,
            c2 - c2 / 4.,
            d2 + d2 / 4.,
            e2 + e2 / 4.,
            f2 + f2/ 4.
        ]
    ];

    Simulation.best_fitness = sys.float_info.max;
    laplacian_id = 0;

    opts = cma.CMAOptions()
    opts.set('tolfun', 1e-4);
    opts['tolx'] = 1e-4;
    opts['bounds'] = bounds;
    #opts['seed'] = 987654321;
    # opts['maxiter'] = 5;
    opts['CMA_stds'] = [1250 * 0.2, 1250 * 0.2, 1250 * 0.2,
        a2 * 0.2, b2 * 0.2, -c2 * 0.2, d2 * 0.2, e2 * 0.2, f2 * 0.2];


    # IND 2.19746627320312	1.16136683253601	0.763740221230013	249.875214879601	0.503314643902767	38.2390121454358	53.2213098006193	    0.084343959685696	0.992778341956513
    # IND 2.07828517522359	8.86453347432242E-05	0.755930952404442	249.992034434311	0.483822894369194	23.3017161885803	53.4611294410208	0.08427650560267	0.992771013383053
    # IND 0.22475860581346976 7.3330722215476225 0.6901736846902663 997.9090436639837 0.629128141527589 807.8482231256484 53.63342291025315 0.07658148986147018 0.9941233938771149
    es = cma.CMAEvolutionStrategy(x0, 0.25, opts);
    es.optimize(Simulation.fitnessFunctionLaplacianLSF);
    elapsed_time = time.time() - start_time
    print("LAPLACIAN-LSF",elapsed_time);

    Simulation.k_core = es.result.xbest[0];
    Simulation.k_fibre = es.result.xbest[1];
    Simulation.k_matrix = es.result.xbest[2];

    Simulation.a2 = es.result.xbest[3];
    Simulation.b2 = es.result.xbest[4];
    Simulation.c2 = es.result.xbest[5];
    Simulation.d2 = es.result.xbest[6];
    Simulation.e2 = es.result.xbest[7];
    Simulation.f2 = es.result.xbest[8];

    np.savetxt(output_directory + "/laplacian3.dat", [Simulation.k_core, Simulation.k_fibre, Simulation.k_matrix], header='k_core, k_fibre, k_matrix');
    np.savetxt(output_directory + "/lsf2.dat", [Simulation.a2, Simulation.b2, Simulation.c2, Simulation.d2, Simulation.e2, Simulation.f2], header='a2, b2, c2, d2, e2, f2');


# Apply the result of the registration
# The response of the detector as the line-spread function (LSF)
t = np.arange(-20., 21., 1.);
Simulation.lsf_kernel=lsf(t*41, Simulation.a2, Simulation.b2, Simulation.c2, Simulation.d2, Simulation.e2, Simulation.f2);
Simulation.lsf_kernel/=Simulation.lsf_kernel.sum();
np.savetxt(output_directory + "/LSF_optimised.txt", Simulation.lsf_kernel);

# Simulate the corresponding CT aquisition
simulated_sinogram, normalised_projections, raw_projections_in_keV = Simulation.simulateSinogram([Simulation.sigma_core, Simulation.sigma_fibre, Simulation.sigma_matrix], [Simulation.k_core, Simulation.k_fibre, Simulation.k_matrix], ["core", "fibre", "matrix"]);

map = (normalised_projections + Simulation.bias) * Simulation.gain;
map[map < 0] = 0;
noise_map = np.random.poisson(map);
normalised_projections += Simulation.scale * noise_map;
simulated_sinogram = Simulation.computeSinogramFromFlatField(normalised_projections);

temp = copy.deepcopy(normalised_projections);
temp.shape = reference_normalised_projections.shape

volume = sitk.GetImageFromArray(temp);
sitk.WriteImage(volume, output_directory + "/LSF_optimised-normalised_projections.mha", useCompression=True);

if not DEBUG_FLAG:

    # Store the corresponding results on the disk
    ZNCC_CT, CT_slice_from_simulated_sinogram = Simulation.reconstructAndStoreResults(simulated_sinogram, output_directory + "/laplacian-LSF");
    print("Laplacian-LSF params:", Simulation.k_core, Simulation.k_fibre, Simulation.k_matrix);
    print("lsf params:", Simulation.a2, Simulation.b2, Simulation.c2, Simulation.d2, Simulation.e2, Simulation.f2);
    print("Laplacian-LSF CT ZNCC:", ZNCC_CT);

    pixel_range = np.linspace(-Simulation.value_range, Simulation.value_range, num=int(Simulation.num_samples), endpoint=True);

    for label, sigma, k in zip(["core", "fibre", "matrix"], [Simulation.sigma_core, Simulation.sigma_fibre, Simulation.sigma_matrix], [Simulation.k_core, Simulation.k_fibre, Simulation.k_matrix]):
        np.savetxt(output_directory + "/laplacian3_kernel_" + label + ".dat", Simulation.laplacian(pixel_range, sigma) * k);

    # Find the fibre in the centre of the reference and simulated slices
    test_fibre_in_centre = np.array(copy.deepcopy(CT_slice_from_simulated_sinogram[Simulation.cylinder_position_in_centre_of_slice[1] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[1] + Simulation.roi_length, Simulation.cylinder_position_in_centre_of_slice[0] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[0] + Simulation.roi_length]));
    reference_fibre_in_centre = np.array(copy.deepcopy(Simulation.reference_CT[Simulation.cylinder_position_in_centre_of_slice[1] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[1] + Simulation.roi_length, Simulation.cylinder_position_in_centre_of_slice[0] - Simulation.roi_length:Simulation.cylinder_position_in_centre_of_slice[0] + Simulation.roi_length]));

    #  Save the corresponding fibres
    volume = sitk.GetImageFromArray(test_fibre_in_centre);
    volume.SetSpacing([Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm]);
    sitk.WriteImage(volume, output_directory + "/laplacian-LSF_simulated_fibre_in_centre.mha", useCompression=True);

    volume = sitk.GetImageFromArray(reference_fibre_in_centre);
    volume.SetSpacing([Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm, Simulation.pixel_spacing_in_mm]);
    sitk.WriteImage(volume, output_directory + "/laplacian-LSF_reference_fibre_in_centre.mha", useCompression=True);

    # Save the corresponding diagonal profiles
    np.savetxt(output_directory + "/laplacian-LSF_profile_reference_fibre_in_centre.txt", np.diag(reference_fibre_in_centre));
    np.savetxt(output_directory + "/laplacian-LSF_profile_simulated_fibre_in_centre.txt", np.diag(test_fibre_in_centre));


    Simulation.printMuStatistics("Laplacian-LSF", reference_fibre_in_centre, test_fibre_in_centre, core_mask, fibre_mask, matrix_mask);
