#!/usr/bin/env python
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
# - `g_reference_CT`: The reference XCT slice;
# - `g_reference_sinogram`: The Radon transform of the reference XCT slice;
# - `g_pixel_spacing_in_micrometre` and `g_pixel_spacing_in_mm`: The physical distance between the centre of two successive pixel;
# - `g_number_of_projections`: The total number of angles in the sinogram;
# - `g_angular_span_in_degrees`: The angular span covered by the sinogram;
# - `g_angular_step`: the angular step; and
# - `g_theta`: The rotation angles in degrees (vertical axis of the sinogram).
#

# In[14]:


g_pixel_spacing_in_micrometre = 1.9;
g_pixel_spacing_in_mm = g_pixel_spacing_in_micrometre * 1e-3;
g_number_of_projections = 900;
g_angular_span_in_degrees = 180.0;
g_angular_step = g_angular_span_in_degrees / g_number_of_projections;
g_theta = np.linspace(0., g_angular_span_in_degrees, g_number_of_projections, endpoint=False);


# ## Load the image data
#
# Load and display the reference projections from a raw binary file, i.e. the target of the registration.

# In[15]:

start_time = time.time();
# Target of the registration
reference_normalised_projections = np.fromfile(args.input[0], dtype=np.float32);
reference_normalised_projections.shape = [g_number_of_projections, int(reference_normalised_projections.shape[0] / g_number_of_projections)];


def computeSinogramFromFlatField(normalised_projections):

    temp = copy.deepcopy(normalised_projections);
    threshold = 0.000001
    temp[temp < threshold] = threshold;

    simulated_sinogram = -np.log(temp);
    simulated_sinogram /= g_pixel_spacing_in_micrometre * gvxr.getUnitOfLength("um") / gvxr.getUnitOfLength("cm");

    return simulated_sinogram;



g_reference_sinogram = computeSinogramFromFlatField(reference_normalised_projections);



# ## CT reconstruction
#
# Now we got a sinogram, we can reconstruct the CT slice. As we used a synchrotron, we can assume we have a parallel source. It means we can use a FBP rather than the FDK algorithm.

# In[ ]:


g_reference_CT = iradon(g_reference_sinogram.T, theta=g_theta, circle=True);

elapsed_time = time.time() - start_time

print("RECONSTRUCTION:", elapsed_time);

# In[ ]:




# ## Set the X-ray simulation environment

# First we create an OpenGL context, here using EGL, i.e. no window.

# In[ ]:


gvxr.createWindow(0, 1, "EGL");
gvxr.setWindowSize(512, 512);


# We set the parameters of the X-ray detector (flat pannel), e.g. number of pixels, pixel, spacing, position and orientation:
#
# ![3D scene to be simulated using gVirtualXray](3d_scene.png)

# In[ ]:


detector_width_in_pixels = g_reference_sinogram.shape[1];
detector_height_in_pixels = 1;
distance_object_detector_in_m =    0.08; # = 80 mm

gvxr.setDetectorPosition(-distance_object_detector_in_m, 0.0, 0.0, "m");
gvxr.setDetectorUpVector(0, 1, 0);
gvxr.setDetectorNumberOfPixels(detector_width_in_pixels, detector_height_in_pixels);
gvxr.setDetectorPixelSize(g_pixel_spacing_in_micrometre, g_pixel_spacing_in_micrometre, "micrometer");


# The beam specturm. Here we have a polychromatic beam, with 97% of the photons at 33 keV, 2% at 66 keV and 1% at 99 keV.

# In[ ]:


energy_spectrum = [(33, 0.97, "keV"), (66, 0.02, "keV"), (99, 0.01, "keV")];

for energy, percentage, unit in energy_spectrum:
    gvxr.addEnergyBinToSpectrum(energy, unit, percentage);


# In[ ]:


energies_in_keV = [];
weights = [];

for energy, percentage, unit in energy_spectrum:
    weights.append(percentage);
    energies_in_keV.append(energy * gvxr.getUnitOfEnergy(unit) / gvxr.getUnitOfEnergy("keV"));

# In[ ]:


# Set up the beam
distance_source_detector_in_m  = 145.0;

gvxr.setSourcePosition(distance_source_detector_in_m - distance_object_detector_in_m,  0.0, 0.0, "mm");
gvxr.usePointSource();
gvxr.useParallelBeam();


# The material properties (chemical composition and density)

# In[ ]:


fibre_radius = 140 / 2; # um
fibre_material = [("Si", 0.5), ("C", 0.5)];
fibre_mu = 2.736; # cm-1
fibre_density = 3.2; # g/cm3

core_radius = 30 / 2; # um
core_material = [("W", 1)];
core_mu = 341.61; # cm-1
core_density = 19.3 # g/cm3

g_matrix_width = 0;
g_matrix_height = 0;
g_matrix_x = 0;
g_matrix_y = 0;
matrix_material = [("Ti", 0.9), ("Al", 0.06), ("V", 0.04)];
matrix_mu = 13.1274; # cm-1
matrix_density = 4.42 # g/cm3


# ### The LSF

# In[ ]:


t = np.arange(-20., 21., 1.);
kernel=lsf(t*41)/lsf(0);
kernel/=kernel.sum();


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


normalised_reference_sinogram = (g_reference_sinogram - g_reference_sinogram.mean()) / g_reference_sinogram.std();
normalised_reference_CT       = (g_reference_CT       - g_reference_CT.mean())       / g_reference_CT.std();


# In[ ]:


# In[ ]:


def setMatrix(apGeneSet):

    gvxr.removePolygonMeshesFromSceneGraph();

    # Matrix
    # Make a cube
    gvxr.makeCube("matrix", 1.0, "micrometer");

    # Translation vector
    x = apGeneSet[0] * detector_width_in_pixels * g_pixel_spacing_in_micrometre;
    y = apGeneSet[1] * detector_width_in_pixels * g_pixel_spacing_in_micrometre;

    gvxr.translateNode("matrix", x, 0.0, y, "micrometer");

    # Rotation angle
    rotation_angle_in_degrees = (apGeneSet[2] + 0.5) * 360.0;
    gvxr.rotateNode("matrix", rotation_angle_in_degrees, 0, 1, 0);

    # Scaling factors
    w = (apGeneSet[3] + 0.5) * detector_width_in_pixels * g_pixel_spacing_in_micrometre;
    h = (apGeneSet[4] + 0.5) * w;
    gvxr.scaleNode("matrix", w, 815, h);

#     print("w:", w, "h:", h, "x:", x, "y:", y);

    # Apply the transformation matrix so that we can save the corresponding STL file
    gvxr.applyCurrentLocalTransformation("matrix");

    # Matrix
    gvxr.setMixture("matrix", "Ti90Al6V4");
    gvxr.setDensity("matrix", matrix_density, "g/cm3");

    gvxr.addPolygonMeshAsInnerSurface("matrix");


# ### Simulate the CT acquisition
#
# Compute the raw projections and save the data. For this  purpose, we define a new function.

# In[ ]:


def tomographyAcquisition():
    raw_projections_in_MeV = [];

    for angle_id in range(0, g_number_of_projections):
        gvxr.resetSceneTransformation();
        gvxr.rotateScene(-g_angular_step * angle_id, 0, 1, 0);

        # Compute the X-ray image
        xray_image = np.array(gvxr.computeXRayImage());

        # Add the projection
        raw_projections_in_MeV.append(xray_image);

    raw_projections_in_MeV = np.array(raw_projections_in_MeV);
    raw_projections_in_keV = raw_projections_in_MeV / gvxr.getUnitOfEnergy("keV");

    return raw_projections_in_keV;


# ### Flat-filed correction
#
# Because the data suffers from a fixed-pattern noise in X-ray imaging in actual experiments, it is necessary to perform the flat-field correction of the raw projections using:
#
# $$normalised\_projections = \frac{raw\_projections − dark\_field}{flat\_field\_image − dark\_field}$$
#
# - $raw\_projections$ are the raw projections with the X-ray beam turned on and with the scanned object,
# - $flat\_field\_image$ is an image with the X-ray beam turned on but without the scanned object, and
# - $dark\_field$ is an image with the X-ray beam turned off.
#
# Note that in our example, $raw\_projections$, $flat\_field\_image$ and $dark\_field$ are in keV whereas $normalised\_projections$ does not have any unit:
#
# $$0 \leq raw\_projections \leq  \sum_E N_0(E) \times E\\0 \leq normalised\_projections \leq 1$$
#
# We define a new function to compute the flat-field correction.

# In[ ]:


def flatFieldCorrection(raw_projections_in_keV):
    dark_field_image = np.zeros(raw_projections_in_keV.shape);
    flat_field_image = np.zeros(raw_projections_in_keV.shape);

    # Retrieve the total energy
    total_energy = 0.0;
    energy_bins = gvxr.getEnergyBins("keV");
    photon_count_per_bin = gvxr.getPhotonCountEnergyBins();

    for energy, count in zip(energy_bins, photon_count_per_bin):
        total_energy += energy * count;
    flat_field_image = np.ones(raw_projections_in_keV.shape) * total_energy;

    normalised_projections = (raw_projections_in_keV - dark_field_image) / (flat_field_image - dark_field_image);

    return normalised_projections;


# In[ ]:

def laplacian(x, sig):
    kernel = (np.power(x, 2.) / np.power(sig, 4.) - 1. / np.power(sig, 2.)) * np.exp(-np.power(x, 2.) / (2 * np.power(sig, 2.)));

    # Make sure the sum of all the kernel elements is NULL
    index_positive = kernel > 0.0;
    index_negative = kernel < 0.0;
    sum_positive = kernel[index_positive].sum();
    sum_negative = kernel[index_negative].sum();

    kernel[index_negative] = -kernel[index_negative] / sum_negative * sum_positive;

    return kernel;

def getLBuffer(object):

    # An empty L-buffer
    L_buffer = [];

    # Get the line of L-buffer for each angle
    for angle_id in range(0, g_number_of_projections):
        gvxr.resetSceneTransformation();
        gvxr.rotateScene(-g_angular_step * angle_id, 0, 1, 0);

        # Compute the X-ray image
        line_of_L_buffer = np.array(gvxr.computeLBuffer(object));

        # Add the projection
        L_buffer.append(line_of_L_buffer);

    # Return as a numpy array
    return np.array(L_buffer);

def simulateSinogram(sigma = None, k = None):


    # Do not simulate the phase contrast using a Laplacian
    if isinstance(sigma, type(None)) or isinstance(k, type(None)):

        # Get the raw projections in keV
        raw_projections_in_keV = tomographyAcquisition();

    # Simulate the phase contrast using a Laplacian
    else:

        # Create the convolution filter
        pixel_range = np.linspace(-value_range, value_range, num=int(num_samples), endpoint=True)
        laplace = laplacian(pixel_range, sigma);

        # Store the L-buffers
        L_buffer_set = {};

        # Look at all the children of the root node
        for label in ["core", "fibre", "matrix"]:
            # Get its L-buffer
            L_buffer_set[label] = getLBuffer(label);

        # For each energy in the beam spectrum
        attenuation = {};
        attenuation_fibre = {};
        projection_per_energy_channel = {};

        # Create a blank image
        raw_projections_in_keV = np.zeros(L_buffer_set["fibre"].shape);

        for energy, photon_count in zip(gvxr.getEnergyBins("keV"), gvxr.getPhotonCountEnergyBins()):

            # Create a blank image
            attenuation[energy] = np.zeros(L_buffer_set["fibre"].shape);

            # Look at all the children of the root node
            #for label in ["core", "fibre", "matrix"]:
            for label in ["core", "fibre", "matrix"]:
            #for label in ["fibre"]:
                # Get mu for this object for this energy
                mu = gvxr.getLinearAttenuationCoefficient(label, energy, "keV");

                # Compute mu * x
                temp = L_buffer_set[label] * mu;
                attenuation[energy] += temp;

                if label == "fibre":
                    attenuation_fibre[energy] = temp;
                elif label == "matrix":
                    attenuation_fibre[energy] += temp;

            # Store the projection for this energy channel
            projection_per_energy_channel[energy] = energy * photon_count * np.exp(-attenuation[energy]);

        # Create the raw projections
        raw_projections_in_keV = np.zeros(L_buffer_set["fibre"].shape);

        for energy in gvxr.getEnergyBins("keV"):

            # Perform the convolution on the attenuation
            shape = [L_buffer_set["fibre"].shape[0], L_buffer_set["fibre"].shape[1], L_buffer_set["fibre"].shape[2]];
            phase_contrast_image = [];

            for y in range(attenuation_fibre[energy].shape[1]):
                for x in range(attenuation_fibre[energy].shape[0]):
                    # phase_contrast_image.append(np.convolve(attenuation_fibre[energy][x][y], laplace, mode='same'));
                    phase_contrast_image.append(np.convolve(attenuation[energy][x][y], laplace, mode='same'));

            phase_contrast_image = np.array(phase_contrast_image);
            phase_contrast_image.shape = raw_projections_in_keV.shape;

            # Normalise it
            phase_contrast_image /= max(np.max(phase_contrast_image), abs(np.min(phase_contrast_image)));

            raw_projections_in_keV += projection_per_energy_channel[energy] - k * phase_contrast_image;

    # Apply the LSF line by line
    for z in range(raw_projections_in_keV.shape[0]):
        for y in range(raw_projections_in_keV.shape[1]):
            raw_projections_in_keV[z][y] = np.convolve(raw_projections_in_keV[z][y], kernel, mode='same');

    normalised_projections = flatFieldCorrection(raw_projections_in_keV);
    simulated_sinogram = computeSinogramFromFlatField(normalised_projections);

    return simulated_sinogram, normalised_projections, raw_projections_in_keV;


# In[ ]:


use_fibres = False;
def fitnessFunctionCube(x):
    global best_fitness, matrix_id, g_reference_sinogram, centroid_set;
    setMatrix(x);

    # Load the cores and fibres
    if use_fibres:
        setFibres(centroid_set);

    # Simulate a sinogram
    simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram();
    normalised_simulated_sinogram = (simulated_sinogram - simulated_sinogram.mean()) / simulated_sinogram.std();


    # Compute the fitness function
    MAE = np.mean(np.abs(np.subtract(normalised_simulated_sinogram.flatten(), normalised_reference_sinogram.flatten())));
    #MAE = np.mean(np.abs(np.subtract(g_reference_sinogram.flatten(), simulated_sinogram.flatten())));
#     ZNCC = np.mean(np.multiply(normalised_simulated_sinogram.flatten(), normalised_reference_sinogram.flatten()));

    return MAE;


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
    es.optimize(fitnessFunctionCube);

    current_best = copy.deepcopy(es.result.xbest); # [-0.12174177  0.07941929 -0.3949529  -0.18708068 -0.23998638]
    np.savetxt(output_directory + "/cube1.dat", current_best, header='x,y,rotation_angle,w,h');
    elapsed_time = time.time() - start_time
    print("CUBE1", elapsed_time);













# ### Apply the result of the registration

# In[30]:


# Save the result
setMatrix(current_best);


simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram();

simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
reconstruction_CT_matrix = iradon(simulated_sinogram.T, theta=g_theta, circle=True);

volume = sitk.GetImageFromArray(reconstruction_CT_matrix);
volume.SetSpacing([g_pixel_spacing_in_mm, g_pixel_spacing_in_mm, g_pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/reconstruction_CT_matrix1.mha", useCompression=True);


print("Matrix1 params:", current_best);
normalised_reconstruction_CT_matrix = (reconstruction_CT_matrix - reconstruction_CT_matrix.mean()) / reconstruction_CT_matrix.std();
ZNCC_CT = np.mean(np.multiply(normalised_reconstruction_CT_matrix.flatten(), normalised_reference_CT.flatten()));
print("Matrix1 CT ZNCC:", ZNCC_CT);


comp_equalized = compare_images(g_reference_CT, reconstruction_CT_matrix, method='checkerboard');
volume = sitk.GetImageFromArray(comp_equalized);
volume.SetSpacing([g_pixel_spacing_in_mm, g_pixel_spacing_in_mm, g_pixel_spacing_in_mm]);
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


def float2uint8(anImage):
    uchar_image = copy.deepcopy(anImage);
    uchar_image -= np.min(uchar_image);
    uchar_image /= np.max(uchar_image);
    uchar_image *= 255;
    return uchar_image.astype(np.uint8);


# We blur the CT scan using a bilateral filter. It preserves edges.

# In[35]:


start_time = time.time()

# Convert in UINT8 and into a SITK image
volume = sitk.GetImageFromArray(float2uint8(g_reference_CT));
volume.SetSpacing([g_pixel_spacing_in_mm, g_pixel_spacing_in_mm, g_pixel_spacing_in_mm]);

# Apply the Otsu's method
otsu_filter = sitk.OtsuThresholdImageFilter();
otsu_filter.SetInsideValue(0);
otsu_filter.SetOutsideValue(1);
seg = otsu_filter.Execute(volume);

# Print the corresponding threshold
print("Threshold:", otsu_filter.GetThreshold());


# In[40]:



# ### Clean up

# In[42]:


cleaned_thresh_img = sitk.BinaryOpeningByReconstruction(seg, [3, 3, 3])
cleaned_thresh_img = sitk.BinaryClosingByReconstruction(cleaned_thresh_img, [3, 3, 3])


# In[43]:




# In[44]:


# ### Size of objects
#
#
# The radius of a tungsten core is 30 / 2 um. The pixel spacing is 1.9 um. The radius in number of pixels is $15/1.9  \approx  7.89$. The area of a core is $(15/1.9)^2  \pi  \approx 196$ pixels.

# In[45]:
# ## Mark each potential tungsten corewith unique label

# In[46]:


# ### Object Analysis

# Once we have the segmented objects we look at their shapes and the intensity distributions inside the objects. Note that sizes are in millimetres.

# In[47]:


shape_stats = sitk.LabelShapeStatisticsImageFilter()
shape_stats.ComputeOrientedBoundingBoxOn()
shape_stats.Execute(sitk.ConnectedComponent(cleaned_thresh_img))

intensity_stats = sitk.LabelIntensityStatisticsImageFilter()
intensity_stats.Execute(sitk.ConnectedComponent(cleaned_thresh_img), volume)


# In[48]:


centroid_set = [];

for i in shape_stats.GetLabels():
    centroid_set.append(cleaned_thresh_img.TransformPhysicalPointToIndex(shape_stats.GetCentroid(i)));


# In[49]:


def setFibres(aCentroidSet):

    global core_radius;
    global fibre_radius;

    # Add the geometries
    gvxr.emptyMesh("fibre");
    gvxr.emptyMesh("core");

    number_of_sectors = 100;

    fibre_centre = [0, 0];

    for i, cyl in enumerate(aCentroidSet):


        x = g_pixel_spacing_in_micrometre * -(cyl[0] - g_reference_CT.shape[1] / 2 + 0.5);
        y = g_pixel_spacing_in_micrometre * (cyl[1] - g_reference_CT.shape[0] / 2 + 0.5);

        fibre_centre[0] += x;
        fibre_centre[1] += y;

        gvxr.emptyMesh("fibre_" + str(i));
        gvxr.emptyMesh("core_"  + str(i));

        gvxr.makeCylinder("fibre_" + str(i), number_of_sectors, 815.0, fibre_radius, "micrometer");
        gvxr.makeCylinder("core_"  + str(i), number_of_sectors, 815.0,  core_radius, "micrometer");

        gvxr.translateNode("fibre_" + str(i), y, 0.0, x, "micrometer");
        gvxr.translateNode("core_"  + str(i), y, 0.0, x, "micrometer");

        gvxr.applyCurrentLocalTransformation("fibre_" + str(i));
        gvxr.applyCurrentLocalTransformation("core_" + str(i));

        gvxr.subtractMesh("matrix", "fibre_" + str(i));
        gvxr.subtractMesh("fibre_" + str(i), "core_" + str(i));


        #gvxr.saveSTLfile("fibre_" + str(i), "Tutorial2/outputs/fibre_" + str(i) + ".stl");
        #gvxr.saveSTLfile("core_" + str(i),  "Tutorial2/outputs/core_"  + str(i) + ".stl");

        gvxr.addMesh("fibre", "fibre_" + str(i));
        gvxr.addMesh("core",  "core_"  + str(i));

    fibre_centre[0] /= len(aCentroidSet);
    fibre_centre[1] /= len(aCentroidSet);

    gvxr.setColor("fibre", 1.0, 0.0, 0.0, 1.0);
    gvxr.setColor("core",  1.0, 0.0, 1.0, 1.0);

    #gvxr.setLinearAttenuationCoefficient("fibre", fibre_mu, "cm-1");
    gvxr.setCompound("fibre", "SiC");
    gvxr.setDensity("fibre", fibre_density, "g/cm3");

    #gvxr.setLinearAttenuationCoefficient("core", core_mu, "cm-1");
    gvxr.setElement("core", "W");

    gvxr.addPolygonMeshAsInnerSurface("core");
    gvxr.addPolygonMeshAsInnerSurface("fibre");


# In[50]:

#
# ### Optimise fibre radius

# In[56]:


def fitnessFunctionFibres(x):
    global best_fitness;
    global radius_fibre_id;
    global fibre_radius;
    global core_radius;

    # Get the radii
    fibre_radius = x[0];
    core_radius = fibre_radius * x[1];

    # Load the matrix
    setMatrix(current_best);

    # Load the cores and fibres
    setFibres(centroid_set);

    # Simulate a sinogram
    simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram();
    normalised_simulated_sinogram = (simulated_sinogram - simulated_sinogram.mean()) / simulated_sinogram.std();
    MAE = np.mean(np.abs(normalised_simulated_sinogram.flatten() - normalised_reference_sinogram.flatten()));
    #MAE = np.mean(np.abs(np.subtract(g_reference_sinogram.flatten(), simulated_sinogram.flatten())));
    # ZNCC = np.mean(np.multiply(normalised_simulated_sinogram.flatten(), normalised_reference_sinogram.flatten()));

    # Reconstruct the corresponding CT slice
#     theta = g_theta / 180.0 * math.pi;
#     rot_center = int(simulated_sinogram.shape[2]/2);
#     reconstruction_tomopy = tomopy.recon(simulated_sinogram, theta, center=rot_center, algorithm="gridrec", sinogram_order=False);


    #simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
    #reconstruction_tomopy = iradon(simulated_sinogram.T, theta=g_theta, circle=True);
    #normalised_simulated_CT = (reconstruction_tomopy - reconstruction_tomopy.mean()) / reconstruction_tomopy.std();
    #MAE_CT = np.mean(np.abs(normalised_simulated_CT.flatten() - normalised_reference_CT.flatten()));
    #ZNCC_CT = np.mean(np.multiply(normalised_simulated_CT.flatten(), normalised_reference_CT.flatten()));

    # Save the data
    fitness = MAE;
    # if best_fitness > fitness:
    #     best_fitness = fitness;




    return fitness;


# In[57]:

    # An individual is made of two floating point numbers:
    # - the radius of the SiC fibre
    # - the ratio    radius of the W core / radius of the SiC fibre


# The registration has already been performed. Load the results.
if os.path.isfile(output_directory + "/fibre_radius1.dat"):
    temp = np.loadtxt(output_directory + "/fibre_radius1.dat");
    core_radius = temp[0];
    fibre_radius = temp[1];
# Perform the registration using CMA-ES
else:
    fibre_radius = 140 / 2; # um
    core_radius = 30 / 2; # um
    ratio = core_radius / fibre_radius;

    x0 = [fibre_radius, ratio];
    bounds = [[5, 0.01], [1.5 * fibre_radius, 0.95]];

    best_fitness = sys.float_info.max;
    radius_fibre_id = 0;

    opts = cma.CMAOptions()
    opts.set('tolfun', 1e-3);
    opts['tolx'] = 1e-3;
    opts['bounds'] = bounds;
    #opts['seed'] = 987654321;
    # opts['maxiter'] = 5;

    es = cma.CMAEvolutionStrategy(x0, 0.9, opts);
    es.optimize(fitnessFunctionFibres);
    elapsed_time = time.time() - start_time
    print("FIBRES1",elapsed_time);
    fibre_radius = es.result.xbest[0];
    core_radius = fibre_radius * es.result.xbest[1];

    np.savetxt(output_directory + "/fibre_radius1.dat", [core_radius, fibre_radius], header='core_radius_in_um,fibre_radius_in_um');



# Load the matrix
setMatrix(current_best);

# Load the cores and fibres
setFibres(centroid_set);





simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram();

simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
reconstruction_CT_fibres = iradon(simulated_sinogram.T, theta=g_theta, circle=True);

volume = sitk.GetImageFromArray(reconstruction_CT_fibres);
volume.SetSpacing([g_pixel_spacing_in_mm, g_pixel_spacing_in_mm, g_pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/reconstruction_CT_fibres1.mha", useCompression=True);



print("Radii1:", core_radius, fibre_radius);
normalised_reconstruction_CT_fibres = (reconstruction_CT_fibres - reconstruction_CT_fibres.mean()) / reconstruction_CT_fibres.std();
ZNCC_CT = np.mean(np.multiply(normalised_reconstruction_CT_fibres.flatten(), normalised_reference_CT.flatten()));
print("Fibres1 CT ZNCC:", ZNCC_CT);

comp_equalized = compare_images(g_reference_CT, reconstruction_CT_fibres, method='checkerboard');
volume = sitk.GetImageFromArray(comp_equalized);
volume.SetSpacing([g_pixel_spacing_in_mm, g_pixel_spacing_in_mm, g_pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/compare_reconstruction_CT_fibres1.mha", useCompression=True);

comp_equalized = compare_images(normalised_reference_CT, normalised_reconstruction_CT_fibres, method='checkerboard');
comp_equalized -= np.min(comp_equalized);
comp_equalized /= np.max(comp_equalized);
comp_equalized *= 255;
comp_equalized = np.array(comp_equalized, dtype=np.uint8);
io.imsave(output_directory + "/compare_reconstruction_CT_fibres1.png", comp_equalized)








use_fibres = True;
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
    es.optimize(fitnessFunctionCube);

    current_best = copy.deepcopy(es.result.xbest); # [-0.12174177  0.07941929 -0.3949529  -0.18708068 -0.23998638]
    np.savetxt(output_directory + "/cube2.dat", current_best, header='x,y,rotation_angle,w,h');
    elapsed_time = time.time() - start_time
    print("CUBE2", elapsed_time);





# Load the matrix
setMatrix(current_best);

# Load the cores and fibres
setFibres(centroid_set);


simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram();

simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
reconstruction_CT_matrix = iradon(simulated_sinogram.T, theta=g_theta, circle=True);

volume = sitk.GetImageFromArray(reconstruction_CT_matrix);
volume.SetSpacing([g_pixel_spacing_in_mm, g_pixel_spacing_in_mm, g_pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/reconstruction_CT_matrix2.mha", useCompression=True);


print("Matrix2 params:", current_best);
normalised_reconstruction_CT_matrix = (reconstruction_CT_matrix - reconstruction_CT_matrix.mean()) / reconstruction_CT_matrix.std();
ZNCC_CT = np.mean(np.multiply(normalised_reconstruction_CT_matrix.flatten(), normalised_reference_CT.flatten()));
print("Matrix2 CT ZNCC:", ZNCC_CT);

comp_equalized = compare_images(g_reference_CT, reconstruction_CT_matrix, method='checkerboard');
volume = sitk.GetImageFromArray(comp_equalized);
volume.SetSpacing([g_pixel_spacing_in_mm, g_pixel_spacing_in_mm, g_pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/compare_reconstruction_CT_matrix2.mha", useCompression=True);

comp_equalized = compare_images(normalised_reference_CT, normalised_reconstruction_CT_matrix, method='checkerboard');
comp_equalized -= np.min(comp_equalized);
comp_equalized /= np.max(comp_equalized);
comp_equalized *= 255;
comp_equalized = np.array(comp_equalized, dtype=np.uint8);
io.imsave(output_directory + "/compare_reconstruction_CT_matrix2.png", comp_equalized)








# Exhaustive local search to refine the centre of each cylinder
roi_length = 40;
new_centroid_set = [];
for i, cyl in enumerate(centroid_set):

    centre = [
        cyl[0],
        cyl[1]
    ];

    # extract ROI from reference image
    reference_image = copy.deepcopy(g_reference_CT[centre[1] - roi_length:centre[1] + roi_length, centre[0] - roi_length:centre[0] + roi_length]);

    # Normalise ROI
    reference_image -= reference_image.mean();
    reference_image /= reference_image.std();

    best_ZNCC = -1;
    best_x_offset = 0;
    best_y_offset = 0;

    for y in range(-10, 11):
        for x in range(-10, 11):

            centre = [
                cyl[0] + x,
                cyl[1] + y
            ];

            # extract ROI from test image
            test_image = copy.deepcopy(reconstruction_CT_fibres[centre[1] - roi_length:centre[1] + roi_length, centre[0] - roi_length:centre[0] + roi_length]);

            # Normalise ROI
            test_image -= test_image.mean();
            test_image /= test_image.std();

            # Compare the ROIs
            zncc = np.mean(np.multiply(reference_image.flatten(), test_image.flatten()));

            if best_ZNCC < zncc:
                best_ZNCC = zncc;
                best_x_offset = x;
                best_y_offset = y;

    # Correct the position of the centre of the fibre
    new_centroid_set. append([cyl[0] - best_x_offset, cyl[1] - best_y_offset]);

centroid_set = new_centroid_set;






# Load the matrix
setMatrix(current_best);

# Load the cores and fibres
setFibres(centroid_set);





simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram();

simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
reconstruction_CT_fibres = iradon(simulated_sinogram.T, theta=g_theta, circle=True);

volume = sitk.GetImageFromArray(reconstruction_CT_fibres);
volume.SetSpacing([g_pixel_spacing_in_mm, g_pixel_spacing_in_mm, g_pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/reconstruction_CT_fibres2.mha", useCompression=True);



print("Radii2:", core_radius, fibre_radius);
normalised_reconstruction_CT_fibres = (reconstruction_CT_fibres - reconstruction_CT_fibres.mean()) / reconstruction_CT_fibres.std();
ZNCC_CT = np.mean(np.multiply(normalised_reconstruction_CT_fibres.flatten(), normalised_reference_CT.flatten()));
print("Fibres2 CT ZNCC:", ZNCC_CT);

comp_equalized = compare_images(g_reference_CT, reconstruction_CT_fibres, method='checkerboard');
volume = sitk.GetImageFromArray(comp_equalized);
volume.SetSpacing([g_pixel_spacing_in_mm, g_pixel_spacing_in_mm, g_pixel_spacing_in_mm]);
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
    fibre_radius = temp[1];
# Perform the registration using CMA-ES
else:
    ratio = core_radius / fibre_radius;

    x0 = [fibre_radius, ratio];
    bounds = [[5, 0.01], [1.5 * fibre_radius, 0.95]];

    best_fitness = sys.float_info.max;
    radius_fibre_id = 0;

    opts = cma.CMAOptions()
    opts.set('tolfun', 1e-3);
    opts['tolx'] = 1e-3;
    opts['bounds'] = bounds;
    #opts['seed'] = 987654321;
    # opts['maxiter'] = 5;

    es = cma.CMAEvolutionStrategy(x0, 0.25, opts);
    es.optimize(fitnessFunctionFibres);
    elapsed_time = time.time() - start_time
    print("FIBRES2",elapsed_time);
    fibre_radius = es.result.xbest[0];
    core_radius = fibre_radius * es.result.xbest[1];

    np.savetxt(output_directory + "/fibre_radius2.dat", [core_radius, fibre_radius], header='core_radius_in_um,fibre_radius_in_um');



# Load the matrix
setMatrix(current_best);

# Load the cores and fibres
setFibres(centroid_set);





simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram();

simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
reconstruction_CT_fibres = iradon(simulated_sinogram.T, theta=g_theta, circle=True);

volume = sitk.GetImageFromArray(reconstruction_CT_fibres);
volume.SetSpacing([g_pixel_spacing_in_mm, g_pixel_spacing_in_mm, g_pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/reconstruction_CT_fibres3.mha", useCompression=True);



print("Radii3:", core_radius, fibre_radius);
normalised_reconstruction_CT_fibres = (reconstruction_CT_fibres - reconstruction_CT_fibres.mean()) / reconstruction_CT_fibres.std();
ZNCC_CT = np.mean(np.multiply(normalised_reconstruction_CT_fibres.flatten(), normalised_reference_CT.flatten()));
print("Fibres3 CT ZNCC:", ZNCC_CT);

comp_equalized = compare_images(g_reference_CT, reconstruction_CT_fibres, method='checkerboard');
volume = sitk.GetImageFromArray(comp_equalized);
volume.SetSpacing([g_pixel_spacing_in_mm, g_pixel_spacing_in_mm, g_pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/compare_reconstruction_CT_fibres3.mha", useCompression=True);

comp_equalized = compare_images(normalised_reference_CT, normalised_reconstruction_CT_fibres, method='checkerboard');
comp_equalized -= np.min(comp_equalized);
comp_equalized /= np.max(comp_equalized);
comp_equalized *= 255;
comp_equalized = np.array(comp_equalized, dtype=np.uint8);
io.imsave(output_directory + "/compare_reconstruction_CT_fibres3.png", comp_equalized)

















def fitnessFunctionLaplacian(x):
    global best_fitness;
    global laplacian_id;
    global fibre_radius;
    global core_radius;
    global value_range;
    global num_samples;
    global best_centre;

    sigma = x[0];
    k = x[1];
    # value_range = x[2];
    # num_samples = x[3];

    # Get the radii
    # fibre_radius = x[4];
    fibre_radius = x[2];

    # Load the matrix
    setMatrix(current_best);

    # Load the cores and fibres
    setFibres(centroid_set);

    # Simulate a sinogram
    simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram(sigma, k);
    normalised_simulated_sinogram = (simulated_sinogram - simulated_sinogram.mean()) / simulated_sinogram.std();
    MAE_sinogram = np.mean(np.abs(normalised_simulated_sinogram.flatten() - normalised_reference_sinogram.flatten()));

    # Reconstruct the corresponding CT slice
    simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
    CT_laplacian = iradon(simulated_sinogram.T, theta=g_theta, circle=True);
    normalised_CT_laplacian = (CT_laplacian - CT_laplacian.mean()) / CT_laplacian.std();

    reference_image = copy.deepcopy(g_reference_CT[best_centre[1] - roi_length:best_centre[1] + roi_length, best_centre[0] - roi_length:best_centre[0] + roi_length]);
    test_image = copy.deepcopy(CT_laplacian[best_centre[1] - roi_length:best_centre[1] + roi_length, best_centre[0] - roi_length:best_centre[0] + roi_length]);

    # MAE_sinogram = np.mean(np.abs(np.subtract(g_reference_sinogram.flatten(), simulated_sinogram.flatten())));
    # ZNCC_sinogram = np.mean(np.multiply(normalised_simulated_sinogram.flatten(), normalised_reference_sinogram.flatten()));
    #
    # normalised_simulated_sinogram.shape = (normalised_simulated_sinogram.size // normalised_simulated_sinogram.shape[2], normalised_simulated_sinogram.shape[2]);
    #
    # SSIM_sinogram = ssim(normalised_simulated_sinogram, normalised_reference_sinogram, data_range=normalised_reference_sinogram.max() - normalised_reference_sinogram.min())

    # Reconstruct the corresponding CT slice
#     theta = g_theta / 180.0 * math.pi;
#     rot_center = int(simulated_sinogram.shape[2]/2);
#     reconstruction_tomopy = tomopy.recon(simulated_sinogram, theta, center=rot_center, algorithm="gridrec", sinogram_order=False);





    # simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
    # CT_laplacian = iradon(simulated_sinogram.T, theta=g_theta, circle=True);


    # offset = min(np.min(CT_laplacian), np.min(g_reference_CT));
    #
    # reconstruction_CT_laplacian = CT_laplacian - offset;
    # reference_CT = g_reference_CT - offset;
    # reconstruction_CT_laplacian += 0.5;
    # reference_CT += 0.5;
    #
    # reconstruction_CT_laplacian = np.log(reconstruction_CT_laplacian);
    # reference_CT = np.log(reference_CT);
    #
    # normalised_simulated_CT = (reconstruction_CT_laplacian - reconstruction_CT_laplacian.mean()) / reconstruction_CT_laplacian.std();
    # temp_reference_CT = (reference_CT - reference_CT.mean()) / reference_CT.std();
    #
    MAE_CT = np.mean(np.abs(np.subtract(g_reference_CT.flatten(), CT_laplacian.flatten())));
    ZNCC_CT = np.mean(np.multiply(normalised_reference_CT.flatten(), normalised_CT_laplacian.flatten()));
    # SSIM_CT = ssim(normalised_simulated_CT, temp_reference_CT, data_range=temp_reference_CT.max() - temp_reference_CT.min())
    #
    #
    #
    # index = np.nonzero(core_mask);
    # diff_core = math.pow(np.mean(reference_image[index]) - np.mean(test_image[index]), 2);
    #
    # index = np.nonzero(fibre_mask);
    # diff_fibre = math.pow(np.mean(reference_image[index]) - np.mean(test_image[index]), 2);
    #
    # index = np.nonzero(matrix_mask);
    # diff_matrix = math.pow(np.mean(reference_image[index]) - np.mean(test_image[index]), 2);
    #
    #
    #
    # reference_image = (reference_image - reference_image.mean()) / reference_image.std();
    # test_image = (test_image - test_image.mean()) / test_image.std();

    MAE_fibre = np.mean(np.abs(np.subtract(reference_image.flatten(), test_image.flatten())));
    ZNCC_fibre = np.mean(np.multiply(reference_image.flatten(), test_image.flatten()));
    # SSIM_fibre = ssim(reference_image, test_image, data_range=reference_image.max() - reference_image.min())

    fitness = MAE_sinogram;
    fitness = MAE_fibre;
    fitness = MAE_CT;
    #fitness = 1 / (ZNCC_CT + 1);

    if best_fitness > fitness:
        best_fitness = fitness;


        volume = sitk.GetImageFromArray(CT_laplacian);
        volume.SetSpacing([g_pixel_spacing_in_mm, g_pixel_spacing_in_mm, g_pixel_spacing_in_mm]);
        sitk.WriteImage(volume, output_directory + "/reconstruction_CT_laplacian_" + str(laplacian_id) + ".mha", useCompression=True);

        volume = sitk.GetImageFromArray(CT_laplacian[best_centre[1] - roi_length:best_centre[1] + roi_length,
                                        best_centre[0] - roi_length:best_centre[0] + roi_length]);
        volume.SetSpacing([g_pixel_spacing_in_mm, g_pixel_spacing_in_mm, g_pixel_spacing_in_mm]);
        sitk.WriteImage(volume, output_directory + "/reconstruction_CT_laplacian_fibre_centre_" + str(laplacian_id) + ".mha", useCompression=True);

        comp_equalized = compare_images(reference_image, test_image, method='checkerboard');
        volume = sitk.GetImageFromArray(comp_equalized)
        sitk.WriteImage(volume, output_directory + "/laplacian_comp_fibre_" + str(laplacian_id) + ".mha", useCompression=True);

        comp_equalized -= np.min(comp_equalized);
        comp_equalized /= np.max(comp_equalized);
        comp_equalized *= 255;
        comp_equalized = np.array(comp_equalized, dtype=np.uint8);
        io.imsave(output_directory + "/laplacian_comp_fibre_" + str(laplacian_id) + ".png", comp_equalized);

        comp_equalized = compare_images(g_reference_CT, CT_laplacian, method='checkerboard');
        volume = sitk.GetImageFromArray(comp_equalized)
        sitk.WriteImage(volume, output_directory + "/laplacian_comp_slice_" + str(laplacian_id) + ".mha", useCompression=True);

        comp_equalized -= np.min(comp_equalized);
        comp_equalized /= np.max(comp_equalized);
        comp_equalized *= 255;
        comp_equalized = np.array(comp_equalized, dtype=np.uint8);
        io.imsave(output_directory + "/laplacian_comp_slice_" + str(laplacian_id) + ".png", comp_equalized);

        laplacian_id += 1;

    return fitness;

# Find the cylinder in the centre of the image
best_centre = None;
best_distance = sys.float_info.max;

for centre in centroid_set:
    distance = math.pow(centre[0] - g_reference_CT.shape[1] / 2,2 ) + math.pow(centre[1] - g_reference_CT.shape[0] / 2, 2);

    if best_distance > distance:
        best_distance = distance;
        best_centre = copy.deepcopy(centre);

value_range = 6;
num_samples = 15;

# The registration has already been performed. Load the results.
if os.path.isfile(output_directory + "/laplacian.dat"):
    temp = np.loadtxt(output_directory + "/laplacian.dat");
    sigma = temp[0];
    k = temp[1];
    fibre_radius = temp[2];
# Perform the registration using CMA-ES
else:
    sigma = 0.5
    k = 1.5;

    x0 = [sigma,k,fibre_radius];
    bounds = [[0.00001, 0.0, 0.75 * fibre_radius], [1, 15, 1.25 * fibre_radius]];

    best_fitness = sys.float_info.max;
    laplacian_id = 0;

    opts = cma.CMAOptions()
    opts.set('tolfun', 1e-3);
    opts['tolx'] = 1e-3;
    opts['bounds'] = bounds;
    #opts['seed'] = 987654321;
    # opts['maxiter'] = 5;
    opts['CMA_stds'] = [0.25, 1.0, fibre_radius * 0.025];


    es = cma.CMAEvolutionStrategy(x0, 0.25, opts);
    es.optimize(fitnessFunctionLaplacian);
    elapsed_time = time.time() - start_time
    print("LAPLACIAN",elapsed_time);

    sigma = es.result.xbest[0];
    k = es.result.xbest[1];
    fibre_radius = es.result.xbest[2];

    np.savetxt(output_directory + "/laplacian.dat", [sigma, k, fibre_radius], header='sigma,k,fibre_radius_in_um');



# Load the matrix
setMatrix(current_best);

# Load the cores and fibres
setFibres(centroid_set);

pixel_range = np.linspace(-value_range, value_range, num=int(num_samples), endpoint=True)
laplace = laplacian(pixel_range, sigma);

simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram(sigma, k);

simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
reconstruction_CT_laplacian = iradon(simulated_sinogram.T, theta=g_theta, circle=True);

volume = sitk.GetImageFromArray(reconstruction_CT_laplacian);
volume.SetSpacing([g_pixel_spacing_in_mm, g_pixel_spacing_in_mm, g_pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/reconstruction_CT_laplacian.mha", useCompression=True);

np.savetxt(output_directory + "/laplace.txt", laplace)


print("Laplacian:", sigma, k, fibre_radius);
normalised_reconstruction_CT_laplacian = (reconstruction_CT_laplacian - reconstruction_CT_laplacian.mean()) / reconstruction_CT_laplacian.std();
ZNCC_CT = np.mean(np.multiply(normalised_reconstruction_CT_laplacian.flatten(), normalised_reference_CT.flatten()));
print("Laplacian CT ZNCC:", ZNCC_CT);

comp_equalized = compare_images(g_reference_CT, reconstruction_CT_laplacian, method='checkerboard');
volume = sitk.GetImageFromArray(comp_equalized);
volume.SetSpacing([g_pixel_spacing_in_mm, g_pixel_spacing_in_mm, g_pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/compare_reconstruction_CT_laplacian.mha", useCompression=True);

comp_equalized = compare_images(normalised_reference_CT, normalised_reconstruction_CT_laplacian, method='checkerboard');
comp_equalized -= np.min(comp_equalized);
comp_equalized /= np.max(comp_equalized);
comp_equalized *= 255;
comp_equalized = np.array(comp_equalized, dtype=np.uint8);
io.imsave(output_directory + "/compare_reconstruction_CT_laplacian.png", comp_equalized)



















roi_length = 40;
fibre_radius_in_px = fibre_radius / g_pixel_spacing_in_micrometre
core_radius_in_px = core_radius / g_pixel_spacing_in_micrometre

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return np.array(mask, dtype=bool);










test_fibre_in_centre = np.array(copy.deepcopy(reconstruction_CT_laplacian[best_centre[1] - roi_length:best_centre[1] + roi_length, best_centre[0] - roi_length:best_centre[0] + roi_length]));

reference_fibre_in_centre = np.array(copy.deepcopy(g_reference_CT[best_centre[1] - roi_length:best_centre[1] + roi_length, best_centre[0] - roi_length:best_centre[0] + roi_length]));






volume = sitk.GetImageFromArray(test_fibre_in_centre);
volume.SetSpacing([g_pixel_spacing_in_mm, g_pixel_spacing_in_mm, g_pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/reconstruction_CT_fibre_in_centre.mha", useCompression=True);




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
