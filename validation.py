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

from skimage.transform import iradon
from skimage.util import compare_images

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
    simulated_sinogram = -np.log(normalised_projections);
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

    gvxr.addPolygonMeshAsOuterSurface("matrix");


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

    # Apply the LSF line by line
    for z in range(raw_projections_in_MeV.shape[0]):
        for y in range(raw_projections_in_MeV.shape[1]):
            raw_projections_in_MeV[z][y] = np.convolve(raw_projections_in_MeV[z][y], kernel, mode='same');

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


def simulateSinogram():
    raw_projections_in_keV = tomographyAcquisition();
    normalised_projections = flatFieldCorrection(raw_projections_in_keV);
    simulated_sinogram = computeSinogramFromFlatField(normalised_projections);

    return simulated_sinogram, normalised_projections, raw_projections_in_keV;


# In[ ]:


def fitnessFunction(x):
    global best_fitness, matrix_id;
    setMatrix(x);

    # Simulate a sinogram
    simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram();
    normalised_simulated_sinogram = (simulated_sinogram - simulated_sinogram.mean()) / simulated_sinogram.std();


    # Compute the fitness function
    MAE = np.mean(np.abs(np.subtract(normalised_simulated_sinogram.flatten(), normalised_reference_sinogram.flatten())));
#     ZNCC = np.mean(np.multiply(normalised_simulated_sinogram.flatten(), normalised_reference_sinogram.flatten()));

    return MAE;


# In[29]:
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
es.optimize(fitnessFunction);

current_best = copy.deepcopy(es.result.xbest); # [-0.12174177  0.07941929 -0.3949529  -0.18708068 -0.23998638]
np.savetxt(output_directory + "/cube.dat", current_best, header='x,y,rotation_angle,w,h');
elapsed_time = time.time() - start_time
print("CUBE", elapsed_time);

# ### Apply the result of the registration

# In[30]:


# Save the result
setMatrix(current_best);


simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram();

simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
reconstruction_CT_matrix = iradon(simulated_sinogram.T, theta=g_theta, circle=True);

volume = sitk.GetImageFromArray(reconstruction_CT_matrix);
volume.SetSpacing([g_pixel_spacing_in_mm, g_pixel_spacing_in_mm, g_pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/reconstruction_CT_matrix.mha", useCompression=True);


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

        gvxr.subtractMesh("fibre_" + str(i), "core_" + str(i));

        gvxr.applyCurrentLocalTransformation("fibre_" + str(i));
        gvxr.applyCurrentLocalTransformation("core_" + str(i));

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


def fitnessFunction(x):
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
    ZNCC = np.mean(np.multiply(normalised_simulated_sinogram.flatten(), normalised_reference_sinogram.flatten()));

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
    if best_fitness > fitness:
        best_fitness = fitness;




    return fitness;


# In[57]:

    # An individual is made of two floating point numbers:
    # - the radius of the SiC fibre
    # - the ratio    radius of the W core / radius of the SiC fibre

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
es.optimize(fitnessFunction);
elapsed_time = time.time() - start_time
print("FIBRES",elapsed_time);
fibre_radius = es.result.xbest[0];
core_radius = fibre_radius * es.result.xbest[1];

np.savetxt(output_directory + "/fibre_radius.dat", [core_radius, fibre_radius], header='core_radius_in_um,fibre_radius_in_um');



# Load the matrix
setMatrix(current_best);

# Load the cores and fibres
setFibres(centroid_set);




simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram();

simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
reconstruction_CT_fibres = iradon(simulated_sinogram.T, theta=g_theta, circle=True);

volume = sitk.GetImageFromArray(reconstruction_CT_fibres);
volume.SetSpacing([g_pixel_spacing_in_mm, g_pixel_spacing_in_mm, g_pixel_spacing_in_mm]);
sitk.WriteImage(volume, output_directory + "/reconstruction_CT_fibres.mha", useCompression=True);


print("Matrix params:", current_best);
normalised_reconstruction_CT_matrix = (reconstruction_CT_matrix - reconstruction_CT_matrix.mean()) / reconstruction_CT_matrix.std();
ZNCC_CT = np.mean(np.multiply(normalised_reconstruction_CT_matrix.flatten(), normalised_reference_CT.flatten()));
print("Matrix CT ZNCC:", ZNCC_CT);

print("Radii:", core_radius, fibre_radius);
normalised_reconstruction_CT_fibres = (reconstruction_CT_fibres - reconstruction_CT_fibres.mean()) / reconstruction_CT_fibres.std();
ZNCC_CT = np.mean(np.multiply(normalised_reconstruction_CT_fibres.flatten(), normalised_reference_CT.flatten()));
print("Fibres CT ZNCC:", ZNCC_CT);


comp_equalized = compare_images(normalised_reference_CT, normalised_reconstruction_CT_fibres, method='checkerboard');

fig=plt.figure();
imgplot = plt.imshow(comp_equalized, cmap='gray', norm=norm);
plt.savefig(output_directory + "/comparison.pdf");
plt.savefig(output_directory + "/comparison.png");
