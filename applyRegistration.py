#!/usr/bin/env python3
# coding: utf-8

import copy, math, os, glob, argparse, sys, time

import numpy as np

import imageio
from skimage.transform import iradon
from skimage.util import compare_images

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

from skimage.util import random_noise

from scipy import ndimage

# import tomopy

import SimpleITK as sitk

import cv2

import cma

import matplotlib
matplotlib.use('TkAGG')   # generate postscript output by default

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LogNorm
from matplotlib import cm

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower

plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False


from lsf import *

import gvxrPython3 as gvxr


# In[2]:


def processCmdLine():
    parser = argparse.ArgumentParser(description='Compute the path length using gVirtualXRay (http://gvirtualxray.sourceforge.net/).')
    parser.add_argument('--input', help='Folder', nargs=1, type=str, required=True);
    parser.add_argument('--output', help='Output dir', nargs=1, type=str, required=True);

    return parser.parse_args()

args = processCmdLine();


output_directory = args.output[0];
if not os.path.exists(output_directory):
    os.makedirs(output_directory);



NoneType = type(None);
pixel_spacing_in_micrometre = 1.9;
pixel_spacing_in_mm = pixel_spacing_in_micrometre * 1e-3;
number_of_projections = 900;
angular_span_in_degrees = 180.0;
angular_step = angular_span_in_degrees / number_of_projections;
theta = np.linspace(0., angular_span_in_degrees, number_of_projections, endpoint=False);
theta_rad = theta / 180.0 * math.pi;

roi_length = 40;

value_range = 6;
num_samples = 15;

sigma_set = None;
k_set = None;
label_set = None;

bias = None;
gain = None;
scale = None;

use_normalisation = True;
use_sinogram = True;

fibre_radius = 140 / 2; # um
core_radius = 30 / 2; # um


# ## Load the image data
#
# Load and display the reference projections from a raw binary file, i.e. the target of the registration.

# In[4]:


# Target of the registration
reference_normalised_projections = np.fromfile(args.input[0], dtype=np.float32);
detector_width_in_pixels = round(reference_normalised_projections.shape[0] / number_of_projections);
reference_normalised_projections.shape = [number_of_projections, detector_width_in_pixels];

# In[5]:
detector_height_in_pixels = 1;
distance_object_detector_in_m =    0.08; # = 80 mm


def saveMHA(fname, image, spacing):
    """
    save the image into a file.

    :param str fname: the filename
    :param 2D_image image: the image to save
    :param [flt, flt, flt] spacing: the space between two successive voxels along the 3 direction
    """

    #np.savetxt("Tutorial2/outputs/reference_CT.txt", g_reference_CT);
    volume = sitk.GetImageFromArray(image);
    volume.SetSpacing(spacing);
    sitk.WriteImage(volume, fname, useCompression=True);

def float2uint8(anImage, min_threshold = None, max_threshold = None):

    uchar_image = copy.deepcopy(anImage);

    if isinstance(min_threshold, NoneType):
        min_threshold = np.min(uchar_image);

    if isinstance(max_threshold, NoneType):
        max_threshold = np.max(uchar_image);

    uchar_image[uchar_image < min_threshold] = min_threshold;
    uchar_image[uchar_image > max_threshold] = max_threshold;

    uchar_image -= min_threshold;
    uchar_image /= max_threshold - min_threshold;
    uchar_image *= 255;

    return uchar_image.astype(np.uint8);


def computeSinogramFromFlatField(normalised_projections):
    """
    This function apply the minus log normalisation
    on the projections that bave been corrected with the flat-field method.

    :param 2D_image normalised_projections: The projections after flat-field corrections
    :return the sinogram.
    """

    # Create a temporary image to hold the sinogram
    simulated_sinogram = copy.deepcopy(normalised_projections);

    # Make sure no value is negative or null (because of the log function)
    # It should not be the case, however, when the Laplacian is used to simulate
    # phase contrast, negative values can be generated.
    threshold = 0.000001
    simulated_sinogram[simulated_sinogram < threshold] = threshold;

    # Apply the minus log normalisation
    simulated_sinogram = -np.log(simulated_sinogram);

    # Rescale the data taking into account the pixel size
    simulated_sinogram /= pixel_spacing_in_micrometre * gvxr.getUnitOfLength("um") / gvxr.getUnitOfLength("cm");

    # Return the new image
    return simulated_sinogram;



reference_sinogram = computeSinogramFromFlatField(reference_normalised_projections);


# rot_center = tomopy.find_center(reference_sinogram, theta_rad, init=512, ind=0, tol=0.5)
# reference_CT = tomopy.recon(reference_sinogram, theta_rad, center=rot_center, algorithm='gridrec', sinogram_order=False, filter_name='shepp')

# reference_CT = tomopy.circ_mask(reference_CT, axis=0, ratio=0.95)






reference_CT = iradon(reference_sinogram.T, theta=theta, circle=True, filter_name='shepp-logan');

normalised_reference_sinogram = (reference_sinogram - reference_sinogram.mean()) / reference_sinogram.std();
normalised_reference_CT       = (reference_CT       - reference_CT.mean())       / reference_CT.std();


saveMHA(output_directory + "/reference_CT.mha", reference_CT, [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);
saveMHA(output_directory + "/reference_CT.png", float2uint8(reference_CT, -35, 35), [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);




temp_sinogram = copy.deepcopy(reference_sinogram)
temp_sinogram.shape = [number_of_projections, detector_width_in_pixels];
temp_CT = iradon(temp_sinogram.T, theta=theta, circle=True, filter_name='shepp-logan');


uint8_reference_CT = float2uint8(temp_CT, 0, 300);
blurred_reference_CT = cv2.bilateralFilter(uint8_reference_CT, 9, 75, 75);

saveMHA(output_directory + '/blurred_reference_CT.mha', blurred_reference_CT, [pixel_spacing_in_mm, pixel_spacing_in_mm]);




# Unlike the previous example, did did not work that well. Here 13 fibres were missed. Many centres are also misplaced. We will use another technique to register the fibres, the popular Otsu's method. It creates a histogram and uses a heuristic to determine a threshold value.

# In[29]:


# Convert the numpy array in float32 into uint, then into a SITK image
volume = sitk.GetImageFromArray(blurred_reference_CT);
volume.SetSpacing([pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);

# Apply the Otsu's method
otsu_filter = sitk.OtsuThresholdImageFilter();
otsu_filter.SetInsideValue(0);
otsu_filter.SetOutsideValue(1);
seg = otsu_filter.Execute(volume);


sitk.WriteImage(seg, output_directory + "/cores_segmentation.mha", useCompression=True);


# In[31]:


fig=plt.figure();
imgplot = plt.imshow(sitk.GetArrayViewFromImage(sitk.LabelOverlay(volume, seg)));
plt.title("Reference image and detected Tungsten cores");
plt.savefig(output_directory+'/fibre_detection_using_otsu_method.pdf');
plt.savefig(output_directory+'/fibre_detection_using_otsu_method.png');
# plt.show()


# Clean-up using mathematical morphology
cleaned_thresh_img = sitk.BinaryOpeningByReconstruction(seg, [3, 3, 3])
cleaned_thresh_img = sitk.BinaryClosingByReconstruction(cleaned_thresh_img, [3, 3, 3])


# sitk.WriteImage(cleaned_thresh_img, output_directory + "/cores_cleaned_segmentation.mha", useCompression=True);


# In[34]:


# fig=plt.figure();
# imgplot = plt.imshow(sitk.GetArrayViewFromImage(sitk.LabelOverlay(volume, cleaned_thresh_img)));
# plt.title("Reference image and detected Tungsten cores");
# plt.savefig('plots/fibre_detection_using_otsu_method_after_cleaning.pdf');
# plt.savefig('plots/fibre_detection_using_otsu_method_after_cleaning.png');


# ## Mark each potential tungsten corewith unique label

# In[35]:


core_labels = sitk.ConnectedComponent(cleaned_thresh_img);


# In[36]:


# fig=plt.figure();
# imgplot = plt.imshow(sitk.GetArrayViewFromImage(sitk.LabelOverlay(volume, core_labels)));
# plt.title("Cleaned Binary Segmentation of the Tungsten cores");
# plt.savefig('plots/fibre_detection_with_label_overlay.pdf');
# plt.savefig('plots/fibre_detection_with_label_overlay.png');


# ### Object Analysis

# Once we have the segmented objects we look at their shapes and the intensity distributions inside the objects. For each labelled tungsten core, we extract the centroid. Note that sizes and positions are given in millimetres.

# In[37]:


shape_stats = sitk.LabelShapeStatisticsImageFilter()
shape_stats.ComputeOrientedBoundingBoxOn()
shape_stats.Execute(core_labels)


# In[38]:


centroid_set = [];

for i in shape_stats.GetLabels():
    centroid_set.append(cleaned_thresh_img.TransformPhysicalPointToIndex(shape_stats.GetCentroid(i)));

print("There are", len(centroid_set), "fibres")

# We now have a list of the centres of all the fibres that can be used as a parameter of the function below to create the cylinders corresponding to the cores and the fibres.
# For each core, a cylinder is creatd and translated:
# ```python
#         gvxr.emptyMesh("core_"  + str(i));
#         gvxr.makeCylinder("core_"  + str(i), number_of_sectors, 815.0,  core_radius, "micrometer");
#         gvxr.translateNode("core_"  + str(i), y, 0.0, x, "micrometer");
# ```
# For each fibre, another cylinder is created and translated:
# ```python
#         gvxr.emptyMesh("fibre_"  + str(i));
#         gvxr.makeCylinder("fibre_"  + str(i), number_of_sectors, 815.0,  fibre_radius, "micrometer");
#         gvxr.translateNode("fibre_"  + str(i), y, 0.0, x, "micrometer");
# ```
# The fibre's cylinder is hollowed to make space for its core:
# ```python
#         gvxr.subtractMesh("fibre_" + str(i), "core_" + str(i));
# ```

# In[39]:


def setFibres(aCentroidSet):
    """This function loads a cylinders in the GPU memory.
    Some are hollow and represent the fibres, some are not and
    correspond to the cores.

    :param array aCentroidSet: a list of cylinder centres.
    """

    global core_radius;
    global fibre_radius;

    # Create empty geometries
    gvxr.emptyMesh("fibre");
    gvxr.emptyMesh("core");

    # Number of sectors to approximate cylinders with triangle meshes
    # It controls the accuracy of the meshes.
    number_of_sectors = 100;

    # Process all the centres from the input list
    for i, cyl in enumerate(aCentroidSet):

        # Convert the centre position from 2D image coordinates in 3D world coordinates
        x = pixel_spacing_in_micrometre * -(cyl[0] - detector_width_in_pixels / 2 + 0.5);
        y = pixel_spacing_in_micrometre * (cyl[1] - detector_width_in_pixels / 2 + 0.5);

        # Create empty geometries (is it needed?)
        gvxr.emptyMesh("fibre_" + str(i));
        gvxr.emptyMesh("core_"  + str(i));

        # Create the two corresponding cylinders (fibre and core)
        gvxr.makeCylinder("fibre_" + str(i), number_of_sectors, 815.0, fibre_radius, "micrometer");
        gvxr.makeCylinder("core_"  + str(i), number_of_sectors, 815.0,  core_radius, "micrometer");

        # Translate the two cylinders to the position of their centre
        gvxr.translateNode("fibre_" + str(i), y, 0.0, x, "micrometer");
        gvxr.translateNode("core_"  + str(i), y, 0.0, x, "micrometer");

        # Apply the local transformation matrix (so that we could save the corresponding STL files)
        gvxr.applyCurrentLocalTransformation("fibre_" + str(i));
        gvxr.applyCurrentLocalTransformation("core_" + str(i));

        # Subtract the fibre from the matrix
        gvxr.subtractMesh("matrix", "fibre_" + str(i));

        # Subtract the core from the fibre
        gvxr.subtractMesh("fibre_" + str(i), "core_" + str(i));

        # Save the corresponding STL files
        #gvxr.saveSTLfile("fibre_" + str(i), "Tutorial2/outputs/fibre_" + str(i) + ".stl");
        #gvxr.saveSTLfile("core_" + str(i),  "Tutorial2/outputs/core_"  + str(i) + ".stl");

        # Add the mesh of the current fibre to the overall fibre mesh
        gvxr.addMesh("fibre", "fibre_" + str(i));

        # Add the mesh of the current core to the overall core mesh
        gvxr.addMesh("core",  "core_"  + str(i));

    # Set the mesh colours (for the interactive visualisation)
    gvxr.setColor("fibre", 1.0, 0.0, 0.0, 1.0);
    gvxr.setColor("core",  1.0, 0.0, 1.0, 1.0);

    # Set the fibre's material properties
    #gvxr.setLinearAttenuationCoefficient("fibre", fibre_mu, "cm-1");
    gvxr.setCompound("fibre", "SiC");
    gvxr.setDensity("fibre", fibre_density, "g/cm3");

    # Set the core's material properties
    #gvxr.setLinearAttenuationCoefficient("core", core_mu, "cm-1");
    gvxr.setElement("core", "W");

    # Add the fibres and cores to the X-ray renderer
    gvxr.addPolygonMeshAsInnerSurface("core");
    gvxr.addPolygonMeshAsInnerSurface("fibre");


# ## Registration of a cube

# We define a function to create the polygon mesh of the Ti90Al6V4 matrix.

# In[40]:


def setMatrix(apGeneSet):
    """This function loads a cube in the GPU memory. The cube represents
    the Ti90Al6V4 matrix.

    apGeneSet[0] is a number between -0.5 and 0.5, related to the translation vector (X component) of the cube. It can be interpreted as a percentage of the detector width.
    apGeneSet[1] is the same as apGeneSet[0], but related to the Y component of the translation vector.
    apGeneSet[2] is a number between -0.5 and 0.5, related to the rotation angle in degrees
    apGeneSet[3] is a scaling factor between -0.5 and 0.5. It can be interpreted as a percentage of the detector width.
    apGeneSet[4] is a scaling factor between -0.5 and 0.5. It can be interpreted as a percentage of apGeneSet[3].
    """

    # Remove all the geometries from the whole scenegraph
    gvxr.removePolygonMeshesFromSceneGraph();

    # Make a cube
    gvxr.makeCube("matrix", 1.0, "micrometer");

    # Translation vector
    x = apGeneSet[0] * detector_width_in_pixels * pixel_spacing_in_micrometre;
    y = apGeneSet[1] * detector_width_in_pixels * pixel_spacing_in_micrometre;
    gvxr.translateNode("matrix", x, 0.0, y, "micrometer");

    # Rotation angle
    rotation_angle_in_degrees = (apGeneSet[2] + 0.5) * 180.0;
    gvxr.rotateNode("matrix", rotation_angle_in_degrees, 0, 1, 0);

    # Scaling factors
    w = (apGeneSet[3] + 0.5) * detector_width_in_pixels * pixel_spacing_in_micrometre;
    h = (apGeneSet[4] + 0.5) * w;
    gvxr.scaleNode("matrix", w, 815, h);

    # Apply the transformation matrix so that we can save the corresponding STL file
    gvxr.applyCurrentLocalTransformation("matrix");

    # Set the matrix's material properties
    gvxr.setMixture("matrix", "Ti90Al6V4");
    gvxr.setDensity("matrix", matrix_density, "g/cm3");

    # Add the matrix to the X-ray renderer
    gvxr.addPolygonMeshAsInnerSurface("matrix");


# ### Simulate the CT acquisition
#
# 1. Set the fibre and cores geometries and material properties (Step 39)
# 2. Set the matrix geometry and material properties (Step 40)
# 3. Simulate the raw projections for each angle:
#    - Without phase contrast (Line 5 of Step 45), or
#    - With phase contrast (Lines 14-55 of Step 45)
# 4. Apply the LSF (Lines 57-60 of Step 45)
# 5. Apply the flat-field correction (Step 62)
# 6. Add Poison noise (Step~\ref{??})
# 7. Apply the minus log normalisation to compute the sinogram (Step 63)
#
# Compute the raw projections and save the data. For this  purpose, we define a new function.

# In[41]:


def tomographyAcquisition():
    """
    This function simulate a CT acquisition.

    :return the raw projections in keV
    """

    # Crete a new array to save every projection in default unit of energy
    raw_projections = [];

    # For each angle, simulate a projection
    for angle_id in range(0, number_of_projections):

        # Reset the transformation matrix and rotate the scnned object
        gvxr.resetSceneTransformation();
        gvxr.rotateScene(-angular_step * angle_id, 0, 1, 0);

        # Compute the X-ray image
        xray_image = np.array(gvxr.computeXRayImage());

        # Add the projection
        raw_projections.append(xray_image);

    # Convert from the default unit of energy to keV
    raw_projections = np.array(raw_projections);
    raw_projections_in_keV = raw_projections / gvxr.getUnitOfEnergy("keV");

    return raw_projections_in_keV;


# ### Flat-filed correction
#
# Because the data suffers from a fixed-pattern noise in X-ray imaging in
# actual experiments, it is necessary to perform the flat-field correction of
# the raw projections using:
#
# $$corrected\_projections = \frac{raw\_projections\_in\_keV − dark\_field\_image}{flat\_field\_image − dark\_field\_image}$$
#
# - $raw\_projections\_in\_keV$ are the raw projections with the X-ray beam turned on and with the scanned object,
# - $flat\_field\_image$ is an image with the X-ray beam turned on but without the scanned object, and
# - $dark\_field\_image$ is an image with the X-ray beam turned off.
#
# Note that in our example, $raw\_projections\_in\_keV$, $flat\_field\_image$ and $dark\_field\_image$ are in keV whereas $corrected\_projections$ does not have any unit:
#
# $$0 \leq raw\_projections\_in\_keV \leq  \sum_E N_0(E) \times E\\0 \leq corrected\_projections \leq 1$$
#
# We define a new function to compute the flat-field correction.

# In[42]:


def flatFieldCorrection(raw_projections_in_keV):
    """
    This function applies the flat-field correction on raw projections.

    :param 2D_image raw_projections_in_keV: the raw X-ray projections in keV
    :return the projections (raw_projections_in_keV) after flat-field correction
    """

    # Create a mock dark field image
    dark_field_image = np.zeros(raw_projections_in_keV.shape);

    # Create a mock flat field image
    flat_field_image = np.ones(raw_projections_in_keV.shape);

    # Retrieve the total energy
    total_energy = 0.0;
    energy_bins = gvxr.getEnergyBins("keV");
    photon_count_per_bin = gvxr.getPhotonCountEnergyBins();

    for energy, count in zip(energy_bins, photon_count_per_bin):
        total_energy += energy * count;
    flat_field_image *= total_energy;

    # Apply the actual flat-field correction on the raw projections
    corrected_projections = (raw_projections_in_keV - dark_field_image) / (flat_field_image - dark_field_image);

    return corrected_projections;


# The function below is used to simulate a sinogram acquisition. Phase contrast in the projections can be taken into account or not.

# In[43]:


def simulateSinogram(sigma_set = None, k_set = None, name_set = None):

    global lsf_kernel;

    # Do not simulate the phase contrast using a Laplacian
    if isinstance(sigma_set, NoneType) or isinstance(k_set, NoneType) or isinstance(name_set, NoneType):

        # Get the raw projections in keV
        raw_projections_in_keV = tomographyAcquisition();

    # Simulate the phase contrast using a Laplacian
    else:

        # Create the convolution filter
        pixel_range = np.linspace(-value_range, value_range, num=int(num_samples), endpoint=True)
        laplacian_kernels = {};

        # Store the L-buffers
        L_buffer_set = {};

        # Look at all the children of the root node
        for label in ["core", "fibre", "matrix"]:
            # Get its L-buffer
            L_buffer_set[label] = getLBuffer(label);

        # Create blank images
        raw_projections_in_keV = np.zeros(L_buffer_set["fibre"].shape);
        phase_contrast_image = np.zeros(L_buffer_set["fibre"].shape);

        for label, k, sigma in zip(name_set, k_set, sigma_set):
            laplacian_kernels[label] = k * laplacian(pixel_range, sigma);

            for z in range(phase_contrast_image.shape[0]):
                for y in range(phase_contrast_image.shape[1]):
                    phase_contrast_image[z][y] += ndimage.convolve((L_buffer_set[label])[z][y], laplacian_kernels[label], mode='wrap');

        for energy, photon_count in zip(gvxr.getEnergyBins("keV"), gvxr.getPhotonCountEnergyBins()):

            # Create a blank image
            attenuation = np.zeros(L_buffer_set["fibre"].shape);

            # Look at all the children of the root node
            #for label in ["core", "fibre", "matrix"]:
            for label in ["core", "fibre", "matrix"]:
                # Get mu for this object for this energy
                mu = gvxr.getLinearAttenuationCoefficient(label, energy, "keV");

                # Compute sum mu * x
                attenuation += L_buffer_set[label] * mu;

            # Store the projection for this energy channel
            raw_projections_in_keV += energy * photon_count * np.exp(-attenuation);

        # Apply the phase contrast
        raw_projections_in_keV -= phase_contrast_image;

    # Apply the LSF line by line
    for z in range(raw_projections_in_keV.shape[0]):
        for y in range(raw_projections_in_keV.shape[1]):
            raw_projections_in_keV[z][y] = ndimage.convolve(raw_projections_in_keV[z][y], lsf_kernel, mode='wrap');

    # Flat-field correction
    normalised_projections = flatFieldCorrection(raw_projections_in_keV);
    normalised_projections[normalised_projections < 0] = 0;

    # Add noise
    if not isinstance(bias, NoneType) and not isinstance(gain, NoneType) and not isinstance(scale, NoneType):

        map = (normalised_projections + (bias + 1)) * gain;
        temp = np.random.poisson(map).astype(np.float);
        temp /= gain;
        temp -= bias + 1;

        # Noise map
        noise_map = (normalised_projections - temp) * scale;
        normalised_projections += noise_map;

    # Linearise
    simulated_sinogram = computeSinogramFromFlatField(normalised_projections);

    return simulated_sinogram, normalised_projections, raw_projections_in_keV;


# The function below is used quantify the differences between two images. It is used in the objective function.

# In[44]:




gvxr.createWindow(0, 1, "EGL");
gvxr.setWindowSize(512, 512);


gvxr.setDetectorPosition(-distance_object_detector_in_m, 0.0, 0.0, "m");
gvxr.setDetectorUpVector(0, 1, 0);
gvxr.setDetectorNumberOfPixels(detector_width_in_pixels, detector_height_in_pixels);
gvxr.setDetectorPixelSize(pixel_spacing_in_micrometre, pixel_spacing_in_micrometre, "micrometer");

distance_source_detector_in_m  = 145.0;
gvxr.setSourcePosition(distance_source_detector_in_m - distance_object_detector_in_m,  0.0, 0.0, "m");
gvxr.usePointSource();
gvxr.useParallelBeam();

energy_spectrum = [(33, 0.97, "keV"), (66, 0.02, "keV"), (99, 0.01, "keV")];

for energy, percentage, unit in energy_spectrum:
    gvxr.addEnergyBinToSpectrum(energy, unit, percentage);

energies_in_keV = [];
weights = [];

for energy, percentage, unit in energy_spectrum:
    weights.append(percentage);
    energies_in_keV.append(energy * gvxr.getUnitOfEnergy(unit) / gvxr.getUnitOfEnergy("keV"));

# fig=plt.figure();
# plt.xlabel("Energy bin (in keV)");
# plt.ylabel("Relative weight");
# plt.xticks(energies_in_keV);
# plt.yticks(weights);
# plt.title("Incident beam spectrum");
# plt.bar(energies_in_keV, weights);
# plt.savefig('plots/beam_spectrum.pdf');
# plt.savefig('plots/beam_spectrum.png');


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


t = np.arange(-20., 21., 1.);
lsf_kernel=lsf(t*41)/lsf(0);
lsf_kernel/=lsf_kernel.sum();


# fig=plt.figure();
# plt.title("Response of the detector (LSF)");
# plt.plot(t, lsf_kernel);
# plt.savefig('plots/LSF.pdf');
# plt.savefig('plots/LSF.png');


# In[46]:

matrix_geometry_parameters = np.loadtxt(output_directory + "/cube.dat");
# ### Apply the result of the registration

# In[47]:


# Save the result
setMatrix(matrix_geometry_parameters);
# gvxr.saveSTLfile("matrix", output_directory + "/matrix.stl");

# Translation vector
x = matrix_geometry_parameters[0] * detector_width_in_pixels * pixel_spacing_in_micrometre;
y = matrix_geometry_parameters[1] * detector_width_in_pixels * pixel_spacing_in_micrometre;

# Rotation angle
rotation_angle_in_degrees = (matrix_geometry_parameters[2] + 0.5) * 180.0;

# Scaling factors
w = (matrix_geometry_parameters[3] + 0.5) * detector_width_in_pixels * pixel_spacing_in_micrometre;
h = (matrix_geometry_parameters[4] + 0.5) * w;

print("Matrix");
print("\tposition:", x, y, "um");
print("\trotation:", rotation_angle_in_degrees, "deg");
print("\tsize:", w, h, "um");


# ### Simulate the correspond CT acquisition

# In[48]:









# Simulate a sinogram
simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram(sigma_set, k_set, label_set);

# Reconstruct the CT slice
print(simulated_sinogram.shape)
print(reference_normalised_projections.shape)
simulated_sinogram.shape = reference_normalised_projections.shape;

saveMHA(output_directory + "/simulated.mha", simulated_sinogram, [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);

simulated_CT = iradon(simulated_sinogram.T, theta=theta, circle=True, filter_name='shepp-logan');
# rot_center = tomopy.find_center(simulated_sinogram, theta_rad, init=512, ind=0, tol=0.5)
# simulated_CT = tomopy.recon(simulated_sinogram, theta_rad, center=rot_center, algorithm='gridrec', sinogram_order=False, filter_name='shepp')
# simulated_CT = tomopy.circ_mask(simulated_CT, axis=0, ratio=0.95)



normalised_simulated_CT = (simulated_CT - simulated_CT.mean()) / simulated_CT.std();

# Compute the ZNCC
print("ZNCC matrix registration:",
      "{:.2f}".format(100.0 * np.mean(np.multiply(normalised_reference_CT, normalised_simulated_CT))));

saveMHA(output_directory + "/registration_matrix.mha", simulated_CT, [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);
saveMHA(output_directory + "/registration_matrix.png", float2uint8(simulated_CT, -35, 35), [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);




setMatrix(matrix_geometry_parameters);
setFibres(centroid_set);


# In[52]:


# Simulate a sinogram
simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram(sigma_set, k_set, label_set);


# In[53]:

simulated_sinogram.shape = reference_normalised_projections.shape;

simulated_CT = iradon(simulated_sinogram.T, theta=theta, circle=True, filter_name='shepp-logan');
# simulated_CT = tomopy.recon(simulated_sinogram, theta_rad, center=rot_center, algorithm='gridrec', sinogram_order=False, filter_name='shepp')
# simulated_CT = tomopy.circ_mask(simulated_CT, axis=0, ratio=0.95)



normalised_simulated_CT = (simulated_CT - simulated_CT.mean()) / simulated_CT.std();

# Compute the ZNCC
print("ZNCC matrix registration with fibres:",
      "{:.2f}".format(100.0 * np.mean(np.multiply(normalised_reference_CT, normalised_simulated_CT))));

saveMHA(output_directory + "/registration_matrix_with_fibres.mha", simulated_CT, [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);
saveMHA(output_directory + "/registration_matrix_with_fibres.png", float2uint8(simulated_CT, -35, 35), [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);

gvxr.saveSTLfile("matrix", output_directory + "/matrix.stl");

# gvxr.saveSTLfile("fibre", output_directory + "/fibre.stl");
# gvxr.saveSTLfile("core",  output_directory + "/core.stl");
# gvxr.renderLoop();




# In[54]:


# simulated_sinogram.shape     = (simulated_sinogram.size     // simulated_sinogram.shape[2],     simulated_sinogram.shape[2]);
# normalised_projections.shape = (normalised_projections.size // normalised_projections.shape[2], normalised_projections.shape[2]);
# raw_projections_in_keV.shape = (raw_projections_in_keV.size // raw_projections_in_keV.shape[2], raw_projections_in_keV.shape[2]);
#
# saveMHA(output_directory + "/simulated_sinogram_with_fibres.mha",
#         simulated_sinogram,
#         [pixel_spacing_in_mm, angular_step, pixel_spacing_in_mm]);
#
# saveMHA(output_directory + "/normalised_projections_with_fibres.mha",
#         normalised_projections,
#         [pixel_spacing_in_mm, angular_step, pixel_spacing_in_mm]);
#
# saveMHA(output_directory + "/raw_projections_in_keV_with_fibres.mha",
#         raw_projections_in_keV,
#         [pixel_spacing_in_mm, angular_step, pixel_spacing_in_mm]);


# In[55]:


# saveMHA(output_directory + "/simulated_CT_with_fibres.mha",
#         simulated_CT,
#         [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);
#
# saveMHA(output_directory + "/normalised_simulated_CT_with_fibres.mha",
#         normalised_simulated_CT,
#         [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);


# In[56]:


# norm = cm.colors.Normalize(vmax=1.25, vmin=-0.5)
#
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# plt.tight_layout()
# fig.suptitle('CT slice with fibres after the registration of the matrix')
#
# ax1.set_title("Reference image");
# imgplot1 = ax1.imshow(normalised_reference_CT, cmap="gray",
#                      norm=norm);
#
# ax2.set_title("Simulated CT slice after automatic registration");
# imgplot2 = ax2.imshow(normalised_simulated_CT,
#                      cmap='gray',
#                      norm=norm);
#
# comp_equalized = compare_images(normalised_reference_CT, normalised_simulated_CT, method='checkerboard');
# ax3.set_title("Checkboard comparison between\n" +
#               "the reference and simulated images\nZNCC: " +
#               "{:.2f}".format(100.0 * np.mean(np.multiply(normalised_reference_CT, normalised_simulated_CT))));
# imgplot3 = ax3.imshow(comp_equalized,
#                      cmap='gray',
#                      norm=norm);
#
# plt.savefig('plots/simulated_CT_slice_with_fibres_after_cube_registration.pdf');
# plt.savefig('plots/simulated_CT_slice_with_fibres_after_cube_registration.png');


#
# ## Optimisation of the cores and fibres radii

# The function below is the objective function used to optimise the radii of the cores and fibres.

# In[57]:



# The registration has already been performed. Load the results.
temp = np.loadtxt(output_directory + "/fibre1_radii.dat");
core_radius = temp[0];
fibre_radius = temp[1];


# In[59]:


# if not os.path.exists("plots/fibre1_registration.gif"):
#     registration_image_set = createAnimation(output_directory + "/fibre1_simulated_CT_",
#                 'plots/fibre1_registration.gif');


# ![Animation of the registration (GIF file)](plots/fibre1_registration.gif)

# ### Apply the result of the registration

# In[60]:


# Load the matrix
setMatrix(matrix_geometry_parameters);

# Load the cores and fibres
setFibres(centroid_set);

gvxr.saveSTLfile("fibre", output_directory + "/fibre1_fibre.stl");
gvxr.saveSTLfile("core",  output_directory + "/fibre1_core.stl");

print("Core diameter:", round(core_radius * 2), "um");
print("Fibre diameter:", round(fibre_radius * 2), "um");

# Simulate the corresponding CT aquisition
simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram(sigma_set, k_set, label_set);


simulated_sinogram.shape = reference_normalised_projections.shape;
simulated_CT = iradon(simulated_sinogram.T, theta=theta, circle=True, filter_name='shepp-logan');

# simulated_CT = tomopy.recon(simulated_sinogram, theta_rad, center=rot_center, algorithm='gridrec', sinogram_order=False, filter_name='shepp')
# simulated_CT = tomopy.circ_mask(simulated_CT, axis=0, ratio=0.95)



normalised_simulated_CT = (simulated_CT - simulated_CT.mean()) / simulated_CT.std();

# Compute the ZNCC
print("ZNCC radii registration 1:",
      "{:.2f}".format(100.0 * np.mean(np.multiply(normalised_reference_CT, normalised_simulated_CT))));

saveMHA(output_directory + "/registration_radii.mha", simulated_CT, [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);
saveMHA(output_directory + "/registration_radii.png", float2uint8(simulated_CT, -35, 35), [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);



# The 3D view of the registration looks like:
#
# ![3D view](./3d-view.png)

# ## Recentre each core/fibre

# Each fibre is extracted from both the reference CT slice and simulated CT slice. The displacement between the corresponding fibres is computed to maximise the ZNCC between the two. The centre of the fibre is then adjusted accordingly.

# In[61]:


def refineCentrePositions(centroid_set, reconstruction_CT_fibres):

    # Exhaustive local search to refine the centre of each cylinder
    roi_length = 40;
    new_centroid_set = [];
    for i, cyl in enumerate(centroid_set):

        centre = [
            cyl[0],
            cyl[1]
        ];

        # extract ROI from reference image
        reference_image = copy.deepcopy(reference_CT[centre[1] - roi_length:centre[1] + roi_length, centre[0] - roi_length:centre[0] + roi_length]);

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
        new_centroid_set.append([cyl[0] - best_x_offset, cyl[1] - best_y_offset]);

    return new_centroid_set;


# In[62]:


centroid_set = refineCentrePositions(centroid_set, normalised_simulated_CT);


# ### Applying the result of recentring

# In[63]:


# Load the matrix
setMatrix(matrix_geometry_parameters);

# Load the cores and fibres
setFibres(centroid_set);


# gvxr.saveSTLfile("fibre", output_directory + "/fibre2_fibre.stl");
# gvxr.saveSTLfile("core",  output_directory + "/fibre2_core.stl");

# Simulate the corresponding CT aquisition
simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram(sigma_set, k_set, label_set);

simulated_sinogram.shape = reference_normalised_projections.shape;
simulated_CT = iradon(simulated_sinogram.T, theta=theta, circle=True, filter_name='shepp-logan');

# simulated_CT = tomopy.recon(simulated_sinogram, theta_rad, center=rot_center, algorithm='gridrec', sinogram_order=False, filter_name='shepp')
# simulated_CT = tomopy.circ_mask(simulated_CT, axis=0, ratio=0.95)



normalised_simulated_CT = (simulated_CT - simulated_CT.mean()) / simulated_CT.std();

# Compute the ZNCC
print("ZNCC recentring registration:",
      "{:.2f}".format(100.0 * np.mean(np.multiply(normalised_reference_CT, normalised_simulated_CT))));

saveMHA(output_directory + "/recentring.mha", simulated_CT, [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);
saveMHA(output_directory + "/recentring.png", float2uint8(simulated_CT, -35, 35), [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);



temp = np.loadtxt(output_directory + "/fibre3_radii.dat");
core_radius = temp[0];
fibre_radius = temp[1];


# Load the matrix
setMatrix(matrix_geometry_parameters);

# Load the cores and fibres
setFibres(centroid_set);

# gvxr.saveSTLfile("fibre", output_directory + "/fibre3_fibre.stl");
# gvxr.saveSTLfile("core",  output_directory + "/fibre3_core.stl");

print("Core diameter:", round(core_radius * 2), "um");
print("Fibre diameter:", round(fibre_radius * 2), "um");


# In[67]:


# Simulate the corresponding CT aquisition
simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram(sigma_set, k_set, label_set);


simulated_sinogram.shape = reference_normalised_projections.shape;
simulated_CT = iradon(simulated_sinogram.T, theta=theta, circle=True, filter_name='shepp-logan');

# simulated_CT = tomopy.recon(simulated_sinogram, theta_rad, center=rot_center, algorithm='gridrec', sinogram_order=False, filter_name='shepp')
# simulated_CT = tomopy.circ_mask(simulated_CT, axis=0, ratio=0.95)



normalised_simulated_CT = (simulated_CT - simulated_CT.mean()) / simulated_CT.std();

# Compute the ZNCC
print("ZNCC radii registration 2:",
      "{:.2f}".format(100.0 * np.mean(np.multiply(normalised_reference_CT, normalised_simulated_CT))));

saveMHA(output_directory + "/registration_radii2.mha", simulated_CT, [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);
saveMHA(output_directory + "/registration_radii2.png", float2uint8(simulated_CT, -35, 35), [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);


# ## Optimisation of the beam spectrum

# In[68]:


temp = np.loadtxt(output_directory + "/spectrum1.dat");
# The beam specturm. Here we have a polychromatic beam.
energy_spectrum = [(33, temp[0], "keV"), (66, temp[1], "keV"), (99, temp[2], "keV")];



# Apply the result of the registration
gvxr.resetBeamSpectrum();
for energy, percentage, unit in energy_spectrum:
    gvxr.addEnergyBinToSpectrum(energy, unit, percentage);


# In[ ]:


for channel in energy_spectrum:
    print(channel);


# In[ ]:


# Simulate the corresponding CT aquisition
simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram(sigma_set, k_set, label_set);

# Reconstruct the CT slice
simulated_sinogram.shape = reference_normalised_projections.shape;
simulated_CT = iradon(simulated_sinogram.T, theta=theta, circle=True, filter_name='shepp-logan');

# simulated_CT = tomopy.recon(simulated_sinogram, theta_rad, center=rot_center, algorithm='gridrec', sinogram_order=False, filter_name='shepp')
# simulated_CT = tomopy.circ_mask(simulated_CT, axis=0, ratio=0.95)



normalised_simulated_CT = (simulated_CT - simulated_CT.mean()) / simulated_CT.std();

# Compute the ZNCC
print("ZNCC spectrum registration:",
      "{:.2f}".format(100.0 * np.mean(np.multiply(normalised_reference_CT, normalised_simulated_CT))));

saveMHA(output_directory + "/registration_spectrum.mha", simulated_CT, [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);
saveMHA(output_directory + "/registration_spectrum.png", float2uint8(simulated_CT, -35, 35), [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);



# ## Optimisation of the phase contrast and the radii

# In[ ]:


def laplacian(x, sigma):
    """
    This function create a Laplacian kernel with

    $$ g''(x) = \left(\frac{x^2}{\sigma^4} - \frac{1}{\sigma^2}\right) \exp\left(-\frac{x^2}{2\sigma^2}\right) $$

    :param array x:
    :param float sigma:
    :return the convolution kernel
    """

    return (np.power(x, 2.) / math.pow(sigma, 4) - 1. / math.pow(sigma, 2)) * np.exp(-np.power(x, 2.) / (2. * math.pow(sigma, 2)));


# In[ ]:


def getLBuffer(object):

    """
    This function compute the L-buffer of the object over all the angles

    :param str object: the name of the object
    :return the L-buffer over all the angles
    """

    # An empty L-buffer
    L_buffer = [];

    # Get the line of L-buffer for each angle
    for angle_id in range(0, number_of_projections):
        gvxr.resetSceneTransformation();
        gvxr.rotateScene(-angular_step * angle_id, 0, 1, 0);

        # Compute the X-ray image
        line_of_L_buffer = np.array(gvxr.computeLBuffer(object));

        # Add the projection
        L_buffer.append(line_of_L_buffer);

    # Return as a numpy array
    return np.array(L_buffer);


# In[ ]:

temp = np.loadtxt(output_directory + "/laplacian1.dat");
sigma_core = temp[0];
k_core = temp[1];
sigma_fibre = temp[2];
k_fibre = temp[3];
sigma_matrix = temp[4];
k_matrix = temp[5];
core_radius = temp[6];
fibre_radius = temp[7];



# Load the matrix
setMatrix(matrix_geometry_parameters);

# Load the cores and fibres
setFibres(centroid_set);

# gvxr.saveSTLfile("fibre", output_directory + "/laplacian1_fibre.stl");
# gvxr.saveSTLfile("core",  output_directory + "/laplacian1_core.stl");

gvxr.saveSTLfile("matrix", output_directory + "/matrix_final.stl");
gvxr.saveSTLfile("fibre", output_directory + "/fibres_final.stl");
gvxr.saveSTLfile("core",  output_directory + "/cores_final.stl");

print("Core diameter:", round(core_radius * 2), "um");
print("Fibre diameter:", round(fibre_radius * 2), "um");


# In[ ]:


# Simulate the corresponding CT aquisition
sigma_set = [sigma_core, sigma_fibre, sigma_matrix];
k_set = [k_core, k_fibre, k_matrix];
label_set = ["core", "fibre", "matrix"];

simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram(sigma_set, k_set, label_set);

simulated_sinogram.shape = reference_normalised_projections.shape;
simulated_CT = iradon(simulated_sinogram.T, theta=theta, circle=True, filter_name='shepp-logan');

# simulated_CT = tomopy.recon(simulated_sinogram, theta_rad, center=rot_center, algorithm='gridrec', sinogram_order=False, filter_name='shepp')
# simulated_CT = tomopy.circ_mask(simulated_CT, axis=0, ratio=0.95)



normalised_simulated_CT = (simulated_CT - simulated_CT.mean()) / simulated_CT.std();

# Compute the ZNCC
print("ZNCC phase contrast registration 1:",
      "{:.2f}".format(100.0 * np.mean(np.multiply(normalised_reference_CT, normalised_simulated_CT))));

saveMHA(output_directory + "/registration_phase_contrast.mha", simulated_CT, [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);
saveMHA(output_directory + "/registration_phase_contrast.png", float2uint8(simulated_CT, -35, 35), [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);




# ## Optimisation of the phase contrast and the LSF

# In[ ]:


old_lsf = copy.deepcopy(lsf_kernel);


# In[ ]:

temp = np.loadtxt(output_directory + "/laplacian2.dat");
k_core = temp[0];
k_fibre = temp[1];
k_matrix = temp[2];

temp = np.loadtxt(output_directory + "/lsf2.dat");
a2 = temp[0];
b2 = temp[1];
c2 = temp[2];
d2 = temp[3];
e2 = temp[4];
f2 = temp[5];



# The response of the detector as the line-spread function (LSF)
t = np.arange(-20., 21., 1.);
lsf_kernel=lsf(t*41, a2, b2, c2, d2, e2, f2);
lsf_kernel/=lsf_kernel.sum();


# In[ ]:


# Simulate the corresponding CT aquisition
sigma_set = [sigma_core, sigma_fibre, sigma_matrix];
k_set = [k_core, k_fibre, k_matrix];
label_set = ["core", "fibre", "matrix"];

simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram(sigma_set, k_set, label_set);

simulated_sinogram.shape = reference_normalised_projections.shape;
simulated_CT = iradon(simulated_sinogram.T, theta=theta, circle=True, filter_name='shepp-logan');

# simulated_CT = tomopy.recon(simulated_sinogram, theta_rad, center=rot_center, algorithm='gridrec', sinogram_order=False, filter_name='shepp')
# simulated_CT = tomopy.circ_mask(simulated_CT, axis=0, ratio=0.95)



normalised_simulated_CT = (simulated_CT - simulated_CT.mean()) / simulated_CT.std();

# Compute the ZNCC
print("ZNCC phase contrast and LSF registration:",
      "{:.2f}".format(100.0 * np.mean(np.multiply(normalised_reference_CT, normalised_simulated_CT))));

saveMHA(output_directory + "/registration_LSF.mha", simulated_CT, [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);
saveMHA(output_directory + "/registration_LSF.png", float2uint8(simulated_CT, -35, 35), [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);




# In[ ]:


# fig=plt.figure();
# plt.title("Response of the detector (LSF)");
# plt.plot(t, old_lsf, label="Before optimisation");
# plt.plot(t, lsf_kernel, label="Before optimisation");
# plt.legend();
# plt.savefig('plots/LSF_optimised.pdf');
# plt.savefig('plots/LSF_optimised.png');



# ### Extract the fibre in the centre of the CT slices

# In[ ]:


# The registration has already been performed. Load the results.
temp = np.loadtxt(output_directory + "/poisson-noise.dat");
bias = temp[0];
gain = temp[1];
scale = temp[2];



print("Noise parameters: ", bias, gain, scale)


# ### Apply the result of the optimisation

# In[ ]:


# Simulate the corresponding CT aquisition
simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram(sigma_set, k_set, label_set);

simulated_sinogram.shape = reference_normalised_projections.shape;
simulated_CT = iradon(simulated_sinogram.T, theta=theta, circle=True, filter_name='shepp-logan');

# simulated_CT = tomopy.recon(simulated_sinogram, theta_rad, center=rot_center, algorithm='gridrec', sinogram_order=False, filter_name='shepp')
# simulated_CT = tomopy.circ_mask(simulated_CT, axis=0, ratio=0.95)



normalised_simulated_CT = (simulated_CT - simulated_CT.mean()) / simulated_CT.std();

# Compute the ZNCC
print("ZNCC noise registration:",
      "{:.2f}".format(100.0 * np.mean(np.multiply(normalised_reference_CT, normalised_simulated_CT))));

saveMHA(output_directory + "/registration_noise.mha", simulated_CT, [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);
saveMHA(output_directory + "/registration_noise.png", float2uint8(simulated_CT, -35, 35), [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);
