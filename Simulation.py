import copy, sys
import numpy as np

from skimage.transform import iradon
import SimpleITK as sitk
import cv2

import gvxrPython3 as gvxr
from lsf import *


# Global variables
pixel_spacing_in_micrometre = 1.9;
pixel_spacing_in_mm = pixel_spacing_in_micrometre * 1e-3;
number_of_projections = 900;
angular_span_in_degrees = 180.0;
angular_step = angular_span_in_degrees / number_of_projections;
theta = np.linspace(0., angular_span_in_degrees, number_of_projections, endpoint=False);

fibre_radius = 140 / 2; # um
core_radius = 30 / 2; # um

value_range = 6;
num_samples = 15;
use_fibres = False;

centroid_set = None;

def createTargetFromRawSinogram(fname):
    """This function read the binary file fname. This file contains
    the projections after flat-field correction.

    It returns 1) the sinogram,
               2) the corresponding CT slice in linear attenuation coefficients (mu),
               3) the sinogram after zero-mean, unit variance normalisation, and
               4) the CT slice after zero-mean, unit variance normalisation.
    """

    global reference_sinogram;
    global reference_CT;
    global normalised_reference_sinogram;
    global normalised_reference_CT;

    # Load the projections after flat-field correction from fname as a binary file in float32.
    reference_normalised_projections = np.fromfile(fname, dtype=np.float32);
    reference_normalised_projections.shape = [number_of_projections, int(reference_normalised_projections.shape[0] / number_of_projections)];

    # Apply the minus log normalisation to produce the sinogram
    reference_sinogram = computeSinogramFromFlatField(reference_normalised_projections);

    # CT reconstruction of the sinogram
    reference_CT = iradon(reference_sinogram.T, theta=theta, circle=True);

    # Zero-mean, unit variance normalisation of the sinogram and CT slice
    normalised_reference_sinogram = (reference_sinogram - reference_sinogram.mean()) / reference_sinogram.std();
    normalised_reference_CT       = (reference_CT       - reference_CT.mean())       / reference_CT.std();

    return reference_sinogram, reference_CT, normalised_reference_sinogram, normalised_reference_CT;

def initGVXR():
    """This function initialises the simulation framework on the GPU and
    some other read-only states, including global variables. It must be called
    before running gVirtualXRay.
    """

    # Global variables

    global detector_width_in_pixels;
    global detector_height_in_pixels;

    # global matrix_material;
    global matrix_density;

    # global fibre_material;
    global fibre_density;

    global lsf_kernel;

    # Create an OpenGL context, here using EGL, i.e. a windowless context
    gvxr.createWindow(0, 1, "EGL");
    gvxr.setWindowSize(512, 512);


    # We set the parameters of the X-ray detector (flat pannel), e.g. number of pixels, pixel, spacing, position and orientation:
    # ![3D scene to be simulated using gVirtualXray](3d_scene.png)
    detector_width_in_pixels = reference_sinogram.shape[1];
    detector_height_in_pixels = 1;
    distance_object_detector_in_m =    0.08; # = 80 mm

    gvxr.setDetectorPosition(-distance_object_detector_in_m, 0.0, 0.0, "m");
    gvxr.setDetectorUpVector(0, 1, 0);
    gvxr.setDetectorNumberOfPixels(detector_width_in_pixels, detector_height_in_pixels);
    gvxr.setDetectorPixelSize(pixel_spacing_in_micrometre, pixel_spacing_in_micrometre, "micrometer");

    # The beam specturm. Here we have a polychromatic beam, with 97% of the photons at 33 keV, 2% at 66 keV and 1% at 99 keV.
    energy_spectrum = [(33, 0.97, "keV"), (66, 0.02, "keV"), (99, 0.01, "keV")];

    for energy, percentage, unit in energy_spectrum:
        gvxr.addEnergyBinToSpectrum(energy, unit, percentage);


    # energies_in_keV = [];
    # weights = [];
    #
    # for energy, percentage, unit in energy_spectrum:
    #     weights.append(percentage);
    #     energies_in_keV.append(energy * gvxr.getUnitOfEnergy(unit) / gvxr.getUnitOfEnergy("keV"));

    # Set up the beam
    distance_source_detector_in_m  = 145.0;

    gvxr.setSourcePosition(distance_source_detector_in_m - distance_object_detector_in_m,  0.0, 0.0, "mm");
    gvxr.usePointSource();
    gvxr.useParallelBeam();


    # The material properties (chemical composition and density)
    fibre_radius = 140 / 2; # um
    # fibre_material = [("Si", 0.5), ("C", 0.5)];
    #fibre_mu = 2.736; # cm-1
    fibre_density = 3.2; # g/cm3

    core_radius = 30 / 2; # um
    # core_material = [("W", 1)];
    # core_mu = 341.61; # cm-1
    # core_density = 19.3 # g/cm3

    # matrix_material = [("Ti", 0.9), ("Al", 0.06), ("V", 0.04)];
    #matrix_mu = 13.1274; # cm-1
    matrix_density = 4.42 # g/cm3

    # The response of the detector as the line-spread function (LSF)
    t = np.arange(-20., 21., 1.);
    lsf_kernel=lsf(t*41)/lsf(0);
    lsf_kernel/=lsf_kernel.sum();


def computeSinogramFromFlatField(normalised_projections):
    """This function apply the minus log normalisation
    on the projections that bave been corrected with the flat-field method.

    It returns the sinogram.
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




def setFibres(aCentroidSet):
    """This function loads a cylinders in the GPU memory.
    Some are hollow and represent the fibres, some are not and
    correspond to the cores.

    aCentroidSet: a list of cylinder centres.
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
        x = pixel_spacing_in_micrometre * -(cyl[0] - reference_CT.shape[1] / 2 + 0.5);
        y = pixel_spacing_in_micrometre * (cyl[1] - reference_CT.shape[0] / 2 + 0.5);

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


def tomographyAcquisition():
    """This function simulate a CT acquisition.

    It returns the raw projections in keV
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
    for angle_id in range(0, number_of_projections):
        gvxr.resetSceneTransformation();
        gvxr.rotateScene(-angular_step * angle_id, 0, 1, 0);

        # Compute the X-ray image
        line_of_L_buffer = np.array(gvxr.computeLBuffer(object));

        # Add the projection
        L_buffer.append(line_of_L_buffer);

    # Return as a numpy array
    return np.array(L_buffer);

def simulateSinogram(sigma_set = None, k_set = None):


    # Do not simulate the phase contrast using a Laplacian
    if isinstance(sigma_set, type(None)) or isinstance(k_set, type(None)):

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
        for label, sigma in zip(["core", "fibre", "matrix"], sigma_set):
            # Get its L-buffer
            L_buffer_set[label] = getLBuffer(label);
            laplacian_kernels[label] = laplacian(pixel_range, sigma);

        # For each energy in the beam spectrum
        attenuation = {};
        attenuation_fibre = {};
        projection_per_energy_channel = {};
        phase_contrast_image = {};

        # Create a blank image
        raw_projections_in_keV = np.zeros(L_buffer_set["fibre"].shape);

        phase_contrast_image_min = sys.float_info.max;
        phase_contrast_image_max = -sys.float_info.max;

        for energy, photon_count in zip(gvxr.getEnergyBins("keV"), gvxr.getPhotonCountEnergyBins()):

            # Create a blank image
            attenuation[energy] = np.zeros(L_buffer_set["fibre"].shape);
            phase_contrast_image[energy] = {};

            # Look at all the children of the root node
            #for label in ["core", "fibre", "matrix"]:
            for label, sigma, in zip(["core", "fibre", "matrix"], sigma_set):
                # Get mu for this object for this energy
                mu = gvxr.getLinearAttenuationCoefficient(label, energy, "keV");

                # Compute mu * x
                temp = L_buffer_set[label] * mu;
                attenuation[energy] += temp;

                phase_contrast_image[energy][label] = [];

                for y in range(attenuation[energy].shape[1]):
                    for x in range(attenuation[energy].shape[0]):
                        phase_contrast_image[energy][label].append(np.convolve(temp[x][y], laplacian_kernels[label], mode='same'));

                phase_contrast_image[energy][label] = np.array(phase_contrast_image[energy][label]);
                phase_contrast_image[energy][label].shape = L_buffer_set[label].shape;

                phase_contrast_image_min = min(phase_contrast_image_min, np.min(phase_contrast_image[energy][label]));
                phase_contrast_image_max = max(phase_contrast_image_min, np.max(phase_contrast_image[energy][label]));

            # Store the projection for this energy channel
            projection_per_energy_channel[energy] = energy * photon_count * np.exp(-attenuation[energy]);

        # Create the raw projections
        raw_projections_in_keV = np.zeros(L_buffer_set["fibre"].shape);

        for energy in gvxr.getEnergyBins("keV"):
            raw_projections_in_keV += projection_per_energy_channel[energy];
            for label, k in zip(["core", "fibre", "matrix"], k_set):
                # Normalise it
                #phase_contrast_image[energy][label] /= max(phase_contrast_image_max, abs(phase_contrast_image_min));

                if label == "fibre":
                    raw_projections_in_keV -= k * phase_contrast_image[energy][label];

    # Apply the LSF line by line
    for z in range(raw_projections_in_keV.shape[0]):
        for y in range(raw_projections_in_keV.shape[1]):
            raw_projections_in_keV[z][y] = np.convolve(raw_projections_in_keV[z][y], lsf_kernel, mode='same');

    normalised_projections = flatFieldCorrection(raw_projections_in_keV);
    simulated_sinogram = computeSinogramFromFlatField(normalised_projections);

    return simulated_sinogram, normalised_projections, raw_projections_in_keV;

def fitnessFunctionCube(x):
    global best_fitness;
    global matrix_id;
    global reference_sinogram;
    global centroid_set;
    global use_fibres;

    setMatrix(x);

    # Load the cores and fibres
    if use_fibres:
        setFibres(centroid_set);

    # Simulate a sinogram
    simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram();
    normalised_simulated_sinogram = (simulated_sinogram - simulated_sinogram.mean()) / simulated_sinogram.std();


    # Compute the fitness function
    MAE = np.mean(np.abs(np.subtract(normalised_simulated_sinogram.flatten(), normalised_reference_sinogram.flatten())));
    #MAE = np.mean(np.abs(np.subtract(reference_sinogram.flatten(), simulated_sinogram.flatten())));
#     ZNCC = np.mean(np.multiply(normalised_simulated_sinogram.flatten(), normalised_reference_sinogram.flatten()));

    return MAE;


def float2uint8(anImage):
    uchar_image = copy.deepcopy(anImage);
    uchar_image -= np.min(uchar_image);
    uchar_image /= np.max(uchar_image);
    uchar_image *= 255;
    return uchar_image.astype(np.uint8);


def findCircles(anInputVolume):
    # Convert in UINT8 and into a SITK image
    volume = sitk.GetImageFromArray(float2uint8(anInputVolume));
    volume.SetSpacing([pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);

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

    return centroid_set;

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
        new_centroid_set. append([cyl[0] - best_x_offset, cyl[1] - best_y_offset]);

    return new_centroid_set;

def fitnessFunctionFibres(x):
    global best_fitness;
    global radius_fibre_id;

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
    #MAE = np.mean(np.abs(np.subtract(reference_sinogram.flatten(), simulated_sinogram.flatten())));
    # ZNCC = np.mean(np.multiply(normalised_simulated_sinogram.flatten(), normalised_reference_sinogram.flatten()));

    # Reconstruct the corresponding CT slice
#     theta = theta / 180.0 * math.pi;
#     rot_center = int(simulated_sinogram.shape[2]/2);
#     reconstruction_tomopy = tomopy.recon(simulated_sinogram, theta, center=rot_center, algorithm="gridrec", sinogram_order=False);


    #simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
    #reconstruction_tomopy = iradon(simulated_sinogram.T, theta=theta, circle=True);
    #normalised_simulated_CT = (reconstruction_tomopy - reconstruction_tomopy.mean()) / reconstruction_tomopy.std();
    #MAE_CT = np.mean(np.abs(normalised_simulated_CT.flatten() - normalised_reference_CT.flatten()));
    #ZNCC_CT = np.mean(np.multiply(normalised_simulated_CT.flatten(), normalised_reference_CT.flatten()));

    # Save the data
    fitness = MAE;
    # if best_fitness > fitness:
    #     best_fitness = fitness;




    return fitness;

def fitnessFunctionLaplacian(x):
    global best_fitness;
    global laplacian_id;
    global value_range;
    global num_samples;
    global best_centre;

    sigma_core = x[0];
    sigma_fibre = x[1];
    sigma_matrix = x[2];
    k_core = x[3];
    k_fibre = x[4];
    k_matrix = x[5];
    # value_range = x[2];
    # num_samples = x[3];

    # Get the radii
    # fibre_radius = x[4];
    fibre_radius = x[6];

    # Load the matrix
    setMatrix(current_best);

    # Load the cores and fibres
    setFibres(centroid_set);

    # Simulate a sinogram
    simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram([sigma_core, sigma_fibre, sigma_matrix], [k_core, k_fibre, k_matrix]);
    normalised_simulated_sinogram = (simulated_sinogram - simulated_sinogram.mean()) / simulated_sinogram.std();
    MAE_sinogram = np.mean(np.abs(normalised_simulated_sinogram.flatten() - normalised_reference_sinogram.flatten()));
    ZNCC_sinogram = np.mean(np.multiply(normalised_simulated_sinogram.flatten(), normalised_reference_sinogram.flatten()));

    # Reconstruct the corresponding CT slice
    simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
    CT_laplacian = iradon(simulated_sinogram.T, theta=theta, circle=True);
    normalised_CT_laplacian = (CT_laplacian - CT_laplacian.mean()) / CT_laplacian.std();

    reference_image = copy.deepcopy(reference_CT[best_centre[1] - roi_length:best_centre[1] + roi_length, best_centre[0] - roi_length:best_centre[0] + roi_length]);
    test_image = copy.deepcopy(CT_laplacian[best_centre[1] - roi_length:best_centre[1] + roi_length, best_centre[0] - roi_length:best_centre[0] + roi_length]);

    # MAE_sinogram = np.mean(np.abs(np.subtract(reference_sinogram.flatten(), simulated_sinogram.flatten())));

    #
    # normalised_simulated_sinogram.shape = (normalised_simulated_sinogram.size // normalised_simulated_sinogram.shape[2], normalised_simulated_sinogram.shape[2]);
    #
    # SSIM_sinogram = ssim(normalised_simulated_sinogram, normalised_reference_sinogram, data_range=normalised_reference_sinogram.max() - normalised_reference_sinogram.min())

    # Reconstruct the corresponding CT slice
#     theta = theta / 180.0 * math.pi;
#     rot_center = int(simulated_sinogram.shape[2]/2);
#     reconstruction_tomopy = tomopy.recon(simulated_sinogram, theta, center=rot_center, algorithm="gridrec", sinogram_order=False);





    # simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
    # CT_laplacian = iradon(simulated_sinogram.T, theta=theta, circle=True);


    # offset = min(np.min(CT_laplacian), np.min(reference_CT));
    #
    # reconstruction_CT_laplacian = CT_laplacian - offset;
    # reference_CT = reference_CT - offset;
    # reconstruction_CT_laplacian += 0.5;
    # reference_CT += 0.5;
    #
    # reconstruction_CT_laplacian = np.log(reconstruction_CT_laplacian);
    # reference_CT = np.log(reference_CT);
    #
    # normalised_simulated_CT = (reconstruction_CT_laplacian - reconstruction_CT_laplacian.mean()) / reconstruction_CT_laplacian.std();
    # temp_reference_CT = (reference_CT - reference_CT.mean()) / reference_CT.std();
    #
    MAE_CT = np.mean(np.abs(np.subtract(reference_CT.flatten(), CT_laplacian.flatten())));
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


    MAE_fibre = np.mean(np.abs(np.subtract(reference_image.flatten(), test_image.flatten())));


    # SSIM_fibre = ssim(reference_image, test_image, data_range=reference_image.max() - reference_image.min())

    fitness = MAE_sinogram;
    fitness = MAE_fibre;
    fitness = MAE_CT;
    #fitness = 1 / (ZNCC_CT + 1);

    if best_fitness > fitness:
        best_fitness = fitness;


        volume = sitk.GetImageFromArray(CT_laplacian);
        volume.SetSpacing([pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);
        sitk.WriteImage(volume, output_directory + "/reconstruction_CT_laplacian_" + str(laplacian_id) + ".mha", useCompression=True);

        volume = sitk.GetImageFromArray(CT_laplacian[best_centre[1] - roi_length:best_centre[1] + roi_length,
                                        best_centre[0] - roi_length:best_centre[0] + roi_length]);
        volume.SetSpacing([pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);
        sitk.WriteImage(volume, output_directory + "/reconstruction_CT_laplacian_fibre_centre_" + str(laplacian_id) + ".mha", useCompression=True);

        comp_equalized = compare_images(reference_image, test_image, method='checkerboard');
        volume = sitk.GetImageFromArray(comp_equalized)
        sitk.WriteImage(volume, output_directory + "/laplacian_comp_fibre_" + str(laplacian_id) + ".mha", useCompression=True);

        comp_equalized -= np.min(comp_equalized);
        comp_equalized /= np.max(comp_equalized);
        comp_equalized *= 255;
        comp_equalized = np.array(comp_equalized, dtype=np.uint8);
        io.imsave(output_directory + "/laplacian_comp_fibre_" + str(laplacian_id) + ".png", comp_equalized);

        comp_equalized = compare_images(reference_CT, CT_laplacian, method='checkerboard');
        volume = sitk.GetImageFromArray(comp_equalized)
        sitk.WriteImage(volume, output_directory + "/laplacian_comp_slice_" + str(laplacian_id) + ".mha", useCompression=True);

        comp_equalized -= np.min(comp_equalized);
        comp_equalized /= np.max(comp_equalized);
        comp_equalized *= 255;
        comp_equalized = np.array(comp_equalized, dtype=np.uint8);
        io.imsave(output_directory + "/laplacian_comp_slice_" + str(laplacian_id) + ".png", comp_equalized);

        laplacian_id += 1;


    reference_image = (reference_image - reference_image.mean()) / reference_image.std();
    test_image = (test_image - test_image.mean()) / test_image.std();
    ZNCC_fibre = np.mean(np.multiply(reference_image.flatten(), test_image.flatten()));
    print("IND", x[0], x[1], x[2], x[3], x[4], x[5], x[6], MAE_sinogram, ZNCC_sinogram, MAE_CT, ZNCC_CT, MAE_fibre, ZNCC_fibre);

    return fitness;
