import copy, sys
import numpy as np

from skimage.transform import iradon
from skimage.util import compare_images
import skimage.io as io

import SimpleITK as sitk
import cv2

import gvxrPython3 as gvxr
from lsf import *


# Global variables
NoneType = type(None);
pixel_spacing_in_micrometre = 1.9;
pixel_spacing_in_mm = pixel_spacing_in_micrometre * 1e-3;
number_of_projections = 900;
angular_span_in_degrees = 180.0;
angular_step = angular_span_in_degrees / number_of_projections;
theta = np.linspace(0., angular_span_in_degrees, number_of_projections, endpoint=False);
roi_length = 40;

fibre_radius = 140 / 2; # um
core_radius = 30 / 2; # um

value_range = 6;
num_samples = 15;
use_fibres = False;

centroid_set = None;
matrix_geometry_parameters = None;
cylinder_position_in_centre_of_slice = None;

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


def flatFieldCorrection(raw_projections_in_keV):
    """This function applies the flat-field correction on raw projections.
    Because the data suffers from a fixed-pattern noise in X-ray imaging in
    actual experiments, it is necessary to perform the flat-field correction of
    the raw projections using:

    $$normalised\_projections = \frac{raw\_projections − dark\_field}{flat\_field\_image − dark\_field}$$

    - $raw\_projections$ are the raw projections with the X-ray beam turned on and with the scanned object,
    - $flat\_field\_image$ is an image with the X-ray beam turned on but without the scanned object, and
    - $dark\_field$ is an image with the X-ray beam turned off.

    Note that in our example, $raw\_projections$, $flat\_field\_image$ and $dark\_field$ are in keV whereas $normalised\_projections$ does not have any unit:

    $$0 \leq raw\_projections \leq  \sum_E N_0(E) \times E\\0 \leq normalised\_projections \leq 1$$

    Return: the projections (raw_projections_in_keV) after flat-field correction
    """

    # Create a mock dark field image
    dark_field_image = np.zeros(raw_projections_in_keV.shape);

    # Create a mock flat field image
    flat_field_image = np.zeros(raw_projections_in_keV.shape);

    # Retrieve the total energy
    total_energy = 0.0;
    energy_bins = gvxr.getEnergyBins("keV");
    photon_count_per_bin = gvxr.getPhotonCountEnergyBins();

    for energy, count in zip(energy_bins, photon_count_per_bin):
        total_energy += energy * count;
    flat_field_image = np.ones(raw_projections_in_keV.shape) * total_energy;

    # Apply the actual flat-field correction on the raw projections
    return (raw_projections_in_keV - dark_field_image) / (flat_field_image - dark_field_image);


def laplacian(x, sigma):
    """This function create a Laplacian kernel with

    $$ g''(x) = \left(\frac{x^2}{\sigma^4} - \frac{1}{\sigma^2}\right) \exp\left(-\frac{x^2}{2\sigma^2}\right) $$
    """
    kernel = (np.power(x, 2.) / math.pow(sigma, 4) - 1. / math.pow(sigma, 2)) * np.exp(-np.power(x, 2.) / (2. * math.pow(sigma, 2)));

    # Make sure the sum of all the kernel elements is NULL
    index_positive = kernel > 0.0;
    index_negative = kernel < 0.0;
    sum_positive = kernel[index_positive].sum();
    sum_negative = kernel[index_negative].sum();

    #kernel[index_negative] = -kernel[index_negative] / sum_negative * sum_positive;
    # kernel[index_positive] = -kernel[index_positive] / sum_positive * sum_negative;

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

temp_count=0;
def simulateSinogram(sigma_set = None, k_set = None, name_set = None):

    global temp_count;

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
            # np.savetxt(output_directory + "/laplacian_" + str(temp_count) + ".dat", laplacian_kernels[label]);

            # print(L_buffer_set[label].shape)
            # print(phase_contrast_image.shape)
            # print(laplacian_kernels[label].shape)
            for z in range(phase_contrast_image.shape[0]):
                for y in range(phase_contrast_image.shape[1]):
                    phase_contrast_image[z][y] += np.convolve((L_buffer_set[label])[z][y], laplacian_kernels[label], mode='same');


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

        # Apply the phase contrastt
        raw_projections_in_keV -= phase_contrast_image;

        # phase_contrast_image.shape = [900, 1024];
        # volume = sitk.GetImageFromArray(phase_contrast_image);
        # # volume.SetSpacing([pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);
        # sitk.WriteImage(volume, output_directory + "/phase_contrast_image.mha", useCompression=True);

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

    global core_radius;
    global fibre_radius;

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

    # Convert the numpy array in float32 into uint8, then into a SITK image
    volume = sitk.GetImageFromArray(float2uint8(anInputVolume));
    volume.SetSpacing([pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);

    # Apply the Otsu's method
    otsu_filter = sitk.OtsuThresholdImageFilter();
    otsu_filter.SetInsideValue(0);
    otsu_filter.SetOutsideValue(1);
    seg = otsu_filter.Execute(volume);

    # Print the corresponding threshold
    print("Threshold:", otsu_filter.GetThreshold());

    # Clean-up using mathematical morphology
    cleaned_thresh_img = sitk.BinaryOpeningByReconstruction(seg, [3, 3, 3])
    cleaned_thresh_img = sitk.BinaryClosingByReconstruction(cleaned_thresh_img, [3, 3, 3])


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
    global fibre_radius;
    global core_radius;

    # Get the radii
    fibre_radius = x[0];
    core_radius = fibre_radius * x[1];

    # Load the matrix
    setMatrix(matrix_geometry_parameters);

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

best_fitness = sys.float_info.max;
laplacian_id = 0;

def fitnessFunctionLaplacian(x):
    global best_fitness;
    global laplacian_id;
    global value_range;
    global num_samples;
    global cylinder_position_in_centre_of_slice;
    global roi_length;
    global fibre_radius;
    global core_radius;

    sigma_core = x[0];
    k_core = x[1];
    sigma_fibre = x[2];
    k_fibre = x[3];
    sigma_matrix = x[4];
    k_matrix = x[5];
    fibre_radius = x[6];

    # Load the matrix
    setMatrix(matrix_geometry_parameters);

    # Load the cores and fibres
    setFibres(centroid_set);

    # Simulate a sinogram
    simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram([sigma_core, sigma_fibre, sigma_matrix], [k_core, k_fibre, k_matrix], ["core", "fibre", "matrix"]);
    normalised_simulated_sinogram = (simulated_sinogram - simulated_sinogram.mean()) / simulated_sinogram.std();
    MAE_sinogram = np.mean(np.abs(normalised_simulated_sinogram.flatten() - normalised_reference_sinogram.flatten()));
    ZNCC_sinogram = np.mean(np.multiply(normalised_simulated_sinogram.flatten(), normalised_reference_sinogram.flatten()));

    # # Reconstruct the corresponding CT slice
    # simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
    # CT_laplacian = iradon(simulated_sinogram.T, theta=theta, circle=True);
    # normalised_CT_laplacian = (CT_laplacian - CT_laplacian.mean()) / CT_laplacian.std();
    #
    # reference_image = copy.deepcopy(normalised_reference_CT[cylinder_position_in_centre_of_slice[1] - roi_length:cylinder_position_in_centre_of_slice[1] + roi_length, cylinder_position_in_centre_of_slice[0] - roi_length:cylinder_position_in_centre_of_slice[0] + roi_length]);
    # test_image = copy.deepcopy(normalised_CT_laplacian[cylinder_position_in_centre_of_slice[1] - roi_length:cylinder_position_in_centre_of_slice[1] + roi_length, cylinder_position_in_centre_of_slice[0] - roi_length:cylinder_position_in_centre_of_slice[0] + roi_length]);

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
    # MAE_CT = np.mean(np.abs(np.subtract(reference_CT.flatten(), CT_laplacian.flatten())));
    # ZNCC_CT = np.mean(np.multiply(normalised_reference_CT.flatten(), normalised_CT_laplacian.flatten()));
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


    # MAE_fibre = np.mean(np.abs(np.subtract(reference_image.flatten(), test_image.flatten())));


    # SSIM_fibre = ssim(reference_image, test_image, data_range=reference_image.max() - reference_image.min())

    fitness = MAE_sinogram;
    # fitness = MAE_fibre;
    # fitness = MAE_CT;
    #fitness = 1 / (ZNCC_CT + 1);

    if best_fitness > fitness:
        best_fitness = fitness;

        simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
        CT_laplacian = iradon(simulated_sinogram.T, theta=theta, circle=True);
        normalised_CT_laplacian = (CT_laplacian - CT_laplacian.mean()) / CT_laplacian.std();

        reference_image = copy.deepcopy(reference_CT[cylinder_position_in_centre_of_slice[1] - roi_length:cylinder_position_in_centre_of_slice[1] + roi_length, cylinder_position_in_centre_of_slice[0] - roi_length:cylinder_position_in_centre_of_slice[0] + roi_length]);
        test_image = copy.deepcopy(CT_laplacian[cylinder_position_in_centre_of_slice[1] - roi_length:cylinder_position_in_centre_of_slice[1] + roi_length, cylinder_position_in_centre_of_slice[0] - roi_length:cylinder_position_in_centre_of_slice[0] + roi_length]);

        volume = sitk.GetImageFromArray(CT_laplacian);
        volume.SetSpacing([pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);
        sitk.WriteImage(volume, output_directory + "/reconstruction_CT_laplacian_" + str(laplacian_id) + ".mha", useCompression=True);

        volume = sitk.GetImageFromArray(CT_laplacian[cylinder_position_in_centre_of_slice[1] - roi_length:cylinder_position_in_centre_of_slice[1] + roi_length,
                                        cylinder_position_in_centre_of_slice[0] - roi_length:cylinder_position_in_centre_of_slice[0] + roi_length]);
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


    # reference_image = (reference_image - reference_image.mean()) / reference_image.std();
    # test_image = (test_image - test_image.mean()) / test_image.std();
    # ZNCC_fibre = np.mean(np.multiply(reference_image.flatten(), test_image.flatten()));
    print("IND", x[0], x[1], x[2], x[3], x[4], x[5], x[6], fitness, ZNCC_sinogram);

    return fitness;



def reconstructAndStoreResults(simulated_sinogram, aPrefix):

    global reference_CT;
    global normalised_reference_CT;
    global cylinder_position_in_centre_of_slice;

    # Reconstruct the CT slice
    simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
    CT_slice_from_simulated_sinogram = iradon(simulated_sinogram.T, theta=theta, circle=True);

    # Save the CT slice
    volume = sitk.GetImageFromArray(CT_slice_from_simulated_sinogram);
    volume.SetSpacing([pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);
    sitk.WriteImage(volume, aPrefix + "_CT_slice.mha", useCompression=True);

    # Compute a mosaic in linear attenuation coefficients
    comp_equalized = compare_images(reference_CT, CT_slice_from_simulated_sinogram, method='checkerboard');
    volume = sitk.GetImageFromArray(comp_equalized);
    volume.SetSpacing([pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);
    sitk.WriteImage(volume, aPrefix + "_compare_experiment_with_simulation.mha", useCompression=True);

    # Compute a mosaic after zero-mean, unit variance normalisation
    normalised_CT_slice = (CT_slice_from_simulated_sinogram - CT_slice_from_simulated_sinogram.mean()) / CT_slice_from_simulated_sinogram.std();
    comp_equalized = compare_images(normalised_reference_CT, normalised_CT_slice, method='checkerboard');
    comp_equalized -= np.min(comp_equalized);
    comp_equalized /= np.max(comp_equalized);
    comp_equalized *= 255;
    comp_equalized = np.array(comp_equalized, dtype=np.uint8);
    io.imsave(aPrefix + "_compare_experiment_with_simulation.png", comp_equalized)

    # Compute the ZNCC
    ZNCC_CT = np.mean(np.multiply(normalised_CT_slice.flatten(), normalised_reference_CT.flatten()));

    return ZNCC_CT, CT_slice_from_simulated_sinogram;


def findFibreInCentreOfCtSlice():
    global centroid_set;
    global reference_CT;
    global cylinder_position_in_centre_of_slice;

    # Find the cylinder in the centre of the image
    cylinder_position_in_centre_of_slice = None;
    best_distance = sys.float_info.max;

    for centre in centroid_set:
        distance = math.pow(centre[0] - reference_CT.shape[1] / 2,2 ) + math.pow(centre[1] - reference_CT.shape[0] / 2, 2);

        if best_distance > distance:
            best_distance = distance;
            cylinder_position_in_centre_of_slice = copy.deepcopy(centre);

    return cylinder_position_in_centre_of_slice;


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return np.array(mask, dtype=bool);

def createMasks(mask_shape):
    fibre_radius_in_px = fibre_radius / pixel_spacing_in_micrometre
    core_radius_in_px = core_radius / pixel_spacing_in_micrometre

    core_mask = create_circular_mask(mask_shape[1], mask_shape[0], None, core_radius_in_px);

    fibre_mask = create_circular_mask(mask_shape[1], mask_shape[0], None, fibre_radius_in_px);
    matrix_mask = np.logical_not(fibre_mask);

    #fibre_mask = np.subtract(fibre_mask, core_mask);
    fibre_mask = np.bitwise_xor(fibre_mask, core_mask);

    #TypeError: numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, the `^` operator, or the logical_xor function instead.

    return core_mask, fibre_mask, matrix_mask

def printMuStatistics(text, reference_fibre_in_centre, test_fibre_in_centre, core_mask, fibre_mask, matrix_mask):

    index = np.nonzero(core_mask);
    print(text, "CORE REF (MIN, MEDIAN, MAX, MEAN, STDDEV):",
            np.min(reference_fibre_in_centre[index]),
            np.median(reference_fibre_in_centre[index]),
            np.max(reference_fibre_in_centre[index]),
            np.mean(reference_fibre_in_centre[index]),
            np.std(reference_fibre_in_centre[index]));

    print(text, "CORE SIMULATED (MIN, MEDIAN, MAX, MEAN, STDDEV):",
            np.min(test_fibre_in_centre[index]),
            np.median(test_fibre_in_centre[index]),
            np.max(test_fibre_in_centre[index]),
            np.mean(test_fibre_in_centre[index]),
            np.std(test_fibre_in_centre[index]));

    index = np.nonzero(fibre_mask);
    print(text, "FIBRE REF (MIN, MEDIAN, MAX, MEAN, STDDEV):",
            np.min(reference_fibre_in_centre[index]),
            np.median(reference_fibre_in_centre[index]),
            np.max(reference_fibre_in_centre[index]),
            np.mean(reference_fibre_in_centre[index]),
            np.std(reference_fibre_in_centre[index]));

    print(text, "FIBRE SIMULATED (MIN, MEDIAN, MAX, MEAN, STDDEV):",
            np.min(test_fibre_in_centre[index]),
            np.median(test_fibre_in_centre[index]),
            np.max(test_fibre_in_centre[index]),
            np.mean(test_fibre_in_centre[index]),
            np.std(test_fibre_in_centre[index]));

    index = np.nonzero(matrix_mask);
    print(text, "MATRIX REF (MIN, MEDIAN, MAX, MEAN, STDDEV):",
            np.min(reference_fibre_in_centre[index]),
            np.median(reference_fibre_in_centre[index]),
            np.max(reference_fibre_in_centre[index]),
            np.mean(reference_fibre_in_centre[index]),
            np.std(reference_fibre_in_centre[index]));

    print(text, "MATRIX SIMULATED (MIN, MEDIAN, MAX, MEAN, STDDEV):",
            np.min(test_fibre_in_centre[index]),
            np.median(test_fibre_in_centre[index]),
            np.max(test_fibre_in_centre[index]),
            np.mean(test_fibre_in_centre[index]),
            np.std(test_fibre_in_centre[index]));
