# From Synchrotron Microtomography to CAD Models using Optimisation and Fast X-ray Simulation on GPU:

## Registration of Tungsten Fibres on XCT Images

### by Franck P. Vidal and Jean-Michel L&eacute;tang

This demo aims to demonstrate how to register polygon meshes onto X-ray microtomography (micro-CT) scans of a tungsten fibre. The code relies on two building blocks:

1.  A global optimisation algorithm. We use the [CMA-ES (Covariance Matrix Adaptation Evolution Strategy)](http://cma.gforge.inria.fr/http://cma.gforge.inria.fr/). It is an evolutionary algorithm for difficult non-linear non-convex optimisation problems.
2.  A fast X-ray simulation toolkit. We use [gVirtualXRay](http://gvirtualxray.sourceforge.net/)[gVirtualXRay](http://gvirtualxray.sourceforge.net/). It is a framework supporting many modern programming languages to generate realistic X-ray images from polygon meshes (triangles or tetrahedrons) on the graphics processor unit (GPU).

Below is an example of CT slice from an experiment we carried out at the [European Synchrotron Radiation Facility (ESRF)European Synchrotron Radiation Facility (ESRF)](https://www.esrf.fr/https://www.esrf.fr/).

![The fibre.](doc/scanned_object.png)

In a previous article, on [*Investigation of artefact sources in synchrotron microtomography via virtual X-ray imaging*](https://doi.org/10.1016/j.nimb.2005.02.003) in [Nuclear Instruments and Methods in Physics Research Section B: Beam Interactions with Materials and Atoms](https://www.sciencedirect.com/journal/nuclear-instruments-and-methods-in-physics-research-section-b-beam-interactions-with-materials-and-atoms), we demonstrated that the image above was corrupted by:

1) beam hardening depsite the use of a monochromator,
2) the response of the camera despite the point spread function (PSF) being almost a Dirac, and
3) phase contrast.

That study was published in 2005, when computer were still relatively slow. Since then, massively parallel processors such as graphics processor units (GPUs) have emerged. Using today's hardware, we will demonstrate that we can now finely tuned the virtual experiments by mathematical optimisation to register polygons meshes on XCT data. Our simulations will include beam-hardening due to polychromatism, take into account the response of the detector, and have phase contrast.

## Registration steps

1. Initialisation
    - [Import Python packages](#Import-packages)
    - [Global variables](#Global-variables) with values corresponding to known parameters
    - [Load the image data from the experiment at ESRF](#Load-the-image-data)
    - [Recontruct the corresponding CT data](#CT-Reconstruction)
    - [Normalise the image data](#Normalise-the-image-data)
    - [Set the X-ray simulation environment](#Set-the-X-ray-simulation-environment)
    - [LSF](#The-LSF)
    - [Find circles to identify the centre of fibres](#Find-circles-to-identify-the-centre-of-fibres)
2. [Simulate the CT acquisition](#Simulate-the-CT-acquisition)
3. [Registration of a cube](#Registration-of-a-cube)
4. [Optimisation of the cores and fibres radii](#Optimisation-of-the-cores-and-fibres-radii)
5. [Recentre each core/fibre](#Recentre-each-core/fibre)
6. [Optimisation the radii after recentring](#Optimisation-the-radii-after-recentring)
7. [Optimisation of the beam spectrum](#Optimisation-of-the-beam-spectrum)
8. [Optimisation of the Poisson noise](#Optimisation-of-the-Poisson-noise)
9. [Optimisation of the phase contrast and the radii](#Optimisation-of-the-phase-contrast-and-the-radii)
10. [Optimisation of the phase contrast and the LSF](#Optimisation-of-the-phase-contrast-and-the-LSF)

## Import packages

We need to import a few libraries (called packages in Python). We use:

- `copy`: duplicating images using deepcopies;
- `glob`: retrieving file names in a directory;
- `math`: the `floor` function;
- `os`: creating a new directory;
- `sys`: retrieving the largest possible floating-point value;
- `cma`: non-linear numerical optimization ([CMA-ES, Covariance Matrix Adaptation Evolution Strategy](https://github.com/CMA-ES/pycma));
- ([OpenCV](https://www.opencv.org/)) (`cv2`): Hough transform and bilateral filter (an edge-preserving smoothing filter);
- `imageio`: creating GIF files;
- `matplotlib`: plotting data;
- `numpy`: who doesn't use numpy?
- `[SimpleITK](https://simpleitk.org/))`: image processing and saving volume data;
- `tomopy`: package for CT reconstruction;
- `scipy`: for the convolution of a 2D image by a 1D kernel;
- `skimage`: comparing the reference CT slice and the simulated one;
- `sklearn`: comparing the reference CT slice and the simulated one;
- `lsf`: the line spread function to filter the X-ray images; and
- `gvxrPython3`: simulation of X-ray images using the Beer-Lambert law on GPU.


```python
%matplotlib inline

import copy
import glob
import math
import os
import sys

import cma
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import tomopy
from matplotlib import cm
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim
from skimage.util import compare_images
from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower

import gvxrPython3 as gvxr

from lsf import *
```


```python
if not os.path.exists("outputs"):
    os.makedirs("outputs");

if not os.path.exists("plots"):
    os.makedirs("plots");
```

## Global variables

We need some global variables:

-  `NoneType`: the type of `None`;
-  `pixel_spacing_in_micrometre`: the physical distance between the centre of two successive pixel;
-  `pixel_spacing_in_mm`: the physical distance between the centre of two successive pixel;
-  `number_of_projections`: the total number of angles in the sinogram;he total number of angles in the sinogram;
-  `angular_span_in_degrees`: the angular span covered by the sinogram;
-  `angular_step`: the angular step;
-  `theta`: the rotation angles in degrees (vertical axis of the sinogram);
-  `theta_rad`: the rotation angles in radians (vertical axis of the sinogram);
-  `roi_length`: control the size of the ROI when displayng the central fibre;
-  `value_range`: control the binning of the Laplacian kernel
-  `num_samples`: control the binning of the Laplacian kernel
-  `sigma_set`: spread of the Laplacian kernels
-  `k_set`: weight of the Laplacian kernels
-  `label_set`: label of the structures on which a Laplacian kernel is applied
-  `bias`: control the bias of the Poisson noise
-  `gain`: control the gain of the Poisson noise: control the bias of the Poisson noise
-  `scale`: control the scale of the Poisson noise: control the bias of the Poisson noise
-  `use_normalisation`: use or do not use zero-mean, unit-variance normalisation in the objective functions;
-  `use_sinogram`: compute the objective functions on the sinogram or flat-field;
-  `metrics_type`: type of image comparison used in the objective functions;
-  `fibre_radius`: radius of the SiC fibres in um
-  `core_radius`: radius of the W fibres in um


```python
NoneType = type(None);
pixel_spacing_in_micrometre = 1.9;
pixel_spacing_in_mm = pixel_spacing_in_micrometre * 1e-3;
number_of_projections = 900;
angular_span_in_degrees = 180.0;
angular_step = angular_span_in_degrees / number_of_projections;
theta = np.linspace(0.,
                    angular_span_in_degrees,
                    number_of_projections,
                    endpoint=False);
theta_rad = theta / 180.0 * math.pi;

roi_length = 60;

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

metrics_type = "RMSE";

fibre_radius = 140 / 2;  # um
core_radius = 30 / 2;  # um
```

## Load the image data

Load and display the reference projections from a raw binary file, i.e. the target of the registration.


```python
# Target of the registration
reference_normalised_projections = np.fromfile("sino.raw", dtype=np.float32);
reference_normalised_projections.shape = [
    number_of_projections,
    int(reference_normalised_projections.shape[0] / number_of_projections)
];
```

We define a function to save raw images in the MHA format:


```python
def saveMHA(fname, image, spacing):
    """
    save the image into a file.

    :param str fname: the filename
    :param 2D_image image: the image to save
    :param [flt, flt, flt] spacing: the space between two successive voxels along the 3 direction
    """

    volume = sitk.GetImageFromArray(image);
    volume.SetSpacing(spacing);
    sitk.WriteImage(volume, fname, useCompression=True);
```

The reference projections in a MHA file


```python
saveMHA('outputs/reference_normalised_projections.mha', reference_normalised_projections, [pixel_spacing_in_mm, angular_step, pixel_spacing_in_mm]);
```

Display the reference projections using Matplotlib


```python
labels = [theta[0], theta[reference_normalised_projections.shape[0] // 2], theta[-1]];
tics = [
    0,
    reference_normalised_projections.shape[0] // 2,
    reference_normalised_projections.shape[0]-1
];
fig = plt.figure();
imgplot = plt.imshow(reference_normalised_projections, cmap="gray");
plt.xlabel("Displacement of projection");
plt.ylabel("Angle of projection (in degrees)");
plt.yticks(tics, labels);
plt.title("Projections after flat-field correction from the experiment at ESRF");
fig.colorbar(imgplot);
plt.savefig('plots/Normalised_projections_from_experiment_ESRF.pdf')
plt.savefig('plots/Normalised_projections_from_experiment_ESRF.png')
```



![png](doc/output_14_0.png)



In the literature, a projection is often modelled as follows:

<!--$$P = \ln\left(\frac{I_0}{I}\right) = -\ln\left(\frac{I}{I_0}\right) = \sum_n \mu(n) \Delta_x$$-->

$$I = \sum_i E \times N \times \text{e}^{-\sum_i(\mu_i \times \Delta_i)}$$

with $I$ the raw X-ray projection, and with the sample and with the the X-ray beam turned on;
$E$ and $N$ the energy in eV and the number of photons at that energy respectively;
$i$ the $i$-th material being scanned, $\mu_i$ its linear attenuation coefficient in cm$^{-1}$ at Energy $E$, and
$\Delta_i$ the path length of the ray crossing the $i$-th material from the X-ray source to the detector.

$$I_0 = E \times N$$

`reference_normalised_projections` above corresponds to the data loaded from the binary file. It corresponds to $\frac{I}{I_0}$, i.e. the flat-field correction has already been performed. It is now necessary to linearise the transmission tomography data using:

$$-\ln\left(\frac{I}{I_0}\right)$$

This new image corresponds to the Radon transform, known as sinogram, of the scanned object in these experimental conditions. Once this is done, we divide the pixels of the sinogram by $\Delta_x$, which is egal to the spacing between two successive pixels along the horizontal axis.

We define a new function to compute the sinogram from flat-field correction and calls it straightaway.


```python
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
```

Compute the sinogram from the flat-field data


```python
reference_sinogram = computeSinogramFromFlatField(reference_normalised_projections);
```

Save the corresponding image


```python
saveMHA('outputs/reference_sinogram.mha', reference_sinogram, [pixel_spacing_in_mm, angular_step, pixel_spacing_in_mm]);
```

Display the sinogram using Matplotlib


```python
labels=[theta[0], theta[reference_sinogram.shape[0] // 2], theta[-1]];
tics=[0, reference_sinogram.shape[0] // 2, reference_sinogram.shape[0]-1];
fig=plt.figure();
imgplot = plt.imshow(reference_sinogram, cmap="gray");
plt.xlabel("Displacement of projection");
plt.ylabel("Angle of projection (in degrees)");
plt.yticks(tics, labels);
plt.title("Sinogram of the reference image");
fig.colorbar(imgplot);
plt.savefig('plots/Sinogram_reference_image.pdf');
plt.savefig('plots/Sinogram_reference_image.png');
```



![png](doc/output_22_0.png)



## CT reconstruction

Now we got a sinogram, we can reconstruct the CT slice. As we used a synchrotron, we can assume we have a parallel source. It means we can use a FBP rather than the FDK algorithm. In fact we use the gridrec algorithm, which is much faster:

Dowd BA, Campbell GH, Marr RB, Nagarkar VV, Tipnis SV, Axe L, and Siddons DP. [Developments in synchrotron x-ray computed microtomography at the national synchrotron light source](https://doi.org/10.1117/12.363725). In Proc. SPIE, volume 3772, 224–236. 1999.


```python
reference_sinogram.shape = [
    reference_sinogram.shape[0],
    1,
    reference_sinogram.shape[1]
];

rot_center = int(reference_sinogram.shape[2]/2);

reference_CT = tomopy.recon(reference_sinogram,
                            theta_rad,
                            center=rot_center,
                            sinogram_order=False,
                            algorithm='gridrec',
                            filter_name='shepp')[0];
```

    Reconstructing 1 slice groups with 1 master threads...


Save the reconstruction in a MHA file


```python
saveMHA('outputs/reference_CT.mha', reference_CT, [pixel_spacing_in_mm, angular_step, pixel_spacing_in_mm]);
```

Plot the CT slice using Matplotlib


```python
fig=plt.figure();
norm = cm.colors.Normalize(vmax=30, vmin=-20)
imgplot = plt.imshow(reference_CT, cmap="gray", norm=norm);
fig.colorbar(imgplot);
plt.title("Reference image (in linear attenuation coefficients, cm$^{-1}$)");
plt.savefig('plots/reference_image_in_mu.pdf');
plt.savefig('plots/reference_image_in_mu.png');
```



![png](doc/output_28_0.png)



## Normalise the image data

Zero-mean, unit-variance normalisation is applied to use the reference images in objective functions and perform the registration. Note that it is called standardisation (or Z-score Normalisation) in machine learning. It is computed as follows:

$$I' = \frac{I - \bar{I}}{\sigma}$$

Where $I'$ is the image after the original image $I$ has been normalised, $\bar{I}$ is the average pixel value of $I$, and $\sigma$ is its standard deviation. We define a function to apply this:


```python
def standardisation(I):
    image = copy.deepcopy(I);

    # Sometimes the CT reconstruction algorithm create NaN on
    # the top and right borders, we filter them out using
    # a median filter ignoring NaN
    nan_index = np.argwhere(np.isnan(image));
    if nan_index.shape[0]:
        temp = np.pad(image, 1, "edge");

        for index in nan_index:
            roi = temp[index[0]-1+1:index[0]+1+2, index[1]-1+1:index[1]+1+2];
            image[index[0], index[1]] = np.nanmedian(roi);

    return (image - image.mean()) / image.std();

```

Normalise the reference sinogram and CT slice


```python
normalised_reference_sinogram = standardisation(reference_sinogram);
normalised_reference_CT       = standardisation(reference_CT);
```

## Set the X-ray simulation environment

First we create an OpenGL context, here using EGL, i.e. no window.


```python
gvxr.createWindow(0, 1, "EGL");
gvxr.setWindowSize(512, 512);
```

We set the parameters of the X-ray detector (flat pannel), e.g. number of pixels, pixel, spacing, position and orientation:

![3D scene to be simulated using gVirtualXray](./doc/3d_scene.png)


```python
detector_width_in_pixels = reference_sinogram.shape[2];
detector_height_in_pixels = 1;
distance_object_detector_in_m =    0.08; # = 80 mm

gvxr.setDetectorPosition(-distance_object_detector_in_m, 0.0, 0.0, "m");
gvxr.setDetectorUpVector(0, 1, 0);
gvxr.setDetectorNumberOfPixels(detector_width_in_pixels, detector_height_in_pixels);
gvxr.setDetectorPixelSize(pixel_spacing_in_micrometre, pixel_spacing_in_micrometre, "micrometer");
```

And the source parameters (beam shape, source position)


```python
# Set up the beam
distance_source_detector_in_m  = 145.0;

gvxr.setSourcePosition(distance_source_detector_in_m - distance_object_detector_in_m,  0.0, 0.0, "m");
gvxr.usePointSource();
# gvxr.useParallelBeam();
```

The beam spectrum. Here we have a polychromatic beam, with 97% of the photons at 33 keV, 2% at 66 keV and 1% at 99 keV.


```python
energy_spectrum = [(33, 0.97, "keV"), (66, 0.02, "keV"), (99, 0.01, "keV")];

for energy, percentage, unit in energy_spectrum:
    gvxr.addEnergyBinToSpectrum(energy, unit, percentage);
```

Plot the beam spectrum using Matplotlib


```python
energies_in_keV = [];
weights = [];

for energy, percentage, unit in energy_spectrum:
    weights.append(percentage);
    energies_in_keV.append(energy * gvxr.getUnitOfEnergy(unit) / gvxr.getUnitOfEnergy("keV"));

fig=plt.figure();
plt.xlabel("Energy bin (in keV)");
plt.ylabel("Relative weight");
plt.xticks(energies_in_keV);
plt.yticks(weights);
plt.title("Incident beam spectrum");
plt.bar(energies_in_keV, weights);
plt.savefig('plots/beam_spectrum.pdf');
plt.savefig('plots/beam_spectrum.png');
```



![png](doc/output_43_0.png)



The material properties (chemical composition and density)


```python
fibre_material = [("Si", 0.5), ("C", 0.5)];
fibre_density = 3.2; # g/cm3

core_radius = 30 / 2; # um
core_material = [("W", 1)];

g_matrix_width = 0;
g_matrix_height = 0;
g_matrix_x = 0;
g_matrix_y = 0;
matrix_material = [("Ti", 0.9), ("Al", 0.06), ("V", 0.04)];
matrix_density = 4.42 # g/cm3
```

### The LSF

In a previous study, we experimentally measured the impulse response of the detector as the line spread function (LSF):

F.P. Vidal, J.M. Létang, G. Peix, P. Cloetens, Investigation of artefact sources in synchrotron microtomography via virtual X-ray imaging, *Nuclear Instruments and Methods in Physics Research Section B: Beam Interactions with Materials and Atoms*, Volume 234, Issue 3, 2005, Pages 333-348, ISSN 0168-583X, DOI [10.1016/j.nimb.2005.02.003](10.1016/j.nimb.2005.02.003).

We use this model during the initial steps of the registration. The LSF model will be tuned in one of the final steps of the registration.


```python
t = np.arange(-20., 21., 1.);
lsf_kernel=lsf(t*41)/lsf(0);
lsf_kernel/=lsf_kernel.sum();
```

Plot the LSF using Matplotlib


```python
fig=plt.figure();
plt.title("Response of the detector (LSF)");
plt.plot(t, lsf_kernel);
plt.savefig('plots/LSF.pdf');
plt.savefig('plots/LSF.png');
```



![png](doc/output_49_0.png)



## Find circles to identify the centre of fibres

We can use the Hoguh transform to detect where circles are in the image. However, the input image in OpenCV's function must be in UINT. We blur it using a bilateral filter (an edge-preserving smoothing filter).

### Convert the image to UINT

We first create a function to convert images in floating point numbers into UINT.


```python
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
```

We blur the CT scan using a bilateral filter. It preserves edges.


```python
uint8_reference_CT = float2uint8(reference_CT, 0, 300);
blurred_reference_CT = cv2.bilateralFilter(uint8_reference_CT, 9, 75, 75);

saveMHA('outputs/blurred_reference_CT.mha', blurred_reference_CT, [pixel_spacing_in_mm, angular_step, pixel_spacing_in_mm]);
```

### Apply the Hough transform

As the fibres and the cores correspond to circles in the CT images, the obvious technique to try is the Hough Circle Transform (HCT). It is a feature extraction technique used in image analysis that can output a list of circles (centres and radii).


```python
circles = cv2.HoughCircles(blurred_reference_CT, cv2.HOUGH_GRADIENT, 2, 80,
                            param1=150, param2=5, minRadius=5, maxRadius=15);
```

### Overlay the detected circles on the top of the image


```python
cimg = cv2.cvtColor(blurred_reference_CT, cv2.COLOR_GRAY2BGR);
circles = np.uint16(np.around(circles));

for i in circles[0,:]:

    # draw the outer circle
    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2);

    # draw the center of the circle
    cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3);
```


```python
fig=plt.figure();
imgplot = plt.imshow(cimg);
plt.title("Reference image and detected Tungsten cores");
plt.savefig('plots/fibre_detection_using_Hough_transform.pdf');
plt.savefig('plots/fibre_detection_using_Hough_transform.png');
```



![png](doc/output_60_0.png)



13 fibres were missed and many centres were misplaced. Controlling the meta-parameters of the algorithm can be difficult to employ in a fully-automatic registration framework. We will use another technique to register the fibres, the popular Otsu's method. It creates a histogram and uses a heuristic to determine a threshold value.


```python
# Convert the numpy array in float32 into uint, then into a SITK image
volume = sitk.GetImageFromArray(blurred_reference_CT);
volume.SetSpacing([pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);

# Apply the Otsu's method
otsu_filter = sitk.OtsuThresholdImageFilter();
otsu_filter.SetInsideValue(0);
otsu_filter.SetOutsideValue(1);
seg = otsu_filter.Execute(volume);

# Print the corresponding threshold
print("Threshold:", otsu_filter.GetThreshold());
```

    Threshold: 91.0



```python
sitk.WriteImage(seg, "outputs/cores_segmentation.mha", useCompression=True);
```


```python
fig = plt.figure();
imgplot = plt.imshow(sitk.GetArrayViewFromImage(sitk.LabelOverlay(volume, seg)));
plt.title("Reference image and detected Tungsten cores");
plt.savefig('plots/fibre_detection_using_otsu_method.pdf');
plt.savefig('plots/fibre_detection_using_otsu_method.png');
```



![png](doc/output_64_0.png)



### Clean up


```python
# Clean-up using mathematical morphology
cleaned_thresh_img = sitk.BinaryOpeningByReconstruction(seg, [3, 3, 3])
cleaned_thresh_img = sitk.BinaryClosingByReconstruction(cleaned_thresh_img, [3, 3, 3])
```


```python
sitk.WriteImage(cleaned_thresh_img, "outputs/cores_cleaned_segmentation.mha", useCompression=True);
```


```python
fig = plt.figure();
imgplot = plt.imshow(sitk.GetArrayViewFromImage(sitk.LabelOverlay(volume, cleaned_thresh_img)));
plt.title("Reference image and detected Tungsten cores");
plt.savefig('plots/fibre_detection_using_otsu_method_after_cleaning.pdf');
plt.savefig('plots/fibre_detection_using_otsu_method_after_cleaning.png');
```



![png](doc/output_68_0.png)



## Mark each potential tungsten corewith unique label

Each distinct tungsten core is assigned a unique label, i.e. a unique pixel intensity


```python
core_labels = sitk.ConnectedComponent(cleaned_thresh_img);
```


```python
fig = plt.figure();
imgplot = plt.imshow(sitk.GetArrayViewFromImage(sitk.LabelOverlay(volume, core_labels)));
plt.title("Cleaned Binary Segmentation of the Tungsten cores");
plt.savefig('plots/fibre_detection_with_label_overlay.pdf');
plt.savefig('plots/fibre_detection_with_label_overlay.png');
```



![png](doc/output_72_0.png)



### Object Analysis

Once we have the segmented objects we look at their shapes and the intensity distributions inside the objects. For each labelled tungsten core, we extract the centroid. Note that sizes and positions are given in millimetres.


```python
shape_stats = sitk.LabelShapeStatisticsImageFilter()
shape_stats.ComputeOrientedBoundingBoxOn()
shape_stats.Execute(core_labels)
```


```python
centroid_set = [];

for i in shape_stats.GetLabels():
    centroid_set.append(cleaned_thresh_img.TransformPhysicalPointToIndex(shape_stats.GetCentroid(i)));
```

We now have a list of the centres of all the fibres that can be used as a parameter of the function below to create the cylinders corresponding to the cores and the fibres.
For each core, a cylinder is creatd and translated:
```python
        gvxr.emptyMesh("core_"  + str(i));
        gvxr.makeCylinder("core_"  + str(i), number_of_sectors, 815.0,  core_radius, "micrometer");
        gvxr.translateNode("core_"  + str(i), y, 0.0, x, "micrometer");
```
For each fibre, another cylinder is created and translated:
```python
        gvxr.emptyMesh("fibre_"  + str(i));
        gvxr.makeCylinder("fibre_"  + str(i), number_of_sectors, 815.0,  fibre_radius, "micrometer");
        gvxr.translateNode("fibre_"  + str(i), y, 0.0, x, "micrometer");
```
The fibre's cylinder is hollowed to make space for its core:
```python
        gvxr.subtractMesh("fibre_" + str(i), "core_" + str(i));
```


```python
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
        x = pixel_spacing_in_micrometre * -(cyl[0] - reference_CT.shape[1] / 2 + 0.5);
        y = pixel_spacing_in_micrometre * (cyl[1] - reference_CT.shape[0] / 2 + 0.5);

        # Create empty geometries (is it needed?)
        gvxr.emptyMesh("fibre_" + str(i));
        gvxr.emptyMesh("core_" + str(i));

        # Create the two corresponding cylinders (fibre and core)
        gvxr.makeCylinder("fibre_" + str(i), number_of_sectors, 815.0, fibre_radius, "micrometer");
        gvxr.makeCylinder("core_"  + str(i), number_of_sectors, 815.0,  core_radius, "micrometer");

        # Translate the two cylinders to the position of their centre
        gvxr.translateNode("fibre_" + str(i), y, 0.0, x, "micrometer");
        gvxr.translateNode("core_" + str(i), y, 0.0, x, "micrometer");

        # Apply the local transformation matrix (so that we could save the corresponding STL files)
        gvxr.applyCurrentLocalTransformation("fibre_" + str(i));
        gvxr.applyCurrentLocalTransformation("core_" + str(i));

        # Subtract the fibre from the matrix
        gvxr.subtractMesh("matrix", "fibre_" + str(i));

        # Subtract the core from the fibre
        gvxr.subtractMesh("fibre_" + str(i), "core_" + str(i));

        # Save the corresponding STL files
        # gvxr.saveSTLfile("fibre_" + str(i), "Tutorial2/outputs/fibre_" + str(i) + ".stl");
        # gvxr.saveSTLfile("core_" + str(i),  "Tutorial2/outputs/core_"  + str(i) + ".stl");

        # Add the mesh of the current fibre to the overall fibre mesh
        gvxr.addMesh("fibre", "fibre_" + str(i));

        # Add the mesh of the current core to the overall core mesh
        gvxr.addMesh("core", "core_"  + str(i));

    # Set the mesh colours (for the interactive visualisation)
    gvxr.setColor("fibre", 1.0, 0.0, 0.0, 1.0);
    gvxr.setColor("core",  1.0, 0.0, 1.0, 1.0);

    # Set the fibre's material properties
    # gvxr.setLinearAttenuationCoefficient("fibre", fibre_mu, "cm-1");
    gvxr.setCompound("fibre", "SiC");
    gvxr.setDensity("fibre", fibre_density, "g/cm3");

    # Set the core's material properties
    # gvxr.setLinearAttenuationCoefficient("core", core_mu, "cm-1");
    gvxr.setElement("core", "W");

    # Add the fibres and cores to the X-ray renderer
    gvxr.addPolygonMeshAsInnerSurface("core");
    gvxr.addPolygonMeshAsInnerSurface("fibre");
```

## Registration of a cube

We define a function to create the polygon mesh of the Ti90Al6V4 matrix.


```python
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
```

### Simulate the CT acquisition

There are 7 successive steps to simulate the XCT data acquisition:

1. Set the fibre and cores geometries and material properties (Step 39)
2. Set the matrix geometry and material properties (Step 40)
3. Simulate the raw projections for each angle:
   - Without phase contrast (Line 5 of Step 45), or
   - With phase contrast (Lines 14-55 of Step 45)
4. Apply the LSF (Lines 57-60 of Step 45)
5. Apply the flat-field correction (Step 62)
6. Add Poison noise (Step~\ref{??})
7. Apply the minus log normalisation to compute the sinogram (Step 63)

Compute the raw projections and save the data. For this  purpose, we define a new function.


```python
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
```

### Flat-filed correction

Because the data suffers from a fixed-pattern noise in X-ray imaging in
actual experiments, it is necessary to perform the flat-field correction of
the raw projections using:

$$I' = \frac{I - D}{F - D}$$

where $F$ (flat fields) and $D$ (dark fields) are projection images without sample and acquired with and without the X-ray beam turned on respectively. $I'$ corresponds to `corrected_projections` in the function below.

Note that in our example, `raw_projections_in_keV`, `flat_field_image` and `dark_field_image` are in keV whereas `corrected_projections` does not have any unit:

$$0 \leq raw\_projections\_in\_keV \leq  \sum_E N_0(E) \times E\\0 \leq corrected\_projections \leq 1$$

We define a new function to compute the flat-field correction.


```python
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
```

The function below is used to simulate a sinogram acquisition. Phase contrast in the projections can be taken into account or not. Also, Poisson noise can be added.


```python
def simulateSinogram(sigma_set=None, k_set=None, name_set=None):

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
            # for label in ["core", "fibre", "matrix"]:
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
        temp = np.random.poisson(map).astype(float);
        temp /= gain;
        temp -= bias + 1;

        # Noise map
        noise_map = (normalised_projections - temp) * scale;
        normalised_projections += noise_map;

    # Linearise
    simulated_sinogram = computeSinogramFromFlatField(normalised_projections);

    return simulated_sinogram, normalised_projections, raw_projections_in_keV;
```

The function below is used quantify the differences between two images. It is used in the objective function.


```python
def metrics(ref, test):

    normalised_ref = ref.flatten();
    normalised_test = test.flatten();

    if use_normalisation or metrics_type == "ZNCC":
        normalised_ref = standardisation(normalised_ref);
        normalised_test = standardisation(normalised_test);

    # Mean absolute error
    if metrics_type == "MAE":
        return mean_absolute_error(normalised_ref, normalised_test);
    # RMSE
    elif metrics_type == "RMSE":
        return math.sqrt(mean_squared_error(normalised_ref, normalised_test));
    # Mean relative error
    elif metrics_type == "MRE" or metrics_type == "MAPE":

        # Prevent division by zero
        denominator = np.abs(np.subtract(normalised_ref, normalised_test)) + 1e-6;
        divisor = np.abs(normalised_ref) + 1e-6;

        return np.mean(np.divide(denominator, divisor));
    elif metrics_type == "SSIM" or metrics_type == "DSSIM":
        normalised_ref.shape = [900, 1024];
        normalised_test.shape = [900, 1024];
        return (1.0 - ssim(normalised_ref, normalised_test,
                  data_range=normalised_ref.max() - normalised_ref.min())) / 2.0;
    elif metrics_type == "ZNCC":
        return (1.0 - np.mean(np.multiply(normalised_ref, normalised_test))) / 2.0;
    else:
        raise "Unknown metrics";
```

The function below is the objective function used to register the matrix.


```python
def fitnessFunctionCube(x):
    global best_fitness;
    global best_fitness_id;
    global prefix;

    global reference_sinogram;
    global centroid_set;
    global use_fibres;

    global core_radius;
    global fibre_radius;

    # Load the matrix geometrical properties from x
    setMatrix(x);

    # Simulate a sinogram
    simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram(sigma_set, k_set, label_set);

    # Compute the objective value
    if use_sinogram:
        objective = metrics(reference_sinogram, simulated_sinogram);
    else:
        objective = metrics(reference_normalised_projections, normalised_projections);

    # The block below is not necessary for the registration.
    # It is used to save the data to create animations.
    if best_fitness > objective:
        best_fitness = objective;

        gvxr.saveSTLfile("matrix", "outputs/matrix_" + str(best_fitness_id) + ".stl");

        # Reconstruct the CT slice
        simulated_CT = tomopy.recon(simulated_sinogram,
                                    theta_rad,
                                    center=rot_center,
                                    sinogram_order=False,
                                    algorithm='gridrec',
                                    filter_name='shepp',
                                    ncore=40)[0];

        # Save the simulated sinogram
        simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
        saveMHA("outputs/" + prefix + "simulated_sinogram_" + str(best_fitness_id) + ".mha",
                simulated_sinogram,
                [pixel_spacing_in_mm, angular_step, pixel_spacing_in_mm]);

        # Save the simulated CT slice
        saveMHA("outputs/" + prefix + "simulated_CT_" + str(best_fitness_id) + ".mha",
                simulated_CT,
                [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);

        np.savetxt("outputs/" + prefix + str(best_fitness_id) + ".dat", x, header='x,y,rotation_angle,w,h');

        best_fitness_id += 1;

    return objective
```


```python
# The registration has already been performed. Load the results.
if os.path.isfile("outputs/cube.dat"):
    matrix_geometry_parameters = np.loadtxt("outputs/cube.dat");
# Perform the registration using CMA-ES
else:
    best_fitness = sys.float_info.max;
    best_fitness_id = 0;
    prefix = "cube_";

    opts = cma.CMAOptions()
    opts.set('tolfun', 1e-2);
    opts['tolx'] = 1e-2;
    opts['bounds'] = [5*[-0.5], 5*[0.5]];

    es = cma.CMAEvolutionStrategy([0.0, 0.0, 0.0, 0.256835938, 0.232903226], 0.5, opts);
    es.optimize(fitnessFunctionCube);

    matrix_geometry_parameters = copy.deepcopy(es.result.xbest);
    np.savetxt("outputs/cube.dat", matrix_geometry_parameters, header='x,y,rotation_angle,w,h');

    # Release memory
    del es;
```

    (4_w,8)-aCMA-ES (mu_w=2.6,w_1=52%) in dimension 5 (seed=351062, Wed May  5 10:45:28 2021)
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Iterat #Fevals   function value  axis ratio  sigma  min&max std  t[m:s]
        1      8 1.176468850243608e+00 1.0e+00 4.68e-01  4e-01  5e-01 0:06.7
    Reconstructing 1 slice groups with 1 master threads...
        2     16 8.714674968631372e-01 1.2e+00 3.93e-01  3e-01  4e-01 0:12.0
        3     24 9.234673861149069e-01 1.3e+00 3.20e-01  3e-01  3e-01 0:17.1
    Reconstructing 1 slice groups with 1 master threads...
        4     32 6.280256386129930e-01 1.4e+00 3.19e-01  2e-01  4e-01 0:22.2
        5     40 6.691429314042504e-01 1.7e+00 3.10e-01  2e-01  4e-01 0:26.7
    Reconstructing 1 slice groups with 1 master threads...
        7     56 5.467190533806590e-01 2.1e+00 3.14e-01  2e-01  4e-01 0:36.2
        9     72 7.983566180450166e-01 2.4e+00 2.34e-01  1e-01  3e-01 0:45.4
    Reconstructing 1 slice groups with 1 master threads...
       11     88 5.356573766442871e-01 2.8e+00 1.89e-01  9e-02  3e-01 0:54.9
       13    104 5.513373075987954e-01 3.2e+00 1.64e-01  7e-02  2e-01 1:04.0
    Reconstructing 1 slice groups with 1 master threads...
       15    120 4.385649543176680e-01 3.8e+00 1.27e-01  5e-02  2e-01 1:13.4
    Reconstructing 1 slice groups with 1 master threads...
       18    144 4.528444213829956e-01 3.6e+00 9.47e-02  3e-02  1e-01 1:27.4
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       21    168 3.615414608506914e-01 3.6e+00 7.23e-02  2e-02  8e-02 1:41.9
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       24    192 2.603746014190437e-01 3.7e+00 4.44e-02  1e-02  4e-02 1:58.4
       27    216 2.645045920324710e-01 3.8e+00 3.03e-02  8e-03  3e-02 2:14.1
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       30    240 2.480670005667387e-01 3.8e+00 1.98e-02  4e-03  2e-02 2:30.5
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       33    264 2.469043623030691e-01 4.1e+00 1.63e-02  4e-03  1e-02 2:49.2
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       36    288 2.455007898383485e-01 4.2e+00 1.15e-02  2e-03  8e-03 3:03.7


### Apply the result of the registration


```python
# Save the result
setMatrix(matrix_geometry_parameters);
gvxr.saveSTLfile("matrix", "outputs/matrix.stl");

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
```

    Matrix
    	position: -58.94657360355499 -21.57106085948489 um
    	rotation: 90.48100733474752 deg
    	size: 1450.9427530956625 1071.591399848632 um


### Simulate the correspond CT acquisition


```python
# Simulate a sinogram
simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram();

# Reconstruct the CT slice
simulated_CT = tomopy.recon(simulated_sinogram,
                            theta_rad,
                            center=rot_center,
                            sinogram_order=False,
                            algorithm='gridrec',
                            filter_name='shepp',
                            ncore=40)[0];
normalised_simulated_CT = standardisation(simulated_CT);

# Compute the ZNCC
print("ZNCC matrix registration:",
      "{:.2f}".format(100.0 * np.mean(np.multiply(normalised_reference_CT, normalised_simulated_CT))));
```

    Reconstructing 1 slice groups with 1 master threads...
    ZNCC matrix registration: 73.41


### Display the result of the registration as an animation


```python
def createAnimation(aPrefix, anOutputFile):
    # Find all the images from the output directory
    files = sorted(
        glob.glob(aPrefix + "[0-9]*.mha"))

    # Store the images
    registration_image_set = [];

    # Create the GIF file
    with imageio.get_writer(anOutputFile, mode='I') as writer:

        # Store the PNG filenames
        png_filename_set = [];

        # Process all the images
        for i in range(len(files)):
            # Create the filenames
            mha_fname = aPrefix + str(i) + ".mha";
            png_filename_set.append(aPrefix + str(i) + ".png");

            # Open the MHA file
            float_image = sitk.ReadImage(mha_fname);

            # Convert in a Numpy array
            narray = sitk.GetArrayFromImage(float_image);

            offset = 60;
            roi_ref = reference_CT[505 - offset:505 + offset + 1,501 - offset:501 + offset + 1];
            roi_sim = narray[505 - offset:505 + offset + 1,501 - offset:501 + offset + 1];

            narray = standardisation(narray);
            registration_image_set.append(narray);

            # Create the figure
            fig, axs = plt.subplots(3, 3)

            # Dispay the reference, registration and error map
            fig.suptitle('Registration: Result ' + str(i+1) + "/" + str(len(files)))
            plt.tight_layout();
            norm = cm.colors.Normalize(vmax=1.25, vmin=-0.5)

            comp_equalized = compare_images(normalised_reference_CT, narray, method='checkerboard');

            roi_normalised_ref = normalised_reference_CT[505 - offset:505 + offset + 1,501 - offset:501 + offset + 1];
            roi_normalised_sim = narray[505 - offset:505 + offset + 1,501 - offset:501 + offset + 1];
            roi_normalised_compare = comp_equalized[505 - offset:505 + offset + 1,501 - offset:501 + offset + 1];


            # Reference
            axs[0, 0].set_title("Reference image");
            axs[0, 0].imshow(normalised_reference_CT, cmap="gray", norm=norm);
            axs[1, 0].imshow(roi_normalised_ref, cmap="gray", norm=norm);
            axs[2, 0].axis('off');

            # Registration
            axs[0, 1].set_title("Simulated CT slice after automatic registration");
            axs[0, 1].imshow(narray, cmap='gray', norm=norm);
            axs[1, 1].imshow(roi_normalised_sim, cmap="gray", norm=norm);

            axs[2, 1].set_title("Diagonal profiles");
            axs[2, 1].plot(np.diag(roi_ref), label="Reference");
            axs[2, 1].plot(np.diag(roi_sim), label="Simulated");
            # axs[2, 1].plot(np.diag(roi_ref) - np.diag(roi_sim), label="Error");
            axs[2, 1].legend();

            # Error map
            ZNCC = 100.0 * np.mean(np.multiply(normalised_reference_CT, narray));
            axs[0, 2].set_title("Checkboard comparison between\nthe reference and simulated images\nZNCC: " + "{:.2f}".format(ZNCC));
            axs[0, 2].imshow(comp_equalized, cmap='gray', norm=norm);
            axs[1, 2].imshow(roi_normalised_compare, cmap='gray', norm=norm);
            axs[2, 2].axis('off');

            plt.tight_layout();

            # Save the figure as a PNG file
            plt.savefig(png_filename_set[i])

            # Close the figure
            plt.close()

            # Open the PNG file with imageio and add it to the GIF file
            image = imageio.imread(png_filename_set[i])
            writer.append_data(image)

            # Delete the PNG file
            os.remove(png_filename_set[i]);

        for i in range(15):
            writer.append_data(image)

    return registration_image_set;
```


```python
if not os.path.exists("plots/cube_registration.gif"):
    cube_registration_image_set = createAnimation("outputs/cube_simulated_CT_",
                'plots/cube_registration.gif');
```

![Animation of the registration (GIF file)](tutorial/plots/cube_registration.gif)

### Adding the fibres

The radius of a tungsten core is 30 / 2 um. The pixel spacing is 1.9 um. The radius in number of pixels is $15/1.9  \approx  7.89$. The area of a core is $(15/1.9)^2  \pi  \approx 196$ pixels.


```python
setMatrix(matrix_geometry_parameters);
setFibres(centroid_set);
```


```python
# Simulate a sinogram
simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram(sigma_set, k_set, label_set);
```


```python
# Reconstruct the CT slice
simulated_CT = tomopy.recon(simulated_sinogram,
                            theta_rad,
                            center=rot_center,
                            sinogram_order=False,
                            algorithm='gridrec',
                            filter_name='shepp',
                            ncore=40)[0];
normalised_simulated_CT = standardisation(simulated_CT);

# Compute the ZNCC
print("ZNCC matrix registration with fibres:",
      "{:.2f}".format(100.0 * np.mean(np.multiply(normalised_reference_CT, normalised_simulated_CT))));
```

    Reconstructing 1 slice groups with 1 master threads...
    ZNCC matrix registration with fibres: 69.46



```python
simulated_sinogram.shape     = (simulated_sinogram.size     // simulated_sinogram.shape[2],     simulated_sinogram.shape[2]);
normalised_projections.shape = (normalised_projections.size // normalised_projections.shape[2], normalised_projections.shape[2]);
raw_projections_in_keV.shape = (raw_projections_in_keV.size // raw_projections_in_keV.shape[2], raw_projections_in_keV.shape[2]);

saveMHA("outputs/simulated_sinogram_with_fibres.mha",
        simulated_sinogram,
        [pixel_spacing_in_mm, angular_step, pixel_spacing_in_mm]);

saveMHA("outputs/normalised_projections_with_fibres.mha",
        normalised_projections,
        [pixel_spacing_in_mm, angular_step, pixel_spacing_in_mm]);

saveMHA("outputs/raw_projections_in_keV_with_fibres.mha",
        raw_projections_in_keV,
        [pixel_spacing_in_mm, angular_step, pixel_spacing_in_mm]);
```


```python
saveMHA("outputs/simulated_CT_with_fibres.mha",
        simulated_CT,
        [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);

saveMHA("outputs/normalised_simulated_CT_with_fibres.mha",
        normalised_simulated_CT,
        [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);
```


```python
norm = cm.colors.Normalize(vmax=1.25, vmin=-0.5)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
plt.tight_layout()
fig.suptitle('CT slice with fibres after the registration')

ax1.set_title("Reference image");
imgplot1 = ax1.imshow(normalised_reference_CT, cmap="gray",
                     norm=norm);

ax2.set_title("Simulated CT slice after automatic registration");
imgplot2 = ax2.imshow(normalised_simulated_CT,
                     cmap='gray',
                     norm=norm);

comp_equalized = compare_images(normalised_reference_CT, normalised_simulated_CT, method='checkerboard');
ax3.set_title("Checkboard comparison between\n" +
              "the reference and simulated images\nZNCC: " +
              "{:.2f}".format(100.0 * np.mean(np.multiply(normalised_reference_CT, normalised_simulated_CT))));
imgplot3 = ax3.imshow(comp_equalized,
                     cmap='gray',
                     norm=norm);

plt.savefig('plots/simulated_CT_slice_with_fibres_after_cube_registration.pdf');
plt.savefig('plots/simulated_CT_slice_with_fibres_after_cube_registration.png');
```



![png](doc/output_107_0.png)




## Optimisation of the cores and fibres radii

The function below is the objective function used to optimise the radii of the cores and fibres.


```python
def fitnessFunctionFibres(x):
    global best_fitness;
    global best_fitness_id;
    global fibre_radius;
    global core_radius;
    global prefix;

    # Get the radii
    fibre_radius = x[0];
    core_radius = fibre_radius * x[1];

    # Load the matrix
    setMatrix(matrix_geometry_parameters);

    # Load the cores and fibres
    setFibres(centroid_set);

    # Simulate a sinogram
    simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram(sigma_set, k_set, label_set);

    # Compute the objective value
    if use_sinogram:
        objective = metrics(reference_sinogram, simulated_sinogram);
    else:
        objective = metrics(reference_normalised_projections, normalised_projections);

    # The block below is not necessary for the registration.
    # It is used to save the data to create animations.
    if best_fitness > objective:
        best_fitness = objective;

        gvxr.saveSTLfile("core",  "outputs/" + prefix + str(best_fitness_id) + "_cores.stl");
        gvxr.saveSTLfile("fibre", "outputs/" + prefix + str(best_fitness_id) + "_fibres.stl");

        # Reconstruct the CT slice
        simulated_CT = tomopy.recon(simulated_sinogram,
                                    theta_rad,
                                    center=rot_center,
                                    sinogram_order=False,
                                    algorithm='gridrec',
                                    filter_name='shepp',
                                    ncore=40)[0];

        # Save the simulated sinogram
        simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
        saveMHA("outputs/" + prefix + "simulated_sinogram_" + str(best_fitness_id) + ".mha",
                simulated_sinogram,
                [pixel_spacing_in_mm, angular_step, pixel_spacing_in_mm]);

        # Save the simulated CT slice
        saveMHA("outputs/" + prefix + "simulated_CT_" + str(best_fitness_id) + ".mha",
                simulated_CT,
                [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);

        np.savetxt("outputs/" + prefix + str(best_fitness_id) + ".dat", x, header='x,y,rotation_angle,w,h');

        best_fitness_id += 1;

    return objective
```


```python
# The registration has already been performed. Load the results.
if os.path.isfile("outputs/fibre1_radii.dat"):
    temp = np.loadtxt("outputs/fibre1_radii.dat");
    core_radius = temp[0];
    fibre_radius = temp[1];
# Perform the registration using CMA-ES
else:
    ratio = core_radius / fibre_radius;

    x0 = [fibre_radius, ratio];
    bounds = [[5, 0.01], [1.5 * fibre_radius, 0.95]];

    best_fitness = sys.float_info.max;
    best_fitness_id = 0;
    prefix = "fibre1_";

    opts = cma.CMAOptions()
    opts.set('tolfun', 1e-3);
    opts['tolx'] = 1e-3;
    opts['bounds'] = bounds;

    es = cma.CMAEvolutionStrategy(x0, 0.9, opts);
    es.optimize(fitnessFunctionFibres);
    fibre_radius = es.result.xbest[0];
    core_radius = fibre_radius * es.result.xbest[1];

    np.savetxt("outputs/fibre1_radii.dat", [core_radius, fibre_radius], header='core_radius_in_um,fibre_radius_in_um');

    # Release memory
    del es;
```

    (3_w,6)-aCMA-ES (mu_w=2.0,w_1=63%) in dimension 2 (seed=322030, Wed May  5 10:48:57 2021)
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Iterat #Fevals   function value  axis ratio  sigma  min&max std  t[m:s]
        1      6 2.056238344210134e-01 1.0e+00 6.82e-01  6e-01  6e-01 0:14.1
        2     12 3.687508937185227e-01 1.1e+00 5.76e-01  4e-01  5e-01 0:24.5
        3     18 2.489597303686306e-01 1.2e+00 5.36e-01  4e-01  4e-01 0:36.7
    Reconstructing 1 slice groups with 1 master threads...
        4     24 1.924598248349651e-01 1.1e+00 5.54e-01  4e-01  4e-01 0:47.8
    Reconstructing 1 slice groups with 1 master threads...
        5     30 1.917642147734157e-01 1.4e+00 5.02e-01  3e-01  4e-01 0:58.4
        6     36 1.926992180118946e-01 1.4e+00 4.39e-01  2e-01  3e-01 1:09.8
        7     42 2.676166977872549e-01 1.5e+00 4.05e-01  2e-01  3e-01 1:20.4
        8     48 1.918502891487001e-01 1.7e+00 3.28e-01  1e-01  2e-01 1:31.8
    Reconstructing 1 slice groups with 1 master threads...
        9     54 1.910763638167751e-01 1.4e+00 3.33e-01  1e-01  2e-01 1:42.7
       10     60 1.995293980518588e-01 2.0e+00 2.78e-01  7e-02  2e-01 1:53.4
       11     66 1.945739091726220e-01 2.4e+00 2.38e-01  5e-02  1e-01 2:03.9
       13     78 1.960170206042391e-01 4.2e+00 2.61e-01  4e-02  2e-01 2:24.3
       15     90 1.942011543026385e-01 5.7e+00 2.47e-01  3e-02  2e-01 2:44.7
    Reconstructing 1 slice groups with 1 master threads...
       17    102 1.909489245147391e-01 7.6e+00 1.81e-01  2e-02  1e-01 3:05.5
       19    114 1.898572220739159e-01 8.3e+00 1.16e-01  8e-03  7e-02 3:26.3
       21    126 1.893649125326281e-01 1.0e+01 8.76e-02  5e-03  4e-02 3:47.1
       23    138 1.895079656739372e-01 1.2e+01 7.15e-02  3e-03  4e-02 4:10.2
       25    150 1.892452516869396e-01 1.4e+01 6.85e-02  2e-03  3e-02 4:30.8
    Reconstructing 1 slice groups with 1 master threads...
       27    162 1.889403454621526e-01 2.0e+01 7.86e-02  2e-03  5e-02 4:52.2
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       29    174 1.890331723861266e-01 2.6e+01 8.68e-02  2e-03  4e-02 5:13.3
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       31    186 1.873002985247594e-01 2.4e+01 2.10e-01  5e-03  1e-01 5:40.5
       33    198 1.878390750096869e-01 3.1e+01 1.98e-01  4e-03  1e-01 6:02.0
    Reconstructing 1 slice groups with 1 master threads...
       36    216 1.872897440411882e-01 2.9e+01 1.11e-01  1e-03  4e-02 6:33.4
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       38    228 1.864781087332659e-01 4.2e+01 1.50e-01  1e-03  8e-02 6:57.7
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       41    246 1.851662643804641e-01 6.8e+01 2.99e-01  3e-03  2e-01 7:30.5
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       44    264 1.841046636172989e-01 7.6e+01 3.65e-01  5e-03  2e-01 8:02.0
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       47    282 1.832952085380684e-01 8.0e+01 3.73e-01  4e-03  2e-01 8:34.5
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       50    300 1.804387068994000e-01 6.9e+01 4.24e-01  3e-03  2e-01 9:06.5
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       53    318 1.740033162621815e-01 7.9e+01 1.50e+00  9e-03  7e-01 9:39.4
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       56    336 1.567678600952053e-01 1.0e+02 3.80e+00  2e-02  2e+00 10:11.3
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       59    354 1.436949544943861e-01 1.6e+02 4.57e+00  1e-02  2e+00 10:43.5
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       62    372 1.196313696336992e-01 2.6e+02 1.06e+01  3e-02  7e+00 11:15.9
    Reconstructing 1 slice groups with 1 master threads...
       66    396 1.192597553608184e-01 3.0e+02 9.06e+00  2e-02  5e+00 11:57.2
    Reconstructing 1 slice groups with 1 master threads...
       70    420 1.186901749525934e-01 3.4e+02 4.89e+00  6e-03  2e+00 12:38.4
    Reconstructing 1 slice groups with 1 master threads...
       74    444 1.185502938762814e-01 2.8e+02 2.60e+00  3e-03  6e-01 13:19.1
    Reconstructing 1 slice groups with 1 master threads...
       78    468 1.185525746051034e-01 4.2e+02 1.67e+00  1e-03  4e-01 14:00.6
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       82    492 1.185418280942284e-01 5.8e+02 9.67e-01  6e-04  2e-01 14:46.0
    Reconstructing 1 slice groups with 1 master threads...
       85    510 1.185394248893163e-01 4.3e+02 4.80e-01  2e-04  5e-02 15:16.4



```python
if not os.path.exists("plots/fibre1_registration.gif"):
    registration_image_set = createAnimation("outputs/fibre1_simulated_CT_",
                'plots/fibre1_registration.gif');
```

![Animation of the registration (GIF file)](tutorial/plots/fibre1_registration.gif)

### Apply the result of the registration


```python
# Load the matrix
setMatrix(matrix_geometry_parameters);

# Load the cores and fibres
setFibres(centroid_set);

gvxr.saveSTLfile("fibre", "outputs/fibre1_fibre.stl");
gvxr.saveSTLfile("core",  "outputs/fibre1_core.stl");

print("Core diameter:", round(core_radius * 2), "um");
print("Fibre diameter:", round(fibre_radius * 2), "um");

# Simulate the corresponding CT aquisition
simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram(sigma_set, k_set, label_set);

# Reconstruct the CT slice
simulated_CT = tomopy.recon(simulated_sinogram,
                            theta_rad,
                            center=rot_center,
                            sinogram_order=False,
                            algorithm='gridrec',
                            filter_name='shepp',
                            ncore=40)[0];
normalised_simulated_CT = standardisation(simulated_CT);


# Compute the ZNCC
print("ZNCC radii registration 1:",
      "{:.2f}".format(100.0 * np.mean(np.multiply(normalised_reference_CT, normalised_simulated_CT))));
```

    Core diameter: 15 um
    Fibre diameter: 103 um
    Reconstructing 1 slice groups with 1 master threads...
    ZNCC radii registration 1: 90.32


The 3D view of the registration looks like:

![3D view](./doc/3d-view.png)

## Recentre each core/fibre

Each fibre is extracted from both the reference CT slice and simulated CT slice. The displacement between the corresponding fibres is computed to maximise the ZNCC between the two. The centre of the fibre is then adjusted accordingly.


```python
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
        reference_image = standardisation(reference_image);

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
                test_image = standardisation(test_image);

                # Compare the ROIs
                zncc = np.mean(np.multiply(reference_image.flatten(), test_image.flatten()));

                if best_ZNCC < zncc:
                    best_ZNCC = zncc;
                    best_x_offset = x;
                    best_y_offset = y;

        # Correct the position of the centre of the fibre
        new_centroid_set.append([cyl[0] - best_x_offset, cyl[1] - best_y_offset]);

    return new_centroid_set;
```


```python
centroid_set = refineCentrePositions(centroid_set, normalised_simulated_CT);
```

### Applying the result of recentring


```python
# Load the matrix
setMatrix(matrix_geometry_parameters);

# Load the cores and fibres
setFibres(centroid_set);

gvxr.saveSTLfile("fibre", "outputs/fibre2_fibre.stl");
gvxr.saveSTLfile("core",  "outputs/fibre2_core.stl");

# Simulate the corresponding CT aquisition
simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram(sigma_set, k_set, label_set);

# Reconstruct the CT slice
simulated_CT = tomopy.recon(simulated_sinogram,
                            theta_rad,
                            center=rot_center,
                            sinogram_order=False,
                            algorithm='gridrec',
                            filter_name='shepp',
                            ncore=40)[0];
normalised_simulated_CT = standardisation(simulated_CT);

# Compute the ZNCC
print("ZNCC recentring registration:",
      "{:.2f}".format(100.0 * np.mean(np.multiply(normalised_reference_CT, normalised_simulated_CT))));
```

    Reconstructing 1 slice groups with 1 master threads...
    ZNCC recentring registration: 89.58


## Optimisation the radii after recentring

After recentring the centres, another run of optimisation is executed to refine the radii of the fibres and cores.


```python
# The registration has already been performed. Load the results.
if os.path.isfile("outputs/fibre3_radii.dat"):
    temp = np.loadtxt("outputs/fibre3_radii.dat");
    core_radius = temp[0];
    fibre_radius = temp[1];
# Perform the registration using CMA-ES
else:
    ratio = core_radius / fibre_radius;

    x0 = [fibre_radius, ratio];
    bounds = [[5, 0.01], [1.5 * fibre_radius, 0.95]];

    best_fitness = sys.float_info.max;
    best_fitness_id = 0;
    prefix = "fibre3_";

    opts = cma.CMAOptions()
    opts.set('tolfun', 1e-2);
    opts['tolx'] = 1e-2;
    opts['bounds'] = bounds;

    es = cma.CMAEvolutionStrategy(x0, 0.9, opts);
    es.optimize(fitnessFunctionFibres);
    fibre_radius = es.result.xbest[0];
    core_radius = fibre_radius * es.result.xbest[1];

    np.savetxt("outputs/fibre3_radii.dat", [core_radius, fibre_radius], header='core_radius_in_um,fibre_radius_in_um');

    # Release memory
    del es;
```

    (3_w,6)-aCMA-ES (mu_w=2.0,w_1=63%) in dimension 2 (seed=370541, Wed May  5 11:05:07 2021)
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Iterat #Fevals   function value  axis ratio  sigma  min&max std  t[m:s]
        1      6 1.285166978617923e-01 1.0e+00 1.25e+00  1e+00  1e+00 0:11.0
        2     12 3.956813782085359e-01 1.1e+00 1.08e+00  1e+00  1e+00 0:21.6
        3     18 2.796703555845466e-01 1.2e+00 1.60e+00  1e+00  2e+00 0:33.2
        4     24 1.576141890793618e-01 1.3e+00 1.72e+00  1e+00  2e+00 0:44.8
        5     30 2.979612052767824e-01 1.6e+00 2.60e+00  2e+00  3e+00 0:55.8
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
        6     36 1.209290806703441e-01 1.6e+00 4.15e+00  3e+00  5e+00 1:10.0
        7     42 1.218518005407195e-01 1.4e+00 4.58e+00  3e+00  5e+00 1:21.5
        8     48 4.209463432453404e-01 1.8e+00 4.82e+00  3e+00  5e+00 1:33.3
        9     54 2.846787171461876e-01 1.7e+00 3.83e+00  2e+00  4e+00 1:45.4
       10     60 2.588711290050154e-01 1.7e+00 3.14e+00  2e+00  3e+00 1:57.3
       11     66 1.242238239652531e-01 1.7e+00 2.42e+00  1e+00  2e+00 2:09.3
       12     72 2.090454495714433e-01 1.7e+00 1.99e+00  8e-01  1e+00 2:20.6
       14     84 2.141632945596124e-01 2.1e+00 1.61e+00  6e-01  1e+00 2:43.3
       16     96 1.257604573651241e-01 2.3e+00 1.77e+00  6e-01  1e+00 3:05.6
    Reconstructing 1 slice groups with 1 master threads...
       18    108 1.202179141331833e-01 3.1e+00 2.72e+00  7e-01  3e+00 3:29.1
       20    120 1.407933182863632e-01 3.0e+00 2.62e+00  8e-01  2e+00 3:52.1
       22    132 3.941537204569511e-01 2.3e+00 3.48e+00  1e+00  2e+00 4:14.6
       24    144 2.543783651414707e-01 1.8e+00 3.19e+00  1e+00  1e+00 4:39.2
       26    156 1.289901459758717e-01 1.5e+00 2.76e+00  9e-01  1e+00 5:02.8
       28    168 1.379355385094951e-01 1.6e+00 3.02e+00  1e+00  1e+00 5:27.1
       30    180 1.730433562512205e-01 1.2e+00 2.14e+00  6e-01  6e-01 5:50.3
       32    192 1.565005633120540e-01 1.2e+00 1.27e+00  3e-01  3e-01 6:14.7
       34    204 1.504535454911944e-01 1.1e+00 1.06e+00  2e-01  2e-01 6:37.5
       37    222 1.603697116291833e-01 2.1e+00 1.29e+00  3e-01  3e-01 7:11.1
       40    240 1.350630144094724e-01 1.5e+00 1.19e+00  2e-01  2e-01 7:45.1
       43    258 1.587792403208920e-01 1.4e+00 1.22e+00  1e-01  3e-01 8:18.3
       46    276 1.265084739551233e-01 2.5e+00 1.07e+00  8e-02  2e-01 8:51.5
       49    294 1.294169148636709e-01 6.0e+00 8.24e-01  3e-02  2e-01 9:24.7
       52    312 1.287426714014878e-01 1.1e+01 7.51e-01  2e-02  2e-01 9:57.9
       55    330 1.256922118605867e-01 1.8e+01 6.65e-01  1e-02  2e-01 10:31.3
       58    348 1.231933904340133e-01 3.2e+01 1.25e+00  1e-02  5e-01 11:05.3
    Reconstructing 1 slice groups with 1 master threads...
       61    366 1.187591185770603e-01 5.7e+01 2.79e+00  2e-02  1e+00 11:40.3
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       64    384 1.174757414718533e-01 7.3e+01 2.21e+00  1e-02  9e-01 12:15.0
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       67    402 1.158741403877490e-01 1.0e+02 1.73e+00  7e-03  7e-01 12:49.2
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       70    420 1.154091373507094e-01 1.8e+02 1.57e+00  4e-03  7e-01 13:26.0
    Reconstructing 1 slice groups with 1 master threads...
       74    444 1.147785314501051e-01 2.9e+02 1.59e+00  3e-03  8e-01 14:11.7
       77    462 1.147025505778406e-01 3.5e+02 9.53e-01  1e-03  4e-01 14:45.4



```python
if not os.path.exists("plots/fibre3_registration.gif"):
    registration_image_set = createAnimation("outputs/fibre3_simulated_CT_",
                'plots/fibre3_registration.gif');
```

![Animation of the registration (GIF file)](tutorial/plots/fibre3_registration.gif)

### Apply the result of the registration


```python
# Load the matrix
setMatrix(matrix_geometry_parameters);

# Load the cores and fibres
setFibres(centroid_set);

gvxr.saveSTLfile("fibre", "outputs/fibre3_fibre.stl");
gvxr.saveSTLfile("core",  "outputs/fibre3_core.stl");

print("Core diameter:", round(core_radius * 2), "um");
print("Fibre diameter:", round(fibre_radius * 2), "um");
```

    Core diameter: 15 um
    Fibre diameter: 104 um



```python
# Simulate the corresponding CT aquisition
simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram(sigma_set, k_set, label_set);

# Reconstruct the CT slice
simulated_CT = tomopy.recon(simulated_sinogram,
                            theta_rad,
                            center=rot_center,
                            sinogram_order=False,
                            algorithm='gridrec',
                            filter_name='shepp',
                            ncore=40)[0];
normalised_simulated_CT = standardisation(simulated_CT);

# Compute the ZNCC
print("ZNCC radii registration 2:",
      "{:.2f}".format(100.0 * np.mean(np.multiply(normalised_reference_CT, normalised_simulated_CT))));
```

    Reconstructing 1 slice groups with 1 master threads...
    ZNCC radii registration 2: 91.01


## Optimisation of the beam spectrum


```python
def fitnessHarmonics(x):

    global energy_spectrum;

    global use_normalisation;

    global best_fitness;
    global best_fitness_id;
    global prefix;

    energy_33_keV = x[0];
    first_order_harmonics = x[1];
    second_order_harmonics = x[2];

    # Normalise the beam spectrum
    total = energy_33_keV + first_order_harmonics + second_order_harmonics;
    energy_33_keV /= total;
    first_order_harmonics /= total;
    second_order_harmonics /= total;

    # The beam specturm. Here we have a polychromatic beam.
    gvxr.resetBeamSpectrum();
    energy_spectrum = [(33, energy_33_keV, "keV"), (66, first_order_harmonics, "keV"), (99, second_order_harmonics, "keV")];

    for energy, percentage, unit in energy_spectrum:
        gvxr.addEnergyBinToSpectrum(energy, unit, percentage);

    # Simulate a sinogram
    simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram(sigma_set, k_set, label_set);

    # Compute the objective value (no normalisation here)
    old_normalisation = use_normalisation;
    use_normalisation = False;
    if use_sinogram:
        objective = metrics(reference_sinogram, simulated_sinogram);
    else:
        objective = metrics(reference_normalised_projections, normalised_projections);
    use_normalisation = old_normalisation;

    # The block below is not necessary for the registration.
    # It is used to save the data to create animations.
    if best_fitness > objective:
        best_fitness = objective;

        # Reconstruct the CT slice
        simulated_CT = tomopy.recon(simulated_sinogram,
                                    theta_rad,
                                    center=rot_center,
                                    sinogram_order=False,
                                    algorithm='gridrec',
                                    filter_name='shepp',
                                    ncore=40)[0];

        # Save the simulated sinogram
        simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
        saveMHA("outputs/" + prefix + "simulated_sinogram_" + str(best_fitness_id) + ".mha",
                simulated_sinogram,
                [pixel_spacing_in_mm, angular_step, pixel_spacing_in_mm]);

        # Save the simulated CT slice
        saveMHA("outputs/" + prefix + "simulated_CT_" + str(best_fitness_id) + ".mha",
                simulated_CT,
                [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);

        np.savetxt("outputs/" + prefix + str(best_fitness_id) + ".dat", np.array(x) / total, header='33keV,66keV,99keV');

        best_fitness_id += 1;

    return objective
```


```python
# The registration has already been performed. Load the results.
if os.path.isfile("outputs/spectrum1.dat"):
    temp = np.loadtxt("outputs/spectrum1.dat");

    # The beam specturm. Here we have a polychromatic beam.
    energy_spectrum = [(33, temp[0], "keV"), (66, temp[1], "keV"), (99, temp[2], "keV")];

# Perform the registration using CMA-ES
else:
    ratio = core_radius / fibre_radius;

    x0 = [0.97, 0.2, 0.1];
    bounds = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];

    best_fitness = sys.float_info.max;
    best_fitness_id = 0;
    prefix = "spectrum1_";

    opts = cma.CMAOptions()
    opts.set('tolfun', 1e-2);
    opts['tolx'] = 1e-2;
    opts['bounds'] = bounds;

    es = cma.CMAEvolutionStrategy(x0, 0.25, opts);
    es.optimize(fitnessHarmonics);

    total = es.result.xbest[0] + es.result.xbest[1] + es.result.xbest[2];
    energy_spectrum = [(33, es.result.xbest[0] / total, "keV"), (66, es.result.xbest[1] / total, "keV"), (99, es.result.xbest[2] / total, "keV")];

    np.savetxt("outputs/spectrum1.dat", [es.result.xbest[0] / total, es.result.xbest[1] / total, es.result.xbest[2] / total], header='weight of main energy,weight of first order harmonics,weight of second order harmonics');

    # Release memory
    del es;
```

    (3_w,7)-aCMA-ES (mu_w=2.3,w_1=58%) in dimension 3 (seed=337843, Wed May  5 11:20:10 2021)
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Iterat #Fevals   function value  axis ratio  sigma  min&max std  t[m:s]
        1      7 6.048137556956906e+02 1.0e+00 2.20e-01  2e-01  2e-01 0:11.9
        2     14 9.221417820991099e+02 1.4e+00 2.28e-01  2e-01  2e-01 0:23.0
    Reconstructing 1 slice groups with 1 master threads...
        3     21 4.547091625468402e+02 1.7e+00 2.33e-01  2e-01  2e-01 0:34.4
        4     28 1.824198288943216e+03 1.6e+00 2.11e-01  2e-01  2e-01 0:45.4
        5     35 1.535813448970954e+03 1.6e+00 1.96e-01  1e-01  2e-01 0:56.4
        6     42 7.565920670178712e+02 1.4e+00 1.85e-01  1e-01  2e-01 1:07.5
    Reconstructing 1 slice groups with 1 master threads...
        7     49 4.064423335333136e+02 1.6e+00 1.54e-01  9e-02  1e-01 1:18.9
        8     56 4.067720531075174e+02 1.5e+00 1.24e-01  6e-02  1e-01 1:29.9
    Reconstructing 1 slice groups with 1 master threads...
        9     63 3.949212437091743e+02 1.7e+00 1.07e-01  5e-02  8e-02 1:41.4
       10     70 5.844376492594570e+02 1.7e+00 1.11e-01  5e-02  1e-01 1:52.4
       11     77 4.093566440346751e+02 2.4e+00 1.20e-01  5e-02  1e-01 2:03.4
       12     84 5.828426177953316e+02 3.1e+00 1.18e-01  5e-02  1e-01 2:14.4
       14     98 3.966277489935002e+02 3.4e+00 1.12e-01  5e-02  1e-01 2:36.6
    Reconstructing 1 slice groups with 1 master threads...
       16    112 4.160212620777321e+02 4.0e+00 1.13e-01  4e-02  1e-01 2:59.0
       18    126 4.710081539848512e+02 4.7e+00 7.32e-02  2e-02  8e-02 3:21.0
       20    140 3.882208917393729e+02 5.0e+00 5.38e-02  2e-02  5e-02 3:43.2
       22    154 3.885488243528806e+02 5.8e+00 4.48e-02  1e-02  5e-02 4:05.2
       24    168 3.947338239840076e+02 8.5e+00 4.97e-02  9e-03  6e-02 4:27.3
       26    182 3.879351289137070e+02 1.1e+01 3.68e-02  6e-03  4e-02 4:49.1
    Reconstructing 1 slice groups with 1 master threads...
       28    196 3.910899057694735e+02 1.4e+01 2.81e-02  3e-03  3e-02 5:11.5
    Reconstructing 1 slice groups with 1 master threads...
       30    210 3.868452569212011e+02 1.8e+01 2.80e-02  3e-03  4e-02 5:34.0
    Reconstructing 1 slice groups with 1 master threads...
       32    224 3.867412339999130e+02 2.3e+01 2.16e-02  2e-03  3e-02 5:56.3
    Reconstructing 1 slice groups with 1 master threads...
       34    238 3.867343646407951e+02 2.6e+01 1.63e-02  1e-03  2e-02 6:18.6
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       37    259 3.867149484747496e+02 2.6e+01 9.13e-03  5e-04  9e-03 6:52.5



```python
if not os.path.exists("plots/spectrum1_registration.gif"):
    registration_image_set = createAnimation("outputs/spectrum1_simulated_CT_",
                'plots/spectrum1_registration.gif');
```

![Animation of the registration (GIF file)](tutorial/plots/spectrum1_registration.gif)


```python
# Apply the result of the registration
gvxr.resetBeamSpectrum();
for energy, percentage, unit in energy_spectrum:
    gvxr.addEnergyBinToSpectrum(energy, unit, percentage);
```


```python
for channel in energy_spectrum:
    print(channel);
```

    (33, 0.9580665254237618, 'keV')
    (66, 0.013316114591326426, 'keV')
    (99, 0.028617359984911786, 'keV')



```python
# Simulate the corresponding CT aquisition
simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram(sigma_set, k_set, label_set);

# Reconstruct the CT slice
simulated_CT = tomopy.recon(simulated_sinogram,
                            theta_rad,
                            center=rot_center,
                            sinogram_order=False,
                            algorithm='gridrec',
                            filter_name='shepp',
                            ncore=40)[0];
normalised_simulated_CT = standardisation(simulated_CT);

# Compute the ZNCC
print("ZNCC spectrum registration 1:",
      "{:.2f}".format(100.0 * np.mean(np.multiply(normalised_reference_CT, normalised_simulated_CT))));
```

    Reconstructing 1 slice groups with 1 master threads...
    ZNCC spectrum registration 1: 91.08


## Optimisation of the phase contrast and the radii


```python
def laplacian(x, sigma):
    """
    This function create a Laplacian kernel with

    $$ g''(x) = \left(\frac{x^2}{\sigma^4} - \frac{1}{\sigma^2}\right) \exp\left(-\frac{x^2}{2\sigma^2}\right) $$

    :param array x:
    :param float sigma:
    :return the convolution kernel
    """

    return (np.power(x, 2.) / math.pow(sigma, 4) - 1. / math.pow(sigma, 2)) * np.exp(-np.power(x, 2.) / (2. * math.pow(sigma, 2)));
```


```python
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
```


```python
def fitnessFunctionLaplacian(x):
    global best_fitness;
    global best_fitness_id;
    global prefix;

    global fibre_radius;
    global core_radius;

    sigma_core = x[0];
    k_core = x[1];
    sigma_fibre = x[2];
    k_fibre = x[3];
    sigma_matrix = x[4];
    k_matrix = x[5];
    core_radius = x[6];
    fibre_radius = x[7];

    # Load the matrix
    setMatrix(matrix_geometry_parameters);

    # Load the cores and fibres
    setFibres(centroid_set);

    # Simulate a sinogram
    simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram(
        [sigma_core, sigma_fibre, sigma_matrix],
        [k_core, k_fibre, k_matrix],
        ["core", "fibre", "matrix"]
    );

    # Compute the objective value
    if use_sinogram:
        objective = metrics(reference_sinogram, simulated_sinogram);
    else:
        objective = metrics(reference_normalised_projections, normalised_projections);

    # The block below is not necessary for the registration.
    # It is used to save the data to create animations.
    if best_fitness > objective:
        best_fitness = objective;

        # Reconstruct the CT slice
        simulated_CT = tomopy.recon(simulated_sinogram,
                                    theta_rad,
                                    center=rot_center,
                                    sinogram_order=False,
                                    algorithm='gridrec',
                                    filter_name='shepp',
                                    ncore=40)[0];

        # Save the simulated sinogram
        simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
        saveMHA("outputs/" + prefix + "simulated_sinogram_" + str(best_fitness_id) + ".mha",
                simulated_sinogram,
                [pixel_spacing_in_mm, angular_step, pixel_spacing_in_mm]);

        # Save the simulated CT slice
        saveMHA("outputs/" + prefix + "simulated_CT_" + str(best_fitness_id) + ".mha",
                simulated_CT,
                [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);

        np.savetxt("outputs/" + prefix + str(best_fitness_id) + ".dat", [sigma_core, k_core, sigma_fibre, k_fibre, sigma_matrix, k_matrix, core_radius, fibre_radius], header='sigma_core, k_core, sigma_fibre, k_fibre, sigma_matrix, k_matrix, core_radius_in_um, fibre_radius_in_um');

        best_fitness_id += 1;

    return objective
```


```python
# The registration has already been performed. Load the results.
if os.path.isfile("outputs/laplacian1.dat"):
    temp = np.loadtxt("outputs/laplacian1.dat");
    sigma_core = temp[0];
    k_core = temp[1];
    sigma_fibre = temp[2];
    k_fibre = temp[3];
    sigma_matrix = temp[4];
    k_matrix = temp[5];
    core_radius = temp[6];
    fibre_radius = temp[7];

# Perform the registration using CMA-ES
else:

    sigma_core = 5.;
    sigma_fibre = 0.75;
    sigma_matrix = 0.6;

    k_core = 1000;
    k_fibre = 1000;
    k_matrix = 1000.0;

    x0 = [
        sigma_core, k_core,
        sigma_fibre, k_fibre,
        sigma_matrix, k_matrix,
        core_radius, fibre_radius
    ];

    bounds = [
        [
            0.005, 0.0,
             0.005, 0.0,
             0.005, 0.0,
             0.95 * core_radius, 0.95 * fibre_radius
        ],
        [
            10.0, 2000,
             2.5, 2000,
             2.5, 2000,
             1.15 * core_radius, 1.15 * fibre_radius
        ]
    ];

    best_fitness = sys.float_info.max;
    best_fitness_id = 0;
    prefix = "laplacian1_";

    opts = cma.CMAOptions()
    opts.set('tolfun', 1e-4);
    opts['tolx'] = 1e-4;
    opts['bounds'] = bounds;
    opts['CMA_stds'] = [0.25, 20.25, 0.25, 20.25, 0.25, 20.25, core_radius * 0.1, fibre_radius * 0.1];

    es = cma.CMAEvolutionStrategy(x0, 0.25, opts);
    es.optimize(fitnessFunctionLaplacian);

    sigma_core = es.result.xbest[0];
    k_core = es.result.xbest[1];
    sigma_fibre = es.result.xbest[2];
    k_fibre = es.result.xbest[3];
    sigma_matrix = es.result.xbest[4];
    k_matrix = es.result.xbest[5];
    core_radius = es.result.xbest[6];
    fibre_radius = es.result.xbest[7];

    np.savetxt("outputs/laplacian1.dat", [sigma_core, k_core, sigma_fibre, k_fibre, sigma_matrix, k_matrix, core_radius, fibre_radius], header='sigma_core, k_core, sigma_fibre, k_fibre, sigma_matrix, k_matrix, core_radius_in_um, fibre_radius_in_um');

    # Release memory
    del es;
```

    (5_w,10)-aCMA-ES (mu_w=3.2,w_1=45%) in dimension 8 (seed=331847, Wed May  5 11:27:17 2021)
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Iterat #Fevals   function value  axis ratio  sigma  min&max std  t[m:s]
        1     10 1.065680914131528e-01 1.0e+00 2.34e-01  6e-02  5e+00 0:19.5
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
        2     20 1.059585545701215e-01 1.1e+00 2.20e-01  5e-02  4e+00 0:40.1
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
        3     30 1.052573982057145e-01 1.2e+00 2.27e-01  6e-02  4e+00 1:00.2
    Reconstructing 1 slice groups with 1 master threads...
        4     40 1.050951583555048e-01 1.3e+00 2.35e-01  6e-02  5e+00 1:19.7
    Reconstructing 1 slice groups with 1 master threads...
        5     50 1.050539077037347e-01 1.6e+00 2.47e-01  6e-02  5e+00 1:39.2
    Reconstructing 1 slice groups with 1 master threads...
        6     60 1.048456705384179e-01 1.6e+00 2.35e-01  5e-02  5e+00 1:58.8
    Reconstructing 1 slice groups with 1 master threads...
        7     70 1.047722603586111e-01 1.6e+00 2.28e-01  5e-02  5e+00 2:17.7
        8     80 1.049834541722742e-01 1.7e+00 2.15e-01  5e-02  4e+00 2:36.0
        9     90 1.047786490794836e-01 1.7e+00 2.03e-01  4e-02  4e+00 2:54.5
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       10    100 1.045309096771790e-01 1.7e+00 1.99e-01  4e-02  4e+00 3:13.4
       11    110 1.046631862045325e-01 1.9e+00 1.98e-01  4e-02  4e+00 3:31.6
       12    120 1.045567270562818e-01 1.9e+00 1.99e-01  4e-02  4e+00 3:49.9
    Reconstructing 1 slice groups with 1 master threads...
       13    130 1.044163183219524e-01 2.1e+00 1.90e-01  4e-02  4e+00 4:08.8
       14    140 1.045133517752244e-01 2.2e+00 1.90e-01  4e-02  4e+00 4:27.1
       15    150 1.045029660762912e-01 2.3e+00 1.77e-01  4e-02  4e+00 4:45.5
       16    160 1.044352260429275e-01 2.3e+00 1.64e-01  3e-02  3e+00 5:04.1
    Reconstructing 1 slice groups with 1 master threads...
       17    170 1.043937780372292e-01 2.4e+00 1.45e-01  3e-02  3e+00 5:23.8
       18    180 1.043991586663496e-01 2.4e+00 1.50e-01  3e-02  3e+00 5:42.8
    Reconstructing 1 slice groups with 1 master threads...
       19    190 1.043638233449991e-01 2.7e+00 1.40e-01  3e-02  3e+00 6:02.2
       21    210 1.044149653326552e-01 3.0e+00 1.23e-01  2e-02  3e+00 6:39.8
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       23    230 1.043247841171232e-01 3.3e+00 1.05e-01  2e-02  2e+00 7:18.0
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       25    250 1.042838880077307e-01 3.4e+00 1.00e-01  2e-02  2e+00 7:55.7
    Reconstructing 1 slice groups with 1 master threads...
       27    270 1.042757155233023e-01 3.8e+00 9.53e-02  1e-02  2e+00 8:33.8
       29    290 1.042838796698735e-01 4.0e+00 7.67e-02  1e-02  2e+00 9:11.2
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       31    310 1.042111614676850e-01 4.3e+00 7.47e-02  1e-02  2e+00 9:50.5
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       33    330 1.041669546337650e-01 4.4e+00 8.70e-02  1e-02  2e+00 10:29.6
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       35    350 1.041271137577909e-01 4.8e+00 1.04e-01  2e-02  3e+00 11:08.3
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       37    370 1.040463232821922e-01 5.2e+00 1.54e-01  2e-02  5e+00 11:47.0
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       39    390 1.039124239230766e-01 5.8e+00 2.06e-01  3e-02  7e+00 12:25.6
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       41    410 1.037129083889837e-01 6.0e+00 2.98e-01  4e-02  1e+01 13:04.5
       43    430 1.038606390794169e-01 6.6e+00 3.44e-01  5e-02  1e+01 13:41.4
    Reconstructing 1 slice groups with 1 master threads...
       45    450 1.035836206340102e-01 6.8e+00 3.76e-01  5e-02  1e+01 14:19.4
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       47    470 1.029904629650162e-01 7.3e+00 5.97e-01  8e-02  2e+01 14:57.8
       49    490 1.034208639604803e-01 7.8e+00 7.97e-01  1e-01  3e+01 15:35.2
       51    510 1.038214609222141e-01 7.9e+00 7.77e-01  1e-01  3e+01 16:13.3
    Reconstructing 1 slice groups with 1 master threads...
       53    530 1.029570012811356e-01 9.5e+00 9.22e-01  1e-01  3e+01 16:51.6
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       55    550 1.019825942277614e-01 1.0e+01 8.94e-01  1e-01  3e+01 17:30.2
    Reconstructing 1 slice groups with 1 master threads...
       57    570 1.019865847498272e-01 1.1e+01 8.93e-01  1e-01  4e+01 18:07.9
    Reconstructing 1 slice groups with 1 master threads...
       60    600 1.014696340914569e-01 1.2e+01 8.00e-01  9e-02  3e+01 19:04.9
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       63    630 1.008923095322233e-01 1.5e+01 8.35e-01  8e-02  4e+01 20:02.4
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       66    660 1.005777963801075e-01 1.9e+01 7.30e-01  7e-02  3e+01 21:00.4
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       69    690 1.006939264068543e-01 2.0e+01 6.08e-01  5e-02  3e+01 22:00.1
    Reconstructing 1 slice groups with 1 master threads...
       72    720 1.002087128522691e-01 2.0e+01 4.44e-01  4e-02  2e+01 22:58.7
       75    750 1.002945339706823e-01 2.3e+01 3.99e-01  3e-02  2e+01 23:55.6
    Reconstructing 1 slice groups with 1 master threads...
       78    780 1.001447563868310e-01 2.6e+01 3.30e-01  3e-02  1e+01 24:51.8
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       81    810 1.000550524746180e-01 2.5e+01 3.78e-01  3e-02  2e+01 25:50.7
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       84    840 9.999203225346749e-02 2.9e+01 4.11e-01  3e-02  2e+01 26:58.9
    Reconstructing 1 slice groups with 1 master threads...
       87    870 9.998239514615145e-02 3.9e+01 4.93e-01  4e-02  3e+01 28:07.1
       90    900 1.000596394488530e-01 4.1e+01 4.57e-01  3e-02  2e+01 29:05.9
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       93    930 9.999281608922900e-02 4.2e+01 3.28e-01  2e-02  2e+01 30:06.5
       96    960 1.000335685195849e-01 4.4e+01 3.47e-01  2e-02  2e+01 31:05.8
       99    990 9.996049340734102e-02 5.2e+01 3.31e-01  2e-02  2e+01 32:03.9
    Reconstructing 1 slice groups with 1 master threads...
      100   1000 9.993417632079502e-02 4.9e+01 3.12e-01  2e-02  2e+01 32:23.8
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
      103   1030 9.992082567468871e-02 5.1e+01 2.71e-01  2e-02  1e+01 33:24.1
      106   1060 9.992868746958689e-02 4.9e+01 2.33e-01  1e-02  1e+01 34:24.2
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
      109   1090 9.992124633869381e-02 5.6e+01 1.97e-01  9e-03  9e+00 35:26.0
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
      112   1120 9.990841646859766e-02 5.9e+01 1.61e-01  7e-03  7e+00 36:29.9
    Reconstructing 1 slice groups with 1 master threads...
      115   1150 9.990651502781348e-02 6.1e+01 1.37e-01  5e-03  6e+00 37:32.0
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
      118   1180 9.990209896734845e-02 6.4e+01 1.29e-01  5e-03  5e+00 38:34.3
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
      121   1210 9.989908295533823e-02 7.2e+01 1.24e-01  4e-03  5e+00 39:37.3
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
      124   1240 9.989670853972313e-02 7.7e+01 1.15e-01  4e-03  4e+00 40:40.1
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
      127   1270 9.989563390427728e-02 8.3e+01 1.27e-01  4e-03  5e+00 41:45.9
      130   1300 9.989731879512545e-02 8.5e+01 9.92e-02  3e-03  4e+00 42:48.0
      131   1310 9.989655806468652e-02 8.8e+01 9.36e-02  3e-03  3e+00 43:07.8



```python
if not os.path.exists("plots/laplacian1_registration.gif"):
    registration_image_set = createAnimation("outputs/laplacian1_simulated_CT_",
                'plots/laplacian1_registration.gif');
```

![Animation of the registration (GIF file)](tutorial/plots/laplacian1_registration.gif)

### Apply the result of the registration


```python
# Load the matrix
setMatrix(matrix_geometry_parameters);

# Load the cores and fibres
setFibres(centroid_set);

gvxr.saveSTLfile("fibre", "outputs/laplacian1_fibre.stl");
gvxr.saveSTLfile("core",  "outputs/laplacian1_core.stl");

print("Core diameter:", round(core_radius * 2), "um");
print("Fibre diameter:", round(fibre_radius * 2), "um");
```

    Core diameter: 16 um
    Fibre diameter: 108 um



```python
# Simulate the corresponding CT aquisition
sigma_set = [sigma_core, sigma_fibre, sigma_matrix];
k_set = [k_core, k_fibre, k_matrix];
label_set = ["core", "fibre", "matrix"];

simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram(sigma_set, k_set, label_set);

# Reconstruct the CT slice
simulated_CT = tomopy.recon(simulated_sinogram,
                            theta_rad,
                            center=rot_center,
                            sinogram_order=False,
                            algorithm='gridrec',
                            filter_name='shepp',
                            ncore=40)[0];
normalised_simulated_CT = standardisation(simulated_CT);

# Compute the ZNCC
print("ZNCC phase contrast registration 1:",
      "{:.2f}".format(100.0 * np.mean(np.multiply(normalised_reference_CT, normalised_simulated_CT))));
```

    Reconstructing 1 slice groups with 1 master threads...
    ZNCC phase contrast registration 1: 93.30


## Optimisation of the phase contrast and the LSF


```python
old_lsf = copy.deepcopy(lsf_kernel);
```


```python
def fitnessFunctionLaplacianLSF(x):
    global best_fitness;
    global best_fitness_id;
    global prefix;

    global lsf_kernel;

    # sigma_core = x[0];
    k_core = x[0];
    # sigma_fibre = x[2];
    k_fibre = x[1];
    # sigma_matrix = x[4];
    k_matrix = x[2];

    a2 = x[3];
    b2 = x[4];
    c2 = x[5];
    d2 = x[6];
    e2 = x[7];
    f2 = x[8];

    # The response of the detector as the line-spread function (LSF)
    t = np.arange(-20., 21., 1.);
    lsf_kernel=lsf(t*41, a2, b2, c2, d2, e2, f2);
    lsf_kernel/=lsf_kernel.sum();

    # Simulate a sinogram
    simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram(
        [sigma_core, sigma_fibre, sigma_matrix],
        [k_core, k_fibre, k_matrix],
        ["core", "fibre", "matrix"]
    );

    # Compute the objective value
    if use_sinogram:
        objective = metrics(reference_sinogram, simulated_sinogram);
    else:
        objective = metrics(reference_normalised_projections, normalised_projections);

    # The block below is not necessary for the registration.
    # It is used to save the data to create animations.
    if best_fitness > objective:
        best_fitness = objective;

        # Reconstruct the CT slice
        simulated_CT = tomopy.recon(simulated_sinogram,
                                    theta_rad,
                                    center=rot_center,
                                    sinogram_order=False,
                                    algorithm='gridrec',
                                    filter_name='shepp',
                                    ncore=40)[0];

        # Save the simulated sinogram
        simulated_sinogram.shape = (simulated_sinogram.size // simulated_sinogram.shape[2], simulated_sinogram.shape[2]);
        saveMHA("outputs/" + prefix + "simulated_sinogram_" + str(best_fitness_id) + ".mha",
                simulated_sinogram,
                [pixel_spacing_in_mm, angular_step, pixel_spacing_in_mm]);

        # Save the simulated CT slice
        saveMHA("outputs/" + prefix + "simulated_CT_" + str(best_fitness_id) + ".mha",
                simulated_CT,
                [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);

        np.savetxt("outputs/" + prefix + "laplacian_" + str(best_fitness_id) + ".dat", [k_core, k_fibre, k_matrix], header='k_core, k_fibre, k_matrix');
        np.savetxt("outputs/" + prefix + "LSF_" + str(best_fitness_id) + ".dat", [a2, b2, c2, d2, e2, f2], header='a2, b2, c2, d2, e2, f2');


        best_fitness_id += 1;

    return objective;
```


```python
# The registration has already been performed. Load the results.
if os.path.isfile("outputs/laplacian2.dat") and os.path.isfile("outputs/lsf2.dat"):
    temp = np.loadtxt("outputs/laplacian2.dat");
    k_core = temp[0];
    k_fibre = temp[1];
    k_matrix = temp[2];

    temp = np.loadtxt("outputs/lsf2.dat");
    a2 = temp[0];
    b2 = temp[1];
    c2 = temp[2];
    d2 = temp[3];
    e2 = temp[4];
    f2 = temp[5];

# Perform the registration using CMA-ES
else:

    a2 = 601.873;
    b2 = 54.9359;
    c2 = -3.58452;
    d2 = 0.469614;
    e2 = 6.32561e+09;
    f2 = 1.0;

    x0 = [
        k_core,
        k_fibre,
        k_matrix,
        a2, b2, c2, d2, e2, f2
    ];

    bounds = [
        [
            k_core-500,
            k_fibre-500,
            k_matrix-500,
            a2 - a2 / 4.,
            b2 - b2 / 4.,
            c2 + c2 / 4.,
            d2 - d2 / 4.,
            e2 - e2 / 4.,
            f2 - f2/ 4.
        ],
        [
            k_core+500,
            k_fibre+500,
            k_matrix+500,
            a2 + a2 / 4.,
            b2 + b2 / 4.,
            c2 - c2 / 4.,
            d2 + d2 / 4.,
            e2 + e2 / 4.,
            f2 + f2/ 4.
        ]
    ];

    best_fitness = sys.float_info.max;
    best_fitness_id = 0;
    prefix = "laplacian2_"

    opts = cma.CMAOptions()
    opts.set('tolfun', 1e-4);
    opts['tolx'] = 1e-4;
    opts['bounds'] = bounds;
    #opts['seed'] = 987654321;
    # opts['maxiter'] = 5;
    opts['CMA_stds'] = [1250 * 0.2, 1250 * 0.2, 1250 * 0.2,
        a2 * 0.2, b2 * 0.2, -c2 * 0.2, d2 * 0.2, e2 * 0.2, f2 * 0.2];

    es = cma.CMAEvolutionStrategy(x0, 0.25, opts);
    es.optimize(fitnessFunctionLaplacianLSF);

    k_core = es.result.xbest[0];
    k_fibre = es.result.xbest[1];
    k_matrix = es.result.xbest[2];

    a2 = es.result.xbest[3];
    b2 = es.result.xbest[4];
    c2 = es.result.xbest[5];
    d2 = es.result.xbest[6];
    e2 = es.result.xbest[7];
    f2 = es.result.xbest[8];

    np.savetxt("outputs/laplacian2.dat", [k_core, k_fibre, k_matrix], header='k_core, k_fibre, k_matrix');
    np.savetxt("outputs/lsf2.dat", [a2, b2, c2, d2, e2, f2], header='a2, b2, c2, d2, e2, f2');

    # Release memory
    del es;
```

    (5_w,10)-aCMA-ES (mu_w=3.2,w_1=45%) in dimension 9 (seed=281012, Wed May  5 12:12:40 2021)
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Iterat #Fevals   function value  axis ratio  sigma  min&max std  t[m:s]
        1     10 9.972437000181190e-02 1.0e+00 2.39e-01  2e-02  3e+08 0:19.1
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
        2     20 9.932980821686133e-02 1.3e+00 2.48e-01  2e-02  3e+08 0:38.5
        3     30 9.958514458818876e-02 1.3e+00 2.53e-01  2e-02  4e+08 0:57.9
        4     40 9.944330678252196e-02 1.4e+00 2.56e-01  2e-02  4e+08 1:15.3
        5     50 9.947764477109031e-02 1.6e+00 2.40e-01  2e-02  4e+08 1:33.7
        6     60 9.943956303459305e-02 1.6e+00 2.62e-01  2e-02  4e+08 1:50.9
        7     70 9.936924952442883e-02 1.8e+00 2.70e-01  3e-02  4e+08 2:06.9
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
        8     80 9.923314964021686e-02 1.9e+00 2.51e-01  2e-02  4e+08 2:23.6
        9     90 9.933579314512903e-02 2.0e+00 2.30e-01  2e-02  3e+08 2:40.6
       10    100 9.936451245134174e-02 2.2e+00 2.23e-01  2e-02  3e+08 2:58.4
    Reconstructing 1 slice groups with 1 master threads...
       11    110 9.919633419066988e-02 2.2e+00 2.09e-01  2e-02  3e+08 3:17.0
    Reconstructing 1 slice groups with 1 master threads...
       12    120 9.911387778575653e-02 2.3e+00 2.21e-01  2e-02  3e+08 3:34.3
       13    130 9.911668594719297e-02 2.4e+00 2.31e-01  3e-02  3e+08 3:51.8
    Reconstructing 1 slice groups with 1 master threads...
       14    140 9.904039008193059e-02 2.4e+00 2.17e-01  2e-02  3e+08 4:10.1
       15    150 9.909888576255736e-02 2.6e+00 1.86e-01  2e-02  3e+08 4:28.4
       16    160 9.920223165949110e-02 2.6e+00 1.77e-01  2e-02  2e+08 4:47.6
       17    170 9.908291308264612e-02 2.6e+00 1.72e-01  2e-02  2e+08 5:04.8
    Reconstructing 1 slice groups with 1 master threads...
       18    180 9.896700390987535e-02 2.7e+00 1.90e-01  2e-02  3e+08 5:23.0
    Reconstructing 1 slice groups with 1 master threads...
       19    190 9.894650269310468e-02 2.9e+00 2.31e-01  2e-02  3e+08 5:41.4
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
    Reconstructing 1 slice groups with 1 master threads...
       21    210 9.888006209449185e-02 3.3e+00 2.62e-01  3e-02  4e+08 6:16.4
    Reconstructing 1 slice groups with 1 master threads...
       23    230 9.899852042704164e-02 3.5e+00 2.55e-01  3e-02  3e+08 6:50.7
       25    250 9.901689918899491e-02 3.5e+00 2.25e-01  2e-02  3e+08 7:24.0
       27    270 9.893269334212311e-02 3.8e+00 2.12e-01  2e-02  3e+08 7:57.2
       29    290 9.890050567171228e-02 4.0e+00 2.13e-01  2e-02  3e+08 8:30.8
       31    310 9.887353419803889e-02 4.0e+00 2.20e-01  2e-02  3e+08 9:03.8
       33    330 9.886796294639294e-02 4.6e+00 2.31e-01  3e-02  3e+08 9:37.0
       35    350 9.887028718850092e-02 5.2e+00 3.22e-01  4e-02  5e+08 10:10.6
       37    370 9.887003373277536e-02 5.4e+00 3.77e-01  5e-02  5e+08 10:43.9
       39    390 9.893323564427291e-02 5.6e+00 4.10e-01  5e-02  6e+08 11:17.0
       41    410 9.887141215532647e-02 5.9e+00 3.87e-01  5e-02  6e+08 11:50.3
       43    430 9.884866871773203e-02 6.6e+00 3.33e-01  5e-02  5e+08 12:24.9
       45    450 9.886302795734891e-02 7.1e+00 2.82e-01  4e-02  4e+08 12:58.2
       47    470 9.886437713275012e-02 7.5e+00 2.52e-01  3e-02  4e+08 13:31.1
    Reconstructing 1 slice groups with 1 master threads...
       49    490 9.885239029903299e-02 7.9e+00 2.12e-01  3e-02  3e+08 14:04.3
    Reconstructing 1 slice groups with 1 master threads...
       52    520 9.884914839888259e-02 8.9e+00 1.91e-01  2e-02  3e+08 14:54.1
       55    550 9.884967339787341e-02 9.4e+00 1.37e-01  2e-02  2e+08 15:43.1
       58    580 9.884309263008563e-02 9.9e+00 1.35e-01  2e-02  2e+08 16:32.6
       61    610 9.884429359989334e-02 1.1e+01 1.30e-01  2e-02  2e+08 17:21.9
       63    630 9.884232174958699e-02 1.1e+01 1.64e-01  2e-02  2e+08 17:54.6



```python
if not os.path.exists("plots/laplacian2_registration.gif"):
    registration_image_set = createAnimation("outputs/laplacian2_simulated_CT_",
                'plots/laplacian2_registration.gif');
```

![Animation of the registration (GIF file)](tutorial/plots/laplacian2_registration.gif)

### Apply the result of the registration


```python
# The response of the detector as the line-spread function (LSF)
t = np.arange(-20., 21., 1.);
lsf_kernel=lsf(t*41, a2, b2, c2, d2, e2, f2);
lsf_kernel/=lsf_kernel.sum();
np.savetxt("outputs/LSF_optimised.txt", lsf_kernel);
```


```python
# Simulate the corresponding CT aquisition
sigma_set = [sigma_core, sigma_fibre, sigma_matrix];
k_set = [k_core, k_fibre, k_matrix];
label_set = ["core", "fibre", "matrix"];

simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram(sigma_set, k_set, label_set);

# Reconstruct the CT slice
simulated_CT = tomopy.recon(simulated_sinogram,
                            theta_rad,
                            center=rot_center,
                            sinogram_order=False,
                            algorithm='gridrec',
                            filter_name='shepp',
                            ncore=40)[0];
normalised_simulated_CT = standardisation(simulated_CT);

offset1 = 86;
offset2 = reference_CT.shape[0] - offset1;
profile_test_whole_image_without_Poisson_noise = copy.deepcopy(np.diag(simulated_CT[offset1:offset2, offset1:offset2]));

# Compute the ZNCC
print("ZNCC phase contrast and LSF registration:",
      "{:.2f}".format(100.0 * np.mean(np.multiply(normalised_reference_CT, normalised_simulated_CT))));
```

    Reconstructing 1 slice groups with 1 master threads...
    ZNCC phase contrast and LSF registration: 93.80



```python
fig=plt.figure();
plt.title("Response of the detector (LSF)");
plt.plot(t, old_lsf, label="Before optimisation");
plt.plot(t, lsf_kernel, label="After optimisation");
plt.legend();
plt.savefig('plots/LSF_optimised.pdf');
plt.savefig('plots/LSF_optimised.png');
```



![png](doc/output_158_0.png)



### Extract the fibre in the centre of the CT slices


```python
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
```


```python
findFibreInCentreOfCtSlice();

reference_fibre_in_centre = np.array(copy.deepcopy(reference_CT[cylinder_position_in_centre_of_slice[1] - roi_length:cylinder_position_in_centre_of_slice[1] + roi_length, cylinder_position_in_centre_of_slice[0] - roi_length:cylinder_position_in_centre_of_slice[0] + roi_length]));
test_fibre_in_centre      = np.array(copy.deepcopy(simulated_CT[cylinder_position_in_centre_of_slice[1] - roi_length:cylinder_position_in_centre_of_slice[1] + roi_length, cylinder_position_in_centre_of_slice[0] - roi_length:cylinder_position_in_centre_of_slice[0] + roi_length]));

profile_reference = copy.deepcopy(np.diag(reference_fibre_in_centre));
profile_test_without_Poisson_noise = copy.deepcopy(np.diag(test_fibre_in_centre));

reference_fibre_in_centre = standardisation(reference_fibre_in_centre);
test_fibre_in_centre = standardisation(test_fibre_in_centre);
```


```python
norm = cm.colors.Normalize(vmax=1.25, vmin=-0.5)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
plt.tight_layout()
fig.suptitle('Fibre in the centre of the CT slices')

ax1.set_title("Reference image");
imgplot1 = ax1.imshow(reference_fibre_in_centre, cmap="gray",
                     norm=norm);

ax2.set_title("Simulated CT slice after automatic registration");
imgplot2 = ax2.imshow(test_fibre_in_centre,
                     cmap='gray',
                     norm=norm);

comp_equalized = compare_images(reference_fibre_in_centre, test_fibre_in_centre, method='checkerboard');
ax3.set_title("Checkboard comparison between\n" +
              "the reference and simulated images\nZNCC: " +
              "{:.2f}".format(100.0 * np.mean(np.multiply(reference_fibre_in_centre, test_fibre_in_centre))));
imgplot3 = ax3.imshow(comp_equalized,
                     cmap='gray',
                     norm=norm);
```



![png](doc/output_162_0.png)



## Optimisation of the Poisson noise


```python
def fitnessFunctionNoise(x):
    global best_fitness;
    global best_fitness_id;
    global prefix;

    bias = x[0];
    gain = x[1];
    scale = x[2];

    # Poisson noise
    map = (normalised_projections_ROI + (bias + 1)) * gain;
    temp = np.random.poisson(map).astype(float);
    temp /= gain;
    temp -= bias + 1;

    # Noise map
    noise_map = normalised_projections_ROI - temp;
    noise_map *= scale;
    noisy_image = normalised_projections_ROI + noise_map;

    # Compute the standard deviation of the pixel values in the ROI extracted from the simulated image with noise
    noisy_image_noise_ROI_stddev = 0;
    for y in range(noisy_image.shape[0]):
        noisy_image_noise_ROI_stddev += noisy_image[y].std();
    noisy_image_noise_ROI_stddev /= noisy_image.shape[0];

    # Difference of std dev between the reference and the simulated image
    diff = reference_noise_ROI_stddev - noisy_image_noise_ROI_stddev;
    objective = diff * diff;

    # The block below is not necessary for the registration.
    # It is used to save the data to create animations.
    if best_fitness > objective:
        best_fitness = objective;

        # Save the simulated CT slice
        saveMHA("outputs/" + prefix + "noisy_image_" + str(best_fitness_id) + ".mha",
                noisy_image,
                [pixel_spacing_in_mm, pixel_spacing_in_mm, pixel_spacing_in_mm]);

        np.savetxt("outputs/" + prefix + str(best_fitness_id) + ".dat", [bias, gain, scale], header='bias, gain, scale');

        best_fitness_id += 1;

    return objective
```


```python
# The registration has already been performed. Load the results.
if os.path.isfile("outputs/poisson-noise.dat"):
    temp = np.loadtxt("outputs/poisson-noise.dat");
    bias = temp[0];
    gain = temp[1];
    scale = temp[2];

# Perform the registration using CMA-ES
else:

    # Extract a ROI from the reference where no object is
    reference_noise_ROI = copy.deepcopy(reference_normalised_projections[450:550,0:125]);

    saveMHA("outputs/reference_noise_ROI.mha",
           reference_noise_ROI,
           [pixel_spacing_in_mm, angular_step, pixel_spacing_in_mm]);

    # Compute the standard deviation of the pixel values in the ROI extracted from the reference
    reference_noise_ROI_stddev = 0;
    for y in range(reference_noise_ROI.shape[0]):
        reference_noise_ROI_stddev += reference_noise_ROI[y].std();
    reference_noise_ROI_stddev /= reference_noise_ROI.shape[0];

    # Copy the simulated projection in a temporary variable
    temp = copy.deepcopy(normalised_projections);
    temp.shape = reference_normalised_projections.shape

    # Extract the corresponding ROI
    normalised_projections_ROI = temp[450:550,0:125];

    saveMHA("outputs/normalised_projections_ROI.mha",
           normalised_projections_ROI,
           [pixel_spacing_in_mm, angular_step, pixel_spacing_in_mm]);

    # Initialise the values
    bias = 0.0;
    gain = 255.0;
    scale = 1;

    x0 = [bias, gain, scale];

    bounds = [
        [-1.0,   0.0, 0.0],
        [ 5.0, 255.0, 255.0]
    ];

    opts = cma.CMAOptions()
    opts.set('tolfun', 1e-8);
    opts['tolx'] = 1e-8;
    opts['bounds'] = bounds;
    opts['CMA_stds'] = [1, 10, 10];

    best_fitness = sys.float_info.max;
    best_fitness_id = 0;
    prefix = "poisson-noise_";

    es = cma.CMAEvolutionStrategy(x0, 0.25, opts);
    es.optimize(fitnessFunctionNoise);

    bias = es.result.xbest[0];
    gain = es.result.xbest[1];
    scale = es.result.xbest[2];

    np.savetxt("outputs/poisson-noise.dat", [bias, gain, scale], header='bias, gain, scale');

    # Release memory
    del es;
```

    (3_w,7)-aCMA-ES (mu_w=2.3,w_1=58%) in dimension 3 (seed=242552, Wed May  5 12:30:59 2021)
    Iterat #Fevals   function value  axis ratio  sigma  min&max std  t[m:s]
        1      7 3.162028159789423e-04 1.0e+00 2.58e-01  3e-01  3e+00 0:00.1
        2     14 7.293268754239140e-04 1.4e+00 2.39e-01  3e-01  3e+00 0:00.1
        3     21 2.905347846539879e-06 1.6e+00 2.50e-01  3e-01  3e+00 0:00.1
       52    364 7.970205606575533e-12 1.2e+03 2.43e-02  4e-04  4e-01 0:00.9



```python
print("Noise parameters: ", bias, gain, scale)
```

    Noise parameters:  0.3971793749613459 253.102676183616 0.0518954062607101


### Apply the result of the optimisation


```python
# Simulate the corresponding CT aquisition
simulated_sinogram, normalised_projections, raw_projections_in_keV = simulateSinogram(sigma_set, k_set, label_set);
```


```python
# Reconstruct the CT slice
simulated_CT = tomopy.recon(simulated_sinogram,
                            theta_rad,
                            center=rot_center,
                            sinogram_order=False,
                            algorithm='gridrec',
                            filter_name='shepp',
                            ncore=40)[0];
normalised_simulated_CT = standardisation(simulated_CT);
profile_test_whole_image_with_Poisson_noise = copy.deepcopy(np.diag(simulated_CT[offset1:offset2, offset1:offset2]));

# Compute the ZNCC
print("ZNCC noise registration:",
      "{:.2f}".format(100.0 * np.mean(np.multiply(normalised_reference_CT, normalised_simulated_CT))));
```

    Reconstructing 1 slice groups with 1 master threads...
    ZNCC noise registration: 92.26



```python
norm = cm.colors.Normalize(vmax=1.25, vmin=-0.5)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
plt.tight_layout()
fig.suptitle('CT slice with fibres after the registration')

ax1.set_title("Reference image");
imgplot1 = ax1.imshow(normalised_reference_CT, cmap="gray",
                     norm=norm);

ax2.set_title("Simulated CT slice after automatic registration");
imgplot2 = ax2.imshow(normalised_simulated_CT,
                     cmap='gray',
                     norm=norm);

comp_equalized = compare_images(normalised_reference_CT, normalised_simulated_CT, method='checkerboard');
ax3.set_title("Checkboard comparison between\n" +
              "the reference and simulated images\nZNCC: " +
              "{:.2f}".format(100.0 * np.mean(np.multiply(normalised_reference_CT, normalised_simulated_CT))));
imgplot3 = ax3.imshow(comp_equalized,
                     cmap='gray',
                     norm=norm);
```



![png](doc/output_170_0.png)




```python
test_fibre_in_centre      = np.array(copy.deepcopy(simulated_CT[cylinder_position_in_centre_of_slice[1] - roi_length:cylinder_position_in_centre_of_slice[1] + roi_length, cylinder_position_in_centre_of_slice[0] - roi_length:cylinder_position_in_centre_of_slice[0] + roi_length]));
profile_test_with_Poisson_noise = copy.deepcopy(np.diag(test_fibre_in_centre));

plt.figure()

plt.title("Diagonal profile of the fibre in the centre of the reference CT and\nthe simulated CT slice without and with Poisson noise")
plt.plot(profile_reference, label="Reference");
plt.plot(profile_test_without_Poisson_noise, label="Simulation without Poisson noise");
plt.plot(profile_test_with_Poisson_noise, label="Simulation with Poisson noise");
plt.legend();
```



![png](doc/output_171_0.png)




```python

```
