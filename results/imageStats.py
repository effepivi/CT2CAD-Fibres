import sys, copy
import numpy as np
import itk

import numpy as np
import pandas as pd
from scipy import ndimage


CPUImageType = itk.Image[itk.F,2]

ReaderType = itk.ImageFileReader[CPUImageType]
reader = ReaderType.New();

reader.SetFileName(sys.argv[1]);
reader.Update();
reference_CT_slice=itk.GetArrayFromImage(reader.GetOutput());

reader.SetFileName(sys.argv[2]);
reader.Update();
simulated_CT_slice=itk.GetArrayFromImage(reader.GetOutput());


roi_length = 40



reference_fibre_in_centre = reference_CT_slice[429 - roi_length:430 + roi_length, 520 - roi_length:521 + roi_length];
test_fibre_in_centre = simulated_CT_slice[429 - roi_length:430 + roi_length, 520 - roi_length:521 + roi_length];



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
    fibre_radius_in_px = int(108 / 1.9) / 2
    core_radius_in_px = int(16 / 1.9) / 2

    core_mask = create_circular_mask(mask_shape[1], mask_shape[0], None, core_radius_in_px);

    fibre_mask = create_circular_mask(mask_shape[1], mask_shape[0], None, fibre_radius_in_px);
    matrix_mask = np.logical_not(fibre_mask);

    #fibre_mask = np.subtract(fibre_mask, core_mask);
    fibre_mask = np.bitwise_xor(fibre_mask, core_mask);

    #TypeError: numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, the `^` operator, or the logical_xor function instead.

    return core_mask, fibre_mask, matrix_mask

mask_shape = reference_fibre_in_centre.shape;
core_mask, fibre_mask, matrix_mask = createMasks(mask_shape);

core_mask = ndimage.binary_erosion(core_mask).astype(core_mask.dtype);

for i in range(4):
    fibre_mask = ndimage.binary_erosion(fibre_mask).astype(fibre_mask.dtype);
    matrix_mask = ndimage.binary_erosion(matrix_mask, border_value=1).astype(matrix_mask.dtype);

core_mask.shape = [core_mask.shape[0], core_mask.shape[1]]
fibre_mask.shape = [fibre_mask.shape[0], fibre_mask.shape[1]]
matrix_mask.shape = [matrix_mask.shape[0], matrix_mask.shape[1]]

def getMuStatistics(reference_fibre_in_centre, test_fibre_in_centre, core_mask, fibre_mask, matrix_mask):

    data = [];
    index = np.nonzero(core_mask);

    data.append(["Theorical",
                "Core",
                "W",
                341.61,
                341.61,
                341.61,
                0.0]);

    data.append(["Experimental",
                "Core",
                "W",
                np.min(reference_fibre_in_centre[index]),
                np.max(reference_fibre_in_centre[index]),
                np.mean(reference_fibre_in_centre[index]),
                np.std(reference_fibre_in_centre[index])]);

    data.append(["Simulated",
                "Core",
                "W",
                np.min(test_fibre_in_centre[index]),
                np.max(test_fibre_in_centre[index]),
                np.mean(test_fibre_in_centre[index]),
                np.std(test_fibre_in_centre[index])]);

    index = np.nonzero(fibre_mask);

    data.append(["Theorical",
                "Fibre",
                "SiC",
                2.736,
                2.736,
                2.736,
                0.0]);

    data.append(["Experimental",
                "Fibre",
                "SiC",
                np.min(reference_fibre_in_centre[index]),
                np.max(reference_fibre_in_centre[index]),
                np.mean(reference_fibre_in_centre[index]),
                np.std(reference_fibre_in_centre[index])]);

    data.append(["Simulated",
                "Fibre",
                "SiC",
                np.min(test_fibre_in_centre[index]),
                np.max(test_fibre_in_centre[index]),
                np.mean(test_fibre_in_centre[index]),
                np.std(test_fibre_in_centre[index])]);

    index = np.nonzero(matrix_mask);
    data.append(["Theorical",
                "Matrix",
                "Ti90Al6V4",
                13.1274,
                13.1274,
                13.1274,
                0.0]);

    data.append(["Experimental",
                "Matrix",
                "Ti90Al6V4",
                np.min(reference_fibre_in_centre[index]),
                np.max(reference_fibre_in_centre[index]),
                np.mean(reference_fibre_in_centre[index]),
                np.std(reference_fibre_in_centre[index])]);

    data.append(["Simulated",
                "Matrix",
                "Ti90Al6V4",
                np.min(test_fibre_in_centre[index]),
                np.max(test_fibre_in_centre[index]),
                np.mean(test_fibre_in_centre[index]),
                np.std(test_fibre_in_centre[index])]);

    return pd.DataFrame(data,
            index=None,
            columns=['CT', 'Structure', "Composition", 'min', 'max', 'mean', 'stddev'])

df = getMuStatistics(reference_fibre_in_centre, test_fibre_in_centre, core_mask, fibre_mask, matrix_mask)

test_experimental=df["CT"] == "Experimental";
test_simulated=df["CT"] == "Simulated";
test_W=df["Composition"] == "W"
test_SiC=df["Composition"] == "SiC"
test_Ti90Al6V4=df["Composition"] == "Ti90Al6V4"


print(df[test_experimental & test_W]["mean"].astype(float)[1],
    df[test_experimental & test_W]["stddev"].astype(float)[1],

    df[test_simulated & test_W]["mean"].astype(float)[2],
    df[test_simulated & test_W]["stddev"].astype(float)[2],

    df[test_experimental & test_SiC]["mean"].astype(float)[4],
    df[test_experimental & test_SiC]["stddev"].astype(float)[4],

    df[test_simulated & test_SiC]["mean"].astype(float)[5],
    df[test_simulated & test_SiC]["stddev"].astype(float)[5],

    df[test_experimental & test_Ti90Al6V4]["mean"].astype(float)[7],
    df[test_experimental & test_Ti90Al6V4]["stddev"].astype(float)[7],

    df[test_simulated & test_Ti90Al6V4]["mean"].astype(float)[8],
    df[test_simulated & test_Ti90Al6V4]["stddev"].astype(float)[8])


# MEAN_CORE_SIM=`grep "After noise CORE SIMULATED (MIN, MEDIAN, MAX, MEAN, STDDEV)" run_SCW_$i/optimisation-$i.out | cut -d " " -f 13`
# STDDEV_CORE_SIM=`grep "After noise CORE SIMULATED (MIN, MEDIAN, MAX, MEAN, STDDEV)" run_SCW_$i/optimisation-$i.out | cut -d " " -f 14`
#
# MEAN_FIBRE_REF=`grep "After noise FIBRE REF (MIN, MEDIAN, MAX, MEAN, STDDEV)" run_SCW_$i/optimisation-$i.out | cut -d " " -f 13`
# STDDEV_FIBRE_REF=`grep "After noise FIBRE REF (MIN, MEDIAN, MAX, MEAN, STDDEV)" run_SCW_$i/optimisation-$i.out | cut -d " " -f 14`
#
# MEAN_FIBRE_SIM=`grep "After noise FIBRE SIMULATED (MIN, MEDIAN, MAX, MEAN, STDDEV)" run_SCW_$i/optimisation-$i.out | cut -d " " -f 13`
# STDDEV_FIBRE_SIM=`grep "After noise FIBRE SIMULATED (MIN, MEDIAN, MAX, MEAN, STDDEV)" run_SCW_$i/optimisation-$i.out | cut -d " " -f 14`
#
# MEAN_MATRIX_REF=`grep "After noise MATRIX REF (MIN, MEDIAN, MAX, MEAN, STDDEV)" run_SCW_$i/optimisation-$i.out | cut -d " " -f 13`
# STDDEV_MATRIX_REF=`grep "After noise MATRIX REF (MIN, MEDIAN, MAX, MEAN, STDDEV)" run_SCW_$i/optimisation-$i.out | cut -d " " -f 14`
#
# MEAN_MATRIX_SIM=`grep "After noise MATRIX SIMULATED (MIN, MEDIAN, MAX, MEAN, STDDEV)" run_SCW_$i/optimisation-$i.out | cut -d " " -f 13`
# STDDEV_MATRIX_SIM=`grep "After noise MATRIX SIMULATED (MIN, MEDIAN, MAX, MEAN, STDDEV)" run_SCW_$i/optimisation-$i.out | cut -d " " -f 14`
#
