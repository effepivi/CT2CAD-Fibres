import sys, copy
import numpy as np
import SimpleITK as sitk


test = sitk.ReadImage(sys.argv[1]);
test_np = sitk.GetArrayFromImage(test);
print(str(test_np.mean()) + "," + str(test_np.std()) + "," + str(test_np.min()) + "," + str(test_np.max()));

