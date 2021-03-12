import math
import numpy as np


def lsf(x, a2 = 601.873, b2 = 54.9359, c2 = -3.58452, d2 = 0.469614, e2 = 6.32561e+09, f2 = 1.0):

    if hasattr(x, "__len__") == False:
        temp = x;

    else:
        temp = np.array(x);
        
    temp_1 = (2.0/(math.sqrt(math.pi)*e2*f2)) * np.exp(-np.square(x)/(e2*e2));
    temp_2 = 1.0 / (b2 * c2) * np.power(1+np.square(x)/(b2*b2),-1);
    temp_3 = np.power(2.0/f2+math.pi/c2, -1);
    value = (temp_1 + temp_2) * temp_3;

    return value
