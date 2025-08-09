import numpy as np
from scipy.ndimage import distance_transform_edt

mask = np.array([[False, False, False],
                 [True,  True,  False],
                 [True,  True,  True ]])

dist = distance_transform_edt(mask)
print(dist)

