	# Rati's data is RGB in nature so be careful and change the mpretrack img = img_timeseries[x,:,:] to img_timeseries[x,:,:,0] 

import mpretrack
import fancytrack
import warnings
import numpy as np
import os

warnings.filterwarnings('ignore')

basepath = '/home/sagar/Documents/codes/finalCodes/correlation/featureTest/'

fname = 'DIW057.tif'
framescan = 1
numframes = framescan   #total number of frames in the stack
time = np.arange(0,numframes)
np.save(os.path.join(basepath, 'fov0/fov0_times.npy'),time)

mpretrack.run(basepath, fname, fovn=0, numframes = framescan, featuresize = 7, masscut = 1000, Imin = 0, barI=None, barCc=None, barRg=None, IdivRg=None, field=2)
