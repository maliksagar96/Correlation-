import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import tiff_file
from bpass import bpass

coord_all = np.load('fov0/MT_featsize_7.npy')

coord = coord_all[np.where(coord_all[:,5]==0)]		#Separating out 0th frame data

x = np.array(coord[:,0])
y = np.array(coord[:,1])

image0 = tiff_file.imread('DIW057.tif')

image = image0[0,:,:,0]

image = np.transpose(image,(1,0))

fig = plt.figure()
implot=plt.imshow(image,'gray',interpolation='none',origin='lower')
n = plt.plot(y,x,'go',markeredgecolor='b',markersize=2,label='Weeks')
#plt.show()
plt.savefig('image0.jpg')

