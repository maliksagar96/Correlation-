#Calculating Four point Strucure factor and Dynamic susceptibility
#Make sure that the number of particles in all the frames are same or else the code won't work 
#and will have complications. So while linking make sure that goodenough is equal to number of frames.

import time
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import warnings
warnings.filterwarnings('ignore')

start = time.time()
mpp = 145 / 512 		 # Microns per pixel, conversion factor
fps = 21 				 # Frame per second
frames = 9998
conc = 0.57
k = 0.5
minus_one = -1

j = np.sqrt(np.complex(minus_one))

fname = '/home/sagar/Documents/codes/finalCodes/correlation/linking/link057AF.npy'

# loading ndarray having information of x_position, y_position , frame number, particle id respectively
data = np.load(fname)   

data[:, 0] *= mpp                # converting pixel values to micrometers
data[:, 1] *= mpp

b = 1

#Step function which just spits out 0 or 1 depending on whether the input particle has moved greater then some distance or not
#Overlap Function
def overlapFn(b, x_t1,x_t2,y_t1,y_t2):
	overlap = 0.3 * b - (((x_t2 - x_t1)**2+(y_t2 - y_t1)**2)**0.5)
	heaviside_overlap=np.where(overlap<0,0,1)
	return heaviside_overlap

    
#Fourier transform of overlap function
#q = Fourier variable
def FT(q,heaviside_overlap,x_t1,x_t2,y_t1,y_t2):
	j = np.sqrt(np.complex(minus_one))
	return 	np.sum(heaviside_overlap*np.exp(-j*(q*x_t1+q*y_t1)))

if __name__ == "__main__":
	        
	N = np.shape(np.where(data[:, 2] == 0))[1]       #Total number of particles in frame 0

	x = np.zeros((N, frames), dtype = float);
	y = np.zeros((N, frames), dtype = float);

	for t in range(0, frames):
		if(t % 100 == 0):
			print("Completed",t," cycles")
		frameIndex = np.where(data[:,2] == t)         # Index of the ith(or tth if you like) frame 

		datafrm = data[frameIndex]                        # Data of that frame
		datafrm[datafrm[:, 3].argsort()]	
		
		x[:, t] = datafrm[:, 0]
		y[:, t] = datafrm[:, 1]

	S4 = []
	chi4 = []
	zeta = []

	#Calculating moving average.
	window  = 600
	skipFrame = 200
	loopCounter = 0

	for i in range(0, frames-window, window):

		Ns = 0 						#Number of slow particles
		Ns2 = 0 					#Ns square
		Wqt = 0 					#Fourier transform of overlap function  
		W_qt = 0          			#It is with minus q remember
		loopCounter = 0				#For time averaging
		print("Started processing ",i,"th frame")
		for j in range(i, i + window):
			
			ovrlap = overlapFn(b,x[:,i],x[:,j],y[:,i],y[:,j])
			ovrlapsum = sum(ovrlap) 
			Ns = Ns + ovrlapsum
			Ns2 = Ns2 + ovrlapsum * ovrlapsum
			ft = FT(k, ovrlap, x[:,i],x[:, j], y[:,i], y[:, j])
			ft_ = FT(-k, ovrlap, x[:,i],x[:, j], y[:,i], y[:, j])
			Wqt = Wqt + ft
			W_qt = W_qt + ft_
			loopCounter = loopCounter + 1

		S4.append([i/fps, (((Wqt * W_qt)/loopCounter) - (Wqt/loopCounter)**2)/N])							
		chi4.append([i/fps, ((Ns2/loopCounter) - (Ns/loopCounter)**2)/N])
		i = i + skipFrame		

	# # # zeta.append()		
	

	S4 = np.array(S4)
	chi4 = np.array(chi4)
	# # #zeta = np.array(zeta)
	np.save('S4_'+str(conc)+'_k_'+str(k)+'_delt',S4)
	np.save('chi4'+str(conc)+'_k_'+str(k)+'_delt',chi4)
	# # #np.save('zeta'+str(conc)+'_k_'+str(k)+'_delt',zeta)

	# # # end=time.time()

# 	# # # print('Runtime : ',end-start)

# 	# # # #plotting S4_q_t

# 	# # plt.title('Four point Correlation function')
# 	# # plt.xscale('log')
# 	# # plt.yscale('log')
# 	# # plt.axis('equal')
# 	# # plt.ylabel('S4_q_t')
# 	# # plt.xlabel('lag time $t$')
# 	# # plt.plot(S4[:,0],S4[:,1], 'bo')
# 	# # plt.show()

# 	# #plotting chi4_t

	plt.title('Dynamic Susceptibility')
	plt.xscale('log')
	plt.yscale('log')
	plt.axis('equal')
	plt.ylabel('chi4_t')
	plt.xlabel('lag time $t$')
	plt.plot(chi4[:,0],chi4[:,1])

	plt.show()

# 	# # # # #plotting eta_t

# 	# # # # plt.title('Dynamic correlation length')
# 	# # # # plt.xscale('log')
# 	# # # # plt.yscale('log')
# 	# # # # plt.axis('equal')
# 	# # # # plt.ylabel('eta_t')
# 	# # # # plt.xlabel('lag time $t$')
# 	# # # # plt.plot(eta_t[:,0],eta_t[:,1])

# 	# # # # plt.show()

