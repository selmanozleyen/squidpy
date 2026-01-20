#!/usr/bin/env python
# coding: utf-8

# # Aligning partial coronal brain sections with the Allen Brain Altas

# In this notebook, we align a STARmap dataset of a the right hemisphere of a mouse brain to the Allen Brain Atlas.
# 
# 

# We will use STalign to achieve this alignment. We will first load the relevant code libraries.

# In[1]:


#Import dependencies

import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import pandas as pd # for csv.
from matplotlib import cm
from matplotlib.lines import Line2D
import os
from os.path import exists,split,join,splitext
from os import makedirs
import glob
import requests
from collections import defaultdict
import nrrd
import torch
from torch.nn.functional import grid_sample
import tornado
from STalign import STalign
import copy
import pandas as pd


# We have already downloaded STARmap datasets and stored them in a folder called starmap_data.
# 

# Now, we can read in the cell infomation using pandas as pd.
# 
# We must load all cell x and y positions into the variable x and y respectively. Importantly, the *scale* of the loaded data must approximately match the Allen Brain Atlas, where ~1 pixel unit is ~1 um

# In[2]:


#Load file

filename = '__'
df = pd.read_csv(r"/home/manju/Documents/STalign-master/STalign_build/docs/starmap_data/well11_spatial.csv.gz")

#Load x position
x = np.array(df['X'])[1:].astype(float) #change to x positions of cells

#Load y position
y = np.array(df['Y'])[1:].astype(float) #change to column y positions of cells


# In[3]:


dx=50
blur = 1


# Now we can load in data from the Allen Brain Altas.
# 
# We need to load in and save files containing the ontology of regions, a 3D cell density atlas, and annotations of brain regions in the aforementioned 3D atlas.
# 
# Depending on which Allen Atlas you would like to use, you can change the url that you download these files from. In this example, we are aligning the adult mouse (P56) Allen Atlas.

# In[4]:


url = 'http://api.brain-map.org/api/v2/data/query.csv?criteria=model::Structure,rma::criteria,[ontology_id$eq1],rma::options[order$eq%27structures.graph_order%27][num_rows$eqall]'

ontology_name,namesdict = STalign.download_aba_ontology(url, 'allen_ontology.csv') #url for adult mouse


# In[5]:


imageurl = 'http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/ara_nissl/ara_nissl_50.nrrd'
labelurl = 'http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_50.nrrd'
imagefile, labelfile = STalign.download_aba_image_labels(imageurl, labelurl, 'aba_nissl.nrrd', 'aba_annotation.nrrd')


# Our first step in this alignment will be to rasterize the single cell centroid positions into an image. This converts our x and y arrays of cells into a pixelated image. For this example, we choose to rasterize at a resolution of 15 (um). 
# 
# 

# In[6]:


#Rasterize Image
X_,Y_,W = STalign.rasterize(x,y,dx=dx, blur = blur,draw=False)     


# In[7]:


X_


# We can visualize the resulting rasterized image.

# In[8]:


#Plot unrasterized/rasterized images
fig,ax = plt.subplots(1,2)
ax[0].scatter(x,y,s=0.5,alpha=0.25)
ax[0].invert_yaxis()
ax[0].set_title('List of cells')
ax[0].set_aspect('equal')

W = W[0]
extent = (X_[0],X_[-1],Y_[0],Y_[-1])    
ax[1].imshow(W,  origin='lower')    
ax[1].invert_yaxis()    
ax[1].set_title('Rasterized')

# save figure
#fig.canvas.draw()    
#fig.savefig(outname[:-4]+'_image.png')


# Now we need to find an approximate slice number in the Allen Atlas to initialize the alignment. Evaluate whether the value of slice is similar to the target image by viewing a side by side comparison of the ABA slice and the target image. If not, change the value of slice. In this example, slice = 0 is anterior and slice = 264 is posterior.

# In[9]:


#find slice
#peruse through images in atlas
# Loading the atlas
slice = 140

vol,hdr = nrrd.read(imagefile)
A = vol    
vol,hdr = nrrd.read(labelfile)
L = vol

dxA = np.diag(hdr['space directions'])
nxA = A.shape
xA = [np.arange(n)*d - (n-1)*d/2.0 for n,d in zip(nxA,dxA)]
XA = np.meshgrid(*xA,indexing='ij')

fig,ax = plt.subplots(1,2)
extentA = STalign.extent_from_x(xA[1:])
ax[0].imshow(A[slice],extent=extentA)
ax[0].set_title('Atlas Slice')

ax[1].imshow(W,extent=extentA)
ax[1].set_title('Target Image')
fig.canvas.draw()    


# In[10]:


len(xA[2])


# Next, we need to find an approximate rotation angle of the Allen Atlas to initialize the alignment. Evaluate whether the rotation of the atlas is similar to the target image by viewing a side by side comparison of the ABA slice and the target image. If not, change the value of theta_deg. Note: the rotation here is defined in degrees.

# In[11]:


from scipy.ndimage import rotate

theta_deg = 90

fig,ax = plt.subplots(1,2)
extentA = STalign.extent_from_x(xA[1:])
ax[0].imshow(rotate(A[slice], angle=theta_deg),extent=extentA)
ax[0].set_title('Atlas Slice')

ax[1].imshow(W,extent=extentA)
ax[1].set_title('Target Image')
fig.canvas.draw()    


# Next, specify if there are any points on the image above that can be approximately matched with eachother and add them in the following format. This helps speed up alignment. (OPTIONAL)
# 

# In[12]:


points_atlas = np.array([[0,0]]) #(-y,-x)
points_target = np.array([[-3700,0]]) # (-y,-x)
Li,Ti = STalign.L_T_from_points(points_atlas,points_target)


# We can define atlas and target points, xI and xJ, as well as atlas and target images, I and J. 

# In[13]:


xJ = [Y_,X_]
J = W[None]/np.mean(np.abs(W))
xI = xA
I = A[None] / np.mean(np.abs(A),keepdims=True)
I = np.concatenate((I,(I-np.mean(I))**2))
Inorm = STalign.normalize(I, t_min=0, t_max=1)
Jnorm = STalign.normalize(J, t_min=0, t_max=1)


# Now that we can chosen the slice number and rotation angle, we are ready to initialize parameters related to our alignment. 

# In[14]:


sigmaA = 0.1 #standard deviation of artifact intensities
sigmaB = 0.1 #standard deviation of background intensities
sigmaM = 0.1 #standard deviation of matching tissue intenities
muA = torch.tensor([0.7,0.7,0.7],device='cpu') #average of artifact intensities
muB = torch.tensor([0,0,0],device='cpu') #average of background intensities


# We can change the parameters above by looking at the intensity histogram of our target image (below). We need to consider intensities of artifacts (tissue that is present in the target image and absent in the atlas), which is usually in the upper range of intensity values. We also need to find the intensity of the background values that does not correspond to any tissue, usually around 0, and standard deviations for these values. The matching tissue intensities are regions that should have tissue that can be aligned in both the atlas and the target image.

# In[15]:


fig,ax = plt.subplots()
ax.hist(Jnorm.ravel())
plt.xlabel('Intensity')
plt.ylabel('Number of Pixels')
plt.title('Intensity Histogram of Target Image')


# The following parameters vary depending on your target image and the duration & accuracy of the alignment. The scale parameters refer to a scaling of the atlas in (x, y, z) to match the size of the target image. The nt parameter ensures smoothness and invertibility of the image transformation. The niter refers to the number of iteration steps in gradient descent to achieve minimum error between the altas and the target image.

# In[16]:


# initialize variables
scale_x = 4 #default = 0.9
scale_y = 4 #default = 0.9
scale_z = 0.9 #default = 0.9
theta0 = (np.pi/180)*theta_deg

# get an initial guess
if 'Ti' in locals():
    T = np.array([-xI[0][slice],np.mean(xJ[0])-(Ti[0]*scale_y),np.mean(xJ[1])-(Ti[1]*scale_x)])
else:
    T = np.array([-xI[0][slice],np.mean(xJ[0]),np.mean(xJ[1])])
#T = np.array([-xI[0][slice],0,0])


scale_atlas = np.array([[scale_z,0,0],
                        [0,scale_x,0],
                        [0,0,scale_y]])
L = np.array([[1.0,0.0,0.0],
             [0.0,np.cos(theta0),-np.sin(theta0)],
              [0.0,np.sin(theta0),np.cos(theta0)]])
L = np.matmul(L,scale_atlas)#np.identity(3)


# Now, we can perform alignment of the atlas to our target slice.

# In[24]:


get_ipython().run_cell_magic('time', '', "#returns mat = affine transform, v = velocity, xv = pixel locations of velocity points\ntransform = STalign.LDDMM_3D_to_slice(\n    xI,Inorm,xJ,Jnorm,\n    T=T,L=L,\n    nt=4,niter=800,\n    a=250,\n    device='cpu',\n    sigmaA = sigmaA, #standard deviation of artifact intensities\n    sigmaB = sigmaB, #standard deviation of background intensities\n    sigmaM = sigmaM, #standard deviation of matching tissue intenities\n    muA = muA, #average of artifact intensities\n    muB = muB #average of background intensities\n)\n")


# In[25]:


A = transform['A']
v = transform['v']
xv = transform['xv']
Xs = transform['Xs']


# In[26]:


print(A)


# Once we have aligned the atlas to our target image, we can project the brain regions defined in the atlas to our target image and export these annotations, along with information about the positions of the atlas that align with the slice.

# In[27]:


df = STalign.analyze3Dalign(labelfile,  xv,v,A, xJ, dx, scale_x=scale_x, scale_y=scale_y,x=x,y=y, X_=X_, Y_=Y_, namesdict=namesdict,device='cpu')


# In[28]:


df


# Now, we can explore our alignment and brain region annotations! 

# We can visualize the MERFISH slice overlayed with the matched section in the Allen Brain Atlas.

# In[29]:


It = torch.tensor(I,device='cpu',dtype=torch.float64)
AI = STalign.interp3D(xI,It,Xs.permute(3,0,1,2),padding_mode="border")
Ishow_source = ((AI-torch.amin(AI,(1,2,3))[...,None,None])/(torch.amax(AI,(1,2,3))-torch.amin(AI,(1,2,3)))[...,None,None,None]).permute(1,2,3,0).clone().detach().cpu()
Jt = torch.tensor(J,device='cpu',dtype=torch.float64)
Ishow_target = Jt.permute(1,2,0).cpu()/torch.max(Jt).item()

import matplotlib as mpl
fig,ax = plt.subplots(1,3, figsize=(15,5))
ax0 = ax[0].imshow(Ishow_target, cmap = mpl.cm.Blues,alpha=0.9)
ax[0].set_title('STARmap Slice')
ax1 = ax[1].imshow(Ishow_source[0,:,:,0], cmap = mpl.cm.Reds,alpha=0.2)
ax[1].set_title('z=0 slice of Aligned 3D Allen Brain Atlas')
ax2 = ax[2].imshow(Ishow_target, cmap = mpl.cm.Blues,alpha=0.9)
ax2 = ax[2].imshow(Ishow_source[0,:,:,0], cmap = mpl.cm.Reds,alpha=0.3)
ax[2].set_title('Overlayed')

plt.show()


# In[30]:


from mpl_toolkits import mplot3d

x=df['coord0']
y=-df['coord1']
z=df['coord2']

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(x,y,z,s=0.05)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


# We can also plot all of the brain regions that are captured in the slice, as defined by our alignment.

# In[31]:


STalign.plot_brain_regions(df)


# If we also interested in only a few regions, in this example let's take VISp4 and VISp5, we can plot those regions using the following function.

# In[32]:


brain_regions = ['CA1']
STalign.plot_subset_brain_regions(df, brain_regions)


# In[ ]:




