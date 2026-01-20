#!/usr/bin/env python
# coding: utf-8

# # Aligning partially matched coronal sections of adult mouse brain from Xenium and STARmap PLUS
# 
# In this notebook, we align two single cell resolution spatial transcriptomics datasets of full and hemi coronal sections of the adult mouse brain from approximately the same locations assayed by Xenium and STARmap PLUS. For more details about how these datasets were generated, please consult the [Xenium mouse brain data release](https://www.10xgenomics.com/products/xenium-in-situ/mouse-brain-dataset-explorer) and the [STARmap PLUS preprint](https://www.biorxiv.org/content/10.1101/2022.06.20.496914v1).
# 
# We will use `STalign` to achieve this alignment. We will first load the relevant code libraries.

# In[1]:


# import dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

# make plots bigger
plt.rcParams["figure.figsize"] = (12,10)


# In[ ]:


# OPTION A: import STalign after pip or pipenv install
from STalign import STalign


# In[2]:


## OPTION B: skip cell if installed STalign with pip or pipenv
import sys
sys.path.append("../../STalign") 

## import STalign from upper directory
import STalign


# We have already downloaded single cell spatial transcriptomics datasets and placed the files in a folder called `xenium_data` and `starmap_data`.
# 
# We can read in the cell information for the first dataset using `pandas` as `pd`.

# In[3]:


# Single cell data 1
# read in data
fname = '../xenium_data/Xenium_V1_FF_Mouse_Brain_MultiSection_1_cells.csv.gz'
df1 = pd.read_csv(fname)
print(df1.head())


# For alignment with `STalign`, we only need the cell centroid information. So we can pull out this information. We can further visualize the cell centroids to get a sense of the variation in cell density that we will be relying on for our alignment by plotting using `matplotlib.pyplot` as `plt`. 

# In[4]:


# get cell centroid coordinates
xI = np.array(df1['x_centroid'])
yI = np.array(df1['y_centroid'])

# plot
fig,ax = plt.subplots()
ax.scatter(xI,yI,s=1,alpha=0.2)


# We will first use `STalign` to rasterize the single cell centroid positions into an image. Assuming the single-cell centroid coordinates are in microns, we will perform this rasterization at a 30 micron resolution. We can visualize the resulting rasterized image.
# 
# Note that points are plotting with the origin at bottom left while images are typically plotted with origin at top left so we've used `invert_yaxis()` to invert the yaxis for visualization consistency. 

# In[5]:


# rasterize at 30um resolution (assuming positions are in um units) and plot
XI,YI,I,fig = STalign.rasterize(xI,yI)

# plot
ax = fig.axes[0]
ax.invert_yaxis()


# Now, we can repeat this for the cell information from the second dataset.

# In[6]:


# Single cell data 2
# read in data
fname = '../starmap_data/well11_spatial.csv.gz'
df2 = pd.read_csv(fname, skiprows=[1]) # first row is data type
print(df2.head())


# In[7]:


# get cell centroids
xJ = np.array(df2['Y'])/5 # convert to similar scale
yJ = np.array(df2['X'])/5

# flip
yJ = yJ.max() - yJ

# plot
fig,ax = plt.subplots()
ax.scatter(xJ,yJ,s=1,alpha=0.2,c='#ff7f0e')


# In[8]:


# rasterize and plot
XJ,YJ,J,fig = STalign.rasterize(xJ,yJ)
ax = fig.axes[0]
ax.invert_yaxis()


# Note that plotting the cell centroid positions from both datasets shows that non-linear local alignment is needed.

# In[9]:


# plot
fig,ax = plt.subplots()
ax.scatter(xI,yI,s=1,alpha=0.1)
ax.scatter(xJ,yJ,s=1,alpha=0.2)


# We can also plot the rasterized images next to each other.

# In[10]:


# get extent of images
extentI = STalign.extent_from_x((YI,XI))
extentJ = STalign.extent_from_x((YJ,XJ))

# plot rasterized images
fig,ax = plt.subplots(2,1)
ax[0].imshow((I.transpose(1,2,0).squeeze()), extent=extentI) 
ax[1].imshow((J.transpose(1,2,0).squeeze()), extent=extentJ) 
ax[0].invert_yaxis()
ax[1].invert_yaxis()


# Now we will perform our alignment. There are many parameters that can be tuned for performing this alignment. If we don't specify parameters, defaults will be used. 

# In[11]:


# set device for building tensors
if torch.cuda.is_available():
    torch.set_default_device('cuda:0')
else:
    torch.set_default_device('cpu')


# In[12]:


get_ipython().run_cell_magic('time', '', "\n# run LDDMM\n# specify device (default device for STalign.LDDMM is cpu)\nif torch.cuda.is_available():\n    device = 'cuda:0'\nelse:\n    device = 'cpu'\n\n# keep all other parameters default\nparams = {\n          'niter': 4000,\n          'device':device,\n          'sigmaM':1.5,\n          'sigmaB':1.0,\n          'sigmaA':1.5,\n          'epV': 50,\n          'muB': torch.tensor([0,0,0]), # black is background in target\n          }\n\nIfoo = np.vstack((I, I, I)) # make RGB instead of greyscale\nJfoo = np.vstack((J, J, J)) # make RGB instead of greyscale\nout = STalign.LDDMM([YI,XI],Ifoo,[YJ,XJ],Jfoo,**params)\n")


# In[13]:


# get necessary output variables
A = out['A']
v = out['v']
xv = out['xv']


# Plots generated throughout the alignment can be used to give you a sense of whether the parameter choices are appropriate and whether your alignment is converging on a solution.
# 
# We can also evaluate the resulting alignment by applying the transformation to visualize how our source and target images were deformed to achieve the alignment. 

# In[14]:


# apply transform
phii = STalign.build_transform(xv,v,A,XJ=[YJ,XJ],direction='b')
phiI = STalign.transform_image_source_to_target(xv,v,A,[YI,XI],Ifoo,[YJ,XJ])

#switch tensor from cuda to cpu for plotting with numpy
if phii.is_cuda:
    phii = phii.cpu()
if phiI.is_cuda:
    phiI = phiI.cpu()

# plot with grids
fig,ax = plt.subplots()
levels = np.arange(-100000,100000,1000)
ax.contour(XJ,YJ,phii[...,0],colors='r',linestyles='-',levels=levels)
ax.contour(XJ,YJ,phii[...,1],colors='g',linestyles='-',levels=levels)
ax.set_aspect('equal')
ax.set_title('source to target')
ax.imshow(phiI.permute(1,2,0)/torch.max(phiI),extent=extentJ)
ax.invert_yaxis()


# Note that because of our use of LDDMM, the resulting transformation is invertible.

# In[15]:


# transform is invertible
phi = STalign.build_transform(xv,v,A,XJ=[YI,XI],direction='f')
phiiJ = STalign.transform_image_target_to_source(xv,v,A,[YJ,XJ],Jfoo,[YI,XI])

#switch tensor from cuda to cpu for plotting with numpy
if phi.is_cuda:
    phi = phi.cpu()
if phiiJ.is_cuda:
    phiiJ = phiiJ.cpu()

# plot with grids
fig,ax = plt.subplots()
levels = np.arange(-100000,100000,1000)
ax.contour(XI,YI,phi[...,0],colors='r',linestyles='-',levels=levels)
ax.contour(XI,YI,phi[...,1],colors='g',linestyles='-',levels=levels)
ax.set_aspect('equal')
ax.set_title('target to source')
ax.imshow(phiiJ.permute(1,2,0)/torch.max(phiiJ),extent=extentI)
ax.invert_yaxis()


# Finally, we can apply our transform to the original sets of single cell centroid positions to achieve their new aligned positions.

# In[16]:


# apply transform to original points
tpointsJ = STalign.transform_points_target_to_source(xv,v,A, np.stack([yJ, xJ], 1))

#switch tensor from cuda to cpu for plotting with numpy
if tpointsJ.is_cuda:
    tpointsJ = tpointsJ.cpu()

# just original points for visualizing later
tpointsI = np.stack([xI, yI]) 


# And we can visualize the results.

# In[17]:


# plot results
fig,ax = plt.subplots()
ax.scatter(tpointsI[0,:],tpointsI[1,:],s=1,alpha=0.1) 
ax.scatter(tpointsJ[:,1],tpointsJ[:,0],s=1,alpha=0.2) # also needs to plot as y,x not x,y


# And save the new aligned positions by appending to our original data using `numpy` with `np.hstack`

# In[18]:


# save results
results = tpointsJ.numpy()


# We will finally create a compressed `.csv.gz` file to create `starmap_data/starmap_STalign_to_xenium.csv.gz`

# In[ ]:


results.to_csv('../starmap_data/starmap_STalign_to_xenium.csv.gz',
               compression='gzip')

