#!/usr/bin/env python
# coding: utf-8

# # Aligning partially matched, serial, single-cell resolution breast cancer spatial transcriptomics data from Xenium
# 
# In this notebook, we take two single cell resolution spatial transcriptomics datasets of serial breast cancer sections profiled by the Xenium technology and align them to each other. See the bioRxiv preprint for more details about this data: https://www.biorxiv.org/content/10.1101/2022.10.06.510405v2 
# 
# We will use `STalign` to achieve this alignment. We will first load the relevant code libraries.

# In[5]:


# import dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

# make plots bigger
plt.rcParams["figure.figsize"] = (10,8)


# In[6]:


# OPTION A: import STalign after pip or pipenv install
from STalign import STalign


# In[4]:


## OPTION B: skip cell if installed STalign with pip or pipenv
import sys
sys.path.append("../../STalign")

## import STalign from upper directory
import STalign


# We have already downloaded single cell spatial transcriptomics data from and placed the files in a folder called `xenium_data`: https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast
# 
# We can read in the cell information for the first dataset using `pandas` as `pd`.

# In[7]:


# Single cell data 1
# read in data
fname = '../xenium_data/Xenium_FFPE_Human_Breast_Cancer_Rep1_cells.csv.gz'
df1 = pd.read_csv(fname)
print(df1.head())


# For alignment with `STalign`, we only need the cell centroid information. So we can pull out this information. We can further visualize the cell centroids to get a sense of the variation in cell density that we will be relying on for our alignment by plotting using `matplotlib.pyplot` as `plt`. 

# In[8]:


# get cell centroid coordinates
xI = np.array(df1['x_centroid'])
yI = np.array(df1['y_centroid'])

# plot
fig,ax = plt.subplots()
ax.scatter(xI,yI,s=1,alpha=0.2)


# We will first use `STalign` to rasterize the single cell centroid positions into an image. Assuming the single-cell centroid coordinates are in microns, we will perform this rasterization at a 30 micron resolution. We can visualize the resulting rasterized image.
# 
# Note that points are plotting with the origin at bottom left while images are typically plotted with origin at top left so we've used `invert_yaxis()` to invert the yaxis for visualization consistency. 

# In[9]:


# rasterize at 30um resolution (assuming positions are in um units) and plot
XI,YI,I,fig = STalign.rasterize(xI,yI,dx=30)

# plot
ax = fig.axes[0]
ax.invert_yaxis()


# Now, we can repeat this for the cell information from the second dataset.

# In[10]:


# Single cell data 2
# read in data
fname = '../xenium_data/Xenium_FFPE_Human_Breast_Cancer_Rep2_cells.csv.gz'
df2 = pd.read_csv(fname)

# get cell centroids
xJ = np.array(df2['x_centroid'])
yJ = np.array(df2['y_centroid'])

# plot
fig,ax = plt.subplots()
ax.scatter(xJ,yJ,s=1,alpha=0.2,c='#ff7f0e')

# rasterize and plot
XJ,YJ,J,fig = STalign.rasterize(xJ,yJ,dx=30)
ax = fig.axes[0]
ax.invert_yaxis()


# Note that plotting the cell centroid positions from both datasets shows that alignment is still needed.

# In[11]:


# plot
fig,ax = plt.subplots()
ax.scatter(xI,yI,s=1,alpha=0.2)
ax.scatter(xJ,yJ,s=1,alpha=0.2)


# `STalign` relies on an interative gradient descent to align these two images. This can be somewhat slow. So we can manually designate a few landmark points to help initialize the alignment. A `curve_annotator.py` script is provided to assist with this. In order to use the `curve_annotator.py` script, we will need to write out our images as `.npz` files. 

# In[56]:


# Optional: write out npz files for landmark point picker
np.savez('../xenium_data/Xenium_Breast_Cancer_Rep1', x=XI,y=YI,I=I)
np.savez('../xenium_data/Xenium_Breast_Cancer_Rep2', x=XJ,y=YJ,I=J)
# outputs Xenium_Breast_Cancer_Rep1.npz and Xenium_Breast_Cancer_Rep2.npz


# Given these `.npz` files, we can then run the following code:
#     
# ```
# python curve_annotator.py Xenium_Breast_Cancer_Rep1.npz
# python curve_annotator.py Xenium_Breast_Cancer_Rep2.npz
# ```
# 
# Which will provide a graphical user interface to selecting landmark points, which will then be saved in `Xenium_Breast_Cancer_Rep1_points.npy` and `Xenium_Breast_Cancer_Rep2_points.npy` respectively. We can then read in these files. 

# In[12]:


# read from file
pointsIlist = np.load('../xenium_data/Xenium_Breast_Cancer_Rep1_points.npy', allow_pickle=True).tolist()
print(pointsIlist)
pointsJlist = np.load('../xenium_data/Xenium_Breast_Cancer_Rep2_points.npy', allow_pickle=True).tolist()
print(pointsJlist)


# Note that these landmark points are read in as lists. We will want to convert them to a simple array for downstream usage. 

# In[13]:


# convert to array
pointsI = []
pointsJ = []

# Jean's note: a bit odd to me that the points are stored as y,x
## instead of x,y but all downstream code uses this orientation
for i in pointsIlist.keys():
    pointsI.append([pointsIlist[i][0][1], pointsIlist[i][0][0]])
for i in pointsJlist.keys():
    pointsJ.append([pointsJlist[i][0][1], pointsJlist[i][0][0]])

pointsI = np.array(pointsI)
pointsJ = np.array(pointsJ)


# In[14]:


# now arrays
print(pointsI)
print(pointsJ)


# Alternatively, you can also just manually create an array of points.
# 
# But it will be good to double check that your landmark points look sensible by plotting them along with the rasterized image we created.

# In[15]:


# get extent of images
extentI = STalign.extent_from_x((YI,XI))
extentJ = STalign.extent_from_x((YJ,XJ))

# plot rasterized images
fig,ax = plt.subplots(2,1)
ax[0].imshow((I.transpose(1,2,0).squeeze()), extent=extentI) # just want 201x276 matrix
ax[1].imshow((J.transpose(1,2,0).squeeze()), extent=extentJ) # just want 201x276 matrix
# with points
ax[0].scatter(pointsI[:,1], pointsI[:,0], c='red')
ax[1].scatter(pointsJ[:,1], pointsJ[:,0], c='red')
for i in range(pointsI.shape[0]):
    ax[0].text(pointsI[i,1],pointsI[i,0],f'{i}', c='red')
    ax[1].text(pointsJ[i,1],pointsJ[i,0],f'{i}', c='red')
ax[0].invert_yaxis()
ax[1].invert_yaxis()


# We can now initialize a simple affine alignment from the landmark points. 

# In[16]:


# set device for building tensors
if torch.cuda.is_available():
    torch.set_default_device('cuda:0')
else:
    torch.set_default_device('cpu')


# In[17]:


# compute initial affine transformation from points
L,T = STalign.L_T_from_points(pointsI, pointsJ)
A = STalign.to_A(torch.tensor(L),torch.tensor(T))


# We can show the results of the simple affine transformation.

# In[18]:


# compute initial affine transformation from points
AI = STalign.transform_image_source_with_A(A, [YI,XI], I, [YJ,XJ])

#switch tensor from cuda to cpu for plotting with numpy
if AI.is_cuda:
    AI = AI.cpu()

fig,ax = plt.subplots(1,2)
ax[0].imshow((AI.permute(1,2,0).squeeze()), extent=extentJ)
ax[1].imshow((J.transpose(1,2,0).squeeze()), extent=extentJ)

ax[0].set_title('source with affine transformation', fontsize=15)
ax[1].set_title('target', fontsize=15)

ax[0].invert_yaxis()
ax[1].invert_yaxis()


# Depending on distortions in the tissue sample (as well as the accuracy of your landmark points), a simple affine alignment may not be sufficient to align the two single-cell spatial transcriptomics datasets. So we will need to perform non-linear local alignments via LDDMM. 
# 
# There are many parameters that can be tuned for performing this alignment. 

# In[21]:


get_ipython().run_cell_magic('time', '', "\n# run LDDMM\n# specify device (default device for STalign.LDDMM is cpu)\nif torch.cuda.is_available():\n    device = 'cuda:0'\nelse:\n    device = 'cpu'\n\n# keep all other parameters default\nparams = {'L':L,'T':T,\n          'niter':300,\n          'pointsI':pointsI,\n          'pointsJ':pointsJ,\n          'device':device,\n          'sigmaM':1.5,\n          'sigmaB':1.0,\n          'sigmaA':1.1,\n          'epV': 100\n          }\n\nout = STalign.LDDMM([YI,XI],I,[YJ,XJ],J,**params)\n")


# In[22]:


# get necessary output variables
A = out['A']
v = out['v']
xv = out['xv']


# Plots generated throughout the alignment can be used to give you a sense of whether the parameter choices are appropriate and whether your alignment is converging on a solution.
# 
# We can also evaluate the resulting alignment by applying the transformation to visualize how our source and target images were deformed to achieve the alignment. 

# In[23]:


# apply transform
phii = STalign.build_transform(xv,v,A,XJ=[YJ,XJ],direction='b')
phiI = STalign.transform_image_source_to_target(xv,v,A,[YI,XI],I,[YJ,XJ])
phipointsI = STalign.transform_points_source_to_target(xv,v,A,pointsI)

#switch tensor from cuda to cpu for plotting with numpy
if phii.is_cuda:
    phii = phii.cpu()
if phiI.is_cuda:
    phiI = phiI.cpu()
if phipointsI.is_cuda:
    phipointsI = phipointsI.cpu()

# plot with grids
fig,ax = plt.subplots()
levels = np.arange(-100000,100000,1000)
ax.contour(XJ,YJ,phii[...,0],colors='r',linestyles='-',levels=levels)
ax.contour(XJ,YJ,phii[...,1],colors='g',linestyles='-',levels=levels)
ax.set_aspect('equal')
ax.set_title('source to target')
ax.imshow(phiI.permute(1,2,0)/torch.max(phiI),extent=extentJ)
ax.scatter(phipointsI[:,1].detach(),phipointsI[:,0].detach(),c="m")
ax.invert_yaxis()


# Note that because of our use of LDDMM, the resulting transformation is invertible.

# In[24]:


# transform is invertible
phi = STalign.build_transform(xv,v,A,XJ=[YI,XI],direction='f')
phiiJ = STalign.transform_image_target_to_source(xv,v,A,[YJ,XJ],J,[YI,XI])
phiipointsJ = STalign.transform_points_target_to_source(xv,v,A,pointsJ)

#switch tensor from cuda to cpu for plotting with numpy
if phi.is_cuda:
    phi = phi.cpu()
if phiiJ.is_cuda:
    phiiJ = phiiJ.cpu()
if phiipointsJ.is_cuda:
    phiipointsJ = phiipointsJ.cpu()

# plot with grids
fig,ax = plt.subplots()
levels = np.arange(-100000,100000,1000)
ax.contour(XI,YI,phi[...,0],colors='r',linestyles='-',levels=levels)
ax.contour(XI,YI,phi[...,1],colors='g',linestyles='-',levels=levels)
ax.set_aspect('equal')
ax.set_title('target to source')
ax.imshow(phiiJ.permute(1,2,0)/torch.max(phiiJ),extent=extentI)
ax.scatter(phiipointsJ[:,1].detach(),phiipointsJ[:,0].detach(),c="m")
ax.invert_yaxis()


# Finally, we can apply our transform to the original sets of single cell centroid positions to achieve their new aligned positions.

# In[25]:


# apply transform to original points of target to source
tpointsJ = STalign.transform_points_target_to_source(xv,v,A, np.stack([yJ, xJ], 1))

#switch tensor from cuda to cpu for plotting with numpy
if tpointsJ.is_cuda:
    tpointsJ = tpointsJ.cpu()


# And we can visualize the results.

# In[26]:


# plot results
fig,ax = plt.subplots()
ax.scatter(xI,yI,s=1,alpha=0.2) 
ax.scatter(tpointsJ[:,1],tpointsJ[:,0],s=1,alpha=0.1) # also needs to plot as y,x not x,y


# And save the new aligned positions by appending to our original data using `numpy` with `np.hstack`

# In[27]:


# save results by appending
# note results are in y,x coordinates
results = np.hstack((df2, tpointsJ.numpy()))


# We will finally create a compressed `.csv.gz` file to create `Xenium_Breast_Cancer_Rep2_STalign_to_Rep1.csv.gz`

# In[ ]:


results.to_csv('../xenium_data/Xenium_Breast_Cancer_Rep2_STalign_to_Rep1.csv.gz',
               compression='gzip')


# Note that the learned transform can be applied from source to target or target to source.

# In[28]:


# apply transform to original points of source to target
tpointsI = STalign.transform_points_source_to_target(xv,v,A, np.stack([yI, xI], 1))

#switch tensor from cuda to cpu for plotting with numpy
if tpointsI.is_cuda:
    tpointsI = tpointsI.cpu()

# plot results
fig,ax = plt.subplots() 
ax.scatter(tpointsI[:,1],tpointsI[:,0],s=1,alpha=0.2) # also needs to plot as y,x not x,y
ax.scatter(xJ,yJ,s=1,alpha=0.2)


# We can also use the matching weights to focus on the tissue region that is overlapping given the partially matching nature of this alignment.

# In[29]:


# get the weights
WM = out['WM']
# compute weight values for transformed source points from target image pixel locations and weight 2D array (matching)
testM = STalign.interp([YI,XI],WM[None].float(),tpointsI[None].permute(-1,0,1).float())


# In[30]:


# note some cells were allocated into the artifact component (not matching and so not included, may need to tune sigmaA)
fig,ax = plt.subplots()
ax.scatter(xJ,yJ,s=1,alpha=0.2)
ax.scatter(tpointsI[:,1],tpointsI[:,0],c=testM[0,0],s=0.1,vmin=0,vmax=1, label='WM values')


# In[31]:


# visualize the distribution to identify reasonable cutoff
fig,ax = plt.subplots()
ax.hist(testM[0,0], bins = 20)


# In[37]:


# set a threshold value
WMthresh = 0.9

# plot just the cells that pass filter (overlapping cells)
filtered = testM[0,0] > WMthresh
tpointsI_filtered = tpointsI[filtered,]

fig,ax = plt.subplots()
ax.scatter(tpointsI[:,1],tpointsI[:,0],s=0.1,alpha=0.5,vmin=0,vmax=1, label='original')
ax.scatter(xJ,yJ,s=1,alpha=0.2)
ax.scatter(tpointsI_filtered[:,1],tpointsI_filtered[:,0],s=0.1,alpha=0.5,vmin=0,vmax=1, label='filtered')


# In[38]:


# choose less stringent threshold value
WMthresh = 0.5

# plot just the cells that pass filter (overlapping cells)
filtered = testM[0,0] > WMthresh
tpointsI_filtered = tpointsI[filtered,]

fig,ax = plt.subplots()
ax.scatter(tpointsI[:,1],tpointsI[:,0],s=0.1,alpha=0.5,vmin=0,vmax=1, label='original')
ax.scatter(xJ,yJ,s=1,alpha=0.2)
ax.scatter(tpointsI_filtered[:,1],tpointsI_filtered[:,0],s=0.1,alpha=0.5,vmin=0,vmax=1, label='filtered')


# In this manner, we can restrict to just the cells that are matching for downstream analysis. 

# In[ ]:





# In[ ]:




