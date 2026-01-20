#!/usr/bin/env python
# coding: utf-8

# # Aligning single-cell resolution spatial transcriptomics data to  H&E staining image from Visium

# In this notebook, we take a single cell resolution spatial transcriptomics datasets of a coronal section of the adult mouse brain profiled by the MERFISH technology and align it to a H&E staining image from a different individual at matched locations with respect to bregma. 
# 
# We will use `STalign` to achieve this alignment. We will first load the relevant code libraries.

# In[28]:


## import dependencies
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd
import torch
import plotly
import requests

# make plots bigger
plt.rcParams["figure.figsize"] = (12,10)


# In[2]:


## skip cell if STalign.py in same folder as notebook
import sys
sys.path.append("../../STalign") 


# In[3]:


## import STalign from upper directory
import STalign


# We can read in the single cell information using `pandas` as `pd`.

# In[4]:


# Single cell data 1
# read in data
fname = '../merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate3_cell_metadata_S2R3.csv.gz'
df1 = pd.read_csv(fname)
print(df1.head())


# For alignment with `STalign`, we only need the cell centroid information. So we can pull out this information. We can further visualize the cell centroids to get a sense of the variation in cell density that we will be relying on for our alignment by plotting using `matplotlib.pyplot` as `plt`. 

# In[5]:


# get cell centroid coordinates
xI = np.array(df1['center_x'])
yI = np.array(df1['center_y'])

# plot
fig,ax = plt.subplots()
ax.scatter(xI,yI,s=1,alpha=0.2)
#ax.set_aspect('equal', 'box')


# We will use STalign to rasterize the single cell centroid positions into an image. Assuming the single-cell centroid coordinates are in microns, we will perform this rasterization at a 30 micron resolution. We can visualize the resulting rasterized image.
# 
# Note that points are plotting with the origin at bottom left while images are typically plotted with origin at top left so we've used `invert_yaxis()` to invert the yaxis for visualization consistency.

# In[6]:


# rasterize at 30um resolution (assuming positions are in um units) and plot
XI,YI,I,fig = STalign.rasterize(xI, yI, dx=30)

ax = fig.axes[0]
ax.invert_yaxis()


# Note that this is a 1D greyscale image. To align with an RGB H&E image, we will need to make our greyscale image into RGB by simply stacking the 1D values 3 times. We will also normalize to get intensity values between 0 to 1.

# In[7]:


print("The initial shape of I is {}".format(I.shape))
I = np.vstack((I, I, I)) # make into 3xNxM
print("The range of I is {} to {}".format(I.min(), I.max() ))

# normalize
I = STalign.normalize(I)
print("The range of I after normalization is {} to {}".format(I.min(), I.max() ))

# double check size of things
print("The new shape of I is {}".format(I.shape))


# We have already downloaded the H&E staining image from https://www.10xgenomics.com/resources/datasets/adult-mouse-brain-ffpe-1-standard-1-3-0 and placed the file in a folder called `visium_data`
# 
# We can read in the H&E staining image using `matplotlib.pyplot` as `plt`. 

# In[8]:


image_file = '../visium_data/tissue_hires_image.png'
V = plt.imread(image_file)

# plot
fig,ax = plt.subplots()
ax.imshow(V)


# Note that this is an RGB image that `matplotlib.pyplot` had read in as an NxMx3 matrix with values ranging from 0 to 1. We will use `STalign` to normalize the image in case there are any outlier intensities. 

# In[9]:


print("The initial shape of V is {}".format(V.shape))
print("The range of V is {} to {}".format(V.min(), V.max() ))

Vnorm = STalign.normalize(V)
print("The range of V after normalization is {} to {}".format(Vnorm.min(), Vnorm.max() ))


# We will transpose `Vnorm` to be a 3xNxM matrix `J` for downstream analyses. We will also create some variances `YJ` and `XJ` to keep track of the image size. 

# In[10]:


J = Vnorm.transpose(2,0,1)
print("The new shape of J is {}".format(J.shape))

YJ = np.array(range(J.shape[1]))*1. # needs to be longs not doubles for STalign.transform later so multiply by 1.
XJ = np.array(range(J.shape[2]))*1. # needs to be longs not doubles for STalign.transform later so multiply by 1.


# We now have a rasterized image corresponding to the single cell positions from the spatial transcriptomics data and an H&E image that we can align. Note, that we have specified the image from cell positions as source `I` and the H&E image as target `J` because the H&E image is only one hemisphere of the brain. We advise choosing the more complete tissue section as the source such that every observation in the target has some correspondence in the source.

# In[11]:


# plot
extentI = STalign.extent_from_x((YI,XI))

fig,ax = plt.subplots(1,2)
ax[0].imshow((I.transpose(1,2,0).squeeze()), extent=extentI) 
ax[0].invert_yaxis()
ax[0].set_title('source', fontsize=15)
ax[1].imshow((J).transpose(1,2,0))
ax[1].set_title('target', fontsize=15)


# `STalign` relies on an interative gradient descent to align these two images. This can be somewhat slow. So we can manually designate a few landmark points to help initialize the alignment. A `curve_annotator.py` script is provided to assist with this. In order to use the `curve_annotator.py` script, we will need to write out our images as `.npz` files. 

# In[ ]:


np.savez('../visium_data/Merfish_S2_R3', x=XI,y=YI,I=I)
np.savez('../visium_data/tissue_hires_image', x=XJ,y=YJ,I=J)


# Given these `.npz` files, we can then run the following code on the command line from inside the `notebooks` folder:
# 
# ```
# python ../../STalign/curve_annotator.py ../visium_data/Merfish_S2_R3.npz
# python ../../STalign/curve_annotator.py ../visium_data/tissue_hires_image.npz
# ```
# 
# Which will provide a graphical user interface to selecting curves. Using the flag `-o` , these curves will saved as `Merfish_S2_R3_curves.npy` and `tissue_hires_image_curves.npy` respectively. We can then read in these files. 

# In[16]:


# read from file
pointsIlist = np.load('../visium_data/Merfish_S2_R3_curves.npy', allow_pickle=True).tolist()
print(pointsIlist)
pointsJlist = np.load('../visium_data/tissue_hires_image_curves.npy', allow_pickle=True).tolist()
print(pointsJlist)


# Note that these landmark points are read in as lists. We will want to convert them to a simple array for downstream usage. 

# In[23]:


# convert to array
pointsI = []
pointsJ = []

# Jean's note: a bit odd to me that the points are stored as y,x
## instead of x,y but all downstream code uses this orientation
for i in pointsIlist.keys():
    for j in range(len(pointsIlist[i])):
        pointsI.append([pointsIlist[i][j][1], pointsIlist[i][j][0]])
for i in pointsJlist.keys():
    for j in range(len(pointsJlist[i])):
        pointsJ.append([pointsJlist[i][j][1], pointsJlist[i][j][0]])

pointsI = np.array(pointsI)
pointsJ = np.array(pointsJ)


# In[24]:


# now arrays
print(pointsI)
print(pointsJ)


# We can double check that our landmark points look sensible by plotting them along with the rasterized image we created.

# In[33]:


# manually make corresponding points
#data = np.load('../visium_data/visium2_points.npz')
#pointsI = np.array(data['pointsI'][...,::-1])
#pointsJ = np.array(data['pointsJ'][...,::-1])


# In[31]:


# plot
extentJ = STalign.extent_from_x((YJ,XJ))
extentI = STalign.extent_from_x((YI,XI))

fig,ax = plt.subplots(1,2)
ax[0].imshow((I.transpose(1,2,0).squeeze()), extent=extentI) 
ax[1].imshow((J.transpose(1,2,0).squeeze()), extent=extentJ)

trans_offset_0 = mtransforms.offset_copy(ax[0].transData, fig=fig,
                                       x=0.05, y=-0.05, units='inches')
trans_offset_1 = mtransforms.offset_copy(ax[1].transData, fig=fig,
                                       x=0.05, y=-0.05, units='inches')

ax[0].scatter(pointsI[:,1],pointsI[:,0], c='red', s=10)
ax[1].scatter(pointsJ[:,1],pointsJ[:,0], c='red', s=10)
for i in range(pointsI.shape[0]):
    ax[0].text(pointsI[i,1],pointsI[i,0],f'{i}', c='red', transform=trans_offset_0)
for i in range(pointsJ.shape[0]):
    ax[1].text(pointsJ[i,1],pointsJ[i,0],f'{i}', c='red', transform=trans_offset_1)

ax[0].set_title('source with pointsI', fontsize=15)
ax[1].set_title('target with pointsJ', fontsize=15)

# invert only rasterized image
ax[0].invert_yaxis()


# From the landmark points, we can generate a linear transformation `L` and translation `T` which will produce a simple initial affine transformation `A`. 

# In[26]:


# compute initial affine transformation from points
L,T = STalign.L_T_from_points(pointsI,pointsJ)
A = STalign.to_A(torch.tensor(L),torch.tensor(T))


# We can show the results of the simple affine transformation.

# In[14]:


# compute initial affine transformation from points
AI= STalign.transform_image_source_with_A(A, [YI,XI], I, [YJ,XJ])

# get extent of images
extentI = STalign.extent_from_x((YI,XI))
extentJ = STalign.extent_from_x((YJ,XJ))

fig,ax = plt.subplots(1,2)
#ax[0].imshow(AI[0], extent=extentI)
ax[0].imshow((AI.permute(1,2,0).squeeze()), extent=extentI) 
ax[1].imshow((J.transpose(1,2,0).squeeze()), extent=extentJ) 

ax[0].set_title('source with affine transformation', fontsize=15)
ax[1].set_title('target', fontsize=15)


# In this case, we can observe that a simple affine alignment is not sufficient to align the single-cell spatial transcriptomics dataset to the H&E staining image. So we will need to perform non-linear local alignments via LDDMM. 
# 
# There are many parameters that can be tuned for performing this alignment. 

# In[ ]:


fig,ax = plt.subplots()
ax.hist(I.ravel())
plt.xlabel('Intensity')
plt.ylabel('Number of Pixels')
plt.title('Intensity Histogram of Source Image')


# In[ ]:


fig,ax = plt.subplots()
ax.hist(J.ravel())
plt.xlabel('Intensity')
plt.ylabel('Number of Pixels')
plt.title('Intensity Histogram of Target Image')


# In[ ]:


WM = torch.ones(J[0].shape)*0.5
WB = torch.ones(J[0].shape)*0.4
WA = torch.ones(J[0].shape)*0.1
muA = torch.sum(WA*J,dim=(-1,-2))/torch.sum(WA)
muB = torch.sum(WB*J,dim=(-1,-2))/torch.sum(WB)


# In[ ]:


print(muA)
print(muB)


# In[ ]:


WM = torch.ones(I[0].shape)*0.5
WB = torch.ones(I[0].shape)*0.4
WA = torch.ones(I[0].shape)*0.1
muA = torch.sum(WA*I,dim=(-1,-2))/torch.sum(WA)
muB = torch.sum(WB*I,dim=(-1,-2))/torch.sum(WB)


# In[ ]:


print(muA)
print(muB)


# In[15]:


get_ipython().run_cell_magic('time', '', "\n# run LDDMM\n# specify device (default device for STalign.LDDMM is cpu)\nif torch.cuda.is_available():\n    device = 'cuda:0'\nelse:\n    device = 'cpu'\n\n# keep all other parameters default\nparams = {'L':L,'T':T,\n          'niter':10000,\n          'pointsI':pointsI,\n          'pointsJ':pointsJ,\n          'device':device\n          }\n\nout = STalign.LDDMM([YI,XI],I,[YJ,XJ],J,**params)\n")


# In[ ]:


# get necessary output variables
A = out['A']
v = out['v']
xv = out['xv']


# Plots generated throughout the alignment can be used to give you a sense of whether the parameter choices are appropriate and whether your alignment is converging on a solution.
# 
# We can also evaluate the resulting alignment by applying the transformation to visualize how our source and target images were deformed to achieve the alignment. 

# In[58]:


# apply transform
phii = STalign.build_transform(xv,v,A,XJ=[YJ,XJ],direction='b')
phiI = STalign.transform_image_source_to_target(xv,v,A,[YI,XI],I,[YJ,XJ])
phiipointsI = STalign.transform_points_source_to_target(xv,v,A,pointsI)

# plot with grids
fig,ax = plt.subplots()
levels = np.arange(-100000,100000,1000)
ax.contour(XJ,YJ,phii[...,0],colors='r',linestyles='-',levels=levels)
ax.contour(XJ,YJ,phii[...,1],colors='g',linestyles='-',levels=levels)
ax.set_aspect('equal')
ax.set_title('source to target')
ax.imshow(phiI.permute(1,2,0)/torch.max(phiI),extent=extentJ)
ax.scatter(phiipointsI[:,1].detach(),phiipointsI[:,0].detach(),c="m")


# Finally, we can apply our transform to the original sets of single cell centroid positions to achieve their new aligned positions.

# In[61]:


# apply transform to original points
tpointsI= STalign.transform_points_source_to_target(xv,v,A, np.stack([yI, xI], 1))

#switch from row column coordinates (y,x) to (x,y)
xI_LDDMM = tpointsI[:,1]
yI_LDDMM = tpointsI[:,0]


# And we can visualize the results.

# In[74]:


# plot
#fig,ax = plt.subplots()
#ax.imshow((I).transpose(1,2,0),extent=extentI)
#ax.scatter(phiipointsJ[:,1].detach(),phiipointsJ[:,0].detach(),c="r")
#ax.scatter(tpointsI[:,1].detach(),tpointsI[:,0].detach(),s=1,alpha=0.1)

# plot results
fig,ax = plt.subplots()
#ax.scatter(xI,yI,s=1,alpha=0.1, label='source')
ax.scatter(xI_LDDMM,yI_LDDMM,s=1,alpha=0.1, label = 'source aligned')
ax.scatter(phiipointsI[:,1].detach(),phiipointsI[:,0].detach(),c="m", label='source landmarks aligned')
ax.scatter(pointsJ[:,1],pointsJ[:,0], c='red', label='target landmarks')
ax.imshow((J).transpose(1,2,0),extent=extentJ)

lgnd = plt.legend(loc="upper right", scatterpoints=1, fontsize=10)
for handle in lgnd.legend_handles:
    handle.set_sizes([10.0])


# And save the new aligned positions by appending to our original data

# In[75]:


df3 = pd.DataFrame(

    {

        "aligned_x": xI_LDDMM,

        "aligned_y": yI_LDDMM,

    },


)

results = pd.concat([df1, df3], axis=1)
results.head()


# We will finally create a compressed `.csv.gz` file named `mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate3_STalign_to_Slice2_Replicate2.csv.gz`

# In[76]:


results.to_csv('../merfish_data/mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate3_STalign_to_Visium_tissue_hires_image.csv.gz',
               compression='gzip')


# In[ ]:




