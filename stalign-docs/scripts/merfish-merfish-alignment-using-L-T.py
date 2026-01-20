#!/usr/bin/env python
# coding: utf-8

# # Aligning two coronal sections of adult mouse brain from MERFISH
# 
# In this notebook, we align two single cell resolution spatial transcriptomics datasets of coronal sections of the adult mouse brain from matched locations with respect to bregma from different individual assayed by MERFISH.
# 
# We will use `STalign` to achieve this alignment. We will first load the relevant code libraries.

# In[1]:


## import dependencies
import numpy as np
import matplotlib.pyplot as plt
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


# We have already downloaded single cell spatial transcriptomics datasets and placed the files in a folder called `merfish_data`.
# 
# We can read in the cell information for the first dataset using `pandas` as `pd`.

# In[4]:


# Single cell data 1
# read in data
fname = '../merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate3_cell_metadata_S2R3.csv.gz'
df1 = pd.read_csv(fname)
print(df1.head())


# For alignment with `STalign`, we only need the cell centroid information so we can pull out this information. We can further visualize the cell centroids to get a sense of the variation in cell density that we will be relying on for our alignment by plotting using `matplotlib.pyplot` as `plt`. 

# In[5]:


# get cell centroid coordinates
xI = np.array(df1['center_x'])
yI = np.array(df1['center_y'])

# plot
fig,ax = plt.subplots()
ax.scatter(xI,yI,s=1,alpha=0.2, label='source')
ax.legend(markerscale = 10)


# Now, we can repeat this to get cell information from the second dataset.

# In[6]:


# Single cell data 2
# read in data
fname = '../merfish_data/datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate2_cell_metadata_S2R2.csv.gz'

df2 = pd.read_csv(fname)

# get cell centroids
xJ = np.array(df2['center_x'])
yJ = np.array(df2['center_y'])

# plot
fig,ax = plt.subplots()
ax.scatter(xJ,yJ,s=1,alpha=0.2,c='#ff7f0e', label='target')
ax.legend(markerscale = 10)


# Note that plotting the cell centroid positions from both datasets shows that non-linear local alignment is needed.

# In[7]:


# plot
fig,ax = plt.subplots()
ax.scatter(xI,yI,s=1,alpha=0.2, label='source')
ax.scatter(xJ,yJ,s=1,alpha=0.1, label= 'target')
ax.legend(markerscale = 10)


# `STalign` relies on an interative gradient descent to align these two images. This performs quicker and better if the source and target are initially at a similar angle.
# 
# Evaluate the similarity of the rotation angle by viewing a side by side comparison. Change the value of `theta_deg`, until the rotation angle is similar. Note: the rotation here is defined in degrees and is in the clockwise direction.
# 
# The angle chosen will be used to construct a 2x2 rotation matrix `L` and a 2 element translation vector `T`.

# In[8]:


theta_deg = 45
theta0 = (np.pi/180)*-theta_deg

#rotation matrix 
#rotates about the origin
L = np.array([[np.cos(theta0),-np.sin(theta0)],
              [np.sin(theta0),np.cos(theta0)]])

source_L = np.matmul(L , np.array([xI, yI]))
xI_L = source_L[0]
yI_L = source_L[1]

#translation matrix
#effectively makes the rotation about the centroid of I (i.e the means of xI and yI]) 
#and also moves the centroid of I to the centroid of J
T = np.array([ np.mean(xI)- np.cos(theta0)*np.mean(xI) +np.sin(theta0)*np.mean(yI) - (np.mean(xI)-np.mean(xJ)),
              np.mean(yI)- np.sin(theta0)*np.mean(xI) -np.cos(theta0)*np.mean(yI) - (np.mean(yI)-np.mean(yJ))])

xI_L_T = xI_L + T[0]
yI_L_T = yI_L + T[1]


fig,ax = plt.subplots()
ax.scatter(xI_L_T,yI_L_T,s=1,alpha=0.1, label='source with initial affine transformation')
ax.scatter(xJ,yJ,s=1,alpha=0.1, label = 'target')
ax.legend(markerscale = 10)


# Now, we will first use `STalign` to rasterize the single cell centroid positions into an image. Assuming the single-cell centroid coordinates are in microns, we will perform this rasterization at a 30 micron resolution. We can visualize the resulting rasterized image.
# 
# Note that points are plotting with the origin at bottom left while images are typically plotted with origin at top left so we've used `invert_yaxis()` to invert the yaxis for visualization consistency. 

# In[9]:


# rasterize at 30um resolution (assuming positions are in um units) and plot
XI,YI,I,fig = STalign.rasterize(xI_L_T,yI_L_T,dx=15,blur=1.5)

# plot
ax = fig.axes[0]
ax.invert_yaxis()


# Repeat rasterization for target dataset.

# In[12]:


# rasterize and plot
XJ,YJ,J,fig = STalign.rasterize(xJ,yJ,dx=15, blur=1.5)
ax = fig.axes[0]
ax.invert_yaxis()


# We can also plot the rasterized images next to each other.

# In[13]:


# get extent of images
extentI = STalign.extent_from_x((YI,XI))
extentJ = STalign.extent_from_x((YJ,XJ))

# plot rasterized images
fig,ax = plt.subplots(1,2)
ax[0].imshow(I[0], extent=extentI) 
ax[1].imshow(J[0], extent=extentJ)
ax[0].invert_yaxis()
ax[1].invert_yaxis()


# Now we will perform our alignment. There are many parameters that can be tuned for performing this alignment. If we don't specify parameters, defaults will be used. 

# In[11]:


get_ipython().run_cell_magic('time', '', "\n# run LDDMM\n# specify device (default device for STalign.LDDMM is cpu)\nif torch.cuda.is_available():\n    device = 'cuda:0'\nelse:\n    device = 'cpu'\n\n# keep all other parameters default\nparams = {\n            'niter': 10000,\n            'device':device,\n            'epV': 50\n          }\n\nIfoo = np.vstack((I, I, I)) # make RGB instead of greyscale\nJfoo = np.vstack((J, J, J)) # make RGB instead of greyscale\nout = STalign.LDDMM([YI,XI],Ifoo,[YJ,XJ],Jfoo,**params)\n")


# In[ ]:


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

# plot with grids
fig,ax = plt.subplots()
levels = np.arange(-100000,100000,1000)
ax.contour(XI,YI,phi[...,0],colors='r',linestyles='-',levels=levels)
ax.contour(XI,YI,phi[...,1],colors='g',linestyles='-',levels=levels)
ax.set_aspect('equal')
ax.set_title('target to source')
ax.imshow(phiiJ.permute(1,2,0)/torch.max(phiiJ),extent=extentI)
ax.invert_yaxis()


# Finally, we can apply our STalign transform to the original sets of single cell centroid positions (with initial affine transformation) to achieve their new aligned positions.

# In[19]:


# apply transform to original points
#tpointsI= STalign.transform_points_source_to_target(xv,v,A, np.stack([yI_L_T, xI_L_T], 1))
tpointsI= STalign.transform_points_source_to_target(xv,v,A, np.stack([yI_L_T, xI_L_T], 1))
#tpointsI= STalign.transform_points_target_to_source(xv,v,A, np.stack([yI_L_T, xI_L_T], 1))

#switch from row column coordinates (y,x) to (x,y)
xI_LDDMM = tpointsI[:,1]
yI_LDDMM = tpointsI[:,0]


# And we can visualize the results.

# In[20]:


# plot results
fig,ax = plt.subplots()
ax.scatter(xI,yI,s=1,alpha=0.1, label='source')
ax.scatter(xI_LDDMM,yI_LDDMM,s=1,alpha=0.1, label = 'source aligned') # also needs to plot as y,x not x,y
ax.scatter(xJ,yJ,s=1,alpha=0.1, label='target')
ax.legend(markerscale = 10)


# In[23]:


fig,ax = plt.subplots(1,2)
#ax[0].scatter(xI,yI,s=1,alpha=0.1, label='source')
ax[0].scatter(xI_L_T,yI_L_T,s=1,alpha=0.1, label='source with initial affine transformation')
ax[0].scatter(xJ,yJ,s=1,alpha=0.1, label='target')
#ax[1].scatter(xI,yI,s=1,alpha=0.1, label='source')
ax[1].scatter(xI_LDDMM,yI_LDDMM,s=1,alpha=0.1, label = 'source STaligned') # also needs to plot as y,x not x,y
ax[1].scatter(xJ,yJ,s=1,alpha=0.1, label='target')
ax[0].legend(markerscale = 10, loc = 'lower left')
ax[1].legend(markerscale = 10, loc = 'lower left')


# And save the new aligned positions by appending to our original data

# In[24]:


df3 = pd.DataFrame(

    {

        "aligned_x": xI_LDDMM,

        "aligned_y": yI_LDDMM,

    },


)

results = pd.concat([df1, df3], axis=1)
results.head()


# We will finally create a compressed `.csv.gz` file named `mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate3_STalign_to_Slice2_Replicate2.csv.gz`

# In[25]:


results.to_csv('../merfish_data/mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate3_STalign_to_Slice2_Replicate2.csv.gz',
               compression='gzip')


# # TO DO: remove section below before uploading

# In[14]:


import importlib
importlib.reload(STalign)


# Alternatively, we can use the determined angle to provide an initial guess for rotation matrix `L` and translation vector `T` to construct the affine transformation component of the mapping.

# In[8]:


# rasterize at 30um resolution (assuming positions are in um units) and plot
XI,YI,I,fig = STalign.rasterize(xI,yI,dx=15,blur=1.5)

# plot
ax = fig.axes[0]
ax.invert_yaxis()


# In[9]:


# rasterize and plot
XJ,YJ,J,fig = STalign.rasterize(xJ,yJ,dx=15, blur=1.5)
ax = fig.axes[0]
ax.invert_yaxis()


# In[11]:


theta_deg = 45
#theta_image_ori = theta_deg+270
theta0 = (np.pi/180)*-(theta_deg)
#theta0 = (np.pi/180)*-(theta_image_ori)

#rotation matrix 
#rotates about the origin
#L = np.array([[np.cos(theta0),-np.sin(theta0)],
#              [np.sin(theta0),np.cos(theta0)]])

L = np.array([[np.sin(theta0),np.cos(theta0)],
              [np.cos(theta0),-np.sin(theta0)]])

#source_L = np.matmul(L , np.array([XI, YI]))
#XI_L = source_L[0]
#YI_L = source_L[1]

#translation matrix
#effectively makes the rotation about the centroid of I (i.e the means of xI and yI]) 
#and also moves the centroid of I to the centroid of J

T = np.array([ np.mean(XI)- np.cos(theta0)*np.mean(XI) +np.sin(theta0)*np.mean(YI) - (np.mean(XI)-np.mean(XJ)),
              np.mean(YI)- np.sin(theta0)*np.mean(YI) -np.cos(theta0)*np.mean(YI) - (np.mean(YI)-np.mean(YJ))])

#T = np.array([ np.mean(YI)- np.cos(theta0)*np.mean(YI) +np.sin(theta0)*np.mean(XI) - (np.mean(YI)-np.mean(YJ)),
#              np.mean(XI)- np.sin(theta0)*np.mean(XI) -np.cos(theta0)*np.mean(XI) - (np.mean(XI)-np.mean(XJ))])

T = np.array([ np.mean(YI)- np.sin(theta0)*np.mean(YI) -np.cos(theta0)*np.mean(XI) - (np.mean(YI)-np.mean(YJ)),
             np.mean(XI)- np.cos(theta0)*np.mean(XI) +np.sin(theta0)*np.mean(XI) - (np.mean(XI)-np.mean(XJ))])


#XI_L_T = XI_L + T[0]
#YI_L_T = YI_L + T[1]


# In[83]:


#T = np.array([ np.mean(XI)- np.cos(theta0)*np.mean(XI) +np.sin(theta0)*np.mean(YI) + 1000 ,
    #np.mean(YI)- np.sin(theta0)*np.mean(XI) -np.cos(theta0)*np.mean(YI) + 1500])


# In[15]:


A = STalign.to_A(torch.tensor(L),torch.tensor(T))
AI= STalign.transform_image_source_with_A(A, [YI,XI], I, [YJ,XJ])


# In[13]:


A = STalign.to_A(torch.tensor(L),torch.tensor(T))

def transform_image_source_with_A(A, XI, I, XJ):
    '''
    Transform an image with an affine matrix

    Parameters

    ----------

    A  : torch tensor
         Affine transform matrix

    XI : list of numpy arrays
         List of arrays storing the pixel location in image I along each image axis. 
         convention is row column order not xy. i.e, 
         locations of pixels along the y axis (rows) followed by
         locations of pixels along the x axis (columns)  

    I  : numpy array
         A rasterized image with len(blur) channels along the first axis

    XJ : list of numpy arrays
         List of arrays storing the pixel location in image I along each image axis. 
         convention is row column order not xy. i.e, 
         locations of pixels along the y axis (rows) followed by
         locations of pixels along the x axis (columns)         

    Returns
    -------
    AI : torch tensor
        image I after affine transformation A, with channels along first axis

    '''
    xv = None
    v = None
    AI= STalign.transform_image_source_to_target(xv, v, A, XI, I, XJ=XJ)
    return AI

#xv = None
#v = None
##XJ_new = torch.stack(torch.meshgrid(torch.tensor(YJ),torch.tensor(XJ),indexing='ij'),-1)
#AI= STalign.transform_image_source_to_target(xv, v, A, [YI,XI], I, XJ=[YJ,XJ])


AI= transform_image_source_with_A(A, [YI,XI], I, [YJ,XJ])


# In[30]:


type(AI)
AI.size()
AI[0].size()


# In[16]:


# get extent of images
extentI = STalign.extent_from_x((YI,XI))
extentJ = STalign.extent_from_x((YJ,XJ))

fig,ax = plt.subplots(1,2)
ax[0].imshow(AI[0], extent=extentJ)
ax[1].imshow(J[0], extent=extentJ)
ax[0].invert_yaxis()
ax[1].invert_yaxis()


# Now we will perform our alignment. There are many parameters that can be tuned for performing this alignment. If we don't specify parameters, defaults will be used. 

# In[30]:


get_ipython().run_cell_magic('time', '', "# run LDDMM\n# running this on my desktop so cpu device\ndevice = 'cpu' \n\n# keep all other parameters default\nparams = {\n            'niter': 4000,\n            'device':device,\n            'epV': 50,\n            'L': L,\n            'T': T\n          }\n\nIfoo = np.vstack((I, I, I)) # make RGB instead of greyscale\nJfoo = np.vstack((J, J, J)) # make RGB instead of greyscale\nA,v,xv = STalign.LDDMM([YI,XI],Ifoo,[YJ,XJ],Jfoo,**params)\n\n")


# In[52]:


A = STalign.to_A(torch.tensor(L),torch.tensor(T))
xv = None
v = None
#XJ_new = torch.stack(torch.meshgrid(torch.tensor(YJ),torch.tensor(XJ),indexing='ij'),-1)
AI= STalign.transform_image_source_to_target(xv, v, A, [YI,XI], I, XJ=[YJ,XJ])


# In[58]:


# get extent of images
extentI = STalign.extent_from_x((YI,XI))
extentJ = STalign.extent_from_x((YJ,XJ))

fig,ax = plt.subplots(1,3)
ax[0].imshow(I[0], extent=extentI)
ax[1].imshow(AI[0], extent=extentJ)
ax[2].imshow((J[0]), extent=extentJ)
#ax[1].imshow((J.transpose(1,2,0).squeeze()), extent=extentJ) # just want 201x276 matrix
ax[0].invert_yaxis()
ax[1].invert_yaxis()
ax[2].invert_yaxis()


# In[43]:


print(A)


# In[54]:


# get extent of images
extentI = STalign.extent_from_x((YI,XI))
extentJ = STalign.extent_from_x((YJ,XJ))

fig,ax = plt.subplots()
ax.imshow(I[0], extent=extentI)


# In[ ]:




