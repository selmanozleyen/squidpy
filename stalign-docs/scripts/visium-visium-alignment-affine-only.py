#!/usr/bin/env python
# coding: utf-8
"""
Aligning two visium datasets using STalign (original implementation)

This script aligns two spot resolution spatial transcriptomics datasets
of serial sections of breast cancer.
"""

from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = DATA_DIR / "output_stalign"
OUTPUT_DIR.mkdir(exist_ok=True)

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')

## import dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

# make plots bigger
plt.rcParams["figure.figsize"] = (12, 10)

# import STalign
from STalign import STalign

print("Loading Visium datasets...")

# Single cell data 1
fname = DATA_DIR / 'visium_data/slice1_coor.csv'
df1 = pd.read_csv(fname)
print(df1.head())


# For alignment with `STalign`, we only need the cell centroid information so we can pull out this information. We can further visualize the cell centroids to get a sense of the variation in cell density that we will be relying on for our alignment by plotting using `matplotlib.pyplot` as `plt`.

# In[4]:


# get cell centroid coordinates
xI = np.array(df1['1.306400000000000006e+01'])
yI = np.array(df1['6.086000000000000298e+00'])

# plot
fig,ax = plt.subplots()
ax.scatter(xI,yI,s=20,alpha=0.2, label='source')
ax.legend(markerscale = 1)
ax.set_aspect('equal')


# Now, we can repeat this to get cell information from the second dataset.

# In[5]:


# Single cell data 2
fname = DATA_DIR / 'visium_data/slice2_coor.csv'
df2 = pd.read_csv(fname)

# get cell centroids
xJ = np.array(df2[df2.columns[0]])
yJ = np.array(df2[df2.columns[1]])

# plot
fig,ax = plt.subplots()
ax.scatter(xJ,yJ,s=20,alpha=0.2,c='#ff7f0e', label='target')
ax.legend(markerscale = 1)
ax.set_aspect('equal')


# Note that plotting the cell centroid positions from both datasets shows that non-linear local alignment is needed.

# In[6]:


# plot
fig,ax = plt.subplots()
ax.scatter(xI,yI,s=20,alpha=0.2, label='source')
ax.scatter(xJ,yJ,s=20,alpha=0.1, label= 'target')
ax.legend(markerscale = 1)
ax.set_aspect('equal')


# `STalign` relies on an interative gradient descent to align these two images. This performs quicker and better if the source and target are initially at a similar angle.
#
# Evaluate the similarity of the rotation angle by viewing a side by side comparison. Change the value of `theta_deg`, until the rotation angle is similar. Note: the rotation here is defined in degrees and is in the clockwise direction.
#
# The angle chosen will be used to construct a 2x2 rotation matrix `L` and a 2 element translation vector `T`.

# In[7]:


theta_deg = 0
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
ax.scatter(xI_L_T,yI_L_T,s=20,alpha=0.1, label='source with initial affine transformation')
ax.scatter(xJ,yJ,s=20,alpha=0.1, label = 'target')

lgnd = plt.legend(scatterpoints=1, fontsize=10)
for handle in lgnd.legend_handles:
    handle.set_sizes([10.0])

ax.set_aspect('equal')


# Now, we will first use `STalign` to rasterize the single cell centroid positions into an image. Assuming the single-cell centroid coordinates are in microns, we will perform this rasterization at a 30 micron resolution. We can visualize the resulting rasterized image.
#
# Note that points are plotting with the origin at bottom left while images are typically plotted with origin at top left so we've used `invert_yaxis()` to invert the yaxis for visualization consistency.

# In[8]:


# rasterize at 30um resolution (assuming positions are in um units) and plot
#XI,YI,I,fig = STalign.rasterize(xI_L_T,yI_L_T,dx=15,blur=1.5)
XI,YI,I,fig = STalign.rasterize(xI,yI,dx=1)

# plot
ax = fig.axes[0]
ax.invert_yaxis()


# Repeat rasterization for target dataset.

# In[9]:


# rasterize and plot
XJ,YJ,J,fig = STalign.rasterize(xJ,yJ,dx=1)
ax = fig.axes[0]
ax.invert_yaxis()


# We can also plot the rasterized images next to each other.

# In[10]:


# get extent of images
extentI = STalign.extent_from_x((YI,XI))
extentJ = STalign.extent_from_x((YJ,XJ))

# plot rasterized images
fig,ax = plt.subplots(1,2)
ax[0].imshow(I.transpose(1,2,0).squeeze(), extent=extentI)
ax[1].imshow(J.transpose(1,2,0).squeeze(), extent=extentJ)
ax[0].invert_yaxis()
ax[1].invert_yaxis()


# Now we will perform our alignment. There are many parameters that can be tuned for performing this alignment. If we don't specify parameters, defaults will be used.

# In[11]:


import time
start_time = time.time()

# run LDDMM
# specify device (default device for STalign.LDDMM is cpu)
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

# keep all other parameters default
params = {
    'a': 5,
    'niter': 1000,
    'diffeo_start': 1001,
    'device': device,
}

print(f"Running STalign LDDMM (affine-only) on {device}...")
out = STalign.LDDMM([YI, XI], I, [YJ, XJ], J, **params)

elapsed = time.time() - start_time
print(f"Alignment completed in {elapsed:.1f} seconds")


# In[12]:


# get necessary output variables
A = out['A']
v = out['v']
xv = out['xv']


# Plots generated throughout the alignment can be used to give you a sense of whether the parameter choices are appropriate and whether your alignment is converging on a solution.
#
# We can also evaluate the resulting alignment by applying the transformation to visualize how our source and target images were deformed to achieve the alignment.

# In[13]:


# apply transform
phii = STalign.build_transform(xv,v,A,XJ=[YJ,XJ],direction='b')
phiI = STalign.transform_image_source_to_target(xv,v,A,[YI,XI],I,[YJ,XJ])

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

# In[14]:


# transform is invertible
phi = STalign.build_transform(xv,v,A,XJ=[YI,XI],direction='f')
phiiJ = STalign.transform_image_target_to_source(xv,v,A,[YJ,XJ],J,[YI,XI])

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

# In[15]:


# apply transform to original points with initial affine transformation
#tpointsI= STalign.transform_points_source_to_target(xv,v,A, np.stack([yI_L_T, xI_L_T], 1))
# apply transform to original points
tpointsI= STalign.transform_points_source_to_target(xv,v,A, np.stack([yI, xI], 1))

#switch from row column coordinates (y,x) to (x,y)
xI_LDDMM = tpointsI[:,1]
yI_LDDMM = tpointsI[:,0]


# And we can visualize the results.

# In[16]:


# plot results
fig,ax = plt.subplots()
ax.scatter(xI,yI,s=20,alpha=0.1, label='source')
ax.scatter(xI_LDDMM,yI_LDDMM,s=20,alpha=0.1, label = 'source aligned')
ax.scatter(xJ,yJ,s=20,alpha=0.1, label='target')

lgnd = plt.legend(scatterpoints=1, fontsize=10)
for handle in lgnd.legend_handles:
    handle.set_sizes([20.0])

ax.set_aspect('equal')


# In[17]:


fig,ax = plt.subplots(1,2)
#ax[0].scatter(xI_L_T,yI_L_T,s=20,alpha=0.1, label='source with initial affine transformation')
ax[0].scatter(xI,yI,s=20,alpha=0.1, label='source')
ax[0].scatter(xJ,yJ,s=20,alpha=0.1, label='target')
ax[1].scatter(xI_LDDMM,yI_LDDMM,s=20,alpha=0.1, label = 'source STaligned')
ax[1].scatter(xJ,yJ,s=20,alpha=0.1, label='target')

lgnd = ax[0].legend(scatterpoints=1, fontsize=10)
for handle in lgnd.legend_handles:
    handle.set_sizes([20.0])

lgnd = ax[1].legend(scatterpoints=1, fontsize=10)
for handle in lgnd.legend_handles:
    handle.set_sizes([20.0])

ax[0].set_aspect('equal')
ax[1].set_aspect('equal')

plt.tight_layout()
output_fig = OUTPUT_DIR / 'stalign_visium_visium_affine_comparison.png'
plt.savefig(output_fig, dpi=150)
plt.close()
print(f"Figure saved to {output_fig}")

# And save the new aligned positions by appending to our original data
df3 = pd.DataFrame({
    "aligned_x": xI_LDDMM,
    "aligned_y": yI_LDDMM,
})

results = pd.concat([df1, df3], axis=1)

# Save results
output_csv = OUTPUT_DIR / 'stalign_visium_visium_affine_results.csv.gz'
results.to_csv(output_csv, compression='gzip')
print(f"Results saved to {output_csv}")

print("\nDone!")




