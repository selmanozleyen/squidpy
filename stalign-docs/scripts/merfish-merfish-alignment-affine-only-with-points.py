#!/usr/bin/env python
# coding: utf-8

# # Aligning two coronal sections of adult mouse brain from MERFISH
# 
# In this notebook, we align two single cell resolution spatial transcriptomics datasets of coronal sections of the adult mouse brain from matched locations with respect to bregma assayed by MERFISH.
# 
# We will use `STalign` to achieve this alignment. We will first load the relevant code libraries.

# In[1]:


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

# In[ ]:


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

# In[ ]:


# plot
fig,ax = plt.subplots()
ax.scatter(xI,yI,s=1,alpha=0.2, label='source')
ax.scatter(xJ,yJ,s=1,alpha=0.1, label= 'target')
ax.legend(markerscale = 10)


# Now, we will first use `STalign` to rasterize the single cell centroid positions into an image. Assuming the single-cell centroid coordinates are in microns, we will perform this rasterization at a 30 micron resolution. We can visualize the resulting rasterized image.
# 
# Note that points are plotting with the origin at bottom left while images are typically plotted with origin at top left so we've used `invert_yaxis()` to invert the yaxis for visualization consistency. 

# In[9]:


# rasterize at 30um resolution (assuming positions are in um units) and plot
XI,YI,I,fig = STalign.rasterize(xI,yI,dx=15,blur=1.5)

# plot
ax = fig.axes[0]
ax.invert_yaxis()


# Repeat rasterization for target dataset.

# In[10]:


# rasterize and plot
XJ,YJ,J,fig = STalign.rasterize(xJ,yJ,dx=15, blur=1.5)
ax = fig.axes[0]
ax.invert_yaxis()


# We can also plot the rasterized images next to each other.

# In[11]:


# get extent of images
extentI = STalign.extent_from_x((YI,XI))
extentJ = STalign.extent_from_x((YJ,XJ))

# plot rasterized images
fig,ax = plt.subplots(1,2)
ax[0].imshow(I.transpose(1,2,0).squeeze(), extent=extentI) 
ax[1].imshow(J.transpose(1,2,0).squeeze(), extent=extentJ)
ax[0].invert_yaxis()
ax[1].invert_yaxis()


# We can manually designate a few landmark points to help initialize the alignment. A `point_annotator.py` script is provided to assist with this. In order to use the `point_annotator.py` script, we will need to write out our images as `.npz` files. 

# In[11]:


np.savez('../merfish_data/Merfish_S2_R3', x=XI,y=YI,I=I)
np.savez('../merfish_data/Merfish_S2_R2', x=XJ,y=YJ,I=J)


# Given these `.npz` files, we can then run the following code on the command line from inside the `notebooks` folder:
# 
# ```
# python ../../STalign/point_annotator.py ../merfish_data/Merfish_S2_R3.npz ../merfish_data/Merfish_S2_R2.npz
# ```
# 
# Which will provide a graphical user interface to selecting points. These points will saved as `Merfish_S2_R3_points.npy` and `Merfish_S2_R2_points.npy` respectively. We can then read in these files. 

# In[12]:


# read from file
pointsIlist = np.load('../merfish_data/Merfish_S2_R3_points.npy', allow_pickle=True).tolist()
print(pointsIlist)
pointsJlist = np.load('../merfish_data/Merfish_S2_R2_points.npy', allow_pickle=True).tolist()
print(pointsJlist)


# Note that these landmark points are read in as lists. We will want to convert them to a simple array for downstream usage. 

# In[13]:


# convert to array
pointsI = []
pointsJ = []

for i in pointsIlist.keys():
    for j in range(len(pointsIlist[i])):
        pointsI.append([pointsIlist[i][j][1], pointsIlist[i][j][0]])
for i in pointsJlist.keys():
    for j in range(len(pointsJlist[i])):
        pointsJ.append([pointsJlist[i][j][1], pointsJlist[i][j][0]])

pointsI = np.array(pointsI)
pointsJ = np.array(pointsJ)


# In[14]:


# now arrays
print(pointsI)
print(pointsJ)


# We can double check that our landmark points look sensible by plotting them along with the rasterized image we created.

# In[15]:


# plot

fig,ax = plt.subplots(1,2)
ax[0].imshow((I.transpose(1,2,0).squeeze()), extent=extentI) 
ax[1].imshow((J.transpose(1,2,0).squeeze()), extent=extentJ)

trans_offset_0 = mtransforms.offset_copy(ax[0].transData, fig=fig,
                                       x=0.05, y=-0.05, units='inches')
trans_offset_1 = mtransforms.offset_copy(ax[1].transData, fig=fig,
                                       x=0.05, y=-0.05, units='inches')

ax[0].scatter(pointsI[:,1],pointsI[:,0], c='red', s=10)
ax[1].scatter(pointsJ[:,1],pointsJ[:,0], c='red', s=10)

for i in pointsIlist.keys():
    for j in range(len(pointsIlist[i])):
        ax[0].text(pointsIlist[i][j][0], pointsIlist[i][j][1],f'{i}{j}', c='red', transform=trans_offset_0, fontsize= 8)
for i in pointsJlist.keys():
    for j in range(len(pointsJlist[i])):
        ax[1].text(pointsJlist[i][j][0], pointsJlist[i][j][1],f'{i}{j}', c='red', transform=trans_offset_1, fontsize= 8)

ax[0].set_title('source with pointsI', fontsize=15)
ax[1].set_title('target with pointsJ', fontsize=15)

# invert only rasterized images
ax[0].invert_yaxis()
ax[1].invert_yaxis()


# From the landmark points, we can generate a linear transformation `L` and translation `T` which will produce a simple initial affine transformation `A`. 

# In[16]:


# set device for building tensors
if torch.cuda.is_available():
    torch.set_default_device('cuda:0')
else:
    torch.set_default_device('cpu')


# In[17]:


# compute initial affine transformation from points
L,T = STalign.L_T_from_points(pointsI,pointsJ)
A = STalign.to_A(torch.tensor(L),torch.tensor(T))


# We can show the results of the simple affine transformation on the rasterized source image.

# In[18]:


# compute initial affine transformation from points
AI= STalign.transform_image_source_with_A(A, [YI,XI], I, [YJ,XJ])

fig,ax = plt.subplots(1,2)

if AI.is_cuda:
    ax[0].imshow((AI.cpu().permute(1,2,0).squeeze()), extent=extentJ) 
else:
    ax[0].imshow((AI.permute(1,2,0).squeeze()), extent=extentJ) 
ax[1].imshow((J.transpose(1,2,0).squeeze()), extent=extentJ) 

ax[0].set_title('source with affine transformation', fontsize=15)
ax[1].set_title('target', fontsize=15)

# invert only rasterized images
ax[0].invert_yaxis()
ax[1].invert_yaxis()


# Finally, we can apply our affine transform to the original sets of single cell centroid positions to achieve their new aligned positions.

# In[19]:


#apply A to sources points in row, column (y,x) orientation
affine = np.matmul(np.array(A.cpu()),np.array([yI, xI, np.ones(len(xI))]))

xIaffine = affine[1,:] 
yIaffine = affine[0,:]


# And we can visualize the results.

# In[20]:


# plot results
fig,ax = plt.subplots()
ax.scatter(xI,yI,s=1,alpha=0.1, label='source')
ax.scatter(xIaffine,yIaffine,s=1,alpha=0.1, label = 'source aligned')
ax.scatter(xJ,yJ,s=1,alpha=0.1, label='target')
ax.legend(markerscale = 10)


# And save the new aligned positions by appending to our original data

# In[21]:


df3 = pd.DataFrame(

    {

        "aligned_x": xIaffine,

        "aligned_y": yIaffine,

    },


)

results = pd.concat([df1, df3], axis=1)
results.head()


# We will finally create a compressed `.csv.gz` file named `mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate3_STalign_to_Slice2_Replicate2.csv.gz`

# In[24]:


results.to_csv('../merfish_data/mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate3_STalign_to_Slice2_Replicate2_affine_only_with_points.csv.gz',
               compression='gzip')


# To further analyze our results we can measure the target registration error (TRE) of the landmarks (between source and target) before and after alignment. To do that, first we need to apply the affine alignment to the source landmark points.

# In[22]:


#apply A to sources landmark points in row, column (y,x) orientation
ypointsI = pointsI[:,0]
xpointsI = pointsI[:,1]
affine = np.matmul(np.array(A.cpu()),np.array([ypointsI, xpointsI, np.ones(len(ypointsI))]))

xpointsIaffine = affine[1,:] 
ypointsIaffine = affine[0,:]
pointsIaffine = np.column_stack((ypointsIaffine,xpointsIaffine))


# In[23]:


print(pointsIaffine)


# We can get the mean TRE across landmark points by using `STalign.calculate_tre`

# In[25]:


treBefore = STalign.calculate_tre(pointsI,pointsJ)
treAffine = STalign.calculate_tre(pointsIaffine,pointsJ)

print("The mean TRE of landmarks before alignment is {:.0f} +/- {:.2f}".format(treBefore[0], treBefore[1] ))
print("The mean TRE of landmarks after affine alignment is {:.0f} +/- {:.2f}".format(treAffine[0], treAffine[1] ))


# Alternatively, we can look at the TRE for each target and source pair of landmark points

# In[26]:


treBefore = np.sqrt(np.sum((pointsI - pointsJ)**2,axis=1))
treAffine = np.sqrt(np.sum((pointsIaffine - pointsJ)**2,axis=1))
print(treBefore)
print(treAffine)


# We can plot the TRE before alignment on each of the target landmarks

# In[47]:


# plot

fig,ax = plt.subplots()
ax.imshow((J.transpose(1,2,0).squeeze()), extent=extentJ) 

trans_offset_0 = mtransforms.offset_copy(ax.transData, fig=fig,
                                       x=0.05, y=-0.05, units='inches')

ax.scatter(pointsJ[:,1],pointsJ[:,0], c='red', s=10, label='Target')
#ax.scatter(pointsI[:,1],pointsI[:,0], c='orange', s=10, label='Source')

#k=0
#for i in pointsIlist.keys():
#    for j in range(len(pointsIlist[i])):
#        ax.text(pointsIlist[i][j][0], pointsIlist[i][j][1],f'{treBefore[k]:.0f}', c='red', transform=trans_offset_0, fontsize= 15)
#        k = k+1

for i in range(pointsJ.shape[0]):
    ax.text(pointsJ[i,1],pointsJ[i,0],f'{treBefore[i]:.0f}', c='red', transform=trans_offset_0, fontsize= 15)

ax.set_title('TRE Before Alignment', fontsize=15)

ax.legend(markerscale = 1)
# invert only rasterized images
ax.invert_yaxis()


# Next plot shows a reminder of what the landmarks of source and target looked like before alignment

# In[48]:


# plot
fig,ax = plt.subplots()
ax.scatter(xI,yI,s=1,alpha=0.2, label='source')
ax.scatter(xJ,yJ,s=1,alpha=0.1, label= 'target')
ax.legend(markerscale = 10)
ax.scatter(pointsI[:,1],pointsI[:,0], c='purple', s=10, label='Source')
ax.scatter(pointsJ[:,1],pointsJ[:,0], c='green', s=10, label='Target')
ax.set_title('Before Alignment', fontsize=15)


# We can plot the TRE after alignment on each of the target landmarks along with the source landmarks after affine alignment.

# In[29]:


# plot

fig,ax = plt.subplots()
ax.imshow((J.transpose(1,2,0).squeeze()), extent=extentJ) 

trans_offset_0 = mtransforms.offset_copy(ax.transData, fig=fig,
                                       x=0.05, y=-0.05, units='inches')

ax.scatter(pointsJ[:,1],pointsJ[:,0], c='red', s=10, label='Target')
ax.scatter(pointsIaffine[:,1],pointsIaffine[:,0], c='orange', s=10, label="Affine Alignment")

#k=0
#for i in pointsIlist.keys():
#    for j in range(len(pointsIlist[i])):
#        ax.text(pointsIlist[i][j][0], pointsIlist[i][j][1],f'{treAffine[k]:.0f}', c='red', transform=trans_offset_0, fontsize= 15)
#        k = k+1

for i in range(pointsJ.shape[0]):
    ax.text(pointsJ[i,1],pointsJ[i,0],f'{treAffine[i]:.0f}', c='red', transform=trans_offset_0, fontsize= 15)

ax.set_title('TRE After Affine Alignment', fontsize=15)

ax.legend(markerscale = 1)

# invert only rasterized images
ax.invert_yaxis()


# More summary plots of alignment below:

# In[76]:


# plot results
fig,ax = plt.subplots()
ax.scatter(xI,yI,s=1,alpha=0.1, label='source')
ax.scatter(xIaffine,yIaffine,s=1,alpha=0.1, label = 'source aligned')
ax.scatter(xJ,yJ,s=1,alpha=0.1, label='target')
ax.legend(markerscale = 10)
#ax.scatter(pointsI[:,1],pointsI[:,0], c='purple', s=10, label='Source')
ax.scatter(pointsJ[:,1],pointsJ[:,0], c='green', s=10, label='Target')
ax.scatter(pointsIaffine[:,1],pointsIaffine[:,0], c='red', s=10, label="Affine Alignment")


# In[64]:


# plot results
fig,ax = plt.subplots(2,2)

#Source
ax[0][0].scatter(xI,yI,s=1,alpha=0.1, label='source cells')
ax[0][0].scatter(pointsI[:,1],pointsI[:,0], c='blue', s=10, label='source landmarks')
ax[0][0].set_title('Source', fontsize=15)
ax[0][0].set_aspect('equal')

lgnd = ax[0][0].legend(scatterpoints=1, fontsize=10)
for handle in lgnd.legend_handles:
    handle.set_sizes([10.0])

#Target
ax[0][1].scatter(xJ,yJ,s=1,alpha=0.1, label='target cells', c='orange')
ax[0][1].scatter(pointsJ[:,1],pointsJ[:,0], c='red', s=10, label='target landmarks')
ax[0][1].set_title('Target', fontsize=15)
ax[0][1].set_aspect('equal')

lgnd = ax[0][1].legend(scatterpoints=1, fontsize=10)
for handle in lgnd.legend_handles:
    handle.set_sizes([10.0])

#Affine Aligned Source
ax[1][0].scatter(xIaffine,yIaffine,s=1,alpha=0.1, label = 'affine aligned source cells', c='blue')
ax[1][0].scatter(pointsIaffine[:,1],pointsIaffine[:,0], c='purple', s=10, label="affine aligned source landmarks")
ax[1][0].set_title('Affine Aligned Source', fontsize=15)
ax[1][0].set_aspect('equal')

lgnd = ax[1][0].legend(scatterpoints=1, fontsize=10)
for handle in lgnd.legend_handles:
    handle.set_sizes([10.0])

#Overlay of Target and Affine Aligned Source
ax[1][1].scatter(xJ,yJ,s=1,alpha=0.1, label='target', c='orange')
ax[1][1].legend(markerscale = 10)
ax[1][1].scatter(pointsJ[:,1],pointsJ[:,0], c='red', s=10, label='target landmarks')

ax[1][1].scatter(xIaffine,yIaffine,s=1,alpha=0.1, label = 'source aligned', c='blue')
ax[1][1].scatter(pointsIaffine[:,1],pointsIaffine[:,0], c='purple', s=10, label="affine aligned source landmarks")
ax[1][1].set_title('Overlay', fontsize=15)
ax[1][1].set_aspect('equal')

lgnd = ax[1][1].legend(scatterpoints=1, fontsize=10)
for handle in lgnd.legend_handles:
    handle.set_sizes([10.0])


# In[ ]:




