#!/usr/bin/env python
# coding: utf-8

# ## CMSC733 HW0
# Name: Young-Shiuan Hsu  
# UID:118339238 
# 
# There is a report in the file where all the images are organized, this is only a notebook of all my code

# In[1]:


import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


# In[2]:


img = mpimg.imread('iribefront.jpg')  ## load the original image
print("Image shape:", img.shape)
row = img.shape[0]
col = img.shape[1]
channel = img.shape[2]
figure(figsize=(7,12))
plt.imshow(img)


# ### 1. Plot the R, G, B values along the scanline on the 250th row of the image

# In[3]:


temp = np.zeros((1,img.shape[1]))
newimg = np.ones((img.shape))
newimg = newimg * 255  ## set the image to white(blank canvas)
for i in range(img.shape[0]): 
    if i == 250:
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                newimg[i,j,k] = img[i,j,k]

figure(figsize=(7,12))
plt.imshow(newimg.astype('uint8'))
plt.savefig('1_scanline.png')


# ### 2. Stack the R, G, B channels of the image vertically

# In[4]:


imgR = img[:,:,0]  ## separate the image into R
imgG = img[:,:,1]  ## separate the image into G
imgB = img[:,:,2]  ## separate the image into B

newimg = []
for i in range(3):
    for j in range(imgR.shape[0]):
        newimg.append(img[j,:,i])

newimg = np.array(newimg)
figure(figsize=(7,12))
plt.imshow(newimg, cmap=plt.get_cmap('gray'))
plt.savefig('2_concat.png')


# ### 3. Load the input color image and swap its red and green color channels

# In[5]:


swapimg = np.zeros(img.shape)
swapimg[:,:,0] = imgG
swapimg[:,:,1] = imgR
swapimg[:,:,2] = imgB

figure(figsize=(7,12))
plt.imshow(swapimg.astype('uint8'))
plt.savefig('3_swapchannel.png')


# ### 4. Convert the input color image to a grayscale image.

# In[6]:


'''
Project RGB values onto a vector ie.assumes the distribution of the three channels are not uniform
New grayscale image = (0.3 * R) + (0.59 * G) + (0.11 * B) 
From this equation we notice that human eyes are more sensitive to green, which is why its weight is greater than the other two colors
reference website: 
https://www.tutorialspoint.com/dip/grayscale_to_rgb_conversion.htm#:~:text=You%20just%20have%20to%20take,Its%20done%20in%20this%20way.
'''

def rgb2gray(img):
    row = img.shape[0]
    col = img.shape[1]
    channel = 3
    newimg = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            newimg[i,j] = 0.3*img[i,j,0] + 0.59*img[i,j,1] + 0.11*img[i,j,2]
    return newimg
                

gray = rgb2gray(img)
figure(figsize=(7,12))
plt.imshow(gray,cmap=plt.get_cmap('gray'))
plt.savefig('4_grayscale.png')  


# ### 5. Take the R, G, B channels of the image. Compute an average over the three channels. 

# In[7]:


newimg = np.zeros((row,col))

## method 1 is to take the average of the three channels ie.assumes the distribution of the three channels are uniform
for i in range(row):
    for j in range(col):
        temp = 0
        for k in range(channel):
            temp += img[i,j,k]/channel
        newimg[i,j] = temp

figure(figsize=(7,12))
plt.imshow(newimg.astype('uint8'),cmap=plt.get_cmap('gray'))
plt.savefig('5_average.png') 


# ### 6. Obtain the negative image of the grayscale image in pb4

# In[8]:


newimg = np.zeros((row,col))
for i in range(row):
    for j in range(col):
        newimg[i,j] = 255 - gray[i,j]
        
figure(figsize=(7,12))
plt.imshow(newimg,cmap=plt.get_cmap('gray'))
plt.savefig('6_negative.png')  


# ### 7. Crop the original image into a squared image of size 372 x 372. Then, rotate the image by 90, 180, and 270 degrees and stack the four images (0, 90, 180, 270 degreess) horizontally.

# In[25]:


def rotate_matrix90(img):  ## rotate the matrix by 90 degrees
    row = img.shape[0]
    col = img.shape[1]
    channels = img.shape[2]
    newimg = np.zeros((col,row, channels))
    for i in range(row):
        for j in range(col):
            for k in range(channels):
                newimg[i,j,k] = img[j,row-i-1,k] 
    return newimg
def rotate_matrix180(img):
    row = img.shape[0]
    col = img.shape[1]
    channels = img.shape[2]
    newimg = np.zeros((row, col, channels))
    for i in range(row):
        for j in range(col):
            for k in range(channels):
                newimg[i,j,:] = img[row-i-1,col-j-1,:]
    return newimg

def rotate_matrix270(img):
    row = img.shape[0]
    col = img.shape[1]
    channels = img.shape[2]
    newimg = np.zeros((col, row, channels))
    for i in range(row):
        for j in range(col):
            for k in range(channels):
                newimg[i,j,:] = img[col-j-1,i,:]
    return newimg

figure(figsize=(5,5))
crop_img = img[200:572,200:572,:]
plt.imshow(crop_img.astype('uint8'))  ## the original cropped image


# In[26]:


rotate1 = rotate_matrix90(crop_img) ## rotate image by 90 degrees
rotate2 = rotate_matrix180(crop_img) ## rotate image by 180 degrees
rotate3 = rotate_matrix270(crop_img) ## rotate image by 270 degrees

newimg = np.zeros((372, 372*3, 3))
for i in range(372):
    newimg[:,i,:] = rotate1[:,i,:]
for i in range(372):
    newimg[:,i+372,:] = rotate2[:,i,:]
for i in range(372):
    newimg[:,i+372*2,:] = rotate3[:,i,:]

figure(figsize=(15,9))
plt.imshow(newimg.astype('uint8'))
plt.savefig('7_rotation.png')


# ### 8. For each channel, set the pixel values as 255 when the  pixel values are greater than 127

# In[27]:


maskimg = np.zeros(img.shape)
for i in range(row):
    for j in range(col):
        for k in range(channel):
            if(img[i,j,k] > 127):
                maskimg[i,j,k] = 255
            else:
                maskimg[i,j,k] = img[i,j,k]
figure(figsize=(7,12))
plt.imshow(maskimg.astype('uint8'))
plt.savefig('8_mask.png')


# ### 9. The mean R, G, B values for those pixels marked by the mask in (8).

# In[28]:


Rmean = 0
Gmean = 0
Bmean = 0

for i in range(row):
    for j in range(col):
        Rmean += maskimg[i,j,0]
        Gmean += maskimg[i,j,1]
        Bmean += maskimg[i,j,2]
Rmean = Rmean / (row*col)
Gmean = Gmean / (row*col)
Bmean = Bmean / (row*col)

print("R mean:", Rmean)
print("G mean:", Gmean)
print("B mean:", Bmean)


# ### 10. Take the grayscale image in (3). Create and initialize another image as all zeros. For each 5 x 5 window in the grayscale image, find out the maximum value and set the pixels with the maximum value in the 5x5 window as 255 in the new image.

# In[29]:


gray = rgb2gray(swapimg)  ## using the self created rgb2gray function
figure(figsize=(7,12))
plt.imshow(gray.astype('uint8'), cmap=plt.get_cmap('gray'))  ## the grayscale image of pb3


# In[30]:


def find_max_reset(matrix):
    maxval = 0
    X = 0
    Y = 0
    for i in range(5):
        for j in range(5):
            if(matrix[i][j] >= maxval):
                maxval = matrix[i][j]
                X = i
                Y = j
    for i in range(5):
        for j in range(5):
            if(matrix[i][j] == maxval):
                matrix[i][j] = 255
    return matrix

def convolve_kernel(img):
    ## a 5*5 kernel
    centerX = 0
    centerY = 0
    patchx = int(img.shape[0] / 5)
    patchy = int(img.shape[1] / 5)
    for i in range(patchx-1):
        for j in range(patchy-1):
            temp = np.ones((5,5))
            for a in range(5):
                for b in range(5):
                    temp[a][b] = img[centerX+a][centerY+b]
            temp = find_max_reset(temp)
            for a in range(5):
                for b in range(5):
                    img[centerX+a][centerY+b] = temp[a][b]
            centerY = centerY+5
        centerX = centerX+5
        centerY = 0
            

newimg = convolve_kernel(gray)
figure(figsize=(7,12))
plt.imshow(gray.astype('uint8'), cmap=plt.get_cmap('gray'))  
plt.savefig('10_nonmax.png')


# In[ ]:




