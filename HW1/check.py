    def _init_(self, x=0, y=0, ori=0):
        self.x = x
        self.y = y
        self.ori = ori
        
    def cal_magnitude(self):
        self.mag = math.sqrt(self.x **2 + self.y**2)
        
        
def zero_padding(img, add):      ## to help calculate the edges in the image
    row = img.shape[0] + (add*2)
    col = img.shape[1] + (add*2)
    chan = img.shape[2]
    new_img = np.zeros((row,col,chan))
    for i in range(img.shape[2]):
        for j in range(img.shape[0]):
            for k in range(img.shape[1]):
                new_img[j+add][k+add][i] = img[j][k][i]
    return new_img

def patch_gradient(patch):
    patchsize = patch.shape[0]
    center = math.floor(patchsize/2)
    
    x_direction = 0
    y_direction = 0
    for i in range(patchsize):
        if i < center:
            y_direction -= patch[i][center]
            x_direction -= patch[center][i]
        if i > center:
            y_direction += patch[i][center]
            x_direction += patch[center][i]
            
    if x_direction == 0:
        if y_direction > 0:
            ori = 90.0
        elif y_direction < 0:
            ori = -90.0
        else:
            ori = math.atan(y_direction) * (180/math.pi)
    elif x_direction > 0:
        ori = math.atan(y_direction/x_direction) * (180/math.pi)
        
    else:
        if y_direction < 0:
            ori = ( math.atan(y_direction/x_direction) * (180/math.pi) ) - 180
        else:
            ori = (math.atan(y_direction/x_direction) * (180/math.pi)) + 180
    
    return x_direction, y_direction, ori
    

def gradientMagnitude(im, sigma):
    '''
    im: input image
    sigma: standard deviation value to smooth the image
    outputs: gradient magnitude and gradient direction of the image
    '''
    print("Calculating Gradient of the image...")
    ## create a gaussian low pass filter
    filter_size = 10
    kernel = gaussian_2D_filter(filter_size, filter_size, sigma)
    blur_img = imgfilter(im, kernel)  ## blur the image 
    
    patch_size = 3
    half_patch_size = math.floor(patch_size/2)
    stride = 1
    ## padding the image to also calcuate the edges of the image
    blur_img = zero_padding(blur_img, half_patch_size)  
    row = blur_img.shape[0]
    col = blur_img.shape[1]
    channels = blur_img.shape[2]
#     print("Padded image size:", row, col, channels)
    
    R_component = []
    G_component = []
    B_component = []
    for i in range(half_patch_size, row-half_patch_size, 1):
        R_row_component = []
        G_row_component = []
        B_row_component = []
        for j in range(half_patch_size, col-half_patch_size, 1):
            ## create patches to calculate the gradient later
            patch = np.zeros((patch_size, patch_size, 3))
            for ii in range(patch_size):    
                for jj in range(patch_size):
                    for kk in range(channels):
                        patch[ii][jj][kk] = blur_img[i-half_patch_size+ii][j-half_patch_size+jj][kk]
            
            for k in range(channels):
                gradient_temp = img_gradient()
                if k == 0:  ## R component
                    gradient_temp.x, gradient_temp.y, gradient_temp.ori = patch_gradient(patch[:,:,0])
                    R_row_component.append(gradient_temp)
                if k == 1: ## G component
                    gradient_temp.x, gradient_temp.y, gradient_temp.ori = patch_gradient(patch[:,:,1])
                    G_row_component.append(gradient_temp)
                elif k == 2:  ## B component
                    gradient_temp.x, gradient_temp.y, gradient_temp.ori = patch_gradient(patch[:,:,2])
                    B_row_component.append(gradient_temp)
                        
        R_component.append(R_row_component)
        G_component.append(G_row_component)
        B_component.append(B_row_component)
    print("Col", len(B_row_component))
    print("Row", len(B_component))
    return R_component, G_component, B_component


# this website is very helpful for non-maximum suppression
## https://justin-liang.com/tutorials/canny/#suppression
def get_direction(pixel):
    if pixel.ori <= 67.5 and pixel.ori > 22.5 :
        return 'SE'
    elif pixel.ori > 67.5 and pixel.ori <= 112.5:
        return 'S'
    elif pixel.ori > 112.5 and pixel.ori <= 157.5:
        return 'SW'
    elif pixel.ori >= -22.5 and pixel.ori < 22.5:
        return 'E'
    elif pixel.ori >= -67.5 and pixel.ori < -22.5:
        return 'NE'
    elif pixel.ori >= -112.5  and pixel.ori < -67.5:
        return 'N'
    elif pixel.ori > -157.5 and pixel.ori < -112.5:
        return 'NW'
    else:
        return 'W'

def thresholding(patch, direction):
    center = patch[1][1].mag
    pixel1 = 0
    pixel2 = 0
    if direction == 'NE' or direction == 'SW':
        pixel1 = patch[0][2].mag
        pixel2 = patch[2][0].mag
    elif direction == 'N' or direction == 'S':
        pixel1 = patch[0][1].mag
        pixel2 = patch[2][1].mag
    elif direction == 'NW' or direction == 'SE':
        pixel1 = patch[0][0].mag
        pixel2 = patch[2][2].mag
    else:
        pixel1 = patch[1][0].mag
        pixel2 = patch[1][2].mag
    
    if center >= pixel1 and center >= pixel2:
        return center
    else:
        return 0
    
def edgeGradient(im):
    '''
    im: input image
    output: a soft boundary map of the image
    '''
    ## assume the input image is a gray scale image
    row = im.shape[0]
    col = im.shape[1]
    patch_size = 3
    half_patch_size = math.floor(patch_size/2)
    new_image = np.zeros((row,col))
    for i in range(1,row-1):
        for j in range(1,col-1):
            direction = ''
            direction = get_direction(im[i][j])
            patch = []
            for ii in range(patch_size):
                patch_temp = []
                for jj in range(patch_size):
                    patch_temp.append(im[i-half_patch_size+ii][j-half_patch_size+jj])
                patch.append(patch_temp)
            
            value = thresholding(patch, direction)        
            new_image[i][j] = value
    print(new_image.shape)     
            
    return new_image