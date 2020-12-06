import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

# A1
def convolutionMaskA(img):
    """
    Convolve the given image with convolution mask (kernel) that computes the average of a 1×9 pixels around each pixel
    The function also displays the image and the convolved image.
    Parameters
    ----------
    img : np.array
        Image to be convolved

    Returns
    -------
    np.array
        The convolved image
    """
    mask = np.ones((1,9))
    print('normalizedConvolutionMaskGeneral A1: ', mask)
    
    
    return normalizedConvolutionMaskGeneral(img, mask)

# A2
def convolutionMaskB(img):
    """
    Convolve the given image with convolution mask (kernel) of size 5×5 such that the maximal value over all possible grey level images      (range 0 to 255) will be obtained in the center of a black 'L' shape region surrounded by white pixels while the rest of the image may contain any values

    Parameters
    ----------
    img : np.array
        Image to be convolved

    Returns
    -------
    np.array
        The convolved image
    """
    mask = np.ones((5,5))

    # L's pixels
    mask[1,1] = -1
    mask[2,1] = -1
    mask[3,1] = -1
    mask[3,2] = -1
    
    return normalizedConvolutionMaskGeneral(img, mask)

# A3
def convolutionMaskC(img):
    """
    Convolve the given image with convolution mask (kernel) of size 5×5 such that the maximal value over all possible grey level images (range 0 to 255) will be obtained in the center of a black 'L' shape region surrounded by white pixels while the rest of the image may     contain any values and the "middle" pixel is a don't care value.

    Parameters
    ----------
    img : np.array
        Image to be convolved

    Returns
    -------
    np.array
        The convolved image
    """
    mask = np.ones((5,5))

    #L's pixels
    mask[1,1] = -1
    mask[2,1] = -1
    mask[3,1] = -1
    mask[3,2] = -1
    
    # L's center
    mask[2,2] = 0 
    
    return normalizedConvolutionMaskGeneral(img, mask)

def normalizedConvolutionMaskGeneral(img, mask):
    """
    *Section A helper function
    Convolve the given image with the given convolution mask (kernel) 
    Print the result's plot which contains the original image and the convolved image side by side
    Parameters
    ----------
    img : np.array
        Image to be convolved
    mask : np.array
        mask (kernel) to convolve the image with

    Returns
    -------
    np.array
        The normalized convolved image
    """
    mask = np.flip(mask)
    resultConv = convolve2d(img, mask, mode='same')
    
    resultConv[resultConv < 0] = 0
    
    # normalize the values
    resultConv = resultConv / resultConv.max()
    
    f, ((ax1, ax2)) = plt.subplots(1, 2, sharex='col', sharey='row')

    ax1.imshow(img, cmap='gray'), ax1.set_title('Original Image')
    ax2.imshow(resultConv, cmap='gray'), ax2.set_title('Applying convolution mask')
    
    # returns the index of a value
    loc = np.unravel_index(np.argmax(resultConv, axis=None), resultConv.shape)

    
    return resultConv



# B1.1
def Deriv_Gauss_x(sigma, mask_size):
    """
    Derive Gauss by x 
    ----------
    sigma : float
    mask_size : integer

    Returns
    -------
    np.array
        The gauss derivative by x
    """
    return Deriv_Gauss(sigma, mask_size, 'x')

# B1.1
def Deriv_Gauss_y(sigma, mask_size):
    """
    Derive Gauss by y
    ----------
    sigma : float
    mask_size : integer

    Returns
    -------
    np.array
        The gauss derivative by y
    """
    return Deriv_Gauss(sigma, mask_size, 'y')


# B1.2
# Changed the signature of the function to accept also G_dx (after reading on Piazza)
def Grad_x(image, gauss_der_x):
    """
    Calculate image's gradient with the derivative of a gaussian with respect to x 
    ----------
    image : np.array
        Image to be calculate
    gauss_der_x : np.array
        mask (kernel) to convolved the image with
    Returns
    -------
    np.array
        The gradiented image by x
    """
    Ix = Gradient(image, gauss_der_x)
    return Ix

# B1.2
# Changed the signature of the function to accept also G_dy (after reading on Piazza)
def Grad_y(image, gauss_der_y):
    """
    Calculate image's gradient with the derivative of a gaussian with respect to y
    ----------
    image : np.array
        Image to be calculate
    gauss_der_y : np.array
        mask (kernel) to convolved the image with
    Returns
    -------
    np.array
        The gradiented image by y
    """
    Iy = Gradient(image, gauss_der_y)
    return Iy


# B1.3
# Changed the signature of the function to accept also Ix and Iy (after reading on Piazza)
def Grad_o(Ix, Iy):
    """
    Calculate gradient orientation by the formula - arctan(Iy, Ix)
    Parameters
    ----------
    Ix : np.array
        image derivation by x
    Iy : np.array
        image derivation by y
        
    Returns
    -------
    np.array
        Image that represents image's orientation 
    """
    angles = np.arctan2(Iy, Ix)
    return angles

# B1.3
# Changed the signature of the function to accept also Ix and Iy (after reading on Piazza)
def Grad_m(Ix, Iy):
    """
    Calculate gradient magnitude by the formula - √(𝐼𝑥^2+𝐼𝑦^2)
    Parameters
    ----------
    Ix : np.array
        image derivation by x
    Iy : np.array
        image derivation by y
        
    Returns
    -------
    np.array
        Image that represents image's magnitue  
    """
    result = np.sqrt(Ix**2 + Iy**2)
#     values are between 0 and 255
    result = (result / result.max()) * 255
    
    return result


# B1.4
def thinning(gradient_magnitude, gradient_orientation):
    """
    Calculate which edges will remain after thinning(which takes only one pixel for every edge with respect to the edge's magnitude)
    
    Parameters
    ----------
    gradient_magnitude : np.array
        Image magnitude of each pixel
    gradient_orientation : np.array
        Image orientation of each pixel in degrees

    Returns
    -------
    np.array
        The image after thining
    """
    angles = np.rad2deg(gradient_orientation)
    angles[angles < 0] += 180
    
    rows, columns = gradient_magnitude.shape
    result = np.zeros((rows, columns))

    'we must loop through the array because we check the two neighbors for each pixels based on the rounded angle in the orientation matrix'
    for i in range(rows):
        for j in range(columns):
            angle = angles[i][j]
            currentValue = gradient_magnitude[i][j]
            
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                'rounding angle to 0'
                first_neighbor = 0 if (j == 0) else gradient_magnitude[i][j - 1]
                second_neighbor = 0 if (j == columns-1) else gradient_magnitude[i][j + 1]
            elif 22.5 <= angle < 67.5:
                'rounding angle to 45'
                first_neighbor = 0 if (i == 0 or j == columns-1) else gradient_magnitude[i - 1][j + 1]
                second_neighbor = 0 if (i == rows-1 or j == 0) else gradient_magnitude[i + 1][j - 1]
            elif 67.5 <= angle < 112.5:
                'rounding angle to 90'
                first_neighbor = 0 if (i == 0) else gradient_magnitude[i - 1][j]
                second_neighbor = 0 if (i == rows-1) else gradient_magnitude[i + 1][j]
            elif 112.5 <= angle < 157.5:
                'rounding angle to 135'
                first_neighbor = 0 if (i == 0 or j == 0) else gradient_magnitude[i - 1][j - 1]
                second_neighbor = 0 if (i == rows-1 or j == columns-1) else gradient_magnitude[i + 1][j + 1]
            
            if (currentValue > first_neighbor) and (currentValue > second_neighbor):
                result[i][j] = currentValue
    
    return result


# Section B helper function
def threshold_hysteresis(thin_edges_image, L_th, H_th):
    """
    Keep the edges that cross the H_th.
    Connect the edges that cross the L_th and are in the same components as thus that cross the high thresholds and remove them otherwise.
    
    1. Calculate the metrix that keeps all edges that cross H_th. 
    2. Calculate the metrix that keeps all edges that cross L_th. 
    3. Dilate the metrix calculated in (1)
    4. Perform dilation with kernel 3*3 on the above matrix.
    5. Create a maximum matrix from (2) and (4) metrices. 
    6. Create a connected component matrix from the above matrix.
    7. Create a set that represent all components IDs that contain only edges that are above the high threshold.
    8. Remove all edges that are not in the same components as in the set that calculated in (7) from the connected component matrix.
    
    Parameters
    ----------
    gradient_magnitude : np.array
        Image magnitude of each pixel
    gradient_orientation : np.array
        Image orientation of each pixel in degrees

    Returns
    -------
    np.array
        The image after thining
    """
    low_values_flag = 1
    high_values_flag = 2
    th1 = cv2.threshold(thin_edges_image, L_th * 255, low_values_flag, cv2.THRESH_BINARY)[1]
    th2 = cv2.threshold(thin_edges_image, H_th * 255, high_values_flag, cv2.THRESH_BINARY)[1]
    th2 = cv2.dilate(th2, np.ones((2, 2)), iterations=1)
    result = np.maximum(th1, th2)
    parsed_array = np.array(result, dtype=np.uint8)
    
    labels, connected_mat = cv2.connectedComponents(parsed_array, connectivity=8)
    
    high_values_set = set()
    
    high_values_set = {connected_mat.flatten()[i] for i, val in enumerate(th2.flatten()) if val == high_values_flag}
    
    for i in range(len(connected_mat)):
        for j in range(len(connected_mat[0])):
            if connected_mat[i][j] in high_values_set:
                connected_mat[i][j] = 1
            else:
                connected_mat[i][j] = 0
    
    parsed_connected_mat = np.array(connected_mat, dtype=np.uint8)

    return parsed_connected_mat


def Deriv_Gauss(sigma, mask_size, axis):
    """
    Create derivative gaussian kernel
    """  
    """
    Calculate Gaussian's Derivative
    Parameters
    ----------
    sigma : float
    mask_size : float
    axis : string

    Returns
    -------
    np.array
        Gaussian kernel
    """
    mask_size = int(mask_size)
    
    if (mask_size % 2) == 0:
        mask_size += 1
   
    # Create a vector with values
    ax = np.linspace(-(mask_size - 1) / 2., (mask_size - 1) / 2., mask_size)
  
    # Assigne two matrices with the appropriate x and y values on which the gaussian function is computed
    xx, yy = np.meshgrid(ax, ax)
    kernel = GaussianDerivative(axis, xx, yy, sigma)
    
    return kernel

def GaussianDerivative(axis, x, y, sigma):
    """
    Calculate Gaussian's derivative based on given axis ('x' or 'y')
    Parameters
    ----------
    axis : string 
    x : np.array
    y : np.array
    sigma : float

    Returns
    -------
    np.array
        The normalized convolved image
    """
    axis_parameter = x if axis == 'x' else y
    return (-axis_parameter/2*np.pi*sigma**4)*np.exp(-(np.square(x) + np.square(y))/(2*sigma**2))

def Gaussian(x, y, sigma):
    """
    Calculate Gaussian's formula with the given parameters
    Parameters
    ----------
    x : np.array
    y : np.array
    sigma: float

    Returns
    -------
    np.array
        The normalized convolved image
    """
    return (1/(2*np.pi*sigma**2))*np.exp(-(np.square(x) + np.square(y))/(2*sigma**2))

def Gradient(image, mask):
    """
    Calculate the matrix's gradient by convolving the image with the given mask
    ----------
    img : np.array
        Image to be convolved
    mask : np.array
        mask (kernel) to convolve with the image

    Returns
    -------
    np.array
        The matrix's gradient
    """
    return convolve2d(image, mask, mode='same')

def canny(img, sigma, L_th, H_th): 
    """
    Perform classic Canny edge detector with the given parameters.
    
    Parameters
    ----------
    img : np.array
        Image for which we want to detect the edges 
    sigma : float
        Determines the width of the Gaussian
    L_th : float
        Sets an high threshold for which pixel will remain after thinning
    H_th : float
        Sets an low threshold for which pixel will remain after thinning 

    Returns
    -------
    np.array
        The image after applying the classic Canny edge detector
    """
    mask_size = sigma * 4

    kernel_x = Deriv_Gauss_x(sigma, mask_size)
    kernel_y = Deriv_Gauss_y(sigma, mask_size)
    
    Ix = Grad_x(img, kernel_x)
    Iy = Grad_y(img, kernel_y)
    
    gradient_orientation = Grad_o(Ix, Iy)
    gradient_magnitude = Grad_m(Ix, Iy)
    
    thin_edges = thinning(gradient_magnitude, gradient_orientation)
    
    # thersholds
    final_image = threshold_hysteresis(thin_edges, L_th, H_th)

    # The output is a binary map where an edge pixel is 1 and the rest are 0
    return final_image

# C2
def evaluate_edges(res, GT):
    """
    Calculate presicion, recall and F measures of the given resulting image(res) and the given image(GT) in order to evalute edge detector algorithm's performance by comparing res's pixels to GT ("ground truth") pixels
    
    Calculate the intersection between res and GT, their edges's counts and calculate presicion, recall and F measures by their formulas
    
    Parameters
    ----------
    res : np.array
        Image to be convolved
    GT : np.array
        edge-detector's result

    Returns
    -------
    np.float
        Precision ( 𝑃 ) 
    np.float
        Recall ( 𝑅 )
    np.float
        F - A measure that combines 𝑃 and 𝑅  
    """
    res_flatten_1_indexes = np.where(res.flatten() == 1)
    GT_flatten_1_indexes = np.where(GT.flatten() == 1)
    intersection_size = len(np.intersect1d(res_flatten_1_indexes, GT_flatten_1_indexes))
    E_size = res.sum()
    GT_size = GT.sum()

    P = intersection_size / E_size if E_size > 0 else np.nan
    R = intersection_size / GT_size if GT_size > 0 else np.nan
    F = np.nan if (np.isnan(P) or np.isnan(R)) else (2 * P * R) / (P + R)
                
    #     print('P: ', P)
    isP_nan = np.isnan(P)
    if isP_nan:
        print('The size of E is 0, cannot calclulate Precision value')
        
#     print('R: ', R)
    isR_nan = np.isnan(R)
    if isR_nan:
        print('The size of E is 0, cannot calclulate Recall value')
        
#     print('F: ', F)
    isF_nan = np.isnan(F)
    if isF_nan:
        print('P or R are 0, cannot calclulate F value')
    
    return P, R, F

    
# C5
def evaluate_edges_shift_tolerant(res, GT):
    """
    Calculate presicion, recall and F measures of the given resulting image(res) and the given image(GT) in order to evalute edge detector algorithm's performance by comparing res's pixels to GT's result consider the fact that a 𝐺𝑇's pixel may be shifted by one pixel with respect to the computed edges.   
    
    Calculate the intersection between res and GT, their edges's counts and calculate presicion, recall and F measures by thier formulas

    Use dilate function with kernel np.ones((3,3)) in order to dilate every edge in res (which surounds each edges's pixels with ones - 8 pixels)).
    compare each GT's edge(pixel) to each res's edges and their 8 neighbors. 
    
    Parameters
    ----------
    res : np.array
        Image to be convolved
    GT : np.array
        edge-detector's result

    Returns
    -------
    np.float
        Precision ( 𝑃 ) 
    np.float
        Recall ( 𝑅 )
    np.float
        F - A measure that combines 𝑃 and 𝑅 
    """
    intersection_size = 0 
    dilated_res = cv2.dilate(res, np.ones((3, 3)), iterations=1)
    GT_size = GT.sum()
    E_size = res.sum()
    row_size = len(GT)
    column_size = len(GT[0])
    
    for i in range(row_size):
        for j in range(column_size):
            
            if dilated_res[i][j] == 1 and GT[i][j] == 1:

                intersection_size += 1

                if i > 0 and res[i-1][j] == 0:
                    dilated_res[i-1][j] = 0
                        
                if i > 0 and j > 0 and res[i-1][j-1] == 0:
                    dilated_res[i-1][j-1] = 0
                        
                if j > 0 and res[i][j-1] == 0:
                    dilated_res[i][j-1] = 0
                        
                if j < column_size -1 and i < row_size -1 and res[i+1][j+1] == 0:
                    dilated_res[i+1][j+1] = 0
                      
                if i < row_size -1 and res[i+1][j] == 0:
                    dilated_res[i+1][j] = 0
                        
                if j < column_size -1 and res[i][j+1] == 0:
                    dilated_res[i][j+1] = 0
                        
                if j < column_size -1 and i > 0 and res[i-1][j+1] == 0:
                    dilated_res[i-1][j+1] = 0
                        
                if i < row_size -1 and j > 0 and res[i+1][j-1] == 0:
                    dilated_res[i+1][j-1] = 0
                        
    print(f'E_size: {E_size}')
    print(f'GT_size: {GT_size}')
    print(f'intersection_size: {intersection_size}')
    
    P = intersection_size / E_size if E_size > 0 else np.nan
    R = intersection_size / GT_size if GT_size > 0 else np.nan
    F = np.nan if (np.isnan(P) or np.isnan(R)) else (2 * P * R) / (P + R)
                
    isP_nan = np.isnan(P)
    if isP_nan:
        print('The size of E is 0, cannot calclulate Precision value')
        
    isR_nan = np.isnan(R)
    if isR_nan:
        print('The size of E is 0, cannot calclulate Recall value')
        
    isF_nan = np.isnan(F)
    if isF_nan:
        print('P or R are 0, cannot calclulate F value')
    
    return P, R, F


# B2
def B2(imageName, sigma, l_th, h_th):
#     Test your functions on an image you choose. Explore various parameters and choose a set such that the result looks “good”. \
#     Submit in the doc/pdf file: display the image you chose, its edges, and the parameters you used.
    img = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
    final_image = canny(img, sigma, l_th, h_th)
    f, ((original_image, final)) = plt.subplots(1, 2, sharex='col', sharey='row')
    original_image.imshow(img, cmap='gray'), original_image.set_title('Original')
    final.imshow(final_image, cmap='gray'), final.set_title(f'Canny, sigma: {sigma}, low_th: {l_th}, high_th: {h_th}')


# C4
def explore_canny(images, smoothnessList, lowThresholdList, highThresholdList):
#     smoothnessList should be a list of values for sigma i.e [1.3, 2, 2.5]
#     lowThresholdList and highThresholdList should lists of values between 0 and 1 i.e [0.15, 0.2, 0.9]
    
    explorations_count = 1 
    
    for imageGT in images:
        for sigma in smoothnessList:
            for l_th in lowThresholdList:
                for h_th in highThresholdList:
#                     print(f'explorations_count: {explorations_count}, image: {imageGT}, sigma: {sigma}, l_th: {l_th}, h_th: {h_th}')
                    explorations_count += 1
                    img = cv2.imread(imageGT, cv2.IMREAD_GRAYSCALE)
                    final_image = canny(img, sigma, l_th, h_th)
            
                    [P, R, F] = evaluate_edges(final_image, img)
                    '''final = plt.subplots(1,2, sharex='col', sharey='row')[1][1]
                    final.imshow(final_image, cmap='gray'), final.set_title(f'Canny img: {imageGT}, sigma: {sigma}, low_th: {l_th}, high_th: {h_th}, P: {P}, R: {R}, F: {F}')'''
                    print(f'Canny img: {imageGT}, sigma: {sigma}, low_th: {l_th}, high_th: {h_th}, P: {P}, R: {R}, F: {F}')