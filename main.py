#imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from moviepy.editor import VideoFileClip

###
#
# Helper Functions
#
###


def grayscale(img):
    """
    Applies the Grayscale transform
    This will return an image with only one color channel
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def calcX(m,n,y):
    """
    Calculates X of an given Y on an linear function with slope m and start n as int
    """
    return int((y-n)/m)

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    """
    left_lane_m = []
    left_lane_n = []
    right_lane_m = []
    right_lane_n = []
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            # Calculate slope and start and append accoring to positive / negative to lists
            # Ignore almost horizontal slopes
            m = (y2-y1)/(x2-x1)
            n = y1-x1 * m
            
            if y1 == y2 or (m > -0.1 and m < 0.1):
                continue
            
            if (m < 0):
                left_lane_m.append(m)
                left_lane_n.append(n)
            elif (m > 0):
                right_lane_m.append(m)
                right_lane_n.append(n)
    
    # calculate mean
    left_lane_slope = sum(left_lane_m) / len(left_lane_m)
    left_lane_start = sum(left_lane_n) / len(left_lane_n)
    right_lane_slope = sum(right_lane_m) / len(right_lane_m)
    right_lane_start = sum(right_lane_n) / len(right_lane_n)
    
    # Create points for lines
    ysize = img.shape[0]
    left_lane_p1 = (calcX(left_lane_slope, left_lane_start, ysize), ysize)
    left_lane_p2 = (calcX(left_lane_slope, left_lane_start, ysize/2+55), int(ysize/2+55))
    right_lane_p1 = (calcX(right_lane_slope, right_lane_start, ysize),ysize)
    right_lane_p2 = (calcX(right_lane_slope, right_lane_start, ysize/2+55), int(ysize/2+55))
    
    # Create lines
    cv2.line(img, left_lane_p1, left_lane_p2, color, 12)
    cv2.line(img, right_lane_p1, right_lane_p2, color, 12)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

###
#
# Classification Function
#
###

def process_image(image):
    """
    This function calculates and adds the estimated lane lines into a given image by a defined pipeline
    INPUT: RGB-image
    OUTPUT: annotated RGB-image
    """

    # Size of Image
    ysize = image.shape[0]
    xsize = image.shape[1]

    ## Preprocessing
    # Grayscale Image and apply gaussian blur
    grey_img = grayscale(image)
    blurred_img = gaussian_blur(grey_img, 1)

    ## Feature extraction
    # Canny filter and mask
    canny_img = canny(blurred_img, 100, 200)

    vertices = np.array([[(100,ysize), (xsize/2 - 25, ysize/2+55), (xsize/2 + 25, ysize/2+55), (xsize, ysize)]], dtype=np.int32)
    masked_img = region_of_interest(canny_img, vertices)

    ## "Classification"
    rho = 1
    theta = 1 * np.pi / 180
    threshold = 35
    min_len = 15
    max_gap = 100
    hough_img = hough_lines(masked_img, rho, theta, threshold, min_len, max_gap)

    identified_img = weighted_img(hough_img, image)
    
    return identified_img


###
#
# Testing
#
###

## Single Image

# Possible index from 0-4
index = 0

images = os.listdir("test_images/")

#reading in the image
image = mpimg.imread('test_images/' + images[index])

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)

processed_img = process_image(image)
plt.imshow(processed_img)


## Video
output = 'test_videos_output/solidWhiteRight.mp4'
clip = VideoFileClip("test_videos/solidWhiteRight.mp4")

### 
# OR
#output = 'test_videos_output/solidYellowLeft.mp4'
#clip = VideoFileClip('test_videos/solidYellowLeft.mp4')
#
# OR
#challenge_output = 'test_videos_output/challenge.mp4'
#clip = VideoFileClip('test_videos/challenge.mp4')
###

processed_clip = clip.fl_image(process_image)
processed_clip.write_videofile(output, audio=False)