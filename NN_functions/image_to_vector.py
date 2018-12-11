import numpy as np
import cv2
import imageio

# convert image to a numpy array (550, 1140, 3) - length, height/width, 3
image_path = 'https://www.gettyimages.ie/gi-resources/images/Homepage/Hero/UK/CMS_Creative_164657191_Kingfisher.jpg'
im = imageio.imread(image_path)
# print(im.shape)
# print(im)

def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    ### START CODE HERE ### (≈ 1 line of code)
    v = image.reshape((image.shape[0])* (image.shape[1])*(image.shape[2]),1)
    ### END CODE HERE ###
    return v

print(image2vector(im))
print(image2vector(im).shape)
