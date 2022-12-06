# # Detecting Well Locations
# We want to use `opencv` for this, since it is reliable and easy to implement. 
# It is also available in Python as well as C++. This means that the code and 
# algorithms we use can be scaled and adapted easily based on our needs.

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename


def detect():
    """
    Takes an image of a 96 well plate and usese the opencv hough circle algorithm to detect the locations of wells.
    Returns an array with each row corresponding to a single circle, with the format [center_x, center_y, radius]

    Returns:
        circles(np.ndarray, dtype='uint16'): Array of [center_x, center_y, radius] for all the cetected circles
    """    
    # ## Image Processing
    # Read in the image
    img_path = askopenfilename(initialdir=os.getcwd(), title="Select a 96 Well Plate HyperSpectral Image")
    img = cv2.imread(img_path)

    # Resize the image (3x the width) to make the wells circular
    resized = cv2.resize(img, (img.shape[1]*3, img.shape[0]))

    # Add blur and convert it to grayscale
    resized = cv2.medianBlur(resized,5)
    gray = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)


    # ## Circle Detection using `HoughCircles`
    # Hough Circle Transform
    circles = cv2.HoughCircles(
        image=gray,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=30,
        maxRadius=50
    )

    # Convert the circles array to 16-bit integer (not 8 bit because we want values greater than 255)
    circles = np.uint16(np.around(circles)) # we use circles[0] since the circles array has
                                            # an extra set of [] outside of the relevant 
                                            # i.e. [ [data] ]


    # ## Drawing the Circles
    # reset cimg to be the original color image (useful for running the loop multiple times)
    drawing = resized.copy()

    # Draw each circle and its center
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(drawing,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(drawing,(i[0],i[1]),2,(0,0,255),3)


    # Display the circles
    plt.imshow(drawing)
    plt.show()

    return circles

if __name__ == "__main__":
    # To use this function in another file:
    # import well_detection as wells
    # circles = wells.detect()
    circles = detect()