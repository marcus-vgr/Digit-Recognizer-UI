import numpy as np
import cv2
import NeuralNetwork as NN
from scipy import ndimage
import cv2

# Loading model
path_model = 'Digit_Recognizer.model'
model = NN.Model.load('Digit_Recognizer.model')

"""
In order to use our trained model, the images to be predicted have to must processed in the same way as the MNIST
dataset. Therefore, for each handwritten image we have to:

1) Resize to 28x28 pixels
2) Put the number in a 20x20 box inside the 28x28 image
3) Centralize the number in the box

Those tasks are done in the ConvertImage_MNIST function, that receives as input the image drawn by the user
"""
def ConvertImage_MNIST(coordinates_img):

    # Given the coordinates of the number, we create a 2D array that will represent our number
    image = np.zeros((512,512))
    for point in coordinates_img:
        for i in range(15):
            for j in range(15):
                
                if 0 <= point[1]+i <= 512 and 0 <= point[0]+j <= 512:
                    image[point[1]+i, point[0]+j] = 255 

                if 0 <= point[1]-i <= 512 and 0 <= point[0]+j <= 512:
                    image[point[1]-i, point[0]+j] = 255 

                if 0 <= point[1]+i <= 512 and 0 <= point[0]-j <= 512:
                    image[point[1]+i, point[0]-j] = 255 

                if 0 <= point[1]-i <= 512 and 0 <= point[0]-j <= 512:
                    image[point[1]-i, point[0]-j] = 255 

    
    # Resize the image to 28x28 pixels
    image = cv2.resize(image, (28,28))

    # Delete every row and column that does not contain information about the number, i.e, 
    # we transform the image to be only the 'block' that contains the number.
    while np.sum(image[0]) == 0:
        image = image[1:]
    while np.sum(image[:,0]) == 0:
        image = np.delete(image,0,1)
    while np.sum(image[-1]) == 0:
        image = image[:-1]
    while np.sum(image[:,-1]) == 0:
        image = np.delete(image,-1,1)

    # Resize the image to certify that it does no exceed the 20x20 box
    rows,cols = image.shape
    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        image = cv2.resize(image, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        image = cv2.resize(image, (cols, rows))

    # Now we add zeros in the array until the image becomes again a 28x28 2D array
    colsPadding = (int(np.ceil((28-cols)/2.0)),int(np.floor((28-cols)/2.0)))
    rowsPadding = (int(np.ceil((28-rows)/2.0)),int(np.floor((28-rows)/2.0)))
    image = np.lib.pad(image,(rowsPadding,colsPadding),'constant')

    
    # Finally, we should centralize the image. To do this, we first obtain the coordinates of the center-of-mass of the
    # image using the ndimage from the scipy library, and then obtain what would be the shift to centralize the image
    rows,cols = image.shape
    cy,cx = ndimage.measurements.center_of_mass(image)
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    # Then the image can be centralized using the cv2 warpAffine function.
    M = np.float32([[1,0,shiftx],[0,1,shifty]])
    image = cv2.warpAffine(image, M, (cols,rows))    
    
    return image

"""
The output of our prediction is given as a plot. Getting user image, performing the forward propagation in the
Neural Network and showing the predictions is a job for the DigitRecognizer function.
"""
def DigitRecognizer(coordinates_img):

    # Convert image to MNIST style.
    image_resized = ConvertImage_MNIST(np.array(coordinates_img))
    
    # Rescale and reshape the image to use as input in the Neural Network
    image_pred = ((image_resized.reshape(1,-1) - 127.5)/ 127.5)
    
    # Perform a forward propagation in our already trained Neural Network and get the predictions
    pred = model.forward(image_pred, training=False)

    return pred.flatten()