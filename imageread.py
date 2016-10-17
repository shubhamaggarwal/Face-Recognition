import cv2
import scipy.misc
from PIL import Image
import numpy


#"""color to bw image using Y' = 0.299 R + 0.587 G + 0.114 B """
img=Image.open("/media/shubham/Work/Projects/FaceRec/willferrel.jpg").convert('L')
img.save("/media/shubham/Work/Projects/FaceRec/willferrelbw.jpg")


##image to numpy array
imgarray=scipy.misc.imread("/media/shubham/Work/Projects/FaceRec/willferrelbw.jpg")

##numpy array to image
img = Image.fromarray(imgarray, 'L')
img.save("/media/shubham/Work/Projects/FaceRec/willferrelarrtobw.jpg")
img.show()