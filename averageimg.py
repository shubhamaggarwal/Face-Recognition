import cv2
import scipy.misc
from PIL import Image
import numpy

#"""color to bw image using Y' = 0.299 R + 0.587 G + 0.114 B """
#img=Image.open("/media/shubham/Work/Projects/FaceRec/willferrel.jpg").convert('L')
#img.save("/media/shubham/Work/Projects/FaceRec/willferrelbw.jpg")


##image to numpy array
imgarray=scipy.misc.imread("/media/shubham/Work/Projects/FaceRec/att_faces/s1/1.pgm")
imgarray2=scipy.misc.imread("/media/shubham/Work/Projects/FaceRec/att_faces/s2/1.pgm")

##size of image
sizeimg=imgarray.shape


###linear flattening of 2d image
ans=numpy.zeros(shape=sizeimg[0]*sizeimg[1],dtype=int)

k=0
for i in range(0,sizeimg[0]):
    for j in range(sizeimg[1]):
        ans[k]+=int(imgarray[i][j])
        ans[k]+=int(imgarray2[i][j])
        ans[k]/=2
        k+=1

##average image using mean
avgimg=numpy.zeros(sizeimg,dtype=numpy.uint8)

d=0
for i in range(sizeimg[0]):
    for j in range(sizeimg[1]):
        avgimg[i][j]=numpy.uint8(ans[d])
        d+=1

##numpy array to image for average image
img = Image.fromarray(avgimg, 'L')
img.save("/home/shubham/Desktop/1.pgm")
img.show()