from PIL import Image
import scipy.misc
import numpy
import cv2
import time
import math
import pylab

timer=time.time()


no_of_people=1
no_of_tests=20

##need size to initialize input image matrix################################################
img=scipy.misc.imread("/home/shubham/Desktop/gwb_cropped/1.jpg")
size=img.shape

##################k*n*n initial image matrix###################################################
input_image_matrix=numpy.zeros(shape=((no_of_tests*no_of_people),size[0]*size[1]),dtype=numpy.float32)

##################################reading all images############################################
no_of_rows_input_matrix=0
for i in range(1,no_of_people+1):
    for j in range(1,no_of_tests+1):
        imgmatrix=scipy.misc.imread("/home/shubham/Desktop/gwb_cropped/"+str(j)+".jpg")
        imgarray=imgmatrix.flatten() ##2 dimensional image to 1 dimensional 
        input_image_matrix[no_of_rows_input_matrix]=imgarray
        no_of_rows_input_matrix+=1

#######################calculating average of all images#########################################
average_image_array=numpy.zeros(shape=size[0]*size[1],dtype=numpy.float32)

for i in range(no_of_rows_input_matrix):
    for j in range(size[0]*size[1]):
        average_image_array[j]+=float(input_image_matrix[i][j])

average_image_array/=float(no_of_rows_input_matrix)

#################converting average image from 1D to 2D#########################################
average_image=numpy.zeros(shape=(size[0],size[1]),dtype=numpy.float32)
k=0
for i in range(size[0]):
    for j in range(size[1]):
        average_image[i][j]=(average_image_array[k])
        k+=1 

########################Displaying average image###############################################
pylab.figure()
pylab.gray()
pylab.imshow(average_image)
pylab.show()

#######################Normalising input image array#############################################
normalised_image_matrix=numpy.zeros(shape=input_image_matrix.shape,dtype=numpy.float32)
for i in range(no_of_rows_input_matrix):
    for j in range(size[0]*size[1]):
        normalised_image_matrix[i][j]=((input_image_matrix[i][j])-(average_image_array[j]))


############Calculating the covariance matrix for dimensionality reduction########################
transpose_image_matrix = normalised_image_matrix.transpose()
covariance_matrix=numpy.dot(normalised_image_matrix,transpose_image_matrix)

###################Calculating eigen vector and eigenvalues using this numpy function#############
e,EV=numpy.linalg.eigh(covariance_matrix)

#######################Transfer back reduced EV to original dimensions##########################
EV_in_original_dimensions=(numpy.dot(normalised_image_matrix.transpose(),EV))
EV_in_original_dimensions=EV_in_original_dimensions.transpose()

##########Bringing out the best features to the top by reverse sorting the array################
EV_in_original_dimensions=EV_in_original_dimensions[::-1]

###############converting eigen vector in original dimensions into image########################
eigen_image=numpy.zeros(shape=(size[0],size[1]),dtype=numpy.float32)
k=0
for i in range(size[0]):
    for j in range(size[1]):
        eigen_image[i][j]=(EV_in_original_dimensions[0][k])
        k+=1

###############Displaying the extracted eigen image#############################################
pylab.figure()
pylab.gray()
pylab.imshow(eigen_image)
pylab.show()
