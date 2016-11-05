from PIL import Image
import scipy.misc
import numpy
import cv2
import time
import math
import pylab

timer=time.time()


no_of_people=40
total_no_of_photos=10
Xtrain=0.7
no_of_tests=int(Xtrain*total_no_of_photos)
##need size to initialize input image matrix################################################
img=scipy.misc.imread("/media/shubham/Work/Projects/FaceRec/att_faces/s1/1.pgm")
size=img.shape
#size=(216,216)
##################k*n*n initial image matrix###################################################
input_image_matrix=numpy.zeros(shape=((no_of_tests*no_of_people),size[0]*size[1]),dtype=numpy.float32)

##################################reading all images############################################
no_of_rows_input_matrix=0
for i in range(1,no_of_people+1):
    for j in range(1,no_of_tests+1):
        imgmatrix=scipy.misc.imread("/media/shubham/Work/Projects/FaceRec/att_faces/s"+str(i)+"/"+str(j)+".pgm")
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
e=e[::-1]
###############converting eigen vector in original dimensions into image########################
eigen_image=numpy.zeros(shape=(size[0],size[1]),dtype=numpy.float32)
###############Displaying the extracted eigen images#############################################
figure1=pylab.figure()
pylab.gray()
t=0
for i in range(EV_in_original_dimensions.shape[0]):
    pixel=0
    for j in range(size[0]):
        for k in range(size[1]):
            eigen_image[j][k]=EV_in_original_dimensions[i][pixel]
            pixel+=1
    figure1.add_subplot(EV_in_original_dimensions.shape[0]/4+1,4,t+1)
    t+=1
    pylab.imshow(eigen_image)
pylab.show()
####################Calculating weight matrix for each Eigenvector/Eigenface######################
Weights= numpy.dot(EV_in_original_dimensions,normalised_image_matrix.transpose())
Weights=Weights.transpose()

########################TRAINING ENDS HERE AND TESTING BEGINS#################################
################################################################################################
################################################################################################

#####################Classifying the given image into one of the test sets######################
acc_cnt=0
total_cnt=0
for pp in range(1,no_of_people+1):
    for tt in range(no_of_tests+1,total_no_of_photos+1):
        read_img=scipy.misc.imread("/media/shubham/Work/Projects/FaceRec/att_faces/s"+str(pp)+"/"+str(tt)+".pgm")
        img_col=numpy.array(read_img,dtype="float32").flatten()

        img_col-=average_image_array #### Subtracting the mean image from the image to be classified.

        S=numpy.dot(EV_in_original_dimensions,img_col) #### Calculating weights for the given image to represent it in terms of Eigenfaces

        diff=numpy.zeros(shape=(Weights.shape[0],Weights.shape[1]),dtype=numpy.float32)
        ####calculating differnce matrix for the image to be classified        
        for i in range(no_of_people*no_of_tests):
            for j in range(no_of_people*no_of_tests):
                diff[i][j]=Weights[i][j]-S[j]

        normal=numpy.linalg.norm(diff,axis=1)
        ####calculating the nearest neighbour
        sort_indices=numpy.argsort(normal)
        closest_face_id=numpy.argmin(normal)
        
        print("image location = "+"/media/shubham/Work/Projects/FaceRec/att_faces/s"+str(pp)+"/"+str(tt)+".pgm")
        print 'PREDICTED CANDIDATE NUMBER = ',closest_face_id/no_of_tests+1
        print 'RESULT = ',str((closest_face_id/no_of_tests+1)==pp)
        if (closest_face_id/no_of_tests+1)==pp:
            acc_cnt+=1
        total_cnt+=1
accuracy_score=acc_cnt/float(total_cnt)*100
print 'Accuracy Score = ',accuracy_score
