import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from __future__ import division



def disparity_matrix(window_size,left_image,right_image):
    length, width, height = right_image.shape
    disparity_matrix = np.zeros((left_image.shape[0],left_image.shape[1]-window_size))
    for i in range(length):
        for j in range(width - window_size):
            temp = []
            for k in range(j,j + window_size):
                temp.append(np.linalg.norm(right_image[i][j] - left_image[i][k]))
            disparity_matrix[i][j] = temp.index(min(temp))
    return disparity_matrix



#GMM clustering using the Expectation Maximization algorithm
def Estep(data, gaussian, num_gaussians, num_samples, mean, std_dev, weights, log_likelihood):
    logL = []
    gaussian = [[] for i in range(num_gaussians)]    
    for k in range(num_gaussians):
        for i in range(num_samples):
            denominator = 1/np.sqrt(6.28319*std_dev[k]*std_dev[k])
            exponent = np.exp( -(data[i]-mean[k])*(data[i]-mean[k]) / (2.0*std_dev[k]*std_dev[k]) )
            gaussian[k].append(denominator*exponent*weights[k])
    gaussian = np.array(gaussian, dtype = 'float64')
    for i in range(len(gaussian)):
        gaussian[i] = gaussian[i]/np.sum(gaussian[i])
    for i in range(num_samples):
        temp = 0.0
        for k in range(num_gaussians):
            temp = temp + gaussian[k][i]
        logL.append(np.log(temp))
    logL = np.array(logL, dtype = 'float64')
    log_likelihood.append(np.sum(logL))
    return gaussian


def Mstep(data, gaussian, num_gaussians, num_samples, mean, std_dev, weights, log_likelihood):
    for k in range(num_gaussians):
        numerator = 0.0
        deviations = 0.0
        for i in range(num_samples):
            numerator += (gaussian[k][i]*data[i])
            deviations += (data[i] - mean[k]) * (data[i] - mean[k]) * (gaussian[k][i])
        denominator = np.sum(gaussian[k])
        mean[k] = numerator/denominator
        std_dev[k] = np.sqrt(deviations/denominator)
        weights[k] = sum(gaussian[k])/len(gaussian[k]) 
    return mean,std_dev,weights


def ExpectationMaximization(disparity_matrix, num_iterations, num_gaussians = 4):   
    num_gaussians = 4
    gaussian = [[] for i in range(num_gaussians)]
    mean = []
    std_dev = []
    weights = []
    log_likelihood = []
    centroids = [31,4,10,21]
    rows, cols = disparity_matrix.shape
    data = np.ndarray.flatten(disparity_matrix)
    num_samples = data.shape[0]
    for i in range(num_gaussians):
        mean.append(centroids[i])
        std_dev.append(2)
        weights.append(1/num_gaussians)
    mean = np.array(mean, dtype = 'float64')
    std_dev = np.array(std_dev, dtype = 'float64')
    weights = np.array(weights, dtype = 'float64')
    for i in range(num_iterations):
        gaussian = Estep(data, gaussian, num_gaussians, num_samples, mean, std_dev, weights, log_likelihood)
        mean,std_dev,weights = Mstep(data, gaussian, num_gaussians, num_samples, mean, std_dev, weights, log_likelihood)
    gaussian = gaussian.T
    for i in range(num_samples):
        for j in range(num_gaussians):
            gaussian[i][j] = abs(data[i] - mean[j])
        gaussian[i] = gaussian[i]/np.sum(gaussian[i])
    for i in range(num_samples):
        gaussian[i] = 1 - gaussian[i]
    return gaussian, mean


def MRF(data):
    num_samples = data.shape[0]
    result = np.array(data.copy(), dtype = 'float64')  
    for i in range(num_samples):
        for n in range(num_gaussians):
            probability = data[i,n]
            if i%390 == 0:
                if i == 0:
                    probability = probability*data[i+1,n]*data[i+390,n]*data[i+390+1,n]
                elif i == num_samples - 390:
                    probability = probability*data[i+1,n]*data[i-390,n]*data[i-390+1,n]
                else:
                    probability = probability*data[i+1,n]*data[i-390,n]*data[i-390+1,n]*data[i+390,n]*data[i+390+1,n]
            elif i%390 == 390 - 1:
                if i == 390 - 1:
                    probability = probability*data[i-1,n]*data[i+390,n]*data[i+390-1,n]
                elif i == num_samples - 1:
                    probability = probability*probability*data[i-1,n]*data[i-390,n]*data[i-390-1,n]
                else:
                    probability = probability*data[i-1,n]*data[i-390,n]*data[i-390-1,n]*data[i+390,n]*data[i+390-1,n]
            else:
                if i < 390 - 1:
                    probability = probability*data[i+1,n]*data[i-1,n]*data[i+390,n]*data[i+390+1,n]*data[i+390-1,n]
                elif i > num_samples - 390:
                    probability = probability*data[i+1,n]*data[i-1,n]*data[i-390,n]*data[i-390+1,n]*data[i-390-1,n]
                else:
                    probability = probability*data[i+1,n]*data[i-1,n]*data[i+390,n]*data[i+390+1,n]*data[i+390-1,n]*data[i-390,n]*data[i-390+1,n]*data[i-390-1,n]
            result[i,n] = probability + 0.000000005
        result[i] = result[i]/np.sum(result[i]) 
    return result


#reading the data
im0 = Image.open('im0.ppm')
im8 = Image.open('im8.ppm')
left_image = np.array(im0, dtype="int32")
right_image = np.array(im8, dtype="int32")

#disparity matrix
disparity_matrix = disparity_matrix(40,left_image,right_image)
length, width = disparity_matrix.shape

#Obtaining probability of being in each cluster for each data point
num_gaussians = 4
GMMresults, mean = ExpectationMaximization(disparity_matrix, 6)
probabilities = np.array(GMMresults, dtype = 'float64')

#performing MRF smoothing
#number of samples is 5
smoothed =  probabilities.copy()
for i in range(5):
    smoothed = MRF(probabilities)
    probabilities = smoothed

#recovering the depth map
indexes = []
for i in range(len(smoothed)):
    values = list(smoothed[i])
    indexes.append(values.index(max(values)))

depth = np.reshape(np.array(indexes, dtype = 'float64'), (381,390))
for i in range(length):
    for j in range(width):
        depth[i][j] = mean[depth[i][j]]


plt.imshow(depth, aspect = 'auto', cmap = 'gray', interpolation = 'nearest')
plt.show()


