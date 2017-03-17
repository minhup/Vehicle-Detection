import numpy as np
import cv2
import glob
import os
import pickle
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  
from skimage.feature import hog
from config import *


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False,  
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    
    return np.hstack( (color1, color2, color3) )

# Define a function to compute color histogram features 
def color_hist(img, nbins=32):#, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)#, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)#, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)#, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='YCrCb', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        #image = mpimg.imread(file)
        image = cv2.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space == 'RGB':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
    
if __name__ == '__main__':

	cars = glob.glob(car_folder)
	notcars = glob.glob(notcar_folder)
	print('Num cars: ',len(cars))
	print('Num notcars: ', len(notcars))


	car_features = extract_features(cars, color_space=color_space,
									spatial_size=spatial_size, hist_bins=hist_bins, 
									orient=orient, pix_per_cell=pix_per_cell, 
									cell_per_block=cell_per_block, 
									hog_channel=hog_channel, spatial_feat=spatial_feat, 
									hist_feat=hist_feat, hog_feat=hog_feat)

	notcar_features = extract_features(notcars, color_space=color_space, 
									spatial_size=spatial_size, hist_bins=hist_bins, 
									orient=orient, pix_per_cell=pix_per_cell, 
									cell_per_block=cell_per_block, 
									hog_channel=hog_channel, spatial_feat=spatial_feat, 
									hist_feat=hist_feat, hog_feat=hog_feat)

	X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
	#Fit a per-column scaler
	X_scaler = StandardScaler().fit(X)
	# Apply the scaler to X
	scaled_X = X_scaler.transform(X)
	# Define the labels vector
	y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

	# Split up data into randomized training and test sets
	#and_state = np.random.randint(0, 100)
	rand_state = 10
	X_train, X_test, y_train, y_test = train_test_split(
	    scaled_X, y, test_size=0.1, random_state=rand_state)
	print('Train on {0} samples'.format(len(y_train)))
	print('Using:',orient,'orientations',pix_per_cell,
	    'pixels per cell and', cell_per_block,'cells per block')
	print('Feature vector length:', len(X_train[0]))
	# Use a linear SVC 
	svc = LinearSVC()
	# Check the training time for the SVC
	t=time.time()
	svc.fit(X_train, y_train)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to train SVC...')
	# Check the score of the SVC
	print('Test on {0} samples'.format(len(y_test)))
	print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
	
	svc_dict = {'svc':svc,
				'scaler': X_scaler,
				'color_space': color_space,
				'orient':orient,
				'pix_per_cell':pix_per_cell,
				'cell_per_block': cell_per_block,
				'hog_channel':hog_channel,
				'spatial_size':spatial_size,
				'hist_bins':hist_bins,
				'spatial_feat':spatial_feat,
				'hist_feat':hist_feat,
				'hog_feat':hog_feat}


	pickle.dump(svc_dict, open("svc_pickle.p", "wb" ))