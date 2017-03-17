import numpy as np
import cv2
import os
import pickle
import time
from skimage.feature import hog
from scipy.ndimage.measurements import label
from config import *
from svm_classify import *
from moviepy.editor import VideoFileClip
from IPython.display import HTML


class Car():
	def __init__(self, heat_map, accumulate, occlude = False):
		self.heat_map = [heat_map]
		self.accumulate = accumulate
		self.occlude = occlude
		self.cal_car_info()

	def cal_car_info(self):
		sum_heat_map = np.sum(self.heat_map, axis = 0)
		nonzero = (sum_heat_map > 0).nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		
		# Define a bounding box based on min/max x and y
		self.bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		self.label = np.zeros_like(self.heat_map[-1])
		self.label[np.min(nonzeroy):np.max(nonzeroy)+1,np.min(nonzerox):np.max(nonzerox)+1] = 1
		#self.center = np.mean(self.bbox, axis =0)
		self.new_heat_map = np.zeros_like(self.heat_map[-1])
	

	def draw_boxes(self, img):
		# Draw the box on the image
		copy_img = np.copy(img)
		cv2.rectangle(copy_img, self.bbox[0], self.bbox[1], (0,0,255), 3)
		# Return the image
		return copy_img

	def add_box(self,box):
		self.new_heat_map[box[0][1]:box[1][1]+1,box[0][0]:box[1][0]+1] += 1

	def not_two_cars(self):
		labels = label(self.new_heat_map)
		if labels[1] >= 2:
			return False
		else:
			return True

	def update(self):
		if (self.occlude == False) or (self.not_two_cars()) :
			self.heat_map.append(self.new_heat_map)
			if len(self.heat_map) > self.accumulate:
				self.heat_map.pop(0)
			self.cal_car_info()
			return None

		else:
			labels = label(self.new_heat_map)
			cars = []
			for i in range(1, labels[1]+1):
				heat_map = np.copy(self.new_heat_map)
				heat_map[labels[0] != i] = 0

				cars.append(Car(heat_map, self.accumulate, occlude = False) )
			return cars

class car_detector():
	def __init__(self, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, roi, scale, accumulate, thresh):

		self.svc = svc
		self.X_scaler = X_scaler
		self.orient = orient
		self.pix_per_cell = pix_per_cell
		self.cell_per_block = cell_per_block
		self.spatial_size = spatial_size
		self.hist_bins = hist_bins
		self.accumulate = accumulate
		self.heat_maps = []
		self.last_heat_map = None
		self.found_car = False 
		self.cars = []
		self.occluded_cars = []
		self.roi = roi
		self.scale = scale
		self.thresh = thresh

	# Define a single function that can extract features using hog sub-sampling and make predictions
	def find_cars(self, img, roi=(None,None) , scale=1):
	    
		img_boxes = []
		heatmap = np.zeros_like(img[:,:,0])
		
		if roi[0] == None:
			ystart, ystop = 380, 636
		else:
			ystart, ystop = roi[0]
		if roi[1] == None:
			img_tosearch = img[ystart:ystop,:,:]
		else:
			xstart, xstop = roi[1]
			img_tosearch = img[ystart:ystop,xstart:xstop,:]

		#img_tosearch = (np.sqrt(img_tosearch.astype(np.float32)/255)*255).astype(np.uint8)
		ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
		if scale != 1:
			imshape = ctrans_tosearch.shape
			ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
		    
		ch1 = ctrans_tosearch[:,:,0]
		ch2 = ctrans_tosearch[:,:,1]
		ch3 = ctrans_tosearch[:,:,2]

		# Define blocks and steps as above
		nxblocks = (ch1.shape[1] // self.pix_per_cell)-1
		nyblocks = (ch1.shape[0] // self.pix_per_cell)-1 
		nfeat_per_block = self.orient*self.cell_per_block**2
		# 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
		window = 64
		nblocks_per_window = (window // self.pix_per_cell)-1 
		cells_per_step = 2  # Instead of overlap, define how many cells to step
		nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
		nysteps = (nyblocks - nblocks_per_window) // cells_per_step

		# Compute individual channel HOG features for the entire image
		hog1 = get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
		hog2 = get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
		hog3 = get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)

		for xb in range(nxsteps):
			for yb in range(nysteps):
				#count += 1

				ypos = yb*cells_per_step
				xpos = xb*cells_per_step
				# Extract HOG for this patch
				hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
				hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
				hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
				hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

				xleft = xpos*pix_per_cell
				ytop = ypos*pix_per_cell

				# Extract the image patch
				subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

				# Get color features
				spatial_features = bin_spatial(subimg, size=self.spatial_size)
				hist_features = color_hist(subimg, nbins=self.hist_bins)

				# Scale features and make a prediction
				test_features = self.X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
				#test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
				test_prediction = self.svc.predict(test_features)

				if test_prediction == 1:
					#print('Car Detected')
					xbox_left = np.int(xleft*scale)
					ytop_draw = np.int(ytop*scale)
					win_draw = np.int(window*scale)
					#cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)

					img_boxes.append( ((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)) )
					heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left:xbox_left+win_draw] += 1
		
		return img_boxes, heatmap

	def intersect(self, box, car):
		label = np.zeros_like(car.label)
		label[box[0][1]:box[1][1]+1, box[0][0]:box[1][0]+1] = 1
		return np.sum(label*car.label)


	def box_of_car(self,box, cars):
		match_list = []
		for i in range(len(cars)):
			#print(box)
			#print(cars[i].label.shape)
			if np.max(cars[i].label[box[0][1]:box[1][1]+1, box[0][0]:box[1][0]+1]) >=1:
				match_list.append(i)
		if len(match_list) == 0:
			return None
		elif len(match_list) == 1:
			return match_list[0]
		else:
			match_list = np.array(match_list)
			intersects = [self.intersect(box, cars[match_list[i]]) for i in range(len(match_list))]
			#return match_list[np.argsort(intersects)][::-1]# sorted(a, reverse=True) 
			return match_list[np.argmax(intersects)]


	def single_image_process(self,img, roi, scale, single_thresh = 2):
		assert len(self.cars) == 0, "Only for the case no car has been found"

		heat_map = np.zeros_like(img[:,:,0])
		for i in range(len(scale)):
			_, hm = self.find_cars(img, roi[i], scale[i])
			heat_map += hm
		#heat_map[heat_map <= 1] = 0
		if np.max(heat_map) > 0:
			labels = label(heat_map)
			for car_number in range(1, labels[1]+1):
				car_heat_map = np.copy(heat_map)
				car_heat_map[labels[0] != car_number] = 0
				if np.max(car_heat_map) >= single_thresh:
					car = Car(car_heat_map, self.accumulate)
					self.cars.append(car)
					self.found_car = True
					img = car.draw_boxes(img)
		return img


	
	def video_car_process(self,frame, roi, scale):
		img = np.copy(frame)
		img_boxes = []
		total_heat_map = np.zeros_like(img[:,:,0])
		for i in range(len(scale)):
			ibs, hm = self.find_cars(img, roi[i], scale[i])
			img_boxes += ibs
			total_heat_map += hm
		if len(self.cars) == 0:
			#thresh = self.accumulate
			self.heat_maps.append(total_heat_map)
			if len(self.heat_maps) > self.accumulate:
				self.heat_maps.pop(0)
				sum_heat_map = np.sum(self.heat_maps, axis=0)
				if np.max(sum_heat_map) > self.thresh: 
					labels = label(sum_heat_map)
					for car_number in range(1, labels[1]+1):
						car_heat_map = np.copy(sum_heat_map)
						car_heat_map[labels[0] != car_number] = 0
						if np.max(car_heat_map) > self.thresh:
							car = Car(car_heat_map, self.accumulate)
							self.cars.append(car)
							img = car.draw_boxes(img)

		else:
			if len(img_boxes) == 0:
				#self.found_car = False
				self.cars = []
			else:
				heat_map = np.zeros_like(img[:,:,0])
				car_ind = np.zeros(len(self.cars))
				for i in range(len(img_boxes)):
					#print(img_boxes[i])
					if self.box_of_car(img_boxes[i], self.cars) == None:
						heat_map[img_boxes[i][0][1]:img_boxes[i][1][1]+1,img_boxes[i][0][0]:img_boxes[i][1][0]+1] += 1
					else:
						ind = self.box_of_car(img_boxes[i], self.cars)
						car_ind[ind] = 1
						self.cars[ind].add_box(img_boxes[i])
					

				lost_cars = [self.cars[i] for i in range(len(car_ind)) if car_ind[i] == 0]
				self.cars = [self.cars[i] for i in range(len(car_ind)) if car_ind[i] == 1]
				

				
				if len(self.cars) > 0:
					if len(lost_cars) > 0:
						for i in range(len(lost_cars)):
							intersects = [self.intersect(lost_cars[i].bbox, self.cars[j]) for j in range(len(self.cars)) ]
							if np.max(intersects) > 0:
								self.cars[np.argmax(intersects)].occlude = True

					new_cars = []
					for i in range(len(self.cars)):
						temp = self.cars[i].update()
						if temp == None:
							new_cars.append(self.cars[i])
						else:
							new_cars += temp
					self.cars = new_cars
					for i in range(len(self.cars)):
						img = self.cars[i].draw_boxes(img)
				
				#thresh = self.accumulate
				self.heat_maps.append(heat_map)
				if len(self.heat_maps) > self.accumulate:
					self.heat_maps.pop(0)
					sum_heat_map = np.sum(self.heat_maps, axis=0)
					if np.max(sum_heat_map) > self.thresh: 
						labels = label(sum_heat_map)
						for car_number in range(1, labels[1]+1):
							car_heat_map = np.copy(sum_heat_map)
							car_heat_map[labels[0] != car_number] = 0
							if np.max(car_heat_map) > self.thresh:
								car = Car(car_heat_map, self.accumulate)
								self.cars.append(car)
								img = car.draw_boxes(img)

			
		return img



	def process_image(self,img):

		return self.video_car_process(img, self.roi, self.scale)



if __name__ == '__main__':
	
	with open("./svc_pickle/svc_pickle.p", 'rb') as f:
		svc_dict = pickle.load(f)
		svc = svc_dict['svc']
		X_scaler = svc_dict['scaler']

	roi = ( (None,None),(None,None))
	scale = (1,1.25)
	accumulate = 10
	thresh = 30


	detector = car_detector(svc=svc, X_scaler=X_scaler, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
		spatial_size=spatial_size, hist_bins = hist_bins, roi = roi, scale=scale, accumulate=accumulate, thresh=thresh )

	test_input = './test_videos/project_video.mp4'
	test_output = './test_videos/project_video_result.mp4'

	t = time.time()
	clip = VideoFileClip(test_input)#.subclip(4,10)
	test_clip = clip.fl_image(detector.process_image)
	test_clip.write_videofile(test_output, audio = False)
	print('Finish detection. Using time: ', round(time.time() - t,2))