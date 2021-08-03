import Augmentor
import pickle 
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
import pandas as pd 
from collections import Counter

training_file = '/Users/Glenwood/autonomous_vehicle/projects/Traffic_Signs/traffic-signs-data/train.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
X_additional=[]
y_additional=[]

X_train, y_train = train['features'], train['labels']
images = X_train
labels= y_train

# create a list of list of images to feed the Augmentator pipeline
list_of_list_images=[[x]for x in images]

# prepare the lists to enumerate each and calculate each class population
list_of_list_images_category=[]
list_of_train_category=[]

#initialize each list
for x in range(43):
	list_of_train_category.append([])
	list_of_list_images_category.append([])

# append images and label to class specific lists
for idx,x in enumerate (images , start =0):
	list_of_list_images_category[labels[idx]].append([x])
	list_of_train_category[labels[idx]].append(labels[idx])

	# path_to_data
	p = Augmentor.Pipeline("/Users/Glenwood/autonomous_vehicle/projects/Traffic_Signs/traffic-signs_augmented/origim")

# this loop will aumgment the population of each class to even it out
for x in range(43): 

	# count each class population
	ind=len(list_of_list_images_category[x])

	# working on class x 
	list_of_list_images=list_of_list_images_category[x]
	label_list=list_of_train_category[x]

	p =Augmentor.DataPipeline(list_of_list_images ,label_list) 

	#processing each image through a series of random transform
	p.rotate(probability=0.2, max_left_rotation=10, max_right_rotation=10)
	p.resize(probability=1.0, width=32, height=32, resample_filter=u'NEAREST')
	p.skew_tilt(probability=0.7,magnitude=0.2)
	p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
	p.shear(probability=0.1,max_shear_left= 5,max_shear_right= 5)
	p.random_color(probability=0.5, min_factor=0.2, max_factor=0.8)	
	p.random_brightness(0.7, 0.2, 0.7)
	p.random_contrast(0.7, 0.2, 0.7)
	
	# only augment each class to raise each final class population to 2012 individuals
	X, y=p.sample(2012-ind)
	
	X_add=[x[0] for x in X]
	y_add= y

	# append processed images and labels to lists 
	X_additional+=X_add
	y_additional+=y_add



X_pickle=np.array(X_additional)
y_pickle=np.array(y_additional)

# checking that erything went right
"""print("X_pickle.shape",X_pickle.shape)
print("X_pickle[-1].shape",X_pickle[-1].shape)
print("y_pickle.shape", y_pickle.shape)
"""

# path to storage 
training_file_augmented = '/Users/Glenwood/autonomous_vehicle/projects/Traffic_Signs/traffic-signs-data/train_augmented_p1.p'

#dictionary for pickling 
augmented={}
augmented['features'], augmented['labels']=X_pickle , y_pickle
print ("len(augmented['features'])",len(augmented["features"]))
print("len(augmented['labels'])",len(augmented['labels']))

# pickling 
with open(training_file_augmented, mode='wb') as g:
    pickle.dump(augmented,g)



