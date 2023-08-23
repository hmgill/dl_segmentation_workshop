"""

Symposium Workshop | Aug 23 2023
'Revolutionizing Medical Imaging with Computer Vision and Artificial Intelligence'
Amity University - IUPUI

"""

import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# our custom modules
from unet import *
from unet_metrics import *



# SETTINGS, VARS, HYPERPARAMETERS
DATASET_IMAGE_PATH = ""
DATASET_MAP_PATH = ""

RESIZE = 512
CHANNELS = 3
INPUT_SHAPE = (RESIZE, RESIZE, CHANNELS)

RANDOM_STATE = 0
TEST_SIZE = 0.2

BATCH_SIZE = 10
EPOCHS = 150
MODEL_NAME = "exudate_segmentation"









class WorkshopPipeline():


    def __init__(self):
        pass


    def match_images_and_maps(self):
        exudate_img_path = DATASET_IMAGE_PATH 
        exudate_map_path = DATASET_MAP_PATH 

        matched = []
        for img in os.listdir(exudate_img_path):
            img_id = img.split(".")[0]
            for segmap in os.listdir(exudate_map_path):
                segmap_id = segmap.split(".")[0]
                
                if img_id == segmap_id:
                    matched.append(
                        {
                            "img": os.path.join(exudate_img_path, img),
                            "map": os.path.join(exudate_map_path, segmap)
                        }
                    )
        
        return matched
                


    def create_generator(self, data):

        generator = ImageDataGenerator()
        #    dict( rescale = 1./255 )
        #)

        generator.fit(
            data,
            seed = RANDOM_STATE
        )

        final_generator = generator.flow(
            data,
            seed = RANDOM_STATE,
            shuffle = False,
            batch_size = BATCH_SIZE
        )

        return final_generator

    


    
    def read_image(self, img_path, new_size = 512, num_channels = 3, use_green_channel = False):

        # read image as color or grayscale
        img = cv2.imread(img_path)
        
        # resize
        img = cv2.resize(
            img,
            (new_size, new_size),
            interpolation = cv2.INTER_NEAREST
        )

        # return as color or grayscale image
        if num_channels != 3:
            if use_green_channel:
                img = img[:,:,1]
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis = -1)
        
        return img




    
    def run_workshop(self):

        # get dataset
        dataset = self.match_images_and_maps()


        
        # read dataset fundus images & segmentation maps
        imgs = np.array( [self.read_image( x["img"], num_channels = CHANNELS, use_green_channel = True ) for x in dataset] )  
        segmaps = np.array( [self.read_image( x["map"], num_channels = 1 ) for x in dataset] )

        imgs = imgs * (1 / 255.)
        segmaps = segmaps * (1 / 255.)
        

        
        # divide imgs and maps into training & validation sets
        x_train, x_val, y_train, y_val = train_test_split(
            imgs,
            segmaps,
            test_size = float(TEST_SIZE),
            random_state = int(RANDOM_STATE)
        )

        # normalize imgs & maps, pass to generators
        train_generator = zip(
            self.create_generator(x_train),
            self.create_generator(y_train)
        )

        val_generator = zip(
            self.create_generator(x_val),
            self.create_generator(y_val)
        )


        

        # compile U-Net model
        model = unet_model( INPUT_SHAPE )
        model.summary()

        optimizer = tf.keras.optimizers.Adam()
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = ["accuracy", iou] # default accuracy metric + our iou metric


        model.compile(
            optimizer = optimizer,
            loss = loss,
            metrics = metrics
        )
        
        
        # train U-Net model
        checkpoint = ModelCheckpoint(
            f"{MODEL_NAME}.hdf5",
            monitor = 'val_iou',
            verbose = 0,
            save_best_only = True,
            mode='max'
        )
        
        steps = len(x_train) // BATCH_SIZE
        val_steps = len(x_val) // BATCH_SIZE

        
        history = model.fit(
            train_generator,
            validation_data = val_generator,
            epochs = EPOCHS,
            steps_per_epoch = steps,
            validation_steps = val_steps,
            callbacks = [checkpoint]
        )
        
        

        

        

if __name__ == "__main__":
    w = WorkshopPipeline()
    w.run_workshop()


    
