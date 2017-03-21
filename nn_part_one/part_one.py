import os
#import sys
#sys.path.append('path/to/your/file')
from network_one import Network

import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

class VideoPreprocessor:
    def __init__(self, path_to_videos, extension, batch_size=None):
        if batch_size == None:
            raise ValueError("Batch size cannot be None.")        
        self.__batch_size = batch_size
        self.__epochs_completed = 0
        self.__videos = []        
        self.__prepared = False # Are datasets prepared?

        for dirpath,_,filenames in os.walk(path_to_videos):
            for f in filenames:
                if f.endswith(extension):
                    video_name = os.path.abspath(os.path.join(dirpath, f))                
                    self.__videos.append(video_name)
        
        if len(self.__videos) == 0:
            raise ValueError("No videos in %s with extension %s." % (path_to_videos, extension))

    # Setters & Getters #
    @property
    def epochs_completed(self):
        return self.__epochs_completed    
    def get_training_frame_shape(self):
        return (self.__training_video_frame_height, self.__training_video_frame_width)
    # Public Methods #
    def prepare_datasets(self, training_ratio, validation_ratio, test_ratio):
        if (training_ratio + validation_ratio + test_ratio) > 1.0:
            raise ValueError("Not valid data ratio.")

        videos_total = len(self.__videos)
        training_set_length = int(videos_total * training_ratio)
        validation_set_length = int(videos_total * validation_ratio)
        testing_set_length = int(videos_total * test_ratio)

        shuffle(self.__videos)
        self.__training_set = self.__videos[0:training_set_length]
        self.__validation_set = self.__videos[training_set_length:(training_set_length + validation_set_length)]
        self.__testing_set = self.__videos[(training_set_length + validation_set_length):]
        
        self.__batch_count = int(training_set_length / self.__batch_size)        
        self.__current_batch_index = 0

        # Get number of frames in a training video and frame shape
        training_video = self.__training_set[0] # Here I assume that every training video has same length (frames)
        cap = cv2.VideoCapture(training_video)
        if not cap.isOpened():
            raise ValueError("Cannot open the training video.")
        # TODO: Move this to next_batch()?
        self.__training_video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.__training_video_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.__training_video_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
        self.__prepared = True

    def next_batch(self, grayscale=False):
        if not self.__prepared:
            raise ValueError("Method 'prepare_datasets()' must be called before calling 'next_batch()'.")

        print("Epoch: %d Batch: %d (out of %d)" % (self.epochs_completed+1, self.__current_batch_index+1, self.__batch_count))

        range_start = self.__current_batch_index * self.__batch_size
        range_end = range_start + self.__batch_size
        current_batch_set = self.__training_set[range_start:range_end]

        frames_in_batch = self.__batch_size * self.__training_video_frame_count
        
        # TODO: Add version for not grayscale option    
        input_frames_batch = np.empty([frames_in_batch, self.__training_video_frame_height, self.__training_video_frame_width, 2], dtype=np.float64)
        gold_output_frames_batch = np.empty([frames_in_batch, self.__training_video_frame_height, self.__training_video_frame_width, 1], dtype=np.float64)
        
        for i in range(len(current_batch_set)):
            print("\tProcessing video: %d (out of %d)..." % (i+1, len(current_batch_set)))
            video = current_batch_set[i]
            cap = cv2.VideoCapture(video)
            for frame_index in range(self.__training_video_frame_count):               
                #print("\tframe: %d" % (frame_index + 1))
                if (self.__training_video_frame_count - frame_index) >= 3:                        
                    input_frame_fst = self.__frame_at_index_with_cap(cap, frame_index, grayscale=grayscale)                        
                    input_frame_snd = self.__frame_at_index_with_cap(cap, frame_index + 1, grayscale=grayscale)


                    input_frames_batch[i * self.__training_video_frame_count + frame_index, :, :, 0] = input_frame_fst
                    input_frames_batch[i * self.__training_video_frame_count + frame_index, :, :, 1] = input_frame_snd

                    gold_output_frame = self.__frame_at_index_with_cap(cap, frame_index + 2, grayscale=grayscale)                    
                    gold_output_frames_batch[i * self.__training_video_frame_count + frame_index, :, :, 0] = gold_output_frame
                    
        self.__current_batch_index += 1
        if self.__current_batch_index == self.__batch_count:            
            self.__epochs_completed += 1
            self.__current_batch_index = 0

        return (input_frames_batch, gold_output_frames_batch)


    def shuffle_training_data(self):
        shuffle(self.__training_set)
    # Private Methods #
    def __frame_at_index_with_cap(self, video_capture, frame_index, grayscale=False):    
        if not video_capture.isOpened():
            raise ValueError("Cannot open the video file")

        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index) # Set video capture to desired frame

        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))    
        frame = np.zeros((frame_height, frame_width), dtype=np.float32)

        success, frame = video_capture.read(frame)
        if not success:
            raise ValueError("Cannot get a frame at index: " + str(frame_index))

        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = frame.astype(np.float32, copy=False)
        return frame
##########
# Global Methods #
def normalize(data):
    data_min = data.min()
    data_max = data.max()
    
    normalized = (data - data_min)/(data_max - data_min)
    return normalized, data_min, data_max

def denormalize(normalized, orig_min, orig_max):
    denormalized = (normalized*(orig_max - orig_min)) + orig_min
    return denormalized


if __name__ == "__main__":    
    vp = VideoPreprocessor("./videos", ".mp4", batch_size=10)
    vp.prepare_datasets(0.6, 0.2, 0.2)
    (height, width) = vp.get_training_frame_shape()

    network = Network(height, width)
    network.construct()
    
    epochs = 5

    for i in range(epochs):
        vp.shuffle_training_data()
        while i == vp.epochs_completed:
            input_frames, gold_output_frames = vp.next_batch(grayscale=True)
            input_frames, _, _ = normalize(input_frames)
            gold_output_frames, _, _ = normalize(gold_output_frames) 

            # Train network here
            print("Training network...")
            print("\tNetwork training step: %d" % (network.training_step + 1))
            #network.train(input_frames, gold_output_frames, network.training_step % 100 == 0)
            loss = network.train(input_frames, gold_output_frames, True, network.training_step == 0)
            print("\tLoss: %f" % loss)

        # Eval network on validation set here

        # Eval network on test set here
    
