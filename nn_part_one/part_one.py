import os
#import sys
#sys.path.append('path/to/your/file')
from network_one import Network

import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from math import ceil

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
        
        self.__batch_count = ceil(training_set_length / self.__batch_size)
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
            raise ValueError("Method 'prepare_datasets()' must be called before.")

        print("Epoch: %d Batch: %d (out of %d)" % (self.epochs_completed+1, self.__current_batch_index+1, self.__batch_count))

        range_start = self.__current_batch_index * self.__batch_size
        range_end = range_start + self.__batch_size
        current_batch_set = self.__training_set[range_start:range_end]

        # -2 because fst_frame is at i, snd_frame at i+1 and gold_output_frame at i+2
        frames_in_batch = self.__batch_size * (self.__training_video_frame_count - 2)
        
        training_frame_shape = (self.__training_video_frame_height, self.__training_video_frame_width)
        input_frames_batch, gold_output_frames_batch = self.__process_videos(current_batch_set, 
                                                                             self.__training_video_frame_count, 
                                                                             training_frame_shape,
                                                                             grayscale=grayscale)                
                    
        self.__current_batch_index += 1
        if self.__current_batch_index == self.__batch_count:            
            self.__epochs_completed += 1
            self.__current_batch_index = 0

        return input_frames_batch, gold_output_frames_batch

    def validation_dataset(self, grayscale):
        if not self.__prepared:
            raise ValueError("Method 'prepare_datasets()' must be called before.")
        
        frame_shape = (self.__training_video_frame_height, self.__training_video_frame_width)
        input_frames_val, gold_output_frames_val = self.__process_videos(self.__validation_set,
                                                                         self.__training_video_frame_count,
                                                                         frame_shape,
                                                                         grayscale=grayscale)
        return input_frames_val, gold_output_frames_val

    def testing_dataset(self, grayscale):
        if not self.__prepared:
            raise ValueError("Method 'prepare_datasets()' must be called before.")
        
        frame_shape = (self.__training_video_frame_height, self.__training_video_frame_width)
        input_frames_test, gold_output_frames_test = self.__process_videos(self.__testing_set,
                                                                           self.__training_video_frame_count,
                                                                           frame_shape,
                                                                           grayscale=grayscale)
        return input_frames_test, gold_output_frames_test

    def shuffle_training_data(self):
        shuffle(self.__training_set)
    # Private Methods #
    def __process_videos(self, videos, frames_in_single_video, frame_shape, grayscale=False):
        # TODO: Add version for not grayscale option    

        (height, width) = frame_shape
        videos_length = len(videos)

        # -2 because fst_frame is at i, snd_frame at i+1 and gold_output_frame at i+2
        frames_total = videos_length * (frames_in_single_video - 2)

        input_frames = np.empty([frames_total, height, width, 2], dtype=np.float64)
        gold_output_frames = np.empty([frames_total, height, width, 1], dtype=np.float64)

        for i in range(videos_length):
            print("\tProcessing video: %d (out of %d)..." % (i+1, videos_length))
            video = videos[i]
            tmp_frames = self.__get_frames_from_video(video, frames_in_single_video, frame_shape, grayscale=True)

            # Insert frames to proper arrays
            for frame_index in range(frames_in_single_video):
                if (frames_in_single_video - frame_index) >= 3:                    
                    input_frames[i * (frames_in_single_video - 2) + frame_index, :, :, 0] = tmp_frames[frame_index]
                    input_frames[i * (frames_in_single_video - 2) + frame_index, :, :, 1] = tmp_frames[frame_index + 1]
                    
                    gold_output_frames[i * (frames_in_single_video - 2) + frame_index, :, :, 0] = tmp_frames[frame_index + 2]
        
        return input_frames, gold_output_frames

    def __get_frames_from_video(self, path_to_video, frames_in_video, frame_shape, grayscale=False):
        cap = cv2.VideoCapture(path_to_video)            
        
        (height, width) = frame_shape

        frames = np.empty([frames_in_video, height, width])
        for frame_index in range(frames_in_video):
            success, frame = cap.read()
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # To grayscale
            if not success:
                raise ValueError("Error while trying to read video %s" % video)
            frames[frame_index] = frame 

        return frames

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


def save_input_frames(path, input_frames):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    input_length = np.size(input_frames, 0)
    for i in range(input_length):
        input_fst = input_frames[i, :, :, 0]
        input_snd = input_frames[i, :, :, 1]

        cv2.imwrite(path+"/fst-%d.png" % i, input_fst)
        cv2.imwrite(path+"/snd-%d.png" % i, input_snd)

def save_gold_output_frames(path, gold_output_frames):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    gold_output_length = np.size(gold_output_frames, 0)    
    for i in range(gold_output_length):
        gold_output = gold_output_frames[i]
        cv2.imwrite(path+"/gold-%d.png" % i, gold_output)

def save_prediction_frames(path, prediction_frames):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    prediction_length = np.size(prediction_frames, 0)
    for i in range(prediction_length):
        prediction = prediction_frames[i]
        cv2.imwrite(path+"/prediction-%d.png" % i, prediction)


def artifacts_workaround(data):
     # TODO: Remove, workaround because video has artifacts
    low_values_indices = data[:] < 200
    data[low_values_indices] = 0
    high_value_indices = data[:] > 200
    data[high_value_indices] = 255

    return data

if __name__ == "__main__":        
    vp = VideoPreprocessor("./videos", ".mp4", batch_size=5)
    vp.prepare_datasets(0.6, 0.2, 0.2)
    (height, width) = vp.get_training_frame_shape()

    network = Network(height, width)
    network.construct()
    
    epochs = 10

    for i in range(epochs):
        epoch_dir_name = "./frames-%d" % i
        if os.path.exists(epoch_dir_name):
            shutil.rmtree(epoch_dir_name)
        os.mkdir(epoch_dir_name)

        vp.shuffle_training_data()
        j = 0
        while i == vp.epochs_completed:
            input_name_path = epoch_dir_name + "/input-frames-%d" % j            
            gold_output_name_path = epoch_dir_name + "/output-frames-%d" % j
            prediction_name_path = epoch_dir_name + "/prediction-frames-%d" % j

            input_frames, gold_output_frames = vp.next_batch(grayscale=True)
            input_frames = artifacts_workaround(input_frames)
            gold_output_frames = artifacts_workaround(gold_output_frames)

            save_input_frames(input_name_path, input_frames)
            save_gold_output_frames(gold_output_name_path, gold_output_frames)

            input_frames, input_min, input_max = normalize(input_frames)
            gold_output_frames, _, _ = normalize(gold_output_frames) 

            # Train network here
            print("Training network...")
            print("\tNetwork training step: %d" % (network.training_step + 1))
            loss, predictions = network.train(input_frames, gold_output_frames, True, network.training_step == 0)
            print("\tLoss: %f" % loss)
            ####################

            predictions = denormalize(predictions, input_min, input_max)
            save_prediction_frames(prediction_name_path, predictions)

            j += 1

        # Eval network on validation set here
        print("\nRunning validation...")
        input_frames_val, gold_output_frames_val = vp.validation_dataset(grayscale=True)
        input_frames_val, _, _ = normalize(input_frames_val)
        gold_output_frames_val, _, _ = normalize(gold_output_frames_val)

        loss, _ = network.evaluate("validation", input_frames_val, gold_output_frames_val, True)
        print("\n\tLoss on validation: ", loss)
        ####################

        # Eval network on test set here
        print("\nRunning testing...")
        input_frames_test, gold_output_frames_test = vp.testing_dataset(grayscale=True)
        # TODO: Save input frames!!!!

        input_frames_test, input_test_min, input_test_max = normalize(input_frames_test)
        gold_output_frames_test, _, _ = normalize(gold_output_frames_test)

        loss, test_predictions = network.evaluate("test", input_frames_test, gold_output_frames_test, True)
        print("\n\tLoss on testing: ", loss)

        test_prediction_name_path = epoch_dir_name + "/test-prediction-frames-%d" % i
        test_predictions = denormalize(test_predictions, input_test_min, input_test_max)
        save_prediction_frames(test_prediction_name_path, test_predictions)
        ####################