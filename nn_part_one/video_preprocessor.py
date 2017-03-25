import os
import cv2
import shutil
import numpy as np
from random import shuffle
from math import ceil

# Global Methods #
'''
def normalize_range(data, start, end):
    data_min = data.min()
    data_max = data.max()
    
    normalized = (end - start)*((data - data_min)/(data_max - data_min)) + start
    return normalized, data_min, data_max

def denormalize_range(normalized, start, end, orig_min, orig_max):
    denormalized = ((normalized - start) * (orig_max - orig_min) / (end - start)) + orig_min
    return denormalized
'''
def normalize(data):
    data_min = data.min()
    data_max = data.max()
    
    normalized = (data - data_min)/(data_max - data_min)    
    return normalized, data_min, data_max   

def denormalize(normalized, orig_min, orig_max):    
    denormalized = (normalized*(orig_max - orig_min)) + orig_min
    return denormalized

def normalize_mean(data):
    mean = np.mean(data)
    data -= mean
    return data, mean

def denormalize_mean(normalized, orig_mean):
    denormalized = normalized + orig_mean
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

    batch_size = np.size(prediction_frames, 0)
    frame_height = np.size(prediction_frames, 1)
    frame_width = np.size(prediction_frames, 2)

    for frame_index in range(batch_size):
        frame = np.zeros([frame_height, frame_width, 1])
        for i in range(frame_height):
            for j in range(frame_width):
                target_pixel = prediction_frames[frame_index, i, j]

                if target_pixel == 1:
                    frame[i, j] = 0
                else:
                    frame[i, j] = 255
        cv2.imwrite(path+"/prediction-%d.png" % frame_index, frame)

def convert_gold_output_to_onehot(gold_outputs):    
    batch_size = np.size(gold_outputs, 0)
    frame_height = np.size(gold_outputs, 1)
    frame_width = np.size(gold_outputs, 2)

    for batch in range(batch_size):
        for i in range(frame_height):
            for j in range(frame_width):
                pixel_val = gold_outputs[batch, i, j, :]
                if pixel_val >= 200: # White
                    gold_outputs[batch, i, j, :] = 0
                else: # Black
                    gold_outputs[batch, i, j, :] = 1
    return gold_outputs

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

        #print("\tEpoch: %d Batch: %d (out of %d)" % (self.epochs_completed+1, self.__current_batch_index+1, self.__batch_count))
        print("\tBatch: %d (out of %d)" % (self.__current_batch_index+1, self.__batch_count))

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
            #print("\tProcessing video: %d (out of %d)..." % (i+1, videos_length))
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
        