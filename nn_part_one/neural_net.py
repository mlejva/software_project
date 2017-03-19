import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

import tensorflow as tf
import tensorflow.contrib.layers as tf_layers

class Network:
    def __init__(self, frame_height, frame_width):
        self.frame_height = frame_height
        self.frame_width = frame_width
        graph = tf.Graph()
        self.session = tf.Session(graph = graph)

    
    def construct(self):
        print("================")
        with self.session.graph.as_default():
            self.input_frames = tf.placeholder(tf.float32, [None, self.frame_height, self.frame_width, 2])
            #print("\tself.input_frames: " + str(self.input_frames.get_shape()))            
            self.gold_output_frames = tf.placeholder(tf.float32, [None, self.frame_height, self.frame_width])
            #print("\tself.output_frames: " + str(self.gold_output_frames.get_shape()))
        
            conv_layer1 = tf_layers.convolution2d(self.input_frames, 30, [5, 5], 2, padding="VALID", normalizer_fn=tf_layers.batch_norm)
            print("\tconv_layer1: " + str(conv_layer1.get_shape()))
            

            max_pool = tf.layers.max_pooling2d(conv_layer1, [4, 4], 2, padding="VALID")
            print("\tmax_pool: " + str(max_pool.get_shape()))


            flattened = tf_layers.flatten(max_pool)
            print("\tflattened: " + str(flattened.get_shape()))
                        
            output_layer = tf_layers.fully_connected(flattened, num_outputs=400, activation_fn=tf.sigmoid)

            self.predictions = output_layer
            print("\toutput_layer: " + str(output_layer.get_shape()))

            gold_output_flattened = tf_layers.flatten(self.gold_output_frames)
            print("\tgold_output_flattened: " + str(gold_output_flattened.get_shape()))

            #self.loss = tf.losses.mean_pairwise_squared_error(gold_output_flattened, self.predictions)
            #self.loss = tf.losses.mean_squared_error(gold_output_flattened, self.predictions)
            self.loss = tf.losses.absolute_difference(gold_output_flattened, self.predictions)
            #self.loss = tf.losses.log_loss(gold_output_flattened, self.predictions)

            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
            self.training = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)

            self.accuracy = tf.metrics.accuracy(self.predictions, gold_output_flattened)

            init = tf.global_variables_initializer()
            self.session.run(init)
            print("================")

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def train(self, input_frames, output_frames):
        args = {"feed_dict": {self.input_frames: input_frames, self.gold_output_frames: output_frames}}
        targets = [self.training, self.loss, self.predictions]
        results = self.session.run(targets, **args)
        return results
        #print(results[1])

    def evaluate(self, input_frames, output_frames):
         args = {"feed_dict": {self.input_frames: input_frames, self.output_frames: output_frames}}
         targets = [self.accuracy]
         results = self.session.run(targets, **args)
         return results[0]

    #def run(self, input_frames, output_frames):
    #    print(self.session.run(self.input_frames, {self.input_frames: input_frames, self.output_frames: output_frames}))


        #print(self.session.run(input_frames, feed_dict={self.input_frame_fst: input_frames[0], self.input_frame_fst: input_frames[1], self.output_frame: output_frame}))

class VideoPreprocessor:
    def __init__(self, path_to_videos, video_file_extension):
        self.video_names = []        
        
        for dirpath,_,filenames in os.walk(path_to_videos):
            for f in filenames:
                if f.endswith(video_file_extension):
                    video_name = os.path.abspath(os.path.join(dirpath, f))                
                    self.video_names.append(video_name)

        self.videos_count = len(self.video_names)

    def prepare_datasets(self, training_ratio, cross_val_ratio, test_ratio):
        self.training_set_count = int(self.videos_count * training_ratio)
        self.cross_val_set_count = int(self.videos_count * cross_val_ratio)
        self.testing_set_count = int(self.videos_count * test_ratio)

        shuffle(self.video_names)
        self.videos_training_set = self.video_names[0:self.training_set_count]
        self.videos_cross_val_set = self.video_names[self.training_set_count:(self.training_set_count + self.cross_val_set_count)]
        self.videos_testing_set = self.video_names[(self.training_set_count + self.cross_val_set_count):]

        return (self.videos_training_set, self.videos_cross_val_set, self.videos_testing_set)
    
    def shuffle_training_set(self):
        shuffle(self.videos_training_set)

    def get_training_video_batch(self, index, size):
        range_start = index * size
        range_end = range_start + size        

        batch = self.videos_training_set[range_start:range_end]        
        return batch

    def get_training_frame_shape(self):
        training_video = self.videos_training_set[0] # Here I assume that every training video has same length (frames)
        cap = cv2.VideoCapture(training_video)
        if not cap.isOpened():
            raise ValueError("Cannot open the video file")

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return (height, width)

    def get_training_frames_count(self):
        training_video = self.videos_training_set[0] # Here I assume that every training video has same length (frames)
        cap = cv2.VideoCapture(training_video)
        if not cap.isOpened():
            raise ValueError("Cannot open the video file")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return frame_count

    def get_training_set_length(self):
        return len(self.videos_training_set)






def get_frame_at_index_with_cap(video_capture, frame_index, grayscale=False):    
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

def normalize(data):
    data_min = data.min()
    data_max = data.max()
    
    normalized = (data - data_min)/(data_max - data_min)
    return normalized, data_min, data_max

def denormalize(normalized, orig_min, orig_max):
    denormalized = (normalized*(orig_max - orig_min)) + orig_min
    return denormalized

if __name__ == "__main__":    
    # Prepare data:
        # Get names of all videos
        # Shuffle names
        # 60% for training set
        # 20% for cross validation set
        # 20% for testing set
    vp = VideoPreprocessor("./videos/", ".mp4")
    (training_set, _, _) = vp.prepare_datasets(0.6, 0.2, 0.2)

    (frame_height, frame_width) = vp.get_training_frame_shape()  # (height, width)    
    frames_count = vp.get_training_frames_count()

    video_batch_size = 5 # 5 worked
    # Each video has 'frames_count' frames --> 'video_batch_size' * 'frames_count' frames in a single batch of videos
    frame_batch_size = video_batch_size * frames_count

    video_batches_count = int(vp.get_training_set_length() / video_batch_size)    
    print("video_batches_count: %d" % video_batches_count)
    # Training:
        # For each batch:
            # Randomize order
            # For each video in batch:
                # One video - one example --> go through all the frames of the video
                # For each frame of a video:
                    # (Flatten frame?)
                    # Run network training
    network = Network(frame_height, frame_width)
    network.construct()

    epoch_count = 2
    for epoch_index in range(epoch_count):
        if os.path.exists("./img-%d" % epoch_index):
            shutil.rmtree("./img-%d" % epoch_index)
        os.makedirs("./img-%d" % epoch_index)

        vp.shuffle_training_set()

        for video_batch_index in range(video_batches_count): # Iterate through each batch of training set
            current_video_batch_set = vp.get_training_video_batch(video_batch_index, video_batch_size)            

            network_input_frames_batch = np.empty([frame_batch_size, frame_height, frame_width, 2], dtype=np.float32)
            network_gold_output_frames_batch = np.empty([frame_batch_size, frame_height, frame_width], dtype=np.float32)

            for i in range(len(current_video_batch_set)): # Iterate through each video in the current batch
                video = current_video_batch_set[i]
                cap = cv2.VideoCapture(video)
                print("Processing video %d (out of %d) in batch %d (out of %d) in epoch %d (out of %d)..." % (i+1, len(current_video_batch_set), video_batch_index+1, video_batches_count, epoch_index+1, epoch_count))           
                for frame_index in range(frames_count): # Iterate through each frame in the current video
                    print("\t Frame: %d" % frame_index)
                    if (frames_count - frame_index) >= 3:                        
                        input_frame_fst = get_frame_at_index_with_cap(cap, frame_index, grayscale=True)                        
                        input_frame_snd = get_frame_at_index_with_cap(cap, frame_index + 1, grayscale=True)           
                                                
                        input_frame_fst, input_min, input_max = normalize(input_frame_fst)
                        input_frame_snd, _, _ = normalize(input_frame_snd)                       

                        #plt.plot(input_frame_fst, 'ro')
                        #plt.plot(input_frame_snd, 'ro')

                        network_input_frames_batch[i * frames_count + frame_index, :, :, 0] = input_frame_fst
                        network_input_frames_batch[i * frames_count + frame_index, :, :, 1] = input_frame_snd                        

                        output_gold_frame = get_frame_at_index_with_cap(cap, frame_index + 2, grayscale=True)
                        output_gold_frame, _, _ = normalize(output_gold_frame)
                        #plt.plot(output_gold_frame, 'ro')
                        network_gold_output_frames_batch[i * frames_count + frame_index] = output_gold_frame           

            #plt.show()
            # Input frame batch into network
            results = network.train(network_input_frames_batch, network_gold_output_frames_batch)
            loss = results[1]
            predictions = results[2]      


            os.makedirs("./img-%d/predict-%d" % (epoch_index, video_batch_index))
            os.makedirs("./img-%d/original-%d" % (epoch_index, video_batch_index))
            for i in range(frame_batch_size):
                frame = denormalize(predictions[i].reshape(frame_height, frame_width), input_min, input_max)
                cv2.imwrite("./img-%d/predict-%d/predict-%d-%d.png" % (epoch_index, video_batch_index, video_batch_index, i), frame)
            
            for i in range(frame_batch_size):
                input_frame_fst = network_input_frames_batch[i, :, :, 0]
                input_frame_fst = denormalize(input_frame_fst, input_min, input_max)

                input_frame_snd = network_input_frames_batch[i, :, :, 1]
                input_frame_snd = denormalize(input_frame_snd, input_min, input_max)

                gold_output = network_gold_output_frames_batch[i]
                gold_output = denormalize(gold_output, input_min, input_max)

                cv2.imwrite("./img-%d/original-%d/fst-%d-%d.png" % (epoch_index, video_batch_index, video_batch_index, i), input_frame_fst)
                cv2.imwrite("./img-%d/original-%d/snd-%d-%d.png" % (epoch_index, video_batch_index, video_batch_index, i), input_frame_snd)
                cv2.imwrite("./img-%d/original-%d/out-%d-%d.png" % (epoch_index, video_batch_index, video_batch_index, i), gold_output)



            print(loss)      
            #print("Loss on batch %d in epoch %d is %r" % (video_batch_index+1, epoch_index+1, loss))
            
            print("===BATCH %d END===" % (video_batch_index + 1))
        


    # Test network on cross validation set
    # Test network on testing set

