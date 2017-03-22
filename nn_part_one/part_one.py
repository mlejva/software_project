import os
#import sys
#sys.path.append('path/to/your/file')
from network_one import Network
from video_preprocessor import VideoPreprocessor

import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Global Methods #
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

    prediction_length = np.size(prediction_frames, 0)
    for i in range(prediction_length):
        prediction = prediction_frames[i]
        cv2.imwrite(path+"/prediction-%d.png" % i, prediction)

if __name__ == "__main__":
    
    batch_size = 7
    epochs = 20

    vp = VideoPreprocessor("./videos", ".mp4", batch_size=batch_size)
    vp.prepare_datasets(0.6, 0.2, 0.2)
    (height, width) = vp.get_training_frame_shape()
    
    exp_name = "base-model_norm-range_batch-%d_epochs-%d" % (batch_size, epochs)
    network = Network(height, width, exp_name)
    network.construct()

    for i in range(epochs):
        epoch_dir_name = "./saved_frames/%s/epoch-%d" % (exp_name, i)
        if os.path.exists(epoch_dir_name):
            shutil.rmtree(epoch_dir_name)
        os.makedirs(epoch_dir_name)

        vp.shuffle_training_data()
        j = 0
        while i == vp.epochs_completed:            
            input_frames, gold_output_frames = vp.next_batch(grayscale=True)            

            input_frames, input_min, input_max = normalize(input_frames)
            gold_output_frames, _, _ = normalize(gold_output_frames)                         
            #input_frames, _ = normalize_mean(input_frames)
            #gold_output_frames, _ = normalize_mean(gold_output_frames)

            # Train network here
            print("Training network...")
            print("\tNetwork training step: %d" % (network.training_step + 1))
            loss, predictions = network.train(input_frames, gold_output_frames, True, network.training_step == 0)
            print("\tLoss: %f" % loss)
            ####################
            j += 1

        # Eval network on validation set here
        print("\nRunning validation...")
        input_frames_val, gold_output_frames_val = vp.validation_dataset(grayscale=True)        

        input_frames_val, _, _ = normalize(input_frames_val)
        gold_output_frames_val, _, _ = normalize(gold_output_frames_val)
        #input_frames_val, _ = normalize_mean(input_frames_val)
        #gold_output_frames_val, _ = normalize_mean(gold_output_frames_val)

        loss, _ = network.evaluate("validation", input_frames_val, gold_output_frames_val, True)
        print("\n\tLoss on validation: ", loss)
        ####################

        # Eval network on test set here
        print("Running testing...")
        test_prediction_name_path = epoch_dir_name + "/test-prediction-frames-%d" % i
        input_name_path = epoch_dir_name + "/test-input-frames-%d" % i            
        gold_output_name_path = epoch_dir_name + "/test-gold-output-frames-%d" % i
       
        input_frames_test, gold_output_frames_test = vp.testing_dataset(grayscale=True)    

        save_input_frames(input_name_path, input_frames_test)
        save_gold_output_frames(gold_output_name_path, gold_output_frames_test)

        input_frames_test, input_test_min, input_test_max = normalize(input_frames_test)
        gold_output_frames_test, _, _ = normalize(gold_output_frames_test)
        #input_frames_test, input_test_mean = normalize_mean(input_frames_test)
        #gold_output_frames_test, _ = normalize_mean(gold_output_frames_test)

        loss, test_predictions = network.evaluate("test", input_frames_test, gold_output_frames_test, True)
        print("\n\tLoss on testing: ", loss)

        test_predictions = denormalize(test_predictions, input_test_min, input_test_max)
        #test_predictions = denormalize_mean(test_predictions, input_test_mean)
        
        save_prediction_frames(test_prediction_name_path, test_predictions)
        ####################