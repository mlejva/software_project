import sys
sys.path.append('../') # Because video_preprocessor is in the parent folder
import video_preprocessor as video_prep
import os
import shutil
import cv2
import numpy as np
from network_base import Network

def compute_accuracy(gold_frames, predictions):
    predicted_frames = video_prep.convert_network_predictions_to_frames(predictions, discrete_conversion=True)

    batch_size = np.size(predicted_frames, 0)
    frame_height = np.size(predicted_frames, 1)
    frame_width = np.size(predicted_frames, 2)

    threshold = 20
    total_right_predictions = 0    
    for frame_index in range(batch_size):
        gold_frame = gold_frames[i]
        predicted_frame = predicted_frames[i]

        target_pixel_pos = np.where(gold_frame <= 50)
        
        if abs(predicted_frame[target_pixel_pos] - gold_frame[target_pixel_pos]) <= 50:        
            total_right_predictions += 1
        

        #target_pixel_pos = np.where(gold_output_frames == 255)         
        '''if predicted_frame[target_pixel_pos] == 255:
            print("Yes")
            total_right_predictions += 1
        else:
            print(predicted_frame[target_pixel_pos])'''
            
    accuracy = total_right_predictions / batch_size
    return accuracy


if __name__ == "__main__":
    batch_size = 7
    epochs = 100

    vp = video_prep.VideoPreprocessor("./videos", ".mp4", batch_size=batch_size)
    vp.prepare_datasets(0.6, 0.2, 0.2)
    (height, width) = vp.get_training_frame_shape()
    
    exp_name = "crossentropy_loss-model_norm-mean_batch-%d_epochs-%d" % (batch_size, epochs)
    exp_name = "dummy"
    network = Network(height, width, exp_name)
    network.construct()

    for i in range(epochs):        
        epoch_dir_name = "./saved_frames/%s/epoch-%d" % (exp_name, i)

        if os.path.exists(epoch_dir_name):
            shutil.rmtree(epoch_dir_name)
        os.makedirs(epoch_dir_name)

        vp.shuffle_training_data()
        j = 0
        print("Training network (epoch: %d)..." % (i+1))
        while i == vp.epochs_completed:            
            input_frames, gold_output_frames = vp.next_batch(grayscale=True)            
            gold_output_frames_one_hot = video_prep.convert_gold_output_to_onehot(gold_output_frames)

            input_frames, _ = video_prep.normalize_mean(input_frames)            

            # Train network here            
            #print("\tNetwork training step: %d" % (network.training_step + 1))
            loss, _ = network.train(input_frames, gold_output_frames_one_hot, True, network.training_step == 0)
            print("\tLoss: %f" % loss)            
            ####################
            j += 1

        # Eval network on validation set here
        print("Running validation (epoch: %d)..." % (i+1))
        input_frames_val, gold_output_frames_val = vp.validation_dataset(grayscale=True)        
        gold_output_frames_val = video_prep.convert_gold_output_to_onehot(gold_output_frames_val)
        
        input_frames_val, _ = video_prep.normalize_mean(input_frames_val)

        loss, _, accuracy = network.evaluate("validation", input_frames_val, gold_output_frames_val, True)
        print("\tLoss on validation: %f" % loss)
        print("\tAccuracy on validation: %f" % (accuracy[1]*100.0))
        ####################

        # Eval network on test set here
        print("Running testing (epoch: %d)..." % (i+1))
        test_prediction_name_path = epoch_dir_name + "/test-prediction-frames-%d" % i
        input_name_path = epoch_dir_name + "/test-input-frames-%d" % i            
        gold_output_name_path = epoch_dir_name + "/test-gold-output-frames-%d" % i
       
        input_frames_test, gold_output_frames_test = vp.testing_dataset(grayscale=True)    
        video_prep.save_gold_output_frames(gold_output_name_path, gold_output_frames_test)
        gold_output_frames_test = video_prep.convert_gold_output_to_onehot(gold_output_frames_test) # convert to one-hot
        
        input_frames_test, _ = video_prep.normalize_mean(input_frames_test)

        loss, test_predictions, accuracy = network.evaluate("test", input_frames_test, gold_output_frames_test, True)
        print("\tLoss on testing: %f" % loss)        
        print("\tAccuracy on testing: %f" % (accuracy[1]*100.0))        
        video_prep.save_prediction_frames(test_prediction_name_path, test_predictions, discrete_conversion=False)
        print("================")
        ####################
        