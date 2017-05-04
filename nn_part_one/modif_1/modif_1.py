import os
import sys
sys.path.append('../') # Because video_preprocessor is in the parent folder
import cv2
import video_preprocessor as video_prep
import shutil
import math
import numpy as np
from network_modif_1 import Network

def make_epoch_dir(name, epoch):
    epoch_dir_name = "./saved_frames/%s/epoch-%d" % (name, epoch)

    if os.path.exists(epoch_dir_name):
        shutil.rmtree(epoch_dir_name)
    os.makedirs(epoch_dir_name)

    return epoch_dir_name

def target_pixel_accuracy(network_predictions, gold):    
    right_predictions = 0    

    batch_size = np.size(gold, 0)
    for frame_index in range(batch_size):
        predicted_frame = network_predictions[frame_index]
        predicted_frame *= 255
        np.clip(predicted_frame, 0, 255)

        gold_frame = gold[frame_index]
        gold_frame *= 255

        predicted_target_pixel_position = np.where(predicted_frame == predicted_frame.min())
        gold_target_pixel_position = np.where(gold_frame == gold_frame.min())
        
        if int(predicted_frame[gold_target_pixel_position]) == int(predicted_frame.min()):
            right_predictions += 1        

    accuracy = right_predictions / batch_size
    return accuracy

def background_distance(network_predictions, gold):
    # Computes average L2 distance of gold and predicted frames in a batch
    
    average_distance = 0
    batch_size = np.size(gold, 0)
    for frame_index in range(batch_size):
        predicted_frame = network_predictions[frame_index]
        #predicted_frame *= 255
        np.clip(predicted_frame, 0, 1)

        gold_frame = gold[frame_index]
        #gold_frame *= 255

        average_distance += np.linalg.norm(gold_frame.astype(int) - predicted_frame.astype(int))

    average_distance /= batch_size
    return average_distance
def perform_train(network, input, gold_output, epoch):        
    loss, _ = network.train(input_frames, gold_output_frames, True, network.training_step == 0)                    
    print("\tLoss: %f" % loss)     

def perform_validation(network, input, gold_output, epoch):
    print("Running validation (epoch: %d)..." % (epoch + 1))

    loss, predictions = network.evaluate("validation", input, gold_output, True)    
    print("\tLoss on validation: %f" % loss)    

    # Compute accuracy and background distance
    target_accuracy = target_pixel_accuracy(predictions, gold_output)
    network.log_scalar_test(target_accuracy, "target_pixel_accuracy", epoch, "validation")
    print("\tTarget pixel accuracy on validation: %.2f" % (target_accuracy * 100.0))

    average_distance = background_distance(predictions, gold_output)
    network.log_scalar_test(average_distance, "background_distance_average", epoch, "validation")
    print("\tAverage distance on validation: %.2f" % average_distance)

def perform_test(network, input, gold_output, base, epoch):
    print("Running testing (epoch: %d)..." % (epoch + 1))

    loss, predictions = network.evaluate("test", input, gold_output, True)        
    print("\tLoss on test: %f" % loss)

    # Save predictions
    test_prediction_name_path = base + "/test-prediction-frames-%d" % epoch
    video_prep.save_prediction_frames(test_prediction_name_path, predictions, version="1.1")

    # Compute accuracy and background distance
    target_accuracy = target_pixel_accuracy(predictions, gold_output)
    network.log_scalar_test(target_accuracy, "target_pixel_accuracy", epoch, "test")
    print("\tTarget pixel accuracy on test: %.2f" % (target_accuracy * 100.0))

    average_distance = background_distance(predictions, gold_output)
    network.log_scalar_test(average_distance, "background_distance_average", epoch, "test")
    print("\tAverage distance on test: %.2f" % average_distance)
    
    print("================") 


def save_test_frames(input, gold_output, base, epoch):
    '''input_name_path = base + "/test-input-frames-%d" % epoch            
    video_prep.save_frames(input_name_path, input)'''

    gold_output_name_path = base + "/test-gold-output-frames-%d" % epoch
    video_prep.save_frames(gold_output_name_path, gold_output)




if __name__ == "__main__":
    batch_size = 7
    epochs = 200

    vp = video_prep.VideoPreprocessor("./videos", ".mp4", batch_size)
    vp.prepare_datasets(0.6, 0.2, 0.2)
    (height, width) = vp.get_training_frame_shape()
        
    exp_name = "model-1"
    network = Network(height, width, exp_name)
    network.construct()

    for epoch in range(epochs):
        epoch_dir_name = make_epoch_dir(exp_name, epoch)

        # Train
        vp.shuffle_training_data()        
        print("Training network (epoch: %d)..." % (epoch + 1))
        while epoch == vp.epochs_completed:              
            input_frames, gold_output_frames = vp.next_batch(grayscale=True)                                                

            input_frames, _, _ = video_prep.normalize(input_frames)            
            gold_output_frames, _, _ = video_prep.normalize(gold_output_frames)
            perform_train(network, input_frames, gold_output_frames, epoch)

        # Validation
        input_frames_val, gold_output_frames_val = vp.validation_dataset(grayscale=True)            

        input_frames_val, _, _ = video_prep.normalize(input_frames_val)
        gold_output_frames_val, _, _ = video_prep.normalize(gold_output_frames_val)
        perform_validation(network, input_frames_val, gold_output_frames_val, epoch)


        # Test
        input_frames_test, gold_output_frames_test = vp.testing_dataset(grayscale=True)
        save_test_frames(input_frames_test, gold_output_frames_test, epoch_dir_name, epoch)

        input_frames_test, _, _ = video_prep.normalize(input_frames_test)
        gold_output_frames_test, _, _ = video_prep.normalize(gold_output_frames_test)
        perform_test(network, input_frames_test, gold_output_frames_test, epoch_dir_name, epoch)
