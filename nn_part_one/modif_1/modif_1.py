import os
import sys
import shutil
import numpy as np
from network_modif_1 import Network
sys.path.append('../')  # Because video_preprocessor is in the parent folder
import video_preprocessor as video_prep

def make_epoch_dir(name, current_epoch):
    """Creates a directory for the current epoch and returns its name"""

    dir_name = "./saved_frames/%s/epoch-%d" % (name, current_epoch)

    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)

    return dir_name

def target_pixel_accuracy(network_predictions, gold):
    """Computes a ratio of right predictions"""

    right_predictions = 0

    batch_size = np.size(gold, 0)
    for frame_index in range(batch_size):
        predicted_frame = network_predictions[frame_index]
        predicted_frame *= 255

        gold_frame = gold[frame_index]
        gold_frame *= 255

        #predicted_target_pixel_position = np.where(predicted_frame == predicted_frame.min())
        gold_target_pixel_position = np.where(gold_frame == gold_frame.min())

        if int(predicted_frame[gold_target_pixel_position]) == int(predicted_frame.min()):
            right_predictions += 1

    accuracy = right_predictions / batch_size
    return accuracy

def background_distance(network_predictions, gold):
    """Computes average L2 distance of gold and predicted frames in a batch"""
    average_distance = 0

    batch_size = np.size(gold, 0)
    for frame_index in range(batch_size):
        predicted_frame = network_predictions[frame_index]
        np.clip(predicted_frame, 0, 1)

        gold_frame = gold[frame_index]

        average_distance += np.linalg.norm(gold_frame.astype(int) - predicted_frame.astype(int))
    average_distance /= batch_size
    return average_distance

def perform_train(network, network_input, network_gold_output):
    """Runs training on a network and prints loss"""

    loss, _ = network.train(network_input,
                            network_gold_output,
                            True,
                            network.training_step == 0)
    print("\tLoss: %f" % loss)

def perform_validation(network, network_input, network_gold_output, current_epoch):
    """Evaluate a validation input on a network and prints loss and accuracy"""

    print("Running validation (epoch: %d)..." % (current_epoch + 1))

    loss, predictions = network.evaluate("validation", network_input, network_gold_output, True)
    print("\tLoss on validation: %f" % loss)

    # Compute accuracy and background distance
    target_accuracy = target_pixel_accuracy(predictions, network_gold_output)
    network.log_scalar_test(target_accuracy, "target_pixel_accuracy", current_epoch, "validation")
    print("\tTarget pixel accuracy on validation: %.2f" % (target_accuracy * 100.0))

    average_distance = background_distance(predictions, network_gold_output)
    network.log_scalar_test(average_distance,
                            "background_distance_average",
                            current_epoch,
                            "validation")
    print("\tAverage distance on validation: %.2f" % average_distance)

def perform_test(network, network_input, network_gold_output, base, current_epoch):
    """Evaluate a test input on a network and prints loss and accuracy.
    Saves network predictions"""

    print("Running testing (epoch: %d)..." % (current_epoch + 1))

    loss, predictions = network.evaluate("test", network_input, network_gold_output, True)
    print("\tLoss on test: %f" % loss)

    # Save predictions
    test_prediction_name_path = base + "/test-prediction-frames-%d" % current_epoch
    video_prep.save_prediction_frames(test_prediction_name_path, predictions, version="1.1")

    # Compute accuracy and background distance
    target_accuracy = target_pixel_accuracy(predictions, network_gold_output)
    network.log_scalar_test(target_accuracy, "target_pixel_accuracy", current_epoch, "test")
    print("\tTarget pixel accuracy on test: %.2f" % (target_accuracy * 100.0))

    average_distance = background_distance(predictions, network_gold_output)
    network.log_scalar_test(average_distance, "background_distance_average", current_epoch, "test")
    print("\tAverage distance on test: %.2f" % average_distance)

    print("================")


def save_test_frames(test_input, test_gold_output, base, current_epoch):
    """Saves input and gold output from testing dataset"""

    '''input_name_path = base + "/test-input-frames-%d" % epoch
    video_prep.save_frames(input_name_path, test_input)'''

    gold_output_name_path = base + "/test-gold-output-frames-%d" % current_epoch
    video_prep.save_frames(gold_output_name_path, test_gold_output)


if __name__ == "__main__":
    BATCH_SIZE = 7
    EPOCHS = 400

    VP = video_prep.VideoPreprocessor("./videos", ".mp4", BATCH_SIZE)
    VP.prepare_datasets(0.6, 0.2, 0.2)
    HEIGHT, WIDTH = VP.get_training_frame_shape()

    EXP_NAME = "model-1"
    NETWORK = Network(HEIGHT, WIDTH, EXP_NAME)
    NETWORK.construct()

    for epoch in range(EPOCHS):
        epoch_dir_name = make_epoch_dir(EXP_NAME, epoch)

        # Train
        VP.shuffle_training_data()
        print("Training network (epoch: %d)..." % (epoch + 1))
        while epoch == VP.epochs_completed:
            input_frames, gold_output_frames = VP.next_batch(grayscale=True)
            # Preprocess
            input_frames, _, _ = video_prep.normalize(input_frames)
            gold_output_frames, _, _ = video_prep.normalize(gold_output_frames)

            perform_train(NETWORK, input_frames, gold_output_frames)
        ####################

        # Validation
        input_frames_val, gold_output_frames_val = VP.validation_dataset(grayscale=True)
        # Preprocess
        input_frames_val, _, _ = video_prep.normalize(input_frames_val)
        gold_output_frames_val, _, _ = video_prep.normalize(gold_output_frames_val)

        perform_validation(NETWORK, input_frames_val, gold_output_frames_val, epoch)
        ####################

        # Test
        input_frames_test, gold_output_frames_test = VP.testing_dataset(grayscale=True)
        save_test_frames(input_frames_test, gold_output_frames_test, epoch_dir_name, epoch)
        # Preprocess
        input_frames_test, _, _ = video_prep.normalize(input_frames_test)
        gold_output_frames_test, _, _ = video_prep.normalize(gold_output_frames_test)

        perform_test(NETWORK, input_frames_test, gold_output_frames_test, epoch_dir_name, epoch)
        ####################
