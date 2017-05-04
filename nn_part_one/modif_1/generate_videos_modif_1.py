import os
import cv2
import numpy as np
import shutil 

from enum import Enum

class Direction(Enum):
    UP_DOWN = 0
    LEFT_RIGHT = 1

def get_frame_at_index(video_path, frame_index):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open the video file \"%s\"" % video_path)
    print(cap)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index) # Set video capture to desired frame

    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    
    frame = np.zeros((frame_height, frame_width), dtype=np.int8)

    success, frame = cap.read(frame)
    if not success:
        raise ValueError("Cannot get a frame at index: " + str(frame_index))
    return frame

def generate_white_img_with_mask(img_height, img_width, mask):
    if len(mask) != 3:
        raise ValueError("Expected array of length 3 as a mask [height, width, value]")
    mask_height = mask[0]
    mask_width = mask[1]
    mask_value = mask[2]

    if mask_height > img_height or mask_width > img_width:
        raise ValueError("Mask position is out of bounds")

    img = np.full((img_height, img_width, 3), 255, dtype="uint8")    
    img[mask_height, mask_width] = mask_value

    return img

def add_noise(img, seed, lower_bound=0.5):
    my_seed = np.random.RandomState(seed)
    noise_channel = my_seed.uniform(lower_bound, 1.0, [img.shape[0], img.shape[1]])
    #noise_channel = my_seed.randint(lower_bound, 1.0, size=[img.shape[0], img.shape[1]])
    noise_channel *= 255
    #noise_channel = my_seed.uniform(lower_bound, 1.0, [img.shape[0], img.shape[1]]) * 255.0

    # Save the original target pixel position
    # becuase it may get masked by the noise
    target_pixel_pos = np.where(img == 0)    
    
    # Copy the noise along the third dimension
    # because images are greyscaled we want same number
    # in all three RGB dimensions
    noise = np.empty(img.shape, dtype="uint8")
    for i in range(np.size(noise, 2)):
        noise[:, :, i] = noise_channel

    # Put the target pixel back
    noise[target_pixel_pos] = 0
    return noise

def generate_black_dot_videos(fourcc, fps, direction, size, reverse):
    frame_height = size
    frame_width = size

    start = 0
    end = size
    step = 1

    if reverse:
        start = size - 1
        end = -1
        step = -1

    if direction == Direction.UP_DOWN:
        type = "up_down"
        if reverse:
            type = "down_up"
    elif direction == Direction.LEFT_RIGHT:
        type = "left_right"
        if reverse:
            type = "right_left"    

    i = start
    while i != end:
        video_name = "./videos/%s-%d.mp4" % (type, i)
        #video_name_noised = "./videos/%s-%d-noised.mp4" % (type, i)
        video_writer = cv2.VideoWriter(video_name, fourcc, fps, (frame_size, frame_size), True) 
        #video_writer_noised = cv2.VideoWriter(video_name_noised, fourcc, fps, (frame_size, frame_size), True)

        seed = i # Same seed for a single video
        j = start
        while j != end:
            if direction == Direction.UP_DOWN:
                mask = [j, i, 0]
            elif direction == Direction.LEFT_RIGHT:
                mask = [i, j, 0]
                
            frame = generate_white_img_with_mask(frame_height, frame_width, mask)    
            frame = add_noise(frame, seed, 0.8)              

            video_writer.write(frame)
            #video_writer_noised.write(frame)
            j += step
        video_writer.release()
        #video_writer_noised.release()
        i += step

if __name__ == "__main__":
    frame_size = 30
    fps = 10.0
    fourcc = cv2.VideoWriter_fourcc(*"hfyu")    

    if os.path.exists("./videos"):
        shutil.rmtree("./videos")
        os.makedirs("./videos")
    else:
        os.makedirs("./videos")

    generate_black_dot_videos(fourcc, fps, Direction.UP_DOWN, frame_size, False)
    generate_black_dot_videos(fourcc, fps, Direction.UP_DOWN, frame_size, True)

    generate_black_dot_videos(fourcc, fps, Direction.LEFT_RIGHT, frame_size, False)
    generate_black_dot_videos(fourcc, fps, Direction.LEFT_RIGHT, frame_size, True)