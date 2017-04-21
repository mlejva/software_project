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
        video_writer = cv2.VideoWriter(video_name, fourcc, fps, (frame_size, frame_size), True)        

        j = start
        while j != end:
            if direction == Direction.UP_DOWN:
                frame = generate_white_img_with_mask(frame_height, frame_width, [j, i, 0])
            elif direction == Direction.LEFT_RIGHT:
                frame = generate_white_img_with_mask(frame_height, frame_width, [i, j, 0])    

            video_writer.write(frame)
            j += step
        video_writer.release()
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