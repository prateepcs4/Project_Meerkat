import tensorflow as tf
import numpy as np
from scipy.ndimage import imread
from glob import glob
import os

import constants as c
from tfutils import log10

def normalize_frames(frames):
    """
    Convert frames from int8 [0, 255] to float32 [-1, 1].
    @param frames: A numpy array. The frames to be converted.
    @return: The normalized frames.
    """
    new_frames = frames.astype(np.float32)
    new_frames /= (255 / 2)
    new_frames -= 1

    return new_frames

def clip_l2_diff(clip):
    """
    @param clip: A numpy array of shape [c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + c.OUT_LEN))].
    @return: The sum of l2 differences between the frame pixels of each sequential pair of frames.
    """
    diff = 0
    for i in xrange(c.HIST_LEN):
        frame = clip[:, :, 3 * i:3 * (i + 1)]
        next_frame = clip[:, :, 3 * (i + 1):3 * (i + 2)]
        # noinspection PyTypeChecker
        diff += np.sum(np.square(next_frame - frame))

    return diff

def get_full_clips(data_dir, num_clips, num_frame_out = c.OUT_LEN):
    # Loads a batch of random clips from the unprocessed train or test data.
    clips = np.empty([num_clips, c.FULL_HEIGHT, c.FULL_WIDTH, 3*(c.HIST_LEN + num_frame_out)])

    # get num_clips random episodes
    ep_dirs = np.random.choice(glob(os.path.join(data_dir, '*')), num_clips)

    # get a random clip of length HIST_LEN + num_rec_out from each episode
    for clip_num, ep_dir in enumerate(ep_dirs):
        ep_frame_paths = sorted(glob(os.path.join(ep_dir, '*')))
        start_index = np.random.choice(len(ep_frame_paths) - (c.HIST_LEN + num_frame_out - 1))
        clip_frame_paths = ep_frame_paths[start_index : start_index + (c.HIST_LEN + num_frame_out)]

        # read in frames
        for frame_num, frame_path in enumerate(clip_frame_paths):
            frame = imread(frame_path, mode='RGB')
            normalized_frame = normalize_frames(frame)

            clips[frame_num, :, :, frame_num * 3:(frame_num + 1) * 3] = normalized_frame
    return clips

def process_clip():
    # Gets a clip from the train dataset, cropped randomly to c.TRAIN_HEIGHT x c.TRAIN_WIDTH.
    clip = get_full_clips(c.TRAIN_DIR, 1)[0]

    # Randomly crop the clip. With 0.05 probability
    chose_first = np.random.choice(2, p=[0.95, 0.05])
    cropped_clip = np.empty([c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + c.OUT_LEN))])

    for i in xrange(100):
        crop_x = np.random.choice(c.FULL_WIDTH - c.TRAIN_WIDTH + 1)
        crop_y = np.random.choice(c.FULL_HEIGHT - c.TRAIN_HEIGHT + 1)
        cropped_clip = clip[crop_y:crop_y + c.TRAIN_HEIGHT, crop_x:crop_x + c.TRAIN_WIDTH, :]

        if chose_first or clip_l2_diff(cropped_clip) > c.MOVEMENT_THRESHOLD:
            break

    return cropped_clip

def get_train_batch():
    """
    Loads c.BATCH_SIZE clips from the database of preprocessed training clips.
    @return: An array of shape
            [c.BATCH_SIZE, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + c.OUT_LEN))].
    """
    clips = np.empty([c.BATCH_SIZE, c.TRAIN_HEIGHT, c.TRAIN_WIDTH, (3 * (c.HIST_LEN + c.OUT_LEN))],
                     dtype=np.float32)
    for i in xrange(c.BATCH_SIZE):
        path = c.TRAIN_DIR_CLIPS + str(np.random.choice(c.NUM_CLIPS)) + '.npz'
        clip = np.load(path)['arr_0']

        clips[i] = clip

    return clips

def get_test_batch(test_batch_size, num_rec_out=1):
    """
    Gets a clip from the test dataset.
    @param test_batch_size: The number of clips.
    @param num_rec_out: The number of outputs to predict. Outputs > 1 are computed recursively,
                        using the previously-generated frames as input. Default = 1.
    @return: An array of shape:
             [test_batch_size, c.TEST_HEIGHT, c.TEST_WIDTH, (3 * (c.HIST_LEN + num_rec_out))].
             A batch of frame sequences with values normalized in range [-1, 1].
    """
    return get_full_clips(c.TEST_DIR, test_batch_size, num_rec_out=num_rec_out)
