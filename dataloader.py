import transform

import cv2
import torch
import numpy as np

import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image

import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import glob
import copy

class Preprocess(object):
    def __call__(self,sample):
        sample['flow'] = np.stack(sample['flow'])
        sample['rgb'] = np.stack(sample['rgb'])
        sample = transform.preprocess_rgb_flow(sample)
        flow = torch.from_numpy(sample['flow'])
        rgb = torch.from_numpy(sample['rgb'])
        flow = flow.permute(3, 0, 1, 2)
        rgb = rgb.permute(3, 0, 1, 2)
        sample['flow'] = flow.float()
        sample['rgb'] = rgb.float()
        return sample
    
class TemporalSample(object):
    def __init__(self, clip_len=8, stride=1, random=True):
        self.clip_len = clip_len
        self.stride = stride
        self.random = random
        
    def __call__(self,sample):
        flow_len = len(sample['flow'])
        start =0
        if self.random:
            max_index = flow_len - self.stride * self.clip_len
            #print(flow_len, max_index)
            #start = np.random.randint(0,flow_len - self.stride * self.clip_len)
            try:
                start = np.random.randint(0, max_index)
            except ValueError:
                print(sample['name'],flow_len, max_index)
        else:
            start = flow_len // 2
        end = start + self.clip_len * self.stride
        sample['flow'] = sample['flow'][start:end:self.stride]
        sample['rgb'] = sample['rgb'][start:end:self.stride]
        return sample

class SpatialSample(object):
    def __init__(self, spatial_idx=-1,
                    min_scale=256,
                    max_scale=320,
                    crop_size=224,
                    random_horizontal_flip=True,
                    inverse_uniform_sampling=False,
                    aspect_ratio=None,
                    scale=None,
                    motion_shift=False,):
        self.spatial_idx=spatial_idx
        self.min_scale=min_scale
        self.max_scale=max_scale
        self.crop_size=crop_size
        self.random_horizontal_flip=random_horizontal_flip
        self.inverse_uniform_sampling=inverse_uniform_sampling
        self.aspect_ratio=aspect_ratio
        self.scale=scale
        self.motion_shift=motion_shift

    def __call__(self, sample):
        return spatial_sampling(sample,
        spatial_idx=self.spatial_idx,
        min_scale=self.min_scale,
        max_scale=self.max_scale,
        crop_size=self.crop_size,
        random_horizontal_flip=self.random_horizontal_flip,
        inverse_uniform_sampling=self.inverse_uniform_sampling,
        aspect_ratio=self.aspect_ratio,
        scale=self.scale,
        motion_shift=self.motion_shift)


def spatial_sampling(
    sample,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            sample = transform.random_short_side_scale_jitter(
                sample=sample,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            sample = transform.random_crop(sample, crop_size)
        if random_horizontal_flip:
            sample = transform.horizontal_flip(0.5, sample)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale}) == 1
        sample = transform.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        sample = transform.uniform_crop(frames, crop_size, spatial_idx)
    return sample

def load_video(video_fname):
    frames = []
    cap = cv2.VideoCapture(video_fname)
    ret = True
    while ret:
        ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
        if ret:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img_rgb)
    #print(len(frames), video_fname)
    return frames

class VideoFlowDataset(Dataset):
    def __init__(self, video_folder='official', flow_folder='official_flow'):
        super().__init__()
        self.video_folder = video_folder
        self.flow_folder = flow_folder
        
        self.all_videos_with_flow = glob.glob(os.path.join(flow_folder,'**/*.mp4'), recursive=True)

        print(f'{len(self.all_videos_with_flow)} videos found')
        self.transform = transforms.Compose([TemporalSample(), Preprocess(), SpatialSample()])
        
    def __getitem__(self, index):
        flow_fname = self.all_videos_with_flow[index]
        video_fname= self.all_videos_with_flow[index].replace(self.flow_folder,self.video_folder)
        #print(flow_fname, video_fname)
        flow = load_video(flow_fname) # list of 3ch images
        rgb = load_video(video_fname) # list of 3ch images
        sample = {'rgb':rgb, 'flow':flow, 'name':video_fname}
        return self.transform(sample)
    
    def __len__(self):
        return len(self.all_videos_with_flow)