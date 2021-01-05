import glob
import os

import numpy as np
import torch
from PIL import Image

from src.dataset.videodataset import (VideoDataset, VideoDatasetMultiClips,
                                      collate_fn)
from src.dataset.ucf101 import UCF101, MultiClips

from torchvision import transforms


def ucf101_name_formatter(x):
    return f'image_{x:05d}.jpg'


class ImageLoaderPIL(object):

    def __call__(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with path.open('rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


class VideoLoader(object):
    def __init__(self, image_name_formatter):
        self.image_name_formatter = image_name_formatter
        self.image_loader = ImageLoaderPIL()

    def __call__(self, video_path, frame_indices):
        video = []
        for i in frame_indices:
            image_path = video_path / self.image_name_formatter(i)
            if image_path.exists():
                video.append(self.image_loader(image_path))

        return video


def get_training_data(opt,
                      spatial_transform=None,
                      temporal_transform=None,
                      target_transform=None):

    elif opt.dataset == 'ucf101':
        training_data = UCF101(opt.video_path,
                            opt.annotation_path,
                            'training',
                            spatial_transform=spatial_transform,
                            temporal_transform=temporal_transform,
                            target_transform=target_transform,
                            video_loader=VideoLoader(ucf101_name_formatter))

    return training_data


def get_inference_data(opt,
                       spatial_transform=None,
                       temporal_transform=None,
                       target_transform=None):

    elif opt.dataset == 'ucf101':
        inference_data = MultiClips(
            opt.video_path,
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            video_loader=VideoLoader(ucf101_name_formatter),
            target_type=opt.inference_target_type)

    return inference_data, collate_fn
