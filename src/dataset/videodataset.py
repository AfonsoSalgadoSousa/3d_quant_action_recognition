import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate


class VideoDataset(data.Dataset):

    def __init__(self,
                 video_path,
                 annotation_path,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 video_loader=None,
                 target_type='label'):
        self.data, self.class_names = self.__make_dataset(
            video_path, annotation_path)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        self.loader = video_loader

        self.target_type = target_type

    def __make_dataset(self, video_path, annotation_path):
        self.video_path = video_path

        self.annotations = np.loadtxt(annotation_path, dtype={'names': (
            'file', 'label'), 'formats': ('U10', np.int8)}, delimiter=' ', skiprows=1)

        n_videos = len(self.annotations)
        dataset = []
        for i, (video_id, label_id) in enumerate(self.annotations):
            if i % (n_videos // 5) == 0:
                print('dataset loading [{}/{}]'.format(i, n_videos))

            segment = [1, 11]
            frame_indices = list(range(segment[0], segment[1]))
            sample = {
                'video': video_path / (video_id + '.mp4'),
                'segment': segment,
                'frame_indices': frame_indices,
                'video_id': video_id + '.mp4',
                'label': label_id - 1
            }
            dataset.append(sample)

        return dataset, {0: 'Neutral', 1: 'Positive', 2: 'Negative'}

    def __loading(self, path, frame_indices):
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip

    def __getitem__(self, index):
        path = self.data[index]['video']
        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        clip = self.__loading(path, frame_indices)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    batch_clips, batch_targets = zip(*batch)

    batch_clips = [clip for multi_clips in batch_clips for clip in multi_clips]
    batch_targets = [
        target for multi_targets in batch_targets for target in multi_targets
    ]

    target_element = batch_targets[0]
    if isinstance(target_element, int) or isinstance(target_element, str):
        return default_collate(batch_clips), default_collate(batch_targets)
    else:
        return default_collate(batch_clips), batch_targets


class VideoDatasetMultiClips(VideoDataset):

    def __loading(self, path, video_frame_indices):
        clips = []
        segments = []
        for clip_frame_indices in video_frame_indices:
            clip = self.loader(path, clip_frame_indices)
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]
            clips.append(torch.stack(clip, 0).permute(1, 0, 2, 3))
            segments.append(
                [min(clip_frame_indices),
                 max(clip_frame_indices) + 1])

        return clips, segments

    def __getitem__(self, index):
        path = self.data[index]['video']

        video_frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            video_frame_indices = self.temporal_transform(video_frame_indices)

        clips, segments = self.__loading(path, video_frame_indices)

        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        if 'segment' in self.target_type:
            if isinstance(self.target_type, list):
                segment_index = self.target_type.index('segment')
                targets = []
                for s in segments:
                    targets.append(copy.deepcopy(target))
                    targets[-1][segment_index] = s
            else:
                targets = segments
        else:
            targets = [target for _ in range(len(segments))]

        return clips, targets
