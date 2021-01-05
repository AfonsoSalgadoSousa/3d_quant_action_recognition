import torch

from src.core.spatial_transforms import (CenterCrop, ColorJitter, Compose,
                                         CornerCrop, MultiScaleCornerCrop,
                                         Normalize, PickFirstChannels,
                                         RandomHorizontalFlip,
                                         RandomResizedCrop, Resize, ScaleValue,
                                         ToTensor)
from src.core.temporal_transforms import Compose as TemporalCompose
from src.core.temporal_transforms import (LoopPadding, SlidingWindow,
                                          TemporalCenterCrop, TemporalEvenCrop,
                                          TemporalRandomCrop,
                                          TemporalSubsampling)
from src.dataset.dataset import get_inference_data, get_training_data


def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)


def get_train_loader(opt):
    spatial_transform = []
    spatial_transform.append(
        RandomResizedCrop(
            opt.sample_size, (opt.train_crop_min_scale, 1.0),
            (opt.train_crop_min_ratio, 1.0 / opt.train_crop_min_ratio)))
    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)
    spatial_transform.append(RandomHorizontalFlip())
    spatial_transform.append(ToTensor())
    spatial_transform.append(ScaleValue(opt.value_scale))
    spatial_transform.append(normalize)
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    temporal_transform.append(TemporalRandomCrop(opt.sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)

    train_data = get_training_data(opt,
                                   spatial_transform=spatial_transform,
                                   temporal_transform=temporal_transform)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.n_threads,
                                               pin_memory=True,
                                               )
    print("Size of train loader is " + str(len(train_loader)))
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    print("Images shape:")
    print(images.shape)
    print("Labels shape:")
    print(labels.shape)

    return train_loader


def get_inference_loader(opt):
    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)

    spatial_transform = [Resize(opt.sample_size)]
    spatial_transform.append(CenterCrop(opt.sample_size))
    spatial_transform.append(ToTensor())
    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    temporal_transform.append(
        SlidingWindow(opt.sample_duration, opt.inference_stride))
    temporal_transform = TemporalCompose(temporal_transform)

    inference_data, collate_fn = get_inference_data(opt,
                                                    spatial_transform,
                                                    temporal_transform)

    inference_loader = torch.utils.data.DataLoader(
        inference_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True,
        collate_fn=collate_fn)

    print("Size of val loader is " + str(len(inference_loader)))

    return inference_loader, inference_data.class_names
