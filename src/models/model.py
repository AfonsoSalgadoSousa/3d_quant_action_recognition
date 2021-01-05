import torch
from torch import nn

from src.models import (densenet, resnet, resnet2p1d,
                        resnext, c3d, squeezenet, shufflenetv2, mobilenetv2)


def generate_model(opt):
    assert opt.model in [
        'resnet', 'resnet2p1d', 'preresnet', 'wideresnet', 'resnext',
        'densenet', 'c3d', 'squeezenet', 'shufflenetv2', 'mobilenetv2'
    ]

    print("Generating {}".format(opt.model))

    if opt.model == 'resnet':
        model = resnet.generate_model(model_depth=opt.model_depth,
                                      n_classes=opt.n_classes,
                                      n_input_channels=opt.n_input_channels,
                                      shortcut_type=opt.resnet_shortcut,
                                      conv1_t_size=opt.conv1_t_size,
                                      conv1_t_stride=opt.conv1_t_stride,
                                      no_max_pool=opt.no_max_pool,
                                      widen_factor=opt.widen_factor,
                                      quantize=opt.quantize)
    elif opt.model == 'resnet2p1d':
        model = resnet2p1d.generate_model(model_depth=opt.model_depth,
                                          n_classes=opt.n_classes,
                                          n_input_channels=opt.n_input_channels,
                                          shortcut_type=opt.resnet_shortcut,
                                          conv1_t_size=opt.conv1_t_size,
                                          conv1_t_stride=opt.conv1_t_stride,
                                          no_max_pool=opt.no_max_pool,
                                          widen_factor=opt.widen_factor)
    elif opt.model == 'resnext':
        model = resnext.generate_model(model_depth=opt.model_depth,
                                       cardinality=opt.resnext_cardinality,
                                       n_classes=opt.n_classes,
                                       n_input_channels=opt.n_input_channels,
                                       shortcut_type=opt.resnet_shortcut,
                                       conv1_t_size=opt.conv1_t_size,
                                       conv1_t_stride=opt.conv1_t_stride,
                                       no_max_pool=opt.no_max_pool)
    elif opt.model == 'densenet':
        model = densenet.generate_model(model_depth=opt.model_depth,
                                        n_classes=opt.n_classes,
                                        n_input_channels=opt.n_input_channels,
                                        conv1_t_size=opt.conv1_t_size,
                                        conv1_t_stride=opt.conv1_t_stride,
                                        no_max_pool=opt.no_max_pool)

    elif opt.model == 'c3d':
        from src.models.c3d import get_fine_tuning_parameters
        model = c3d.get_model(num_classes=opt.n_classes,
                              sample_size=opt.sample_size,
                              sample_duration=opt.sample_duration)
    elif opt.model == 'squeezenet':
        from src.models.squeezenet import get_fine_tuning_parameters
        model = squeezenet.get_model(version=opt.version,
                                     num_classes=opt.n_classes,
                                     sample_size=opt.sample_size,
                                     sample_duration=opt.sample_duration,
                                     quantize=opt.quantize)
    elif opt.model == 'shufflenetv2':
        from src.models.shufflenetv2 import get_fine_tuning_parameters
        model = shufflenetv2.get_model(num_classes=opt.n_classes,
                                       sample_size=opt.sample_size,
                                       width_mult=opt.widen_factor)
    elif opt.model == 'mobilenetv2':
        from src.models.mobilenetv2 import get_fine_tuning_parameters
        model = mobilenetv2.get_model(num_classes=opt.n_classes,
                                      sample_size=opt.sample_size,
                                      width_mult=opt.widen_factor,
                                      quantize=opt.quantize)

    # Fine tuning
    if opt.pretrain_path:
        print('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        # assert opt.arch == pretrain['arch']
        model.load_state_dict(pretrain['state_dict'])

        if opt.model in ['mobilenet', 'mobilenetv2', 'shufflenet', 'shufflenetv2']:
            model.classifier = nn.Sequential(
                nn.Dropout(0.9),
                nn.Linear(
                    model.classifier[1].in_features, opt.n_finetune_classes)
            )
        elif opt.model == 'squeezenet':
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Conv3d(model.classifier[1].in_channels,
                          opt.n_finetune_classes, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AvgPool3d((1, 4, 4), stride=1))
        else:
            model.fc = nn.Linear(model.fc.in_features, opt.n_finetune_classes)

        # parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
        parameters = get_fine_tuning_parameters(model, opt.ft_portion)
        return model, parameters

    return model, model.parameters()


def make_data_parallel(model, device):
    print("Device: ", device)
    if device.type == 'cuda':
        model = nn.DataParallel(model, device_ids=None).cuda()

    return model


def resume_model(resume_path, model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    # assert arch == checkpoint['arch']

    if hasattr(model, 'module'):  # for models saved as DataParallel
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    return model
