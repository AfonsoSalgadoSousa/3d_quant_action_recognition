import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from torchvision.models import resnet50

from src.core import inference, utils
from src.core.opts import parse_opts
from src.dataset.loader import get_train_loader, get_inference_loader
from src.models.model import generate_model, make_data_parallel, resume_model


def get_opt():
    opt = parse_opts()

    opt.begin_epoch = 1
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.n_input_channels = 3
    opt.mean, opt.std = utils.get_mean_std(
        opt.value_scale, dataset=opt.mean_dataset)
    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    opt.inference_target_type = ['video_id', 'segment']

    return opt


def main():
    opt = get_opt()

    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    
    # Regular Training
    if opt.mode == 'train':
        from src.training import train_3d
        train_3d(opt)

    # Quantization
    elif opt.mode == 'quant_aware':
        from src.core.trainer import train_epoch
        opt.quantize = True
        model, parameters = generate_model(opt)
        qat_model = resume_model(
            (opt.model_path / opt.model_name), model)
        print("Size of model before quantization")
        utils.print_size_of_model(qat_model)
        model = make_data_parallel(model, opt.device)
        qat_model.qconfig = torch.quantization.get_default_qat_qconfig(
            'fbgemm')
        print(qat_model.qconfig)
        torch.quantization.prepare_qat(qat_model, inplace=True)

        criterion = nn.CrossEntropyLoss()
        train_loader = get_train_loader(opt)

        optimizer = SGD(parameters,
                        lr=opt.learning_rate,
                        momentum=opt.momentum,
                        weight_decay=opt.weight_decay)

        for i in range(8):
            train_epoch(i, train_loader, model, criterion, optimizer,
                        opt.device)
            if i > 3:
                # Freeze quantizer parameters
                qat_model.apply(torch.quantization.disable_observer)
            if i > 2:
                # Freeze batch norm mean and variance estimates
                qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        save_file_path = (opt.model_path / 'quantized' /
                          ('qat_' + opt.model_name.split("/")[1])).as_posix()
        torch.jit.save(torch.jit.script(qat_model), save_file_path)
        mq = torch.jit.load(save_file_path)
        print(mq)

    elif opt.mode == 'post_train_quant':
        from src.core.inference import evaluate3d
        opt.quantize = True
        model, _ = generate_model(opt)
        fp_model = resume_model(
            (opt.model_path / opt.model_name), model)
        print("Size of model before quantization")
        utils.print_size_of_model(fp_model)
        fp_model.eval()
        if opt.fuse:
            fp_model.fuse_model()
        fp_model.qconfig = torch.quantization.default_qconfig
        print(fp_model.qconfig)
        torch.quantization.prepare(fp_model, inplace=True)

        criterion = nn.CrossEntropyLoss()
        train_loader = get_train_loader(opt)
        evaluate3d(fp_model, criterion, train_loader, neval_batches=1)
        torch.quantization.convert(fp_model, inplace=True)
        print("Size of model after quantization")
        utils.print_size_of_model(fp_model)
        save_file_path = (opt.model_path / 'quantized' /
                          ('quant_' + opt.model_name.split("/")[1])).as_posix()
        torch.jit.save(torch.jit.script(fp_model), save_file_path)
        mq = torch.jit.load(save_file_path)
        print(mq)

    elif opt.mode == 'test':
        from src.testing import test_3d
        test_3d(opt)

if __name__ == '__main__':
    main()
