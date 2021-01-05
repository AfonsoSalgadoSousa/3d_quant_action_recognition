import time

import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from src.core import inference, utils
from src.core.inference import evaluate3d
from src.dataset.loader import get_inference_loader, get_train_loader
from src.models.model import generate_model, resume_model


def test_3d(opt):
    if opt.quantize:
        model = torch.jit.load((opt.model_path / opt.model_name).as_posix())
    elif opt.load_full:
        model = torch.load(
            (opt.model_path / opt.model_name), map_location='cpu')
    else:
        model, _ = generate_model(opt)
        model = resume_model(
            (opt.model_path / opt.model_name), model)

    # utils.print_size_of_model(model)

    model.to(opt.device)
    print(opt.device)

    inference_loader, inference_class_names = get_inference_loader(opt)

    start_time = int(round(time.time()*1000))
    inference_result_path = opt.model_path / 'val.json'
    inference.inference(inference_loader, model, inference_result_path,
                        inference_class_names, opt.output_topk, device=opt.device)

    elapsed_time = int(round(time.time()*1000)) - start_time
    print('Train elapsed time: {:.2f} seconds.'.format(
        elapsed_time / 1000))


def test_2d(opt):
    start_time = int(round(time.time()*1000))

    model = resnet18(pretrained=False)
    # model = resume_model(opt.model_path / '2d_resnet18_10ep.pth', model)
    # model = resume_model(opt.model_path / 'quant_resnet18.pth', model)
    checkpoint = torch.load(opt.model_path / 'quant_resnet18.pth')
    model.load_state_dict(checkpoint['net_state_dict'])

    params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Number of Parameters: {:.1f}M".format(params/1e6))
    utils.print_size_of_model(model)

    model.to(opt.device)

    criterion = CrossEntropyLoss().to(opt.device)
    opt.inference_target_type = 'label'
    inference_loader, _ = get_inference_loader(opt)
    top1 = inference.evaluate(
        model, criterion, inference_loader, opt.device, neval_batches=opt.batch_size)

    print("Top-1 Average: {}".format(top1.avg))
    elapsed_time = int(round(time.time()*1000)) - start_time
    print('Train elapsed time: {:.2f} seconds.'.format(
        elapsed_time / 1000))
    params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Number of Parameters: {:.1f}M".format(params/1e6))
