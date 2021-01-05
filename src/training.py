from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler

from src.models.model import (generate_model, make_data_parallel,
                                  resume_model)
from src.core.trainer import train_2D_epoch, train_epoch
from src.dataset.loader import get_train_loader
from src.core import utils


def train_3d(opt):
    model, parameters = generate_model(opt)
    model = make_data_parallel(model, opt.device)
    train_loader = get_train_loader(opt)
    criterion = CrossEntropyLoss().to(opt.device)
    optimizer = SGD(parameters,
                    lr=opt.learning_rate,
                    momentum=opt.momentum,
                    weight_decay=opt.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         opt.multistep_milestones)

    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        train_epoch(i, train_loader, model, criterion, optimizer,
                    opt.device)
        if i % opt.checkpoint == 0:
            save_file_path = opt.model_path / 'save_{}.pth'.format(i)
            utils.save_checkpoint(save_file_path, i, opt.arch, model, optimizer,
                                  scheduler)
        scheduler.step()


def train_2d(opt):
    float_model = quant_resnet.resnet50()
    float_model = make_data_parallel(float_model, opt.device)

    train_loader = get_train_loader(opt)
    criterion = CrossEntropyLoss().to(opt.device)
    optimizer = SGD(float_model.parameters(),
                    lr=opt.learning_rate,
                    momentum=opt.momentum,
                    weight_decay=opt.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         opt.multistep_milestones)

    for i in range(opt.begin_epoch, 11):
        train_2D_epoch(i, train_loader, float_model, criterion, optimizer,
                       opt.device)
        scheduler.step()
    save_file_path = opt.model_path / '2d_resnet50_10ep.pth'
    utils.save_checkpoint(save_file_path, i, opt.arch, float_model, optimizer,
                          scheduler)


def inflated_train(opt):
    float_model = quant_resnet.resnet50()
    float_model = resume_model(
        (opt.model_path / '2d_resnet50_10ep.pth'), opt.arch, float_model)

    model = I3ResNet(copy.deepcopy(float_model),
                     opt.sample_duration, class_nb=opt.n_classes)
    model = make_data_parallel(model, opt.device)

    train_loader = get_train_loader(opt)
    criterion = CrossEntropyLoss().to(opt.device)
    optimizer = SGD(model.parameters(),
                    lr=opt.learning_rate,
                    momentum=opt.momentum,
                    weight_decay=opt.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         opt.multistep_milestones)

    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        train_inflate_epoch(i, train_loader, model, float_model, criterion, optimizer,
                            opt.device)
        if i % opt.checkpoint == 0:
            save_file_path = opt.model_path / 'save_{}.pth'.format(i)
            utils.save_checkpoint(save_file_path, i, opt.arch, model, optimizer,
                                  scheduler)
        scheduler.step()
