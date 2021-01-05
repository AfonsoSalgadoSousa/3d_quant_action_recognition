import time
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

from src.core.utils import AverageMeter


def get_video_results(outputs, class_names, output_topk):
    sorted_scores, locs = torch.topk(outputs,
                                     k=min(output_topk, len(class_names)))

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[locs[i].item()],
            'score': sorted_scores[i].item()
        })

    return video_results


def inference(data_loader, model, result_path, class_names,
              output_topk, no_average=False, device='cuda'):
    print('inference')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    results = {'results': defaultdict(list)}

    end_time = time.time()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            if device == 'cuda':
                inputs = inputs.cuda()
            data_time.update(time.time() - end_time)

            video_ids, segments = zip(*targets)
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1).cpu()

            for j in range(outputs.size(0)):
                results['results'][video_ids[j]].append({
                    'segment': segments[j],
                    'output': outputs[j]
                })

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('[{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time))

    inference_results = {'results': {}}
    if not no_average:
        for video_id, video_results in results['results'].items():
            video_outputs = [
                segment_result['output'] for segment_result in video_results
            ]
            video_outputs = torch.stack(video_outputs)
            average_scores = torch.mean(video_outputs, dim=0)
            inference_results['results'][video_id] = get_video_results(
                average_scores, class_names, output_topk)
    else:
        for video_id, video_results in results['results'].items():
            inference_results['results'][video_id] = []
            for segment_result in video_results:
                segment = segment_result['segment']
                result = get_video_results(segment_result['output'],
                                           class_names, output_topk)
                inference_results['results'][video_id].append({
                    'segment': segment,
                    'result': result
                })

    with result_path.open('w') as f:
        json.dump(inference_results, f)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader, device, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            targets = torch.from_numpy(np.asarray(targets)) 
            targets = targets.to(device)
            for frame_idx in range(inputs.shape[2]):
                frame = inputs[:, :, frame_idx,:,:]
                frame = frame.to(device)
                output = model(frame)
                loss = criterion(output, targets)
                cnt += 1
                acc1 = accuracy(output, targets, topk=(1,))
                print('.', end = '')
                top1.update(acc1[0], frame.size(0))
                if cnt >= neval_batches:
                    return top1
    return top1

def evaluate3d(model, criterion, data_loader, neval_batches=None):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for inputs, target in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, target)
            cnt += 1
            acc1, acc5 = accuracy(outputs, target, topk=(1, 5))
            print('.', end = '')
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            if neval_batches and cnt >= neval_batches:
                 return top1, top5

    return top1, top5