import argparse
from pathlib import Path


def parse_opts():

    default_root = Path(__file__).resolve().parent.parent.parent

    parser = argparse.ArgumentParser()
    # Path Arguments
    # parser.add_argument('--root_path',
    #                     default=default_root,
    #                     type=Path,
    #                     help='Root directory path')
    parser.add_argument('--video_path', default='', type=Path,
                        help='Directory path of videos')
    parser.add_argument('--annotation_path', default='',
                        type=Path, help='Annotation file path')
    parser.add_argument('--dataset',
                        default='emotiw',
                        type=str,
                        help='Used dataset (emotiw | ucf101)')
    # Emotiw specific
    # parser.add_argument('--train_frames_path',
    #                     default=default_root / 'data' / 'emotiw' / 'Train_Frames_10/',
    #                     type=Path,
    #                     help='Train frames directory path')
    # parser.add_argument('--val_frames_path',
    #                     default=default_root / 'data' / 'emotiw' / 'Val_Frames_10/',
    #                     type=Path,
    #                     help='Val frames directory path')
    # parser.add_argument('--train_labels_path',
    #                     default=default_root / 'data' / 'emotiw' / 'Train_labels.txt',
    #                     type=Path,
    #                     help='Train labels file path')
    # parser.add_argument('--val_labels_path',
    #                     default=default_root / 'data' / 'emotiw' / 'Val_labels.txt',
    #                     type=Path,
    #                     help='Val labels file path')
    parser.add_argument('--model_path',
                        default=default_root / 'model',
                        type=Path,
                        help='Model directory path')
    # Model Arguments
    parser.add_argument('--model',
                        default='resnet',
                        type=str,
                        help='(resnet | resnet2p1d | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument('--model_depth',
                        default=50,
                        type=int,
                        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--conv1_t_size',
                        default=7,
                        type=int,
                        help='Kernel size in t dim of conv1.')
    parser.add_argument('--conv1_t_stride',
                        default=1,
                        type=int,
                        help='Stride in t dim of conv1.')
    parser.add_argument('--no_max_pool',
                        action='store_true',
                        help='If true, the max pooling after conv1 is removed.')
    parser.add_argument('--n_classes',
                        default=3,
                        type=int,
                        help='Number of classes (activitynet: 200, kinetics: 400 or 600, ucf101: 101, hmdb51: 51)')
    parser.add_argument('--resnet_shortcut',
                        default='B',
                        type=str,
                        help='Shortcut type of resnet (A | B)')
    parser.add_argument('--widen_factor',
                        default=1.0,
                        type=float,
                        help='The number of feature maps is multiplied by this value')
    parser.add_argument('--resnext_cardinality',
                        default=32,
                        type=int,
                        help='ResNeXt cardinality')
    parser.add_argument('--version', default=1.1,
                        type=float, help='Version of SqueezeNet')

    parser.add_argument('--mode',
                        type=str,)
    # required=True,
    # choices=['train', 'soft_prune', 'prune', 'test', 'post_train_quant',
    # 'quant_aware', 'knowledge_distillation', 'count_flops'])
    parser.add_argument('--model_name',
                        type=str)
    # choices=['resnet50.pth', 'soft_prune_resnet50.pth', 'pruned_resnet50.pth'])
    
    # Fine Tuning Arguments
    parser.add_argument('--pretrain_path', default='',
                        type=str, help='Pretrained model (.pth)')
    parser.add_argument('--n_finetune_classes', default=400, type=int,
                        help='Number of classes for fine-tuning. n_classes is set to the number when pretraining.')
    parser.add_argument('--ft_portion', default='complete', type=str,
                        help='The portion of the model to apply fine tuning, either complete or last_layer')
    # Training Arguments
    parser.add_argument('--learning_rate',
                        default=0.1,
                        type=float,
                        help=('Initial learning rate'
                              '(divided by 10 while training by lr scheduler)'))
    parser.add_argument('--multistep_milestones',
                        default=[50, 100, 150],
                        type=int,
                        nargs='+',
                        help='Milestones of LR scheduler. See documentation of MultistepLR.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weight_decay',
                        default=1e-3,
                        type=float,
                        help='Weight Decay')
    parser.add_argument('--n_epochs',
                        default=50,
                        type=int,
                        help='Number of total epochs to run')
    parser.add_argument('--batch_size',
                        default=128,
                        type=int,
                        help='Batch Size')
    parser.add_argument('--manual_seed',
                        default=42,
                        type=int,
                        help='Manually set random seed')
    parser.add_argument('--checkpoint',
                        default=10,
                        type=int,
                        help='Trained model is saved at every this epochs.')
    parser.add_argument('--no_cuda',
                        action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--n_threads',
                        default=4,
                        type=int,
                        help='Number of threads for multi-thread loading')
    parser.add_argument('--sample_size',
                        default=112,
                        type=int,
                        help='Height and width of inputs')
    parser.add_argument('--n_frames',
                        default=10,
                        type=int,
                        help='Number of frames per video')
    parser.add_argument('--sample_duration',
                        default=16,
                        type=int,
                        help='Temporal duration of inputs')
    # Data Augmentation Arguments
    parser.add_argument('--train_crop_min_scale',
                        default=0.25,
                        type=float,
                        help='Min scale for random cropping in training')
    parser.add_argument('--train_crop_min_ratio',
                        default=0.75,
                        type=float,
                        help='Min aspect ratio for random cropping in training')
    # Pruning Arguments
    parser.add_argument('--pruning_rate', default=0.9,
                        type=float, help='Pruning rate')
    parser.add_argument('--epoch_prune', default=1,
                        type=float, help='Prune every X epochs')
    # Preprocessing Arguments
    parser.add_argument('--value_scale',
                        default=1,
                        type=int,
                        help='If 1, range of inputs is [0-1]. If 255, range of inputs is [0-255].')
    parser.add_argument('--mean_dataset',
                        default='kinetics',
                        type=str,
                        help=('dataset for mean values of mean subtraction'
                              '(activitynet | kinetics | 0.5)'))
    parser.add_argument('--no_mean_norm',
                        action='store_true',
                        help='If true, inputs are not normalized by mean.')
    parser.add_argument('--no_std_norm',
                        action='store_true',
                        help='If true, inputs are not normalized by standard deviation.')
    # Inference Arguments
    parser.add_argument('--inference_stride',
                        default=16,
                        type=int,
                        help='Stride of sliding window in inference.')
    parser.add_argument('--output_topk',
                        default=5,
                        type=int,
                        help='Top-k scores are saved in json file.')
    parser.add_argument('--load_full',
                        action='store_true',
                        help='If true, load full model.')
    parser.add_argument(
        '--inference_no_average',
        action='store_true',
        help='If true, outputs for segments in a video are not averaged.')
    # Quantization Arguments
    parser.add_argument('--fuse',
                        action='store_true',
                        help='If true, subsequent conv/bn/relu modules are fused.')
    parser.add_argument('--temperature',
                        default=20,
                        type=int,
                        help='Temperature of quatization function.')
    parser.add_argument('--quantize_input',
                        action='store_true',
                        help='If true, input is quantized.')
    parser.add_argument('--quantize',
                        action='store_true',
                        help='If true, quantize model.')

    args = parser.parse_args()

    # if (args.mode == 'prune' or args.mode == 'test') and args.model_name is None:
    #    parser.error("Pruning and testing requires model name.")
    return args
