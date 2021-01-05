# python main.py --mode train --n_epochs 50 \
#  --video_path data/ucf101/ucf101_videos/jpg \
#  --annotation_path data/ucf101/ucf101_01.json \
#  --pretrain_path model/pretrained/kinetics_mobilenetv2_1.0x.pth \
#  --n_classes 600 --n_finetune_classes 101 --ft_portion last_layer \
#  --model mobilenetv2 --dataset ucf101

# python main.py --mode train --n_epochs 50 \
#  --video_path data/ucf101/ucf101_videos/jpg \
#  --annotation_path data/ucf101/ucf101_01.json \
#  --pretrain_path model/pretrained/kinetics_squeezenet.pth \
#  --n_classes 600 --n_finetune_classes 101 --ft_portion last_layer \
#  --model squeezenet --dataset ucf101

python main.py --mode train --n_epochs 50 \
 --video_path data/ucf101/ucf101_videos/jpg \
 --annotation_path data/ucf101/ucf101_01.json \
 --pretrain_path model/pretrained/kinetics_shufflenetv2_1.0x.pth \
 --n_classes 600 --n_finetune_classes 101 --ft_portion last_layer \
 --model shufflenetv2 --dataset ucf101
