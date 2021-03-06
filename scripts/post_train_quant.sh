# python main.py --mode post_train_quant --quantize \
#   --video_path data/ucf101/ucf101_videos/jpg \
#   --annotation_path data/ucf101/ucf101_01.json \
#   --model_name full_precision/mobilenet3d_ft_50ep.pth \
#   --n_classes 101 --model mobilenetv2 --dataset ucf101 \
#   --no_cuda

# python main.py --mode post_train_quant --quantize \
#  --video_path data/ucf101/ucf101_videos/jpg \
#  --annotation_path data/ucf101/ucf101_01.json \
#  --model_name full_precision/squeezenet_ft_50ep.pth \
#  --n_classes 101 --model squeezenet --dataset ucf101 \
#  --no_cuda --fuse

python main.py --mode post_train_quant --quantize \
 --video_path data/ucf101/ucf101_videos/jpg \
 --annotation_path data/ucf101/ucf101_01.json \
 --model_name full_precision/my_r3d50_U_200ep.pth \
 --n_classes 101 --model resnet --dataset ucf101 \
 --no_cuda

