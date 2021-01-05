# python main.py --mode test --video_path data/ucf101/ucf101_videos/jpg \
# --annotation_path data/ucf101/ucf101_01.json \
# --model_name full_precision/mobilenet3d_ft_50ep.pth --n_classes 101 \
# --output_topk 5 --model mobilenetv2 --dataset ucf101 --batch_size 8 \
# --no_cuda

# python main.py --mode test --video_path data/ucf101/ucf101_videos/jpg \
# --annotation_path data/ucf101/ucf101_01.json \
# --model_name full_precision/squeezenet_ft_50ep.pth --n_classes 101 \
# --output_topk 5 --model squeezenet --dataset ucf101 --batch_size 8 \
# --no_cuda

# python main.py --mode test --video_path data/ucf101/ucf101_videos/jpg \
# --annotation_path data/ucf101/ucf101_01.json \
# --model_name full_precision/shufflenetv2_ft_50ep.pth --n_classes 101 \
# --output_topk 5 --model shufflenetv2 --dataset ucf101 --batch_size 8 \
# --no_cuda

python main.py --mode test --video_path data/ucf101/ucf101_videos/jpg \
--annotation_path data/ucf101/ucf101_01.json \
--model_name full_precision/my_r3d50_U_200ep.pth --n_classes 101 \
--output_topk 5 --model resnet --dataset ucf101 --batch_size 4 \
--no_cuda



# FOR INFERENCE ON QUANTIZED MODEL

# python main.py --mode test --video_path data/ucf101/ucf101_videos/jpg \
# --annotation_path data/ucf101/ucf101_01.json \
# --model_name quantized/quant_mobilenet3d_ft_50ep.pth --n_classes 101 \
# --output_topk 5 --model mobilenetv2 --dataset ucf101 --batch_size 8 \
# --no_cuda --quantize

# python main.py --mode test --video_path data/ucf101/ucf101_videos/jpg \
# --annotation_path data/ucf101/ucf101_01.json \
# --model_name quantized/quant_squeezenet_ft_50ep.pth --n_classes 101 \
# --output_topk 5 --model squeezenet --dataset ucf101 --batch_size 8 \
# --no_cuda --quantize

# python main.py --mode test --video_path data/ucf101/ucf101_videos/jpg \
# --annotation_path data/ucf101/ucf101_01.json \
# --model_name quantized/quant_shufflenetv2_ft_50ep.pth --n_classes 101 \
# --output_topk 5 --model shufflev2 --dataset ucf101 --batch_size 8 \
# --no_cuda --quantize

# python main.py --mode test --video_path data/ucf101/ucf101_videos/jpg \
# --annotation_path data/ucf101/ucf101_01.json \
# --model_name quantized/qat_squeezenet_ft_50ep.pth --n_classes 101 \
# --output_topk 5 --model squeezenet --dataset ucf101 --batch_size 8 \
# --no_cuda --quantize

# python main.py --mode test --video_path data/ucf101/ucf101_videos/jpg \
# --annotation_path data/ucf101/ucf101_01.json \
# --model_name quantized/quant_my_r3d50_U_200ep.pth --n_classes 101 \
# --output_topk 5 --model resnet --dataset ucf101 --batch_size 8 \
# --no_cuda --quantize


