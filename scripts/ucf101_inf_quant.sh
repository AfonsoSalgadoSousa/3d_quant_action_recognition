python main.py --mode test --video_path data/ucf101/ucf101_videos/jpg \
--annotation_path data/ucf101/ucf101_01.json \
--model_name quantized/quantized_mobilenet_sd.pth --n_classes 101 \
--output_topk 5 --model mobilenetv2 --dataset ucf101 --batch_size 8 \
--no_cuda --quantize
