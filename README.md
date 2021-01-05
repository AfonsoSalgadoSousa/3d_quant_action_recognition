# Analysis of Uniform Quantization for Action Recognition
PyTorch implementation of uniform quantization on 3D CNN architectures for action recognition using UCF-101 dataset.

## Requirements

* PyTorch
* OpenCV
* FFmpeg, FFprobe
* Python 3

## Available models:
 - 3D Resnet
 - 3D SqueezeNet
 - 3D MobileNetv2

## Downloading UCF-101 Dataset

* Download videos and train/test splits [here](http://crcv.ucf.edu/data/UCF101.php).
* Convert from avi to jpg files using ```util_scripts/generate_video_jpgs.py```

```bash
python -m util_scripts.generate_video_jpgs avi_video_dir_path jpg_video_dir_path ucf101
```

* Generate annotation file in json format similar to ActivityNet using ```util_scripts/ucf101_json.py```
  * ```annotation_dir_path``` includes classInd.txt, trainlist0{1, 2, 3}.txt, testlist0{1, 2, 3}.txt

```bash
python -m util_scripts.ucf101_json annotation_dir_path jpg_video_dir_path dst_json_path
```

## Running the code
The primary function of the code is available from the [main file](https://github.com/AfonsoSalgadoSousa/3d_quant_action_recognition/main.py). Use the following command:
```bash
python main.py --mode <option>
```
The available modes are:
- train - train 3D networks
- post_train_quant - post training quantization
- quant_aware - quantization aware training
- test - test 3D networks

Please follow the command with *--help* to check a detailed description of every available parameter.

### Execution Examples
Example for every usecase can be found in the [script folder](https://github.com/AfonsoSalgadoSousa/3d_quant_action_recognition/scripts).

## Acknowledgement
The code was built on top of the works from [Hara et al.](https://github.com/kenshohara/3D-ResNets-PyTorch) and [Kopuklu et al.](https://github.com/okankop/Efficient-3DCNNs).
