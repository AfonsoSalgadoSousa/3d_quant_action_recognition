import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument('file_path', type=str)
parser.add_argument('--dst_file_path', default=None, type=str)
args = parser.parse_args()

if args.dst_file_path is None:
    args.dst_file_path = args.file_path

x = torch.load(args.file_path)
state_dict = x['state_dict']

new_state_dict = {str.replace(str(k), 'module.', ''): v for k, v in state_dict.items()}

x['state_dict'] = new_state_dict

torch.save(x, args.dst_file_path)
