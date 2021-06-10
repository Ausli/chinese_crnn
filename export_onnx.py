import argparse
from easydict import EasyDict as edict
import yaml
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import lib.models.crnn as crnn
import lib.utils.utils as utils
from lib.dataset import get_dataset
from lib.core import function
import lib.config.alphabets as alphabets
from lib.utils.utils import model_info

from torch.autograd import Variable
from tensorboardX import SummaryWriter

def parse_arg():
    parser = argparse.ArgumentParser(description="train crnn")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='lib/config/360CC_config.yaml')
    parser.add_argument('--checkpoint', type=str,
                        default='output\checkpoints\checkpoint_124_acc_1.0000.pth',
                        help='the path to your checkpoints')
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        # config = yaml.load(f, Loader=yaml.FullLoader)
        config = yaml.load(f,Loader=yaml.FullLoader)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config,args
if __name__ == '__main__':
    config, args = parse_arg()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = crnn.get_crnn(config).to(device)
    print('loading pretrained model from {0}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    batch_size = 1
    input_shape = (1, 32, 160)

    dummy_input = torch.randn(batch_size, *input_shape).to(device)

    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(model, dummy_input, "crnn.onnx", verbose=False,
                      input_names=input_names, output_names=output_names,opset_version=11)
