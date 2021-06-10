"""
@author: Aus
@file: onnx_test.py
@time: 2021/3/28 15:31
"""
from io import BytesIO
import onnxruntime
import numpy as np
from PIL import Image
import argparse
import cv2
import yaml
import lib.config.alphabets as alphabets
from easydict import EasyDict as edict
from torch.autograd import Variable
import torch
import lib.utils.utils as utils
def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='lib/config/360CC_config.yaml')
    parser.add_argument('--image_path', type=str, default='img/7ca442da0d6ccdb2603fff36e98fe7db.jpg', help='the path to your image')
    parser.add_argument('--checkpoint', type=str, default='output/360CC/crnn/2021-06-08-14-21/checkpoints\checkpoint_97_acc_1.0000.pth',
                        help='the path to your checkpoints')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config
class ONNXModel(object):
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        # print("input_name:{}".format(self.input_name))
        # print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def to_numpy(self, file, shape, gray=False):
        if isinstance(file, np.ndarray):
            img = Image.fromarray(file)
        elif isinstance(file, bytes):
            img = Image.open(BytesIO(file))
            pass
        else:
            img = Image.open(file)

        widht, hight = shape
         # 改变大小 并保证其不失真
        img = img.convert('RGB')
        if gray:
            img = img.convert('L')
        img = img.resize((widht, hight), Image.ANTIALIAS)

        # 转换成矩阵
        image_numpy = np.array(img) # (widht, hight, 3)
        if gray:
            image_numpy = np.expand_dims(image_numpy,0)
            image_numpy = image_numpy.transpose(0, 1, 2)
        else:
            image_numpy = image_numpy.transpose(2,0,1) # 转置 (3, widht, hight)
        image_numpy = np.expand_dims(image_numpy,0)
        # 数据归一化
        image_numpy = image_numpy.astype(np.float32) / 255.0
        return image_numpy




class CRNN(ONNXModel):
    def __init__(self, onnx_path="model_onnx.onnx", char_dict="lib/dataset/txt/char_std_5990.txt"):
        super(CRNN, self).__init__(onnx_path)
        #with open(char_dict, 'rb') as file:
            #char_dict = {num: char.strip().decode('gbk', 'ignore') for num, char in enumerate(file.readlines())}

        with open(char_dict, 'r', encoding='utf-8') as f:
            data = f.read()
            self.characters = data.split('\n')


    def decect(self, img):
        config=parse_arg()
        h, w = img.shape
        # fisrt step: resize the height and width of image to (32, x)
        img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.OW / w , fy=config.MODEL.IMAGE_SIZE.H / h,
                         interpolation=cv2.INTER_CUBIC)
        # second step: keep the ratio of image's text same with training
        h, w = img.shape
        w_cur = int(img.shape[1] / (config.MODEL.IMAGE_SIZE.OW / config.MODEL.IMAGE_SIZE.W))
        img = cv2.resize(img, (0, 0), fx=w_cur / w, fy=1.0, interpolation=cv2.INTER_CUBIC)
        img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, w_cur, 1))
        # normalize
        img = img.astype(np.float32)
        img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
        img = img.transpose([2, 0, 1])
        image_numpy = np.expand_dims(img, 0)
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        out = self.onnx_session.run(self.output_name, input_feed=input_feed)[0]
        preds=torch.from_numpy(out)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        #print(sim_pred)
        return sim_pred


if __name__ == '__main__':
    import time
    crnn_model_path="crnn.onnx"
    file = "000ae60d6d5e9cc885f67164b0bbbd22.jpg"
    rnet2 = CRNN(crnn_model_path)
    s = time.time()
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = rnet2.decect(img)
    print(time.time() -s)
