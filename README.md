# chinese_crnn
中文crnn识别以及其模式转onnx
## 更正（Correction）
更正了[原demo](https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec/blob/stable/demo.py)图片resize错误的问题
### 环境配置（Dev Environments）
Win10 + torch1.8.1+cu111+cudnn8.1.1
### 数据（Data）
#### Synthetic Chinese String Dataset
1. Download the [dataset](https://pan.baidu.com/s/1ufYbnZAZ1q0AlK7yZ08cvQ)
2. Edit **lib/config/360CC_config.yaml** DATA:ROOT to you image path

```angular2html
    DATASET:
      ROOT: 'to/your/images/path'
```

3. Download the [labels](https://pan.baidu.com/s/1oOKFDt7t0Wg6ew2uZUN9xg) (password: eaqb)
4. Put *char_std_5990.txt* in **lib/dataset/txt/**
5. And put *train.txt* and *test.txt* in **lib/dataset/txt/**

    eg. test.txt
```
    20456343_4045240981.jpg 89 201 241 178 19 94 19 22 26 656
    20457281_3395886438.jpg 120 1061 2 376 78 249 272 272 120 1061
    ...
```
#### Or your own data
1. Edit **lib/config/OWN_config.yaml** DATA:ROOT to you image path
```angular2html
    DATASET:
      ROOT: 'to/your/images/path'
```
2. And put your *train_own.txt* and *test_own.txt* in **lib/dataset/txt/**

    eg. test_own.txt
```
    20456343_4045240981.jpg 你好啊！祖国！
    20457281_3395886438.jpg 晚安啊！世界！
    ...
```
**note**: fixed-length training is supported. yet you can modify dataloader to support random length training.   

## Train
注：将训练的字符添加至**lib/config/alphabets.py**后，再进行训练否则训练出错
```angular2html
   [run] python train.py --cfg lib/config/360CC_config.yaml
or [run] python train.py --cfg lib/config/OWN_config.yaml
```
```
#### loss curve

```angular2html
   [run] cd output/360CC/crnn/xxxx-xx-xx-xx-xx/
   [run] tensorboard --logdir log
```

## Demo
```angular2html
   [run] python demo.py --image_path images/test.png --checkpoint output/checkpoints/mixed_second_finetune_acc_97P7.pth
```
# 参考资料（References）
https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec
