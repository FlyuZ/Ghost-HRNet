# Ghost-HRNet
Ghost HRNet


使用GhostModule替换Conv2d 搭建HRNet分类网络

Total params: 22,875,512

Trainable params: 22,875,512

Non-trainable params: 0

Total mult-adds (G): 6.38

===================================================================================================================
Input size (MB): 0.79

Forward/backward pass size (MB): 460.23

Params size (MB): 91.50

Estimated Total Size (MB): 552.52


但是效果一般，准确率不高，暂不清楚原因

SHRNet为HRNet的重构版，不再需要yaml文件，参数量相同，但是GFLOPs会大一点点，暂不清楚原因

参考：
https://github.com/samcw/ResNet18-GhostNet/blob/master/resnet18.py

https://github.com/KopiSoftware/Ghost_ResNet56/blob/master/resnet.py

https://github.com/HRNet/HRNet-Image-Classification/edit/master/lib/models/cls_hrnet.py

https://github.com/iamhankai/ghostnet.pytorch/blob/master/ghost_net.py

https://github.com/stefanopini/simple-HRNet/blob/master/models/hrnet.py
