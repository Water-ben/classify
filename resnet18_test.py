

import torch.utils.data.distributed
import torchvision.transforms as transforms

from torch.autograd import Variable
import os
from PIL import Image
import cv2.cv2 as cv2
import numpy as np

classes = ('cat', 'dog')

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("model.pth", map_location=DEVICE)
model.eval()
model.to(DEVICE)
path='/Volumes/software/project/cv_dataset/classify/dogs-vs-cats/dataset/test/'
testList=os.listdir(path)
for file in testList:
    pil_img=Image.open(path+file)
    tensor_img=transform_test(pil_img)
    tensor_img.unsqueeze_(0)
    img = Variable(tensor_img).to(DEVICE)
    out=model(img)
    # Predict
    _, pred = torch.max(out.data, 1)
    print('Image Name:{},predict:{}'.format(file,classes[pred.data.item()]))

    # pil_img.show()
    '''
    每次识别路径下的一张图片，并显示，图片title是识别结果，键盘渐入任意键，开始识别下一张图片，不需要看图片注释下面代码
    '''
    cv_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imshow(f'{classes[pred.data.item()]}',cv_img)
    cv2.waitKey(0)