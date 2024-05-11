import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
from torchvision import models
import numpy as np
import cv2
import requests
from setup import config
from pytorch_grad_cam import GradCAM,ScoreCAM,GradCAMPlusPlus,EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from PIL import Image
from visualization import build_eval_model
from models.mps import format_reverse


# model = models.resnet50(pretrained=True)
model = build_eval_model(config,120)
model.eval()
model.cuda()
image_url = 'C:\\Experiment\\Code\\Fine-Grained\MPS\\visualize\\sample_img\\dogs\\n02086646_567.jpg'

img = np.array(Image.open(image_url))
img = cv2.resize(img, (384, 384))
img = np.float32(img) / 255
input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).cuda()

# The target for the CAM is the Bear category.
# As usual for classication, the target is the logit output
# before softmax, for that category.
target_layers = [model.norm]
with GradCAM(model=model, target_layers=target_layers,reshape_transform=format_reverse,use_cuda=True) as cam:
    grayscale_cams = cam(input_tensor=input_tensor, targets=None)
    print(grayscale_cams.shape)
    cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
cam = np.uint8(255*grayscale_cams[0, :])
cam = cv2.merge([cam, cam, cam])
images = np.hstack((np.uint8(255*img), cam , cam_image))
print(cam.shape)
plt.imshow(Image.fromarray(images))
plt.show()