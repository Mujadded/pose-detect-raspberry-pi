import torch
import torchvision
import numpy as np
import cv2
import argparse
import utils
from PIL import Image
from torchvision.transforms import transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, 
                    help='path to the input data')
args = vars(parser.parse_args())

transform = transforms.Compose([
    transforms.ToTensor()
])

# initialize the model
# model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True,
#                                                                num_keypoints=17)
weights = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
auto_transforms = weights.transforms()
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=weights,
                                                               num_keypoints=17)
# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the modle on to the computation device and set to eval mode
model.to(device).eval()

image_path = args['input']
image = Image.open(image_path).convert('RGB')
# NumPy copy of the image for OpenCV functions
orig_numpy = np.array(image, dtype=np.float32)
# convert the NumPy image to OpenCV BGR format
orig_numpy = cv2.cvtColor(orig_numpy, cv2.COLOR_RGB2BGR) / 255.
# transform the image
image = auto_transforms(image)
# add a batch dimension
image = image.unsqueeze(0).to(device)

with torch.inference_mode():
    model.eval()
    outputs = model(image)
output_image = utils.draw_keypoints(outputs, orig_numpy)

# # visualize the image
# cv2.imshow('Keypoint image', output_image)
# cv2.waitKey(0)

save_path = f"/home/kronos/src/health_monitoring/keypoint_detection/outputs/{args['input'].split('/')[-1].split('.')[0]}.jpg"
cv2.imwrite(save_path, output_image*255.)