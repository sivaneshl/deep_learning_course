import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms
from glob import glob
from tqdm import tqdm
from PIL import Image

# Load file names for humans and dogs
# Download dog imags from https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
# Download human images from https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip
data_dir = 'C:\\deep_learning\\deep_learning_course\\convolutional_neural_networks\\dog_breed_classifier\\'
human_files = np.array(glob(data_dir + 'data\\lfw\\*\\*'))
dog_files = np.array(glob(data_dir + 'data\\dog_images\\*\\*\\*'))

# print the number of images in each dataset
print('There are %d total human images' % len(human_files))
print('There are %d total dog images' % len(dog_files))

## Step 1 - Detect humans
# Use the OpenCV's implementation of heat feature based cascade classifiers to detect human faces in images.
# OpenCV provide many pre-trainied face detectors stored as XML files on
# https://github.com/opencv/opencv/tree/master/data/haarcascades.

# Extract pre-trained face detector
face_cascade = cv2.CascadeClassifier(data_dir + '\\haarcascades\\haarcascade_frontalface_alt.xml')

# Load color BGR image
img = cv2.imread(human_files[0])
# Convert BGR image to grayscale
# Before using any of the face detectors, it is standard procedure to convert the images to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find faces in the image. The detectMultiScale function executes the classifier stored in face_cascade and takes
# the grayscale image as a parameter.
faces = face_cascade.detectMultiScale(gray)

# print the number of faces detected in the image
print('Number of faces detected: ', len(faces))

# Get bounding boxes for each image.
# faces is a numpy array of detected faces, where each row corresponds to a detected face. Each detected face is a 1D
# array with four entries that specifies the bounding box of the detected face. The first two entries in the array
# (extracted in the above code as x and y) specify the horizontal and vertical positions of the top left corner of the
# bounding box. The last two entries in the array (extracted here as w and h) specify the width and height of the box.
for (x, y, w, h) in faces:
    # add a bounding box to each color image
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image along with the bounding box
plt.imshow(cv_rgb)
plt.show()


## Function to check if a face is detected
def face_detector(img_path):
    """
    Returns True if a face is detected in the given image
    :param img_path: 
    :return: 
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


## Question 1: Assess the human face detector
# What percentage of the first 100 images in human_files have a detected human face?
# What percentage of the first 100 images in dog_files have a detected human face?
human_files_short = human_files[:100]
dog_files_short = dog_files[:100]

n_false = sum([not (face_detector(x)) for x in human_files_short])
print('Accuracy of the face detector on human faces: {:.2f}%'.format(n_false))
n_false = sum([face_detector(x) for x in dog_files_short])
print('Accuracy of the face detector on dog faces: {:.2f}%'.format(n_false))


## Step 2 - Detect dogs
# obtain the pre-trained vgg16 model
vgg16 = models.vgg16(pretrained=True)

# check if cuda is available and use it
use_cuda = torch.cuda.is_available()
if use_cuda:
    vgg16.cuda()

# Freeze model weights
for param in vgg16.parameters():
    param.requires_grad = False

# define the transform for pre-processing images for pytorch models
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])


# make predictions with pre-trained model
def vgg16_predict(img_path):
    """
    Use pre-trained VGG16 model to obtain index corresponding to predicted ImageNet class for image at specified path
    :param img_path: path to an image
    :return: corresponding index to VGG16 model's prediction
    """
    img = Image.open(img_path)
    img_processed = torch.unsqueeze(transform(img), 0)

    vgg16.eval()

    if use_cuda:
        img_processed = img_processed.cuda()

    output = vgg16(img_processed)
    _, pred_tensor = torch.max(output, 1)
    pred = np.squeeze(pred_tensor.numpy()) if not use_cuda else np.squeeze(pred_tensor.cpu().numpy())

    return pred.item()  # predicted class index

# In ImageNet the dog classes are index between 151 and 268 (inclusive).
def dog_detector(img_path):
    """
    Returns True if a dog is detected in the image
    :param img_path:
    :return:
    """
    imagenet_class = vgg16_predict(img_path)
    if 151 <= imagenet_class <= 268:
        return True
    return False


## Question 2: Assess the performance of the dog_detector function
# What percentage of the images in human_files_short have a detected dog?
# What percentage of the images in dog_files_short have a detected dog?
human_has_dog = sum([dog_detector(x) for x in human_files_short])
print('Percentage of humans detected as dog: {:.2f}%'.format(human_has_dog))
dog_has_dog = sum([dog_detector(x) for x in dog_files_short])
print('Percentage of dogs detected as dog: {:.2f}%'.format(dog_has_dog))



