import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import gridspec
import os
from datetime import datetime, time
import cv2
from model import Deeplabv3

def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap

def label_to_color_image(label):
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  #grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  #plt.subplot(grid_spec[0])
  #plt.imshow(image)
  #plt.axis('off')
  #plt.title('input image')

  #plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  #plt.imshow(seg_image)
  #plt.axis('off')
  #plt.title('segmentation map')

  #plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  #unique_labels = np.unique(seg_map)
  #ax = plt.subplot(grid_spec[3])
  #ax.yaxis.tick_right()
  #plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  #plt.xticks([], [])
  #ax.tick_params(width=0.0)
  #plt.grid('off')
  plt.show()

# Generates labels using most basic setup.  Supports various image sizes.  Returns image labels in same format
# as original image.  Normalization matches MobileNetV2

deeplab_model = Deeplabv3()

def predict_image(array):
  trained_image_width=512
  mean_subtraction_value=127.5
  image = array
  #image=cv2.resize(image,(480,360))
  image=np.array(image)
  # resize to max dimension of images from training dataset
  w, h, _ = image.shape
  ratio = float(trained_image_width) / np.max([w, h])
  resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h), int(ratio * w))))

  # apply normalization for trained dataset images
  resized_image = (resized_image / mean_subtraction_value) - 1.

  # pad array to square image to match training images
  pad_x = int(trained_image_width - resized_image.shape[0])
  pad_y = int(trained_image_width - resized_image.shape[1])
  resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')

  # make prediction
  
  res = deeplab_model.predict(np.expand_dims(resized_image, 0))
  labels = np.argmax(res.squeeze(), -1)

# remove padding and resize back to original image
  if pad_x > 0:
      labels = labels[:-pad_x]
  if pad_y > 0:
      labels = labels[:, :-pad_y]
  labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))
  return image,labels

#vis_segmentation(image,labels)
#plt.waitforbuttonpress()

'''
a=[[[0,0,0]]]
ax = plt.subplot()
IM = ax.imshow(a)
IM2=ax.imshow(a,alpha=0.6)

plt.ion()

cam_cap=cv2.VideoCapture(0)

while True:
  ret,frame=cam_cap.read()
  image,seg_image=predict_image(frame)
  #IM.set_data(image)
  #IM2.set_data(seg_image)
  #plt.pause(0.5)
  plt.imshow(image)
  plt.imshow(seg_image)
  plt.show()
  if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#print(datetime.now()-t_start)
#plt.show()

'''
cam_cap=cv2.VideoCapture(0)
i=0
t_start=datetime.now()
while True:
  i+=1
  ret,frame=cam_cap.read()
  image,seg_map=predict_image(frame)
  print(i/(datetime.now()-t_start).seconds)
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  #seg_image=cv2.cvtColor(seg_image,cv2.COLOR_GRAY2BGR)
  output = cv2.addWeighted(image,0.6,seg_image,0.4,0)
  cv2.imshow('video',output)
  if cv2.waitKey(1) & 0xFF == ord('q'):
      break