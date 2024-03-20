# Project 2: Comparing the Impact of Different Image Preprocessing Methods on Model Precisions
### Authors: Yuxuan Zhang, Yuhan Hou
## Data Preperation


The following file contains the codes for data preperation. For the purpose of training YOLO, all images and labels need to be organized into the following order:

```
root 
├── ...
├── notebooks
│ └── Train.ipynb
└── dataset
├── data.yaml # Describes the dataset
├── train
│ ├── images # contains all the training images
│ │ └── ...
│ └── labels 
```

In addition, we need a .yaml file to instruct the model to find our dataset during the training process. This file will be stored inside the same folder as the dataset, and has the following format:

```
path: path\to\project
train: path\to\training\images
val: path\to\validation\images
test: path\to\testing\images

nc: 1
names: ["fractured"]
```
**To reproduce this project, make sure to modify the path in all YAML files for the training process to work correctly**
**There is some known issue with the [default system-wide dataset folder](https://superuser.com/questions/37449/what-are-and-in-a-directory). An alternative approach it to delete 'path:' and copy the absolute directory path of the images in each split in the YAML file.**

The steps below shows the general pipeline of this project. You can open each python notebook to reproduce the data preperation and training process.

### Converting COCO Segmentation Masks into YOLO format
Since the author of the original dataset only included the bounding box in YOLO format, to perform segmentation tasks, we will need to convert the COCO segmentation masks into YOLO format.

```python
import json
with open('FracAtlas\FracAtlas\Annotations\COCO JSON\COCO_fracture_masks.json') as f:
    coco_data = json.load(f)

for index, image in enumerate(coco_data['images']):
    id = image['id']
    width = image['width']
    height = image['height']
    YOLO_file_name = image['file_name'].replace('.jpg', '.txt')
    directory_path = "D:\\project2\\YOLO_Segmentation\\"
    YOLO_file_path = directory_path + YOLO_file_name
    
    for annotation in coco_data['annotations']:
        if annotation['image_id'] == id:
            for i in range(len(annotation['segmentation'])):
                segmentation_lst = annotation['segmentation'][i]
                normalized = []
                for i, value in enumerate(segmentation_lst):
                    if i % 2 == 0:
                        normalized_value = value / width
                    else:
                        normalized_value = value / height
                    normalized.append(normalized_value)
                segmentation = ' '.join(map(str, normalized))
                segmentation = '0 ' + segmentation + '\n'
                with open(YOLO_file_path, 'w') as file:
                    file.write(segmentation)
```

### Next, we will perform different image processing methods to train the models.

### Vanilla

For the vanilla version, we only need to arrange the images from the original dataset according to the structure specified before.
```python
import pandas as pd
import os
import shutil

# For detection
# Images
train_csv = "D:/project2/FracAtlas/FracAtlas/Utilities/Fracture Split/train.csv"
valid_csv = "D:/project2/FracAtlas/FracAtlas/Utilities/Fracture Split/valid.csv"
test_csv = "D:/project2/FracAtlas/FracAtlas/Utilities/Fracture Split/test.csv"

source_dir = "D:/project2/FracAtlas/FracAtlas/images/All_Images"
dest_dirs = {
    'train': "D:/project2/Training_model/dataset/train/images",
    'valid': "D:/project2/Training_model/dataset/valid/images",
    'test': "D:/project2/Training_model/dataset/test/images", 
}

def copy_images(csv_file, target_dir):
    df = pd.read_csv(csv_file)
    os.makedirs(target_dir, exist_ok=True)
    for image_name in df['image_id']:
        src_path = os.path.join(source_dir, image_name)
        dst_path = os.path.join(target_dir, image_name)
        shutil.copy(src_path, dst_path)

copy_images(train_csv, dest_dirs['train'])
copy_images(valid_csv, dest_dirs['valid'])
copy_images(test_csv, dest_dirs['test'])

# Labels
source_dir = "D:/project2/FracAtlas/FracAtlas/Annotations/YOLO"

dest_dirs = {
    'train': "D:/project2/Training_model/dataset/train/labels",
    'valid': "D:/project2/Training_model/dataset/valid/labels",
    'test': "D:/project2/Training_model/dataset/test/labels", 
}

def copy_labels(csv_file, target_dir):
    df = pd.read_csv(csv_file)
    for image_name in df['image_id']:
        label_name = image_name.replace('.jpg', '.txt')
        shutil.copy(os.path.join(source_dir, label_name), os.path.join(target_dir, label_name))

copy_labels(train_csv, dest_dirs['train'])
copy_labels(valid_csv, dest_dirs['valid'])
copy_labels(test_csv, dest_dirs['test'])

# For Segmentation
# Images
source_dir = "D:/project2/FracAtlas/FracAtlas/images/All_Images"
dest_dirs = {
    'train': "D:/project2/Training_model_Seg/dataset/train/images",
    'valid': "D:/project2/Training_model_Seg/dataset/valid/images",
    'test': "D:/project2/Training_model_Seg/dataset/test/images", 
}

def copy_images(csv_file, target_dir):
    df = pd.read_csv(csv_file)
    os.makedirs(target_dir, exist_ok=True)
    for image_name in df['image_id']:
        src_path = os.path.join(source_dir, image_name)
        dst_path = os.path.join(target_dir, image_name)
        shutil.copy(src_path, dst_path)

copy_images(train_csv, dest_dirs['train'])
copy_images(valid_csv, dest_dirs['valid'])
copy_images(test_csv, dest_dirs['test'])

# Labels
source_dir = "D:\project2\YOLO_Segmentation"
dest_dirs = {
    'train': "D:/project2/Blured_Training_Segmentation/dataset/train/labels",
    'valid': "D:/project2/Blured_Training_Segmentation/dataset/valid/labels",
    'test': "D:/project2/Blured_Training_Segmentation/dataset/test/labels", 
}

def create_directories(directory_paths):
    for path in directory_paths.values():
        os.makedirs(path, exist_ok=True)
        
create_directories(dest_dirs)

def copy_labels(csv_file, target_dir):
    df = pd.read_csv(csv_file)

    for image_name in df['image_id']:
        label_name = image_name.replace('.jpg', '.txt')
        shutil.copy(os.path.join(source_dir, label_name), os.path.join(target_dir, label_name))

copy_labels(train_csv, dest_dirs['train'])
copy_labels(valid_csv, dest_dirs['valid'])
copy_labels(test_csv, dest_dirs['test'])
```

**The data distribution processes for blurred, edge, and CLAHE images are the same, and we won't repeat them below.**

### Gaussian Blur
```python
from PIL import ImageFile, ImageFilter
import pandas as pd
import os
import numpy as np
from datasets import load_dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True

def blur_image(image):
    # Apply Gaussian blur directly on the PIL image object
    return image.filter(ImageFilter.GaussianBlur)

dataset = load_dataset("yh0701/FracAtlas_dataset", trust_remote_code=True)
for j in ['test', 'validation', 'train']:
    for i in range(len(dataset[j])):
        pil_image = dataset[j][i]['image']
        blurred_image = blur_image(pil_image)
        image_name = dataset[j][i]['image_id']
        blurred_image.save(f'D:\project2\Blured_Images\{image_name}')
```

### CLAHE
```python
import cv2 as cv
for j in ['train', 'test', 'validation']:
    for i, item in enumerate(dataset[j]):
        image_data = np.array(item['image'])
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        if len(image_data.shape) == 3 and image_data.shape[2]>1:
            image_data = cv.cvtColor(image_data, cv.COLOR_RGB2GRAY)
            equ = clahe.apply(image_data)
        else:
            equ = clahe.apply(image_data)
        image_id = item['image_id']
        image_path = f'D:\project2\CLAHE\{image_id}'
        cv.imwrite(image_path, equ)
```

### Edge
```python
for j in ['train', 'test', 'validation']:
    for i, item in enumerate(dataset[j]):
        image_data = np.array(item['image'])
        if len(image_data.shape) == 3 and image_data.shape[2]>1:
            image_data = cv.cvtColor(np.array(image_data), cv.COLOR_BGR2GRAY)
        edges = cv.Canny(image_data,100,200)
        image_id = item['image_id']
        image_path = f'D:\project2\Edge\{image_id}'
        cv.imwrite(image_path, edges)
```

### Training
After finished preparing the datasets, we can proceed to the training process. In this project, we are training YOLOv8s from Ultralytics. Before training, **make sure to check the compatibility of pytorch and CUDA** for maximum efficiency.
```python
import ultralytics
ultralytics.checks()
!yolo task=detect mode=train model=yolov8s.pt data=dataset/data.yaml epochs=30 imgsz=600
```
The dataset is comparatively small. With Nvidia 4090 Laptop version and CUDA 12.1, it takes around 3 minutes for me to finish the training.
To train the segmentation task, replace 'task=detect' with 'task=segment'.
```python
!yolo task=segment mode=train model=yolov8s.pt data=dataset/data.yaml epochs=30 imgsz=600
```
The results and trained weights should be stored inside the folder 'runs', where you can check for the precisions, mAP, and other metrics. 
