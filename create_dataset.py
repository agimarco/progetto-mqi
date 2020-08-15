from pycocotools.coco import COCO

import requests
from requests.exceptions import HTTPError

import cv2
import numpy as np
import os

from collections import Counter

categories = ["car", "motorcycle", "truck", "bus", "bicycle", "person", "traffic light", "stop sign"]

#final categories will then be car, bicycle (which is motorcycle and bicycle), truck (which is truck and bus)

N_CAT = 5

N_train = 400
N_test = 100

IMG_SIZE = 512

def xml_generator(im, mode, dimension, annotations, xml_directory=None, image_directory=None, xml_filename=None, image_filename=None, save=False):
    #we use the boolean "delete" to check if the image has at least one viable bounding box, if there are not bboxes we must delete the image
    #otherwise the specific YOLO model won't work with images without bboxes
    delete = True
    #we open the image to write bboxes on it
    image = cv2.imread(image_directory + image_filename)
    #we get the original width and height
    original_h = im['height']
    original_w = im['width']
    # PASCAL VOC file format
    lines = []
    lines.append("<annotation>")
    if(mode == 'train'):
        lines.append("  " + "<folder>" + mode + "</folder>")
    else:
        lines.append("  " + "<folder>" + 'test' + "</folder>")
    lines.append("  " + "<filename>" + image_filename + "</filename>")
    lines.append("  " + "<path>" + image_directory + "</path>")
    lines.append("  " + "<source>")
    lines.append("      " + "<database>Unknown</database>")
    lines.append("  " + "</source>")
    lines.append("  " + "<size>")
    lines.append("      " + "<width>" + str(dimension) + "</width>")
    lines.append("      " + "<height>" + str(dimension) + "</height>")
    lines.append("      " + "<depth>3</depth>")
    lines.append("  " + "</size>")
    lines.append("  " + "<segmented>0</segmented>")
    # for each bbox
    for annotation in annotations:
        #old bbox coordinates
        bbox = annotation['bbox']
        #box coordinates in normalized image coordinates, 0 is the leftmost pixel, 1 is the rightmost (for x)
        bboxxmin = bbox[0] / original_w
        bboxymin = bbox[1] / original_h
        bboxxmax = (bbox[0] + bbox[2]) / original_w
        bboxymax = (bbox[1] + bbox[3]) / original_h
        # new bbox coordinates
        if original_h < original_w:
            xRidimensionata = dimension
            yRidimensionata = dimension / original_w * original_h
            yOffset = int((dimension - yRidimensionata) / 2)
            xmin = int(float(bboxxmin) * dimension)
            xmax = int(float(bboxxmax) * dimension)
            ymin = int(float(bboxymin) * yRidimensionata + yOffset)
            ymax = int(float(bboxymax) * yRidimensionata + yOffset)
        else:
            xRidimensionata = dimension / original_h * original_w
            yRidimensionata = dimension
            xOffset = int((dimension - xRidimensionata) / 2)
            xmin = int(float(bboxxmin) * xRidimensionata + xOffset)
            xmax = int(float(bboxxmax) * xRidimensionata + xOffset)
            ymin = int(float(bboxymin) * dimension)
            ymax = int(float(bboxymax) * dimension)
        percentage = (xmax - xmin) * (ymax - ymin) / (xRidimensionata * yRidimensionata)
        if percentage > 0.01:
            #we set delete to false since the image has at least one viable bounding box and therefore we don't have to delete it
            delete = False
            lines.append("  " + "<object>")
            # get the annotation category\class (using test file 'cocote' since categories have no difference)
            classes = cocote.loadCats(annotation['category_id'])
            current_class = classes[0]['name']
            #if there is a bus then label it as truck
            if(current_class == "bus"):
                current_class = "truck"
            #if there is a motorcycle then label it as bicycle
            if(current_class == "motorcycle"):
                current_class = "bicycle"
            lines.append("      " + "<name>" + current_class + "</name>")
            lines.append("      " + "<pose>Unspecified</pose>")
            lines.append("      " + "<truncated>0</truncated>")
            lines.append("      " + "<difficult>0</difficult>")
            lines.append("      " + "<bndbox>")
            lines.append("          " + "<xmin>" + str(xmin) + "</xmin>")
            lines.append("          " + "<ymin>" + str(ymin) + "</ymin>")
            lines.append("          " + "<xmax>" + str(xmax) + "</xmax>")
            lines.append("          " + "<ymax>" + str(ymax) + "</ymax>")
            lines.append("      " + "</bndbox>")
            lines.append("  " + "</object>")
            #draw rectangles on picture, each class has a color associated, save picture in folder
            '''
            color = { "car": (0, 0, 255), "bicycle": (0, 255, 0), "truck": (255, 0, 0) } 
            image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color[current_class], 2)
            cv2.imwrite("bboxes/" + image_filename, image)
            '''
    lines.append("</annotation>")
    if save and not delete:
        # save xml file
        g = open(xml_directory + xml_filename, "w", encoding="UTF-8")
        for x in lines:
            g.write(x)
            g.write("\n")
        g.close()
    #if the image has no viable bboxes we delete it
    if delete:
        os.unlink(image_directory + image_filename)

def image_resize(image, dimension, directory=None, filename=None, save=False):
    # original picture height and width
    (h, w) = image.shape[:2]
    # which one to modify to keep aspect ratio
    if h < w:
        yRidimensionata = dimension / w * h
        dim = (dimension, int(yRidimensionata))
        xOffset = 0
        yOffset = int((dimension - yRidimensionata) / 2)
    else:
        xRidimensionata = dimension / h * w
        dim = (int(xRidimensionata), dimension)
        xOffset = int((dimension - xRidimensionata) / 2)
        yOffset = 0
    # first resize
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # turn it into a square image
    square = np.zeros((dimension, dimension, 3), np.uint8)
    square[yOffset:yOffset + resized.shape[0], xOffset:xOffset + resized.shape[1]] = resized
    if save:
        cv2.imwrite(directory + filename, square)
    return square

#download image, resize image, compute new bounding box and create PASCAL VOC file
def download_image(im, mode):
    directory_image = 'data/' + mode + '/image/'
    directory_annotation = 'data/' + mode + '/annotation/'
    image_filename = im['file_name'] 
    xml_filename = im['file_name'].split('.')[0] + ".xml"
    url = im['coco_url'] 
    while True:
        try:
            response = requests.get(url)
            # If the response was successful, no Exception will be raised
            response.raise_for_status()
        except HTTPError as http_err:
            print(f'HTTP error occurred: {http_err}')  # Python 3.6
        except Exception as err:
            print(f'Other error occurred: {err}')  # Python 3.6
        break
    img_data = response.content
    #download image
    with open(directory_image + image_filename, 'wb') as handler:
        handler.write(img_data)
    #open dowloaded image with cv2
    image = cv2.imread(directory_image + image_filename)
    #resize maintaining aspect ratio
    image = image_resize(image, IMG_SIZE, directory_image, image_filename, True)
    # generate xml and compute new bboxes
    #get annotations 
    catIds = cocote.getCatIds(catNms=categories[:N_CAT])
    if(mode == 'train'):
        annIds = cocotr.getAnnIds(imgIds = im['id'], catIds = catIds, iscrowd = False)
        annotations = cocotr.loadAnns(annIds)
    else:
        annIds = cocote.getAnnIds(imgIds = im['id'], catIds = catIds, iscrowd = False)
        annotations = cocote.loadAnns(annIds)
    #print(annotations)
    #call function that computes new bboxes, creates the xml file and saves it. Creates images with bboxes in bboxes folder
    xml_generator(im, mode, IMG_SIZE, annotations, directory_annotation, directory_image, xml_filename, image_filename, True)
    
def remove_duplicates(l):
    seen = set()
    new_l = []
    for d in l:
        t = tuple(d.items())
        if t not in seen:
            seen.add(t)
            new_l.append(d)
    return new_l

# instantiate COCO specifying the annotations json path
cocotr = COCO('instances_train2017.json')
cocote = COCO('instances_val2017.json')

train = []
test = []

# 1 more picture to get 340 total testing pictures
catIdste1pic = cocote.getCatIds(catNms=categories[:N_CAT])
imgIdste1pic = cocote.getImgIds(catIds=catIdste)
imageste1pic = cocote.loadImgs(imgIdste)[:1]
test = test + imageste1pic

#for each class get images, resize them, make the XML file in PASCAL VOC format
for category in categories[:N_CAT]:
    # Specify a list of category names of interest using COCO API
    catIdstr = cocotr.getCatIds(catNms=[category])
    catIdste = cocote.getCatIds(catNms=[category])
    # Get the corresponding image ids and images using loadImgs
    imgIdstr = cocotr.getImgIds(catIds=catIdstr)
    imgIdste = cocote.getImgIds(catIds=catIdste)

    imagestr = cocotr.loadImgs(imgIdstr)[:N_train]
    imageste = cocote.loadImgs(imgIdste)[:N_test]

    train = train + imagestr
    test = test + imageste

print("\nExample of element of train images: ", train[0])
print("\nExample of element of test images: ", test[0])
print("\nTraining images: ", len(train))
print("Validation images: ", len(test))
train = remove_duplicates(train)
test = remove_duplicates(test)
print("Training images after removing duplicates: ", len(train))
print("Validation images after removing duplicates: ", len(test), "\n")

# Save the images into a local folder
for im in train:
    download_image(im,'train')

# Save the images into a local folder
for im in test:
    download_image(im,'val')
