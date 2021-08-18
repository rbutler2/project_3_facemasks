#!/usr/bin/env python
# coding: utf-8

import dlib
from matplotlib import pyplot as plt
import imutils
import numpy as np
import cv2
import PIL
import requests
from imutils import face_utils
from PIL import Image


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def plt_imshow(title, image, size=4):
  # convert the image frame BGR to RGB color space and display it
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(size, size))
    plt.imshow(image)
    plt.title(title)
    plt.grid(False)
    plt.show()

url = 'https://as01.epimg.net/en/imagenes/2021/07/28/latest_news/1627501245_454307_1627501474_noticia_normal_recorte1.jpg'
url = input('url:\n')

code_live = True

while code_live == True:
    try:
        image = Image.open(requests.get(url, stream=True).raw)
        image.save('temp_image/new.jpg')
        image = 'temp_image/new.jpg'

    except:
        print("URL Error")
        code_live = False


    args = {
    "shape_predictor": "shape_predictor_68_face_landmarks.dat",
    "image": image
    }
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])
    image = cv2.imread(args["image"])
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    if len(rects) < 1:
        print('No faces detected')
        code_live = False
    else:
        print(f'{len(rects)} faces detected')
        face_rects = {}
        
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            face_rects[i] = [x,y,w,h]


        # Saves images to dictionary

        image_dict = {}
        count = 0

        for k,v in face_rects.items():
            x = v[0]
            y = v[1]
            w = v[2]
            h = v[3]

            im_to_add = image[y:y+h, x:x+w]
            image_dict[x] = im_to_add
            count += 1



        # Shows dictionary of faces
        counter = 1
        for k,v in image_dict.items():
            try:
                plt_imshow(f'Face {counter}', v)
                v = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)
                rescaled = (255.0 / v.max() * (v - v.min())).astype(np.uint8)
        #         im = Image.fromarray(rescaled)
                im = Image.fromarray(v)
                im.save(f'web_faces_to_test/{counter}.png')
                counter += 1
            except:
                continue
        
        # plt_imshow(f'Image', image,8)

        code_live = False