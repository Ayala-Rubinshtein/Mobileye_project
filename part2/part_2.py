import os
from random import randint
from sys import argv

import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt



def generate_random_coordinate(image):
    image_length = image.shape
    return randint(0, image_length[0]), randint(0, image_length[1])


def crop_not_traffic(image, image_label):
    while True:
        x, y = generate_random_coordinate(image_label)

        try:
            if image_label[x][y] != 19:
                border_image = ImageOps.expand(image, border=41, fill='black')
                crop = border_image.crop((round(y) + 1, round(x) + 1, round(y) + 82, round(x) + 82))
                crop_not_traffic = np.asarray(crop)
                return crop_not_traffic
        except:
            return np.array(0)
def create_image_with_border(image):
    return ImageOps.expand(image, border=81, fill='black')

def create_tfls(color_image_path, label_image_path):
    label_image = create_image_with_border(Image.open(label_image_path))
    color_image = create_image_with_border(Image.open(color_image_path))
    label_image_in_np = np.array(label_image)
    tfl = np.argwhere(label_image_in_np == 19)
    if len(tfl)==0:
        return
    unique_tfl = []
    unique_tfl.append(tfl[0])
    for i in tfl:
        if (i[0] - unique_tfl[-1][0] > 83) and i[1] - unique_tfl[-1][1] > 83:
            unique_tfl.append(i)

    for i in unique_tfl:
        tmp_image = color_image.crop((i[1], i[0], i[1] + 81, i[0] + 81))
        add_new_data(tmp_image)
        add_new_label(np.array(1))
        tmp_image=crop_not_traffic(color_image, np.array(label_image))
        try:
            if len(tmp_image)==1:
                continue
        except:
                continue
        add_new_data(tmp_image)
        add_new_label(np.array(0))




def add_new_data(im):
    filename = "train_data.bin"
    with open(filename, mode='ab') as fileobj:
        np.array(im, dtype=np.uint8).tofile(fileobj)


def add_new_label(num):
    filename = "train_labels.bin"
    with open(filename, mode='ab') as fileobj:
        np.array(num, dtype=np.uint8).tofile(fileobj)


def read_from_data():
    data = np.fromfile("data.bin", dtype='uint8')
    for i in range(len(data) // 19683):
        im = data[i * 19683:(i + 1) * 19683].reshape(81, 81, 3)
        plt.imshow(im)
        plt.show()

def read_one_image_from_data(i):
    data = np.fromfile("val_data.bin", dtype='uint8')
    im = data[i * 19683:(i + 1) * 19683].reshape(81, 81, 3)
    plt.imshow(im)
    plt.show()

def read_from_labels():
    data = np.fromfile("val_labels.bin", dtype='uint8')
    print(data)

def read_one_label_from_labels(i):
    data = np.fromfile("labels.bin", dtype='uint8')
    print(data[i])

def get_label_path(path):
    label_path = path.replace("leftImg8bit", "gtFine", 1).replace("_leftImg8bit", "_gtFine_labelIds", 1)
    return label_path

def init_data():
    path = "m/leftImg8bit/train"
    for root, dirs, files in os.walk(path):
        for file in files:
            path = os.path.join(root, file)
            if path.endswith('_leftImg8bit.png'):
                print(path)
                label_path=get_label_path(path)
                create_tfls(path,label_path)

def main():
    init_data()
    # read_from_data()
    # read_from_labels()
    # read_one_image_from_data(10)
    # read_one_label_from_labels(10)

if __name__ == '__main__':
    main()
