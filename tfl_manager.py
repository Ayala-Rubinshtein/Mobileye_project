from part2.part2 import get_model, get_label
from part2.part_2 import create_image_with_border
from part3.SFM_standAlone import run_sfm
from run_attention import find_tfl_lights

import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def crop_red(image, coordinate):
    img = ImageOps.expand(image, border=41, fill='black')
    im = img.crop((coordinate[0] - 10, coordinate[1] -10, coordinate[0] + 71, coordinate[1] + 71))
    im=np.array(im)
    return im

def crop_green(image, coordinate):
    img = ImageOps.expand(image, border=41, fill='black')
    im = img.crop((coordinate[0] - 10, coordinate[1] -10, coordinate[0] + 71, coordinate[1] + 71))
    im=np.array(im)
    return im

def is_tfl(model,image: Image)-> bool:
    if get_label(model,image)[0]==1:
        return True
    return False

def check_red_tfl(model,image:Image,x_red: list,y_red: list)->tuple:
    x_red2 = []
    y_red2 = []
    for i in range(len(x_red)):
        tmp_image=crop_red(image,[x_red[i],y_red[i]])
        if is_tfl(model,tmp_image):
            x_red2.append(x_red[i])
            y_red2.append(y_red[i])
    return x_red2,y_red2

def check_green_tfl(model,image:Image,x_green: list,y_green: list)->tuple:
    del_list = []
    x_green2=[]
    y_green2=[]
    for i in range(len(x_green)):
        tmp_image=crop_green(image,[x_green[i],y_green[i]])
        if is_tfl(model,tmp_image):
            x_green2.append(x_green[i])
            y_green2.append(y_green[i])
            del_list.append(i)
    return x_green2,y_green2


def part_2(image, x_red: list, y_red: list, x_green: list, y_green: list) -> tuple:
    model=get_model()
    x_red,y_red=check_red_tfl(model,image,x_red,y_red)
    print(np.array(image).shape)
    x_green,y_green=check_green_tfl(model,image,x_green,y_green)
    return x_red,y_red,x_green,y_green
    # return x_red, y_red, [], []


class TFLMan:
    def __init__(self, pkl, frame_list):
        self.pkl = pkl
        self.frame_list = frame_list
        self.image=Null

    def part_1(self,i: int) -> tuple:
        image = np.array(Image.open(self.frame_list[i]))
        self.image=create_image_with_border(Image.open(self.frame_list[i]))
        x_red, y_red, x_green, y_green = find_tfl_lights(image)
        plt.imshow(image)
        plt.plot(x_red, y_red, 'ro', color='r', markersize=4)
        plt.plot(x_green, y_green, 'ro', color='g', markersize=4)
        plt.show()
        return x_red, y_red, x_green, y_green

    def part_3(self,curr_id,prev_id,curr_x_red, curr_y_red, curr_x_green, curr_y_green,prev_x_red,prev_y_red,prev_x_green, prev_y_green):
        run_sfm(curr_id,prev_id,self.frame_list,self.pkl,curr_x_red, curr_y_red, curr_x_green, curr_y_green,prev_x_red,prev_y_red,prev_x_green, prev_y_green)

    def run(self, i: int) -> None:
        curr_x_red, curr_y_red, curr_x_green, curr_y_green = self.part_1(i)
        curr_image=Image.open(self.frame_list[i])
        curr_x_red, curr_y_red, curr_x_green, curr_y_green = part_2(curr_image,curr_x_red, curr_y_red, curr_x_green, curr_y_green)
        prev_x_red,prev_y_red, prev_x_green,prev_y_green = self.part_1(i-1)
        prev_image = Image.open(self.frame_list[i-1])
        prev_x_red,prev_y_red,prev_x_green, prev_y_green = part_2(prev_image,prev_x_red,prev_y_red,prev_x_green, prev_y_green)
        self.part_3(i,i-1,curr_x_red, curr_y_red, curr_x_green, curr_y_green,prev_x_red,prev_y_red,prev_x_green, prev_y_green)