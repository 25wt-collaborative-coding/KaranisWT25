import cv2
import os 
import random
import numpy as np

# Get the list of all files and directories
letter_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
train_image_list = []
test_image_list = [] 
train_image_list_names = [] #test
test_image_list_names = [] #test

def resize_image(image, target_height, target_width):
    h, w = image.shape
    scale = target_height / h
    new_width = int(w * scale)
    resized = cv2.resize(image, (new_width, target_height))

    if new_width < target_width:
        pad_width = target_width - new_width
        padded = cv2.copyMakeBorder(resized, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=0)
        return padded
    else:
        return resized[:, :target_width]

#standardize images
for letter in letter_list:
    path = "./dataset/" + letter
    dir_list = os.listdir(path)
    index = 0
    for i in dir_list:
        if i == ".DS_Store":
            dir_list.pop(index)
        index = index + 1    
    dir_list.pop(0)
    for i in dir_list:
        image_path = "./dataset/"+ letter + "/" + i
        image = cv2.imread("./dataset/"+ letter + "/" + i, cv2.IMREAD_GRAYSCALE)
        image = image / 255.0
        resize_image(image, 32, 128)


#split data
for letter in letter_list:
    path = "./dataset/" + letter
    dir_list = os.listdir(path)
    index = 0
    for i in dir_list:
        if i == ".DS_Store":
            dir_list.pop(index)
        index = index + 1    
    dir_list.pop(0)
    file_list = []
    file_name_list = [] #test 
    for i in dir_list:
        image_path = "./dataset/"+ letter + "/" + i
        image = cv2.imread("./dataset/"+ letter + "/" + i)
        image_name = os.path.basename(image_path) #test
        file_name_list.append(image_name) #test
        file_list.append(image) 
    n = 2
    random.shuffle(file_list)
    random.shuffle(file_name_list) #test
    train_image_list.append(file_list[:n])
    test_image_list.append(file_list[n:])
    train_image_list_names.append(file_name_list[:n]) #test
    test_image_list_names.append(file_name_list[n:]) #test
    
#test
print("TRAINING SET: ")
print(train_image_list_names)
print()
print("TEST SET:")
print(test_image_list_names)




    




    


