# Import modules
import cv2
import pandas as pd
import os
import shutil
print("Enter image size:\t")
image_size = int(input())
base_path = os.getcwd()
print(base_path)
source_path = f'{base_path}/ultra-mnist'
destination_path = f'{base_path}/ultra-mnist_{image_size}'
if os.path.exists(destination_path):
    print("Directory already exists.")
elif not os.path.exists(source_path):
    print("Ultramnist dataset missing.")
else:
    os.mkdir(destination_path)
    os.mkdir(f'{destination_path}/train')
    os.mkdir(f'{destination_path}/test')
    shutil.copy2(f'{source_path}/train.csv', f'{destination_path}/train.csv') 
    shutil.copy2(f'{source_path}/sample_submission.csv', f'{destination_path}/sample_submission.csv')
    train = pd.read_csv(f'{destination_path}/train.csv')
    train.head()
    test = pd.read_csv(f'{destination_path}/sample_submission.csv')
    test.head()
    num_train = len(train)
    print("Number of training sample: ", num_train)
    num_test = len(test)
    print("Number of test images: ", num_test)
    for i in range(num_train):
        pth = f'{source_path}/train/{train.iloc[i]["id"]}.jpeg'
        print((i+1), end="\r")
        image = cv2.imread(pth)
        resized_down = cv2.resize(image, (image_size,image_size), interpolation= cv2.INTER_AREA)
        cv2.imwrite(f'{destination_path}/train/{train.iloc[i]["id"]}.jpeg', resized_down)
    for i in range(num_test):
        pth = f'{source_path}/test/{test.iloc[i]["id"]}.jpeg'
        print(i+1, end="\r")
        image = cv2.imread(pth)
        resized_down = cv2.resize(image, (256,256), interpolation= cv2.INTER_AREA)
        cv2.imwrite(f'{destination_path}/test/{test.iloc[i]["id"]}.jpeg', resized_down)