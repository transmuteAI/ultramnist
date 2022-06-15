# importing required modules
import gc
import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from torchvision.models import resnet50
import pathlib
import yaml
import warnings

warnings.filterwarnings('ignore')

# This helps make all other paths relative
base_path = pathlib.Path().absolute()

# Input for the experiment whose results have to be reproduced
print("Which model fo you want to run:(1/2/3): EfficientNetB0, EfficientNetB3, Resnet50")
model_select = 0
model_name = ""
load_epoch = 0
while(True):
    # model_select = int(input())
    model_select = 3
    if model_select == 1:
        model_name = "efficientnet-b0"
    elif model_select == 2:
        model_name = "efficientnet-b3"
    elif model_select == 3:
        model_name = "resnet50"
    else:
        continue
    break

print("What image size do you want?(256/512/1024)")
while(True):
    # image_size = int(input())
    image_size =1024
    if (image_size == 256) | (image_size == 512) | (image_size == 1024):
        break

print("Enter the epoch number to load.")
while(True):
    # load_epoch = int(input())
    load_epoch = 18
    if(model_select >= 0) & (model_select < 51):
        break

# Input of the required hyperparameters
with open(f"models/gpu_11GB/{model_name}_{image_size}.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
BATCH_SIZE = 4*cfg["params"]["BATCH_SIZE"]
mid_features = cfg["params"]["mid_features"]

# Fixed hyperparameters
SEED = 42
EPOCHS = 50
num_workers = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
root_dir = f"{base_path}/ultra-mnist_{image_size}/train"
EXPERIMENT_NAME = f"{model_name}_{image_size}"
PATH = f"{base_path}/{EXPERIMENT_NAME}"
LOG_PATH = f'{PATH}/log_file.txt'
train_csv_path = f'{base_path}/ultra-mnist_{image_size}/train.csv'

# DEFINING hyperameters
pretrained_model = f'{base_path}/{model_name}_{image_size}/checkpoint_{load_epoch}.pth.tar'
print(pretrained_model)
sample_csv_path = f'{base_path}/ultra-mnist_{image_size}/sample_submission.csv'
sample_df = pd.read_csv(sample_csv_path)

X_final = sample_df['id']
y_final = sample_df["digit_sum"]

# Training dataset
class CustomDataset(Dataset):
    def __init__(self, root_dir, X_train, y_train, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.X_train = X_train
        self.y_train = y_train

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, index):
        label = self.y_train.iloc[index]

        image_path = f"{self.root_dir}/{self.X_train.iloc[index]}.jpeg"
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image)
        return image, torch.tensor(label)

def run():
    # Data transforms
    test_transforms = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize((image_size, image_size)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5],
                                                               [0.5, 0.5, 0.5])])

    test_dataset = CustomDataset(f'{base_path}/ultra-mnist_{image_size}/test',
                                 X_final, y_final,
                                 test_transforms)

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    # Loading the pretrained model here
    if 'efficientnet' in model_name:
        model = EfficientNet.from_pretrained(model_name)
    else:
        model = resnet50(pretrained = True)

    if "efficientnet" in model_name:
        model.fc = nn.Sequential(
            nn.Linear(mid_features, 100),
            nn.ReLU(),
            nn.Linear(100, 28)
        )
    else:
        model._fc = nn.Sequential(
            nn.Linear(mid_features, 100),
            nn.ReLU(),
            nn.Linear(100, 28)
        )

    # loading the trained weights
    model.load_state_dict(torch.load(pretrained_model)['model_state_dict'])
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model.to(device)

    # Testing loop
    label = np.array([])
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)
            output = model(images)
            _, pred = torch.max(output, 1)
            label = np.concatenate((label, np.array(pred.cpu().data)), axis=0)

    # Preparing submission file
    sample_df['digit_sum'] = label
    sample_df['digit_sum'] = sample_df["digit_sum"].astype(int)
    sample_df.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    run()