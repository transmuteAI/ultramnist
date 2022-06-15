# Importing required modules
from calendar import EPOCH
from random import seed
import warnings
import gc
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
import torch.optim as optim
import torch.nn as nn
import torch
import cv2
import pandas as pd
import numpy as np
import pathlib
import logging
import os
import sys
from torchvision.models import resnet50
import yaml
import wandb
warnings.filterwarnings('ignore')

# Seeding to help make results reproduceable
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Building custom dataset
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
    torch.cuda.empty_cache()
    seed_everything(SEED)

    # loggig info into file
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    with open(LOG_PATH, 'w') as fp:
        fp.write(EXPERIMENT_NAME)
        fp.write("\n")
    logging.basicConfig(filename=LOG_PATH, level=logging.INFO)

    #  Loading the train data
    train_df = pd.read_csv(train_csv_path)
    # train_df = train_df.sample(10)
    train_df.head()

    # building training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(train_df['id'], train_df['digit_sum'], test_size=0.1, random_state=SEED)
    print('Data lengths: ', len(X_train), len(X_valid), len(y_train), len(y_valid))

    # Data transforms
    train_transforms = transforms.Compose([transforms.ToPILImage(),
                                           transforms.Resize((image_size, image_size)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])])
    test_transforms = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize((image_size, image_size)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])])

    # DataLoader
    train_dataset = CustomDataset(root_dir,X_train, y_train,train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    valid_dataset = CustomDataset(root_dir,X_valid, y_valid,test_transforms)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)

    # Loading the pretrained model here
    if 'efficientnet' in model_name:
        model = EfficientNet.from_pretrained(model_name)
        model.fc = nn.Sequential(
            nn.Linear(mid_features, 100),
            nn.ReLU(),
            nn.Linear(100, 28)
        )
    else:
        model = resnet50(pretrained = True)
        model._fc = nn.Sequential(
            nn.Linear(mid_features, 100),
            nn.ReLU(),
            nn.Linear(100, 28)
        )
       
    for params in model.parameters():
        params.requires_grad = True

    # Training model
    epoch = 0
    learning_rate = cfg["params"]["learning_rate"]
    if LOAD_CHECKPOINT:
        CHECKPOINT_PATH = f"{base_path}/{model_name}_{image_size}/checkpoint_{epoch}.pth.tar"
        model.load_state_dict(torch.load(CHECKPOINT_PATH)['model_state_dict'])
        learning_rate = 0.0001847341009527235
    best_s = 0.0
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma_value)
    EPOCHS = num_epoch
    

    while epoch < num_epoch:
        print(f'Epoch: {epoch+1}/{EPOCHS}')

        correct = 0
        total = 0
        losses = []

        for batch_idx, data in enumerate(tqdm(train_loader, total=len(train_loader))):
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)
            output = model(images)
            loss = criterion(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred = torch.max(output, 1)
            correct += (pred == targets).sum().item()
            total += pred.size(0)
            losses.append(loss.item())
            loss.detach()
            del images, targets, output, loss
            gc.collect()

        train_loss = np.mean(losses)
        train_acc = correct * 100.0 / total
        del losses
        total = 0
        correct = 0
        valid_acc = 0

        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(valid_loader, total=len(valid_loader))):
                images, targets = data
                images = images.to(device)
                targets = targets.to(device)
                output = model(images)
                _, pred = torch.max(output, 1)
                correct += (pred == targets).sum().item()
                total += pred.size(0)
        valid_acc = correct/total * 100

        # Logging results in file, printing and updating on wandb
        logging.info("\n")
        logging.info(f"EPOCH : {epoch + 1}/{EPOCHS} | LR: {scheduler.get_lr()} | Train Loss = {train_loss} | Accuracy = {train_acc} | Valid Accuracy: {valid_acc}")
        
        #Saving checkpoint, each time the current valid_acc overshoots the previous best.
        if valid_acc > best_s:
            checkpoint_name = PATH + '/checkpoint_' + str(epoch+1) + '.pth.tar'
            torch.save({'model_state_dict': model.state_dict(),}, checkpoint_name)
            best_s = valid_acc
        print(f'Train Loss: {train_loss}\tTrain Acc: {train_acc}\tLR: {scheduler.get_lr()}\tValid Accuracy: {valid_acc}', end = '\r')
        wandb.log({"Train loss": train_loss, "Train Acc": train_acc, "Learning Rate": (scheduler.get_lr()[0]), "Valid Accuracy": valid_acc})
        scheduler.step()
        epoch+=1

if __name__ == "__main__":
    # This helps make all other paths relative
    base_path = pathlib.Path().absolute()

    # Input for the experiment whose results have to be reproduced
    # arg1:  efficientnet-b0, efficientnet-b3, resnet50")
    model_name = sys.argv[1]
    # arg2: (256/512/1024)
    image_size = int(sys.argv[2])
    # arg3: (11 or 24)
    model_select = int(sys.argv[3])

    # Input of the required hyperparameters
    yml_path = f"models/gpu_{model_select}GB/{model_name}_{image_size}.yml"
    if not os.path.exists(yml_path):
        print("No such config file exists.")
        exit()
    with open(yml_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    BATCH_SIZE = cfg["params"]["BATCH_SIZE"]
    mid_features = cfg["params"]["mid_features"]
    learning_rate = cfg["params"]["learning_rate"]
    gamma_value = cfg["params"]["gamma_value"]

    # Fixed hyperparameters
    LOAD_CHECKPOINT = False

    SEED = 42
    num_epoch = 50
    num_workers = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root_dir = f"{base_path}/ultra-mnist_{image_size}/train"
    if not os.path.exists(root_dir):
        print("Dataset missing.")
    EXPERIMENT_NAME = f"{model_name}_{image_size}"
    PATH = f"{base_path}/{EXPERIMENT_NAME}"
    LOG_PATH = f'{PATH}/log_file.txt'
    train_csv_path = f'{base_path}/ultra-mnist_{image_size}/train.csv'


    wandb.login()
    wandb.init(project="ultramnist-dgx", entity="gakash2001")
    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": num_epoch,
        "batch_size": BATCH_SIZE
    }
    run()
