# ultramnist

Official Repository related to UltraMNIST classification benchmark

## Getting Started

You will need [Python 3.9.12](https://www.python.org/downloads) and the packages specified in _requirements.txt_.
We recommend setting up a [virtual environment with pip](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
and installing the packages there.

Install packages with:
```
$ pip install -r requirements.txt
```
## Configure and Run

All configurations concerning data, model, training, etc. can be called using commandline arguments.

The main script offers many options, which can be used to reproduce all the 18 results of the paper.

### Dataset placement

To reproduce the result, we first need to download the [UltraMNIST dataset from kaggle,](https://www.kaggle.com/competitions/ultra-mnist/) which has images of size 4000x4000 each. Accept the competition rule, download the dataset and then it has to be moved inside the cloned repository, where requirements.txt, train,py and other files lie.
The directory structure file can be used as a refrence to place the folder properly.

### Dataset generation

To create dataset of images of any specified size, the data_generator.py has to be used.
The syntax is:
     python data_generator.py (image_size)
The example code for 256x256 data generation would be:
```
$ python data_generator.py 256
```

### Training 
The systax to train any of the configurations is:
     python train.py <model_name> <image_size> <gpu_configuration>

**model_name** : It can be effcientnet-b0, efficientnet-b3 or resnet50.
**image_size** : It can take 256, 512 and 1024 as valid input.
**gpu_configuration** : It can take 11 and 24 as input.

And example code to train efficientnet-b0 for 256x256 image size on a 11GB GPU would be:
```
$ python train.py efficientnet-b0 256 11
```
If wandb is not logged in, the code might prompt wandb login.

### Testing
After training new folders namely wandb and the folder for the specified configurations would be created. The folder would have a file for logging the results of each epoch, along with the weights for the epochs whenever it beats the previous best.

For creating the submission file, the test.py has to be run using the syntax:
     python test.py <model_name> <image_size> <epoch>
**epoch** : This parameter is to tell the epoch whose weights has to be used to produce the submission file.

An example code to load 27th epoch results for efficientnet-b0 with image size 256x256 would be:
```
$ python test.py efficientnet-b0 256 27
```
### Submission
Now, this submission file can be submitted on kaggle to get the test results.