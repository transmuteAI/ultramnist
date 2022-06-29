import os
import argparse
from ultramnist import CreateUltraMNIST

ap = argparse.ArgumentParser()
ap.add_argument("--root_path", type=str, default="../data/", help="path to the root directory where the generated data will be stored")
args = ap.parse_args()

obj_umnist = CreateUltraMNIST(root=os.path.join(args.root_path, 'ultramnist'), 
                                base_data_path=os.path.join(args.root_path, 'mnist'), 
                                n_samples = [28000, 28000], 
                                img_size=4000)
obj_umnist.generate_dataset()