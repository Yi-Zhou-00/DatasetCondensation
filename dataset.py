
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample

        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image), torch.from_numpy(landmarks)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, landmarks = sample
        image = transforms.Normalize(image, mean=self.mean, std=self.std)
        # print("Image type: ", type(image))
        # print("Image shape: ", image.shape)
        return image, landmarks


labels = ['Exiting', 'activity_walking', 'vehicle_moving', 'activity_carrying', 
          'vehicle_turning_left', 'vehicle_stopping', 'Talking', 'activity_standing', 
          'Interacts', 'Closing', 'Opening', 'vehicle_turning_right', 'activity_gesturing', 
          'Pull', 'specialized_talking_phone', 'Transport_HeavyCarry', 'vehicle_starting', 
          'Entering', 'Loading', 'specialized_miscellaneous', 'specialized_texting_phone', 
          'Riding', 'specialized_using_tool', 'activity_running', 'Push', 'Misc']
labels_dict = dict(zip(labels, range(len(labels))))


'''  '''
class TinyViratDataset(Dataset):
    def __init__(self, train=True, transform=ToTensor(), kaggle=False, test="NA"):
        self.kaggle = kaggle
        if self.kaggle:
            self.root_path = "../input/tinyvirat-frames-20-64/TinyVIRAT-frames-20-64"
        else:
            self.root_path = "./TinyVIRAT-frames-20-64"

        self.train = train
        if self.train:
            if test=="NA":
                self.file_name = "tiny_train_20_frames.csv"
            elif test=="10_1":
                self.file_name = "tiny_train_20_frames_test_10_1.csv"
            elif test=="10_10":
                self.file_name = "tiny_train_20_frames_test_10_10.csv"
            self.extra_frame_path = os.path.join(self.root_path, "train")
        else:
            if test=="NA":
                self.file_name = "tiny_test_20_frames.csv"
            elif test=="10_1":
                self.file_name = "tiny_test_20_frames_test_10_1.csv"
            elif test=="10_10":
                self.file_name = "tiny_test_20_frames_test_10_10.csv"
            self.extra_frame_path = os.path.join(self.root_path, "test")

        self.frame_paths = pd.read_csv(os.path.join(self.root_path, self.file_name))["path"]
        self.labels = pd.read_csv(os.path.join(self.root_path, self.file_name))["label"]
        self.n_sample = len(self.labels)      
        self.transform = transform
        self.all_frames = self.__get_all__()
        
    def __getitem__(self, index):
        frame_path = os.path.join(self.extra_frame_path, self.frame_paths[index])
        
        frame = Image.open(frame_path)
        plt.imshow(frame)
        # frame = np.asarray(frame, dtype="float32")  
        # label = np.array([labels_dict[self.labels[index]]])
        frame = np.asarray(frame, dtype="float32")/255
        label = labels_dict[self.labels[index]]
        if self.transform:
            frame = self.transform(frame)

        sample = frame, label

        # print("ITEM: frame type: ", type(frame))
        # print("ITEM: label type: ", type(label))
        # if self.transform:
        #     sample = self.transform(sample)
        
        return sample
    
    def __len__(self):
        return self.n_sample
    
    def classes(self):
        return list(set(self.labels))
    
    def __get_all__(self):
        all_img = []

        for i in range(self.n_sample):
            frame_path = os.path.join(self.extra_frame_path, self.frame_paths[i])
            frame = Image.open(frame_path)
            # plt.imshow(frame)
            frame = np.asarray(frame, dtype="float32")/255
            #label = np.array([labels_dict[self.labels[i]]])
            label = labels_dict[self.labels[i]]

            # label =  torch.tensor(labels_dict[self.labels[i]])
            # print("Frame tpye", type(frame))
            # print("Label tpye", type(label))
            #sample = frame, label #np.array([frame, label])
            if self.transform:
                frame = self.transform(frame)
            # if label == 1: print("Frame type: ", type(frame))
            sample = frame, label
            all_img.append(sample)
        # all_img = torch.from_numpy(np.array(all_img))
        all_img = np.array(all_img)
        return all_img