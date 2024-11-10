import pandas as pd
import numpy as np
import os
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModel,AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import clip
import argparse
import warnings
warnings.filterwarnings('ignore')

# Set seed.
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


# Set the device to GPU if available, else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("Device:",device)

 
start_time = time.time()


class BHM(Dataset):
        def __init__(self, dataframe, data_dir, max_seq_length, tokenizer=None, transform=None, method_name = None):
            self.data = dataframe
            self.max_seq_length = max_seq_length
            self.data_dir = data_dir
            self.tokenizer = tokenizer
            self.transform = transform
            self.method_name = method_name

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img_name = os.path.join(self.data_dir, self.data.loc[idx, 'image_name'])
            image = Image.open(img_name)
            caption = self.data.loc[idx, 'Captions']
            label = int(self.data.loc[idx, 'Labels'])

            if self.method_name =='mclip':
                # print(f"Applying {self.method_name.upper()} Transformation")
                if self.transform:
                    image = self.transform(image.convert("RGB"))

                return {
                    'image': image,
                    'text': caption,
                    'label': label
                }

            else:

                if self.transform:
                    image = self.transform(image)

                # Tokenize the caption using tokenizer
                inputs = self.tokenizer(caption, return_tensors='pt',
                                        padding='max_length', truncation=True, max_length=self.max_seq_length)

                return {
                    'image': image,
                    'input_ids': inputs['input_ids'].squeeze(),
                    'attention_mask': inputs['attention_mask'].squeeze(),
                    'label': label
                }

# clip_model, preprocess = clip.load("ViT-B/32", device=device)

# # Convert model weights to the same data type as the input data
# clip_model = clip_model.half()


class MUTE(Dataset):
        def __init__(self, dataframe, data_dir, max_seq_length, tokenizer=None, transform=None, method_name = None):
            self.data = dataframe
            self.max_seq_length = max_seq_length
            self.data_dir = data_dir
            self.tokenizer = tokenizer
            self.transform = transform
            self.method_name = method_name

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img_name = os.path.join(self.data_dir, self.data.loc[idx, 'image_name'])
            image = Image.open(img_name)
            caption = self.data.loc[idx, 'Captions']
            label = int(self.data.loc[idx, 'Label'])

            if self.method_name =='mclip':
                # print(f"Applying {self.method_name.upper()} Transformation")
                if self.transform:
                    image = self.transform(image.convert("RGB"))

                return {
                    'image': image,
                    'text': caption,
                    'label': label
                }
            
            else:

                if self.transform:
                    image = self.transform(image)

                # Tokenize the caption using tokenizer
                inputs = self.tokenizer(caption, return_tensors='pt',
                                        padding='max_length', truncation=True, max_length=self.max_seq_length)

                return {
                    'image': image,
                    'input_ids': inputs['input_ids'].squeeze(),
                    'attention_mask': inputs['attention_mask'].squeeze(),
                    'label': label
                }


class MIMOSA(Dataset):
        def __init__(self, dataframe, data_dir, max_seq_length, tokenizer=None, transform=None, method_name=None):
            self.data = dataframe
            self.max_seq_length = max_seq_length
            self.data_dir = data_dir
            self.tokenizer = tokenizer
            self.transform = transform
            self.method_name= method_name

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img_name = os.path.join(self.data_dir, self.data.loc[idx, 'image_name'])
            image = Image.open(img_name)
            caption = self.data.loc[idx, 'Captions']
            label = int(self.data.loc[idx, 'Label'])

            if self.method_name =='mclip':
                # print(f"Applying {self.method_name.upper()} Transformation")
                if self.transform:
                    image = self.transform(image.convert("RGB"))

                return {
                    'image': image,
                    'text': caption,
                    'label': label
                }

            else:
                # print(f"Applying {self.method_name.upper()} Transformation")
                if self.transform:
                    image = self.transform(image)

                # Tokenize the caption using tokenizer
                inputs = self.tokenizer(caption, return_tensors='pt',
                                        padding='max_length', truncation=True, max_length=self.max_seq_length)

                return {
                    'image': image,
                    'input_ids': inputs['input_ids'].squeeze(),
                    'attention_mask': inputs['attention_mask'].squeeze(),
                    'label': label
                }

def load_dataset(dataset_name, max_len, batch_size, tokenizer=None, method_name=None):

    curr_dir = os.getcwd()
    # print(curr_dir)
    root_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))
    # print(root_dir)
    dataset_dir = os.path.abspath(os.path.join(root_dir, 'Datasets'))
    # print(dataset_dir)

    # Data preprocessing and augmentation
    data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"Preparing {dataset_name.upper()} Data Loaders...")

    if dataset_name =='bhm':
        bhm_dir = os.path.join(dataset_dir,'BHM')
        
        files_dir = os.path.join(bhm_dir,'Files')
        memes_path = os.path.join(bhm_dir,'Memes')

        # Read the Dataset
        train_data = pd.read_excel(os.path.join(files_dir,'train_task1.xlsx'))
        valid_data = pd.read_excel(os.path.join(files_dir,'valid_task1.xlsx'))
        test_data =  pd.read_excel(os.path.join(files_dir,'test_task1.xlsx'))

        # Label Encoding
        train_data['Labels'] = train_data['Labels'].replace({'non-hate':0,'hate':1})
        valid_data['Labels'] = valid_data['Labels'].replace({'non-hate':0,'hate':1})
        test_data['Labels']= test_data['Labels'].replace({'non-hate':0,'hate':1})

        
        # Create data loaders
        train_dataset = BHM(dataframe = train_data, tokenizer = tokenizer,data_dir = memes_path,
                                    max_seq_length=max_len,transform=data_transform,method_name=method_name)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = BHM(dataframe = valid_data, tokenizer = tokenizer,data_dir = memes_path,
                                    max_seq_length=max_len,transform=data_transform,method_name=method_name )
        val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=False)

        test_dataset = BHM(dataframe = test_data, tokenizer = tokenizer,data_dir = memes_path,
                                    max_seq_length=max_len,transform=data_transform,method_name=method_name )
        test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)


    elif dataset_name =='mute':
        mute_dir = os.path.join(dataset_dir,'MUTE')
        memes_path = os.path.join(mute_dir,'Memes')    

         # Read the Dataset
        train_data = pd.read_excel(os.path.join(mute_dir, 'train_hate.xlsx'))
        valid_data = pd.read_excel(os.path.join(mute_dir, 'valid_hate.xlsx'))
        test_data =  pd.read_excel(os.path.join(mute_dir, 'test_hate.xlsx'))

        # Label Encoding
        train_data['Label'] = train_data['Label'].replace({'not-hate':0,'hate':1})
        valid_data['Label'] = valid_data['Label'].replace({'not-hate':0,'hate':1})
        test_data['Label'] = test_data['Label'].replace({'not-hate':0,'hate':1})

        # Create data loaders
        train_dataset = MUTE(dataframe = train_data, tokenizer = tokenizer,data_dir = memes_path,
                                    max_seq_length=max_len,transform=data_transform, method_name=method_name )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = MUTE(dataframe = valid_data, tokenizer = tokenizer,data_dir = memes_path,
                                    max_seq_length=max_len,transform=data_transform,method_name=method_name )
        val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=False)

        test_dataset = MUTE(dataframe = test_data, tokenizer = tokenizer,data_dir = memes_path,
                                    max_seq_length=max_len,transform=data_transform,method_name=method_name )
        test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)


    elif dataset_name=='mimosa':
        mimosa_dir = os.path.join(dataset_dir,'MIMOSA')
        memes_path = os.path.join(mimosa_dir,'Memes')    

         # Read the Dataset
        train_data = pd.read_csv(os.path.join(mimosa_dir, 'aggressive_memes_train.csv'))
        valid_data = pd.read_csv(os.path.join(mimosa_dir, 'aggressive_memes_val.csv'))
        test_data =  pd.read_csv(os.path.join(mimosa_dir, 'aggressive_memes_test.csv'))

        # print(type(train_data['Label'][1]))
        #print(train_data['Label'].value_counts())

        # Label Encoding
        train_data['Label'] = train_data['Label'].replace({0:0,1:1,2:1,3:1,4:1})
        valid_data['Label'] = valid_data['Label'].replace({0:0,1:1,2:1,3:1,4:1})
        test_data['Label'] = test_data['Label'].replace({0:0,1:1,2:1,3:1,4:1})

        # print(train_data['Label'].value_counts())

        # Create data loaders
        train_dataset = MIMOSA(dataframe = train_data, data_dir = memes_path,
                                    max_seq_length=max_len, tokenizer = tokenizer,transform=data_transform ,method_name = method_name)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = MIMOSA(dataframe = valid_data,data_dir = memes_path,
                                    max_seq_length=max_len, tokenizer = tokenizer,transform=data_transform, method_name = method_name )
        val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=False)

        test_dataset = MIMOSA(dataframe = test_data,data_dir = memes_path,
                                    max_seq_length=max_len,tokenizer = tokenizer, transform=data_transform,method_name = method_name)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)  
  

    print("Done.")

    return train_loader, val_loader, test_loader    
