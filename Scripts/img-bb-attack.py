import pandas as pd
import numpy as np
import os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModel,AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import clip
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add the Scripts directory to sys.path
scripts_path = os.path.join(os.path.dirname(os.getcwd()), 'Scripts')
sys.path.append(scripts_path)

import dora,maf,mclip


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

## image Noises 

# Salt and Pepper Noise
def add_salt_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    total_pixels = image.size
    salt_pixels = int(total_pixels * salt_prob)
    pepper_pixels = int(total_pixels * pepper_prob)

    # Adding Salt
    for _ in range(salt_pixels):
        x, y = np.random.randint(0, image.shape[0]), np.random.randint(0, image.shape[1])
        noisy_image[x, y] = 255

    # Adding Pepper
    for _ in range(pepper_pixels):
        x, y = np.random.randint(0, image.shape[0]), np.random.randint(0, image.shape[1])
        noisy_image[x, y] = 0

    return noisy_image


# Gaussian Noise
def add_gaussian_noise(image, mean, std):
    gaussian_noise = np.random.normal(mean, std, image.shape)
    noisy_image = np.clip(image + gaussian_noise, 0, 255).astype(np.uint8)
    return noisy_image


# NewsPrint Noise
def add_newsprint_noise(image,label):
    # label -->  1,2,3,4,5
    # dont need numpy array
    noisy_image = ImageOps.posterize(image,label)  # Simulates newsprint-like effect
    return noisy_image

# Random Noise
def add_random_noise(image, intensity):
    # intensity=0.5
    random_noise = np.random.rand(*image.shape) * 255 * intensity
    noisy_image = np.clip(image + random_noise, 0, 255).astype(np.uint8)
    return noisy_image


# Dataset Class

class AnyDataset(Dataset):
        def __init__(self, dataframe, data_dir, max_seq_length, label_column, tokenizer=None, transform=None, 
                     method_name = None, noise = None, noise_params = None):
            self.data = dataframe
            self.label_field = label_column
            self.max_seq_length = max_seq_length
            self.data_dir = data_dir
            self.tokenizer = tokenizer
            self.transform = transform
            self.method_name = method_name
            self.noise = noise
            self.noise_params = noise_params

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img_name = os.path.join(self.data_dir, self.data.loc[idx, 'image_name'])
            image = Image.open(img_name)

            if self.noise == 'salt-peper':
                salt = self.noise_params['salt']
                peper = self.noise_params['peper']
                image = np.array(image)  # Convert to numpy array for noise application
                image = add_salt_pepper_noise(image, salt, peper)  # salt, pepper varied 10 20 30
                image = Image.fromarray(image.astype('uint8'), 'RGB')  # Convert back to PIL Image

            elif self.noise == 'gaussian':
                mean = self.noise_params['mean']
                std = self.noise_params['std']
                image = np.array(image)  # Convert to numpy array for noise application
                image = add_gaussian_noise(image, mean,std)  # mean = 0, std = 25
                image = Image.fromarray(image.astype('uint8'), 'RGB')  # Convert back to PIL Image    

            elif self.noise == 'newsprint':
                label = self.noise_params
                image = add_newsprint_noise(image,label)  # label
                # 
            elif self.noise == 'random':
                intensity = self.noise_params
                image = np.array(image)  # Convert to numpy array for noise application
                image = add_random_noise(image,intensity)  # intensity=0.5
                image = Image.fromarray(image.astype('uint8'), 'RGB')  # Convert back to PIL Image    

                
            caption = self.data.loc[idx, 'Captions']
            label = int(self.data.loc[idx, self.label_field ])

            if self.method_name =='mclip':
                if self.transform:
                    image = self.transform(image.convert("RGB"))
                return {'image': image,'text': caption, 'label': label }

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

def main(args):

    curr_dir = os.getcwd()
    # print(curr_dir)
    root_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))
    # print(root_dir)
    dataset_dir = os.path.join(os.path.join(root_dir, 'Datasets'))
    # print(dataset_dir)
    models_dir = os.path.join(os.path.join(root_dir, 'Saved_Models'))
    # print(models_dir)

    # BHM dir
    bhm_dir = os.path.join(dataset_dir,'BHM')        
    files_dir = os.path.join(bhm_dir,'Files')
    # bhm_memes_path = os.path.join(bhm_dir,'Memes')
    # MIMOSA dir
    mimosa_dir = os.path.join(dataset_dir,'MIMOSA')
    # mimosa_memes_path = os.path.join(mimosa_dir,'Memes')    

    if args.dataset =='bhm':
        # Read the BHM test Set
        test_bhm =  pd.read_excel(os.path.join(files_dir,'test_task1.xlsx'))
        test_bhm['Labels']= test_bhm['Labels'].replace({'non-hate':0,'hate':1})
        memes_path = os.path.join(bhm_dir,'Memes')
        dataset = test_bhm
        label_column = 'Labels'

    elif args.dataset =='mimosa':
        # Read the MIMOSA Test Set
        test_mimosa =  pd.read_csv(os.path.join(mimosa_dir, 'aggressive_memes_test.csv'))
        test_mimosa['Label'] = test_mimosa['Label'].replace({0:0,1:1,2:1,3:1,4:1})
        memes_path = os.path.join(mimosa_dir,'Memes')    
        dataset = test_mimosa
        label_column = 'Label'

    # Tokinizer will depend on method name

    def get_dataloader(dataset, image_dir, max_len, label_column, tokenizer,method_name, transform, noise, noise_params):
    
        # Create data loaders for 
        test_dataset = AnyDataset(dataframe = dataset, data_dir = image_dir, max_seq_length=max_len, 
                                label_column = label_column, tokenizer = tokenizer,
                                transform=transform ,method_name=method_name, noise = noise,
                                noise_params = noise_params)
        test_loader = DataLoader(test_dataset, batch_size=16,shuffle=False)

        return test_loader


    ## Tokenizer
    # DORA
    if args.method == 'dora':
        max_len = 50
        tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-564M")
    # MAF
    elif args.method == 'maf':    
        tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")
        max_len = 70 

    # M-CLIP  
    elif args.method == 'mclip':    
        max_len = None    
        tokenizer = None

       
    # Data preprocessing and augmentation
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
  
    
    values = []
    if args.noise == 'salt-peper':
        values = [0.05,0.1,0.2,0.3,0.4,0.5]
    elif args.noise == 'gaussian':
        values = [25,50,75,100]  # std values
    elif args.noise == 'newsprint':
        values = [1,2,3,4,5]  #label   
    elif args.noise == 'random':
        values = [0.25,0.5,0.75]  #label    

    if values:
        print("Dataset: ",args.dataset.upper(),'\t','Method:',args.method.upper())
        for i in range(len(values)):

            # # salt-peper params
            if args.noise == 'salt-peper':
                params = {'salt':values[i], 'peper':values[i]}
            # # guass params
            elif args.noise == 'gaussian':
                params = {'mean':0, 'std':values[i]}
            # # news params
            elif args.noise == 'newsprint':
                params = values[i]

            else:
                params = values[i]

            print("Noise: ",args.noise.upper(), '\n')    

            # get data loader 
            test_loader = get_dataloader(dataset = dataset, image_dir = memes_path, max_len= max_len,
                                            label_column = label_column, tokenizer = tokenizer, 
                                            method_name = args.method, transform = data_transform,
                                            noise = args.noise, noise_params = params)
            
            
            
            model_path = f"{args.method}_{args.dataset}_{2e-5}.pth"
            if args.method == 'maf':
                maf.predict(os.path.join(models_dir,model_path), test_loader, heads = 16)
            elif args.method == 'dora':
                dora.predict(os.path.join(models_dir,model_path), test_loader, heads = 2)    
            else:
                model_path = f"clip_{args.dataset}_{0.0005}.pth"
                mclip.predict(os.path.join(models_dir,model_path),test_loader)


    if args.noise == 'all':
        noises = ['salt-peper', 'gaussian','newsprint','random']
        
        print("Dataset: ",args.dataset.upper(),'\t','Method:',args.method.upper())
        
        for n in noises:
            # print(n)

            if n == 'salt-peper':
                values = [0.01,0.03,0.05]
            elif n == 'gaussian':
                values = [25,50,75,100]  # std values
            elif n== 'newsprint':
                values = [2,3,4]  #label   
            elif n=='random':
                values = [0.2,0.4,0.5]  #label  

            print(values)

            print("Noise: ",n.upper(), '\n')
            for i in range(len(values)):

                # salt-peper params
                if n == 'salt-peper':
                    params = {'salt':values[i], 'peper':values[i]}
                # # guass params
                elif n == 'gaussian':
                    params = {'mean':0, 'std':values[i]}
                # # news params
                elif n == 'newsprint':
                    params = values[i]
                else:
                    params = values[i]

                print("Parameters:",params)    

                # get data loader for BHM-MAF 
                test_loader = get_dataloader(dataset = dataset, image_dir = memes_path, max_len= max_len,
                                                 label_column = label_column, 
                                                tokenizer = tokenizer, method_name = args.method, 
                                                transform = data_transform,
                                                noise = n, noise_params = params)
                
                
                
                model_path = f"{args.method}_{args.dataset}_{2e-5}.pth"
                if args.method == 'maf':
                    maf.predict(os.path.join(models_dir,model_path), test_loader, heads = 16)
                elif args.method == 'dora':
                    dora.predict(os.path.join(models_dir,model_path), test_loader, heads = 2)    
                else:
                    model_path = f"clip_{args.dataset}_{0.0005}.pth"
                    mclip.predict(os.path.join(models_dir,model_path),test_loader)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adversarial Attack on Hateful Memes')

    parser.add_argument('--dataset', default='bhm', choices=['bhm', 'mimosa'],
                        help='dataset name')
    parser.add_argument('--method', default='dora', choices=['dora', 'maf','mclip'],
                        help='method name')
    parser.add_argument('--noise', default = 'gaussian', choices=['salt-peper', 'gaussian','newsprint','random','all'],
                           help='noise name')
  
    args = parser.parse_args()
    main(args)
