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
from transformers import AutoModel,AutoTokenizer
from multilingual_clip import pt_multilingual_clip
import argparse
from tqdm import tqdm
from sklearn.metrics import classification_report
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

class AnyDataset(Dataset):
        def __init__(self, dataframe, data_dir, max_seq_length, label_column, tokenizer=None, transform=None, 
                     method_name = None):
            self.data = dataframe
            self.label_field = label_column
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


# Tokinizer will depend on method name

def get_dataloader(dataset, image_dir, max_len, label_column, tokenizer,method_name, transform ):

    # Create data loaders for 
    test_dataset = AnyDataset(dataframe = dataset, data_dir = image_dir, max_seq_length=max_len, 
                            label_column = label_column, tokenizer = tokenizer,
                            transform=transform ,method_name=method_name)
    test_loader = DataLoader(test_dataset, batch_size=16,shuffle=False)

    return test_loader



# Load pre-trained ResNet model (e.g., ResNet-18)
model_resnet = models.resnet18(pretrained=True)
model_resnet.eval()  # Set the model to evaluation mode

# Move the model to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_resnet = model_resnet.to(device)



def fgsm_transfer_attack(model, image, labels, epsilon=0.03):
    """
    Perform FGSM attack on the image using the ResNet model.
    model: the ResNet model
    image: the input image tensor
    labels: the true labels (for calculating gradients)
    epsilon: the perturbation strength
    """
    # Ensure the image requires gradients
    image.requires_grad = True

    # Forward pass through the model
    output = model(image)  # Get model output
    loss = F.cross_entropy(output, labels)  # Calculate loss using CrossEntropyLoss

    # Zero all gradients
    model.zero_grad()

    # Backward pass to compute gradients
    loss.backward()

    # Get the gradient of the image
    grad = image.grad.data

    # Apply the perturbation: epsilon * sign of the gradient
    adversarial_image = image + epsilon * grad.sign()

    # Return the adversarial image
    return adversarial_image


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


    ## Tokenizer
    # DORA
    if args.method == 'dora':
        max_len = 50
        tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-564M")
    # MAF
    elif args.method == 'maf':    
        tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")
        max_len = 70
    elif args.method == 'mclip':
        # Assuming you have already defined and loaded your model
        tokenizer = AutoTokenizer.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-L-14')
        clip_text = pt_multilingual_clip.MultilingualCLIP.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-L-14')    
        max_len = None

    # if args.method in ['dora','maf']:    
        # Data preprocessing and augmentation
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # else:
    #     data_transform = None

    
    
    # get data loader 
    test_loader = get_dataloader(dataset = dataset, image_dir = memes_path, max_len= max_len,
                                    label_column = label_column, tokenizer = tokenizer, 
                                    method_name = args.method, transform = data_transform)
    
    # for batch in test_loader:
    #     print(batch['image'])
    #     print(batch['text'])
            
    
    ## Model Loading 
    model_path = f"{args.method}_{args.dataset}_{2e-5}.pth"
    if args.method == 'maf':
        model = maf.load_model(os.path.join(models_dir,model_path),heads = 16)
    elif args.method == 'dora':
        model = dora.load_model(os.path.join(models_dir,model_path),heads = 2)
    else:
        model_path = f"clip_{args.dataset}_{0.0005}.pth"
        model = mclip.load_model(os.path.join(models_dir,model_path))

    print("--------------------------------")
    print('Adversarial Evaluation.')
    print("--------------------------------")
    print("Dataset: ",args.dataset.upper(),'\t','Method:',args.method.upper())
    print("Perturbation: ",args.epsilon,)

    if args.method !='mclip':
        # Initialize variables to store labels and predictions
        test_labels = []
        test_preds = []

        # Set the model to evaluation mode
        model.eval()

        # Iterate over the test loader without using no_grad during the attack loop
        for batch in tqdm(test_loader, desc="Testing", unit="batch"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
     
            # Apply PGD attack to generate adversarial images
            adversarial_images = fgsm_transfer_attack(model_resnet, images,labels, epsilon=args.epsilon)                               

            # Get the model's predictions for the adversarial images
            with torch.no_grad():  # We don't need gradients for the model during inference
                outputs = model(adversarial_images, input_ids, attention_mask).squeeze().cpu().numpy()

            # Convert the outputs to binary predictions
            preds = (outputs > 0.5).astype(int)
            
            # Store the true labels and predicted labels
            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(preds)

        # Print evaluation results
        
        print("--------------------------------")
        print(classification_report(test_labels, test_preds, digits=3))


    
    if args.method =='mclip':
        # Initialize variables to store labels and predictions
        test_labels = []
        test_preds = []

        # Set the model to evaluation mode
        model.eval()

        # Iterate over the test loader without using no_grad during the attack loop
        for batch in tqdm(test_loader, desc="Testing", unit="batch"):
            images = batch['image'].to(device)
            texts= batch['text']
            text_embed = clip_text.forward(texts,tokenizer).to(device)
            labels = batch['label'].to(device)
            
            # Apply FGSM attack to generate adversarial images
            adversarial_images = fgsm_transfer_attack(model_resnet, images,labels, epsilon=args.epsilon)

            # Get the model's predictions for the adversarial images
            with torch.no_grad():  # We don't need gradients for the model during inference
                outputs = model(adversarial_images,  text_embed)

            # Convert the outputs to binary predictions
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Store the true labels and predicted labels
            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(preds)

        # Print evaluation results
        print('Evaluation Done.')
        print("--------------------------------")
        print(classification_report(test_labels, test_preds, digits=3))      
     


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adversarial Attack on Hateful Memes')

    parser.add_argument('--dataset', default='bhm', choices=['bhm', 'mimosa'],
                        help='dataset name')
    parser.add_argument('--method', default='dora', choices=['dora', 'maf','mclip'],
                        help='method name')
    parser.add_argument('--epsilon', type=float, default = 0.015,
                           help='perturbation amount')
   
  
    args = parser.parse_args()
    main(args)
