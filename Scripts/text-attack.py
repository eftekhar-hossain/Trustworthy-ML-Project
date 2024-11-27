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
import random
from translate import Translator
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

## text attacks
emoji_list = ['😀', '🎉', '❤️', '🔥', '😊', '🐍', '🚀', '🌟', '🥳', '😎', '😢', '😔', '😭', '😞', '😩', '😲', '😯', '😮', '😳', '😱', '💕', '😍', '😘', '💖', '😠', '😡', '🤬', '👿', '😤']

def add_random_emoji(caption, num_emojis):
    return caption + ' ' + ''.join(random.choices(emoji_list, k=num_emojis))

positive_tokens = ["খুব ভালো", "আনন্দ", "প্রিয়", "সুন্দর", "ভালবাসা", "অসাধারণ", "সাফল্য", "শুভ", "আশা", "জয়"]

# Function to add a variable number of random positive tokens to the text
def add_random_positive_tokens(caption, num_tokens_to_add):
    # Split the text into words to facilitate insertion
    text_words = caption.split()

    # Ensure the number of tokens to add is not more than the available positive tokens
    num_tokens_to_add = min(num_tokens_to_add, len(positive_tokens))  # Limit to the number of positive tokens available

    # Randomly select positive tokens to add
    selected_tokens = random.sample(positive_tokens, num_tokens_to_add)

    # Insert the selected positive tokens at random positions in the text
    for token in selected_tokens:
        random_position = random.randint(0, len(text_words))  # Random position to insert the token
        text_words.insert(random_position, token)

    # Join the words back into a single string and return the result
    return ' '.join(text_words)

def insert_typos(caption, typo_probability=0.1):
    bangla_chars = list(caption)  # Convert text into a list of characters
    typo_text = []

    for char in bangla_chars:
        # Randomly decide whether to introduce a typo
        if random.random() < typo_probability:
            typo_action = random.choice(['remove', 'replace', 'insert'])

            if typo_action == 'remove':
                # Randomly remove a character (skip this character)
                continue
            elif typo_action == 'replace':
                # Replace the character with a random Bangla character or a similar-looking character
                random_bangla_char = random.choice(['অ', 'ই', 'উ', 'এ', 'ঐ', 'ঊ', 'ও', 'অঁ', 'য', 'র', 'ল', 'ম', 'ন', 'প', 'ব', 'ভ'])
                typo_text.append(random_bangla_char)
            elif typo_action == 'insert':
                # Insert a random Bangla character at the current position
                random_bangla_char = random.choice(['অ', 'ই', 'উ', 'এ', 'ঐ', 'ঊ', 'ও', 'অঁ', 'য', 'র', 'ল', 'ম', 'ন', 'প', 'ব', 'ভ'])
                typo_text.append(char)  # Keep the original character
                typo_text.append(random_bangla_char)  # Insert a new character

        else:
            typo_text.append(char)

    return ''.join(typo_text)
# Initialize the Translator
translator = Translator(from_lang="bn", to_lang="en")

# Function to dynamically translate a Bangla word to English
def translate_to_english(word):
    try:
        return translator.translate(word)
    except Exception as e:
        print(f"Translation error for word '{word}': {e}")
        return word  # Return the original word if translation fails

def replace_with_english_dynamic(bangla_text, num_words_to_translate):
    words = bangla_text.split()

    # Ensure we don't exceed the number of words in the text
    num_words_to_translate = min(num_words_to_translate, len(words))

    # Randomly select unique words to translate
    random_words = random.sample(words, num_words_to_translate)

    # Translate and replace each selected word
    for random_word in random_words:
        english_translation = translate_to_english(random_word)
        bangla_text = bangla_text.replace(random_word, english_translation, 1)

    return bangla_text

def all_attacks_together(caption, frequency=1):
    caption = add_random_emoji(caption, frequency)
    caption = replace_with_english_dynamic(caption, frequency)
    caption = insert_typos(caption, 0.1)
    caption = add_random_positive_tokens(caption,1)
    return caption

# Dataset Class

class AnyDataset(Dataset):
        def __init__(self, dataframe, data_dir, max_seq_length, label_column, tokenizer=None, transform=None, 
                     method_name = None, text_attack = None, frequency = None):
            self.data = dataframe
            self.label_field = label_column
            self.max_seq_length = max_seq_length
            self.data_dir = data_dir
            self.tokenizer = tokenizer
            self.transform = transform
            self.method_name = method_name
            self.text_attack = text_attack
            self.frequency = frequency

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img_name = os.path.join(self.data_dir, self.data.loc[idx, 'image_name'])
            image = Image.open(img_name)
            caption = self.data.loc[idx, 'Captions']

            if self.text_attack == 'emoji':
                caption = add_random_emoji(caption, self.frequency)

            elif self.text_attack== 'positive_token':
                caption = add_random_positive_tokens(caption, self.frequency)   

            elif self.text_attack == 'typo':
                caption = insert_typos(caption, self.frequency)
            
            elif self.text_attack == 'translate':
                caption = replace_with_english_dynamic(caption, self.frequency)
            
            elif self.text_attack == 'all':
                caption = all_attacks_together(caption, self.frequency)    
                
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

    def get_dataloader(dataset, image_dir, max_len, label_column, tokenizer,method_name, transform, text_attack, frequency):
    
        # Create data loaders for 
        test_dataset = AnyDataset(dataframe = dataset, data_dir = image_dir, max_seq_length=max_len, 
                                label_column = label_column, tokenizer = tokenizer,
                                transform=transform ,method_name=method_name, text_attack=text_attack, frequency=frequency)
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
    if args.text_attack == 'typo':
        values = [0.1, 0.3, 0.5, 0.7]
    elif args.text_attack == 'translate':
        values = [1,2,3]      
    else:
        values = [1,2,3,4,5]  #label    

    if values:
        print("Dataset: ",args.dataset.upper(),'\t','Method:',args.method.upper())
        for i in range(len(values)):
            params = values[i]

            print("Text attack: ",args.text_attack.upper(), '\n')    

            # get data loader 
            test_loader = get_dataloader(dataset = dataset, image_dir = memes_path, max_len= max_len,
                                            label_column = label_column, tokenizer = tokenizer, 
                                            method_name = args.method, transform = data_transform,
                                            text_attack = args.text_attack, frequency = params)
            
            
            
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
    parser.add_argument('--text_attack', default = 'emoji', choices=['emoji','typo', 'translate','positive_token','all'],
                           help='text attack name')
  
    args = parser.parse_args()
    main(args)
