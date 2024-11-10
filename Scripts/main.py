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

import dataset, dora, maf, mclip


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

def main(args):
    

    # Initialize XGLM

    if args.method == 'dora':

        tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-564M")
        train_loader, valid_loader, test_loader = dataset.load_dataset(args.dataset,  max_len=args.max_len, 
                                                                       batch_size = 16, tokenizer=tokenizer,)
        dora.fit(args.dataset, train_loader, valid_loader, heads = args.heads, 
                    epochs = args.epochs, lr_rate = args.learning_rate )
        model_path = f"dora_{args.dataset}_{args.learning_rate}.pth"
        dora.predict(model_path, test_loader, heads = args.heads)

        # for batch in test_loader:
        #     label = batch['label']
        #     print(label)
    elif args.method == 'maf':
        # maf requires max_len = 70
        tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")
        train_loader, valid_loader, test_loader = dataset.load_dataset(args.dataset,max_len=args.max_len, 
                                                                       batch_size = 16, tokenizer=tokenizer, )
        maf.fit(args.dataset, train_loader, valid_loader, heads = args.heads, 
                epochs = args.epochs, lr_rate = args.learning_rate )
        model_path = f"maf_{args.dataset}_{args.learning_rate}.pth"
        maf.predict(model_path, test_loader, heads = args.heads)
        # for batch in test_loader:
        #     input_ids = batch['input_ids']
        #     # label = batch['label']
        #     print(input_ids)
        #     # print(image)
        #     # print(label)

    elif args.method =='mclip':
        # max_len here will be set by CLIP by default
        # CLIP learning rate is very crucial default will not work here, 5e-4 works well in some tasks

        train_loader, valid_loader, test_loader = dataset.load_dataset(args.dataset, max_len=args.max_len, 
                                                                       batch_size = 16, method_name = args.method)
        mclip.fit(args.dataset, train_loader, valid_loader, epochs = args.epochs, lr_rate = args.learning_rate )
        model_path = f"clip_{args.dataset}_{args.learning_rate}.pth"
        mclip.predict(model_path, test_loader)
        # for batch in test_loader:
        #     image = batch['image']
        #     # label = batch['label']
        #     print(image)
        #     break
            # print(image)
            # print(label)        





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adversarial Attack on Hateful Memes')

    parser.add_argument('--dataset', default='bhm', choices=['bhm', 'mute', 'mimosa', 'abuse'])
    parser.add_argument('--method', default='dora', choices=['dora', 'maf','mclip'])
    parser.add_argument('--max_len', type=int, default = 50,
                           help='the maximum text length')
    parser.add_argument('--heads', type=int, default = 4,
                        help='number of heads - default 4')                       
    parser.add_argument('--epochs', type=int, default = 1,
                         help='Number of Epochs - default 1')
    parser.add_argument('--learning_rate', type=float, default = 2e-5,
                        help='Learning rate - default 2e-5')
    # parser.add_argument('--batch_size',dest="batch", type=int, default = 4,
    #                     help='Batch Size - default 4')   
    # parser.add_argument('--model', dest='model_path', type=str, default = 'Saved_Models',
    #                     help='the directory of the saved model folder')
    
                     

    
    args = parser.parse_args()
    main(args)
