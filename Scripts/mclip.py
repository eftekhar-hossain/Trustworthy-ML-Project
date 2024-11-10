# imports
import torch
import numpy as np
import os
import pandas as pd
import clip
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from multilingual_clip import pt_multilingual_clip
from transformers import AutoModel, AutoTokenizer, AdamW
import sys
import argparse


# Set seed.
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

curr_dir = os.getcwd()
root_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))
if not os.path.exists(os.path.join(root_dir,'Saved_Models')):
              os.makedirs(os.path.join(root_dir,'Saved_Models'))
model_dir = os.path.join(root_dir,'Saved_Models')  


clip_imodel, preprocess = clip.load("ViT-B/32", device=device)
tokenizer = AutoTokenizer.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-L-14')
clip_text = pt_multilingual_clip.MultilingualCLIP.from_pretrained('M-CLIP/XLM-Roberta-Large-Vit-L-14')  
      

    # # for batch in train_loader:
    # #     image = batch['image']
    # #     text = batch['text']
    # #     label = batch['label']

    # #     print(image.shape)
    # #     #print(text.shape)
    # #     print(label.shape)

    # #     break

    # Freeze the parameters of the CLIP model
for param in clip_imodel.parameters():
    param.requires_grad = False  


class CLIPClassifier(nn.Module):
    def __init__(self, device='cpu') -> None:
        super(CLIPClassifier, self).__init__()
        self.device = device
        
        self.clip_image= clip_imodel # Changed JIT to True for just inference
        # output of clip is 512

        
        # cat image and text for 1024
        self.fc = nn.Sequential(
            nn.Linear(1280, 128),  # Input size includes visual features, embeddings
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),  # Updated for Binary classification
        )
        
    def forward(self, image, text):
        image_features = self.clip_image.encode_image(image).float()
        #print(image_features.shape)
        text_features = text
        #print(text_features.shape)
        features = torch.cat((image_features, text_features), dim=1)
        #print(features.shape)

        x = self.fc(features)
        # print(x)

        return x
    
model = CLIPClassifier(device=device)
model  = model.to(device)    


    # for batch in train_loader:
    #     image = batch['image'].to(device)
    #     text = batch['text']
    #     text_embed = clip_text.forward(text, tokenizer).to(device)
    #     # label = batch['label'].to(device)
    #     with torch.no_grad():

    #         features = model(image,text_embed)


    # Define a function to calculate accuracy
def calculate_accuracy(predictions, targets):
    # For multi-class classification, you can use torch.argmax to get the predicted class
    predictions = torch.argmax(predictions, dim=1)
    correct = (predictions == targets).float()
    accuracy = correct.sum() / len(correct)
    return accuracy



def fit(dataset_name, train_loader, val_loader, epochs, lr_rate):

    # Create an instance of the model
    learning_rate = lr_rate
    num_epochs = epochs
    momentum = 0.9

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # AdamW gives nan value
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate, momentum=momentum)

    print(f"Start Training CLIP on {dataset_name.upper()}")
    print("--------------------------------")
    print("--------------------------------")
    print("Epochs:",epochs)
    print("Learning Rate:",lr_rate )
    # Training loop
    best_val_accuracy = 0.0
    patience = 5
    early_stopping_counter = 0

    for epoch in range(num_epochs):

        model.train()
        total_loss = 0
        total_accuracy = 0

        # Wrap the train_loader with tqdm for the progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as t:
            for batch in t:
                images = batch['image'].to(device)
                #print(images.shape)
                texts= batch['text']
                text_embed = clip_text.forward(texts, tokenizer).to(device)
                # print(text_embed.shape)
                labels = batch['label'].to(device)
                # print("Actual Labels: ",labels)

                optimizer.zero_grad()
                outputs = model(images, text_embed)
                # print("Model Output:",outputs)
                loss = criterion(outputs, labels)
                # print("Loss: ",loss)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_accuracy += calculate_accuracy(outputs, labels).item()

                # Update the tqdm progress bar
                t.set_postfix(loss=total_loss / (t.n + 1), acc=total_accuracy / (t.n + 1))

        # Calculate training accuracy and loss
        avg_train_loss = total_loss / len(train_loader)
        avg_train_accuracy = total_accuracy / len(train_loader)

        # Validation loop
        model.eval()
        val_labels = []
        val_preds = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation", unit="batch"):
                images = batch['image'].to(device)
                #print(images.shape)
                texts= batch['text']
                text_embed = clip_text.forward(texts,tokenizer).to(device)
                # print(text_embed.shape)
                labels = batch['label'].to(device)

                outputs = model(images, text_embed)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds)

        # Calculate validation accuracy and loss
        val_accuracy = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy * 100:.2f}%, Val Acc: {val_accuracy * 100:.2f}%")

            # Early stopping logic
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            early_stopping_counter = 0  # Reset the counter if validation improves

            torch.save(model.state_dict(), os.path.join(model_dir, f'clip_{dataset_name}_{lr_rate}.pth'))
            print("Model Saved.")
        else:
            early_stopping_counter += 1  # Increment the counter if validation does not improve
            print(f"No improvement in validation accuracy. Early stopping counter: {early_stopping_counter}/{patience}")
    
        # Stop training if early stopping criteria is met
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Best Validation Accuracy: {best_val_accuracy * 100:.2f}%")
    print("--------------------------------")
    print("Training is Done.")
    
def predict(model_path,test_loader):

    # Load the saved model
    model = CLIPClassifier(device=device)
    model  = model.to(device)    
    print("Model is Loading..")
    model.load_state_dict(torch.load(os.path.join(model_dir, model_path)))
    model.eval()
    print("Loaded.")

    test_labels = []
    test_preds = []

    print("--------------------------------")
    print("Start Evaluating..")
    # testing
    with torch.no_grad(), tqdm(test_loader, desc="Testing", unit="batch") as t:
        for batch in t:
            images = batch['image'].to(device)
            texts= batch['text']
            text_embed = clip_text.forward(texts,tokenizer).to(device)
            labels = batch['label'].to(device)

            outputs = model(images, text_embed)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(preds)

    print('Evaluation Done.')
    print("--------------------------------")
    print(classification_report(test_labels,test_preds,digits=3))
