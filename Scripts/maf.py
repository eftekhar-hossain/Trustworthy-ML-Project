import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torch.optim as optim
from torch.optim import lr_scheduler
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from madgrad import MADGRAD
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report
import clip
import os
import numpy as np

# Set seed.
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
############################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

curr_dir = os.getcwd()
root_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))
if not os.path.exists(os.path.join(root_dir,'Saved_Models')):
        os.makedirs(os.path.join(root_dir,'Saved_Models'))
model_dir = os.path.join(root_dir,'Saved_Models')  


# Define the MultiheadAttention class
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, query, key, value, mask=None):
        output, _ = self.attention(query, key, value, attn_mask=mask)
        return output


clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model = clip_model.visual

# Convert model's weights to single-precision
clip_model = clip_model.float()

# Create an instance of the CLIP model
clip_model = clip_model.to(device)

# Freeze the parameters of the CLIP model
for param in clip_model.parameters():
    param.requires_grad = False   



# Define the model in PyTorch
class MAF(nn.Module):
    def __init__(self, clip_model, num_classes, num_heads):
        super(MAF, self).__init__()

        # Visual feature extractor (CLIP)
        self.clip = clip_model # Load the CLIP model
        self.visual_linear = nn.Linear(512, 768)

        # Textual feature extractor (BERT)
        self.bert = AutoModel.from_pretrained("sagorsarker/bangla-bert-base")

        # Multihead attention
        self.attention = MultiheadAttention(d_model=768, nhead=num_heads)

        # Fully connected layers (updated for binary classification)
        self.fc = nn.Sequential(
            nn.Linear(768+768+768, 128),  # Input size includes visual features, BERT embeddings, and attention output
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes), 
            nn.Sigmoid()  # Updated for binary classification
        )

    def forward(self, image_input, input_ids, attention_mask):

        # Extract visual features using CLIP
        image_features = self.clip(image_input)
        image_features = self.visual_linear(image_features)
        image_features = image_features.unsqueeze(1)
        # Apply average pooling to reduce the sequence length to 50
        image_features = F.adaptive_avg_pool1d(image_features.permute(0, 2, 1), 70).permute(0, 2, 1)
        # print(image_features.shape)


        # Extract BERT embeddings
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_output = bert_outputs.last_hidden_state
        # print(bert_output.shape)

        # # Apply multihead attention between visual_features and BERT embeddings
        # # Assuming that visual_features and bert_output have shape (seq_length, batch_size, feature_size)
        attention_output = self.attention(
            query=image_features.permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
            key=bert_output.permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
            value=image_features.permute(1, 0, 2),  # Swap batch_size and seq_length dimensions
            mask=None  # You can add a mask if needed
        )

        # Swap back the dimensions to (batch_size, seq_length, feature_size)
        attention_output = attention_output.permute(1, 0, 2)
        # print(attention_output.shape)

        # Concatenate the context vector, visual features, BERT embeddings, and attention output
        fusion_input = torch.cat([attention_output, image_features, bert_output], dim=2)
        # print(fusion_input.shape)

        output = self.fc(fusion_input.mean(1))  # Pool over the sequence dimension
        return output



# Define a function to calculate accuracy 
def calculate_accuracy(predictions, targets):
    predictions = (predictions > 0.5).float()
    correct = (predictions == targets).float()
    accuracy = correct.sum() / len(correct)
    return accuracy


def fit(dataset_name, train_loader, val_loader,  heads, epochs, lr_rate):

  # Create an instance of the model
  num_classes = 1  # Number of output classes
  num_heads = heads  # Number of attention heads for multihead attention
  model = MAF(clip_model, num_classes, num_heads)
  model = model.to(device)  

  # Define loss and optimizer
  criterion = nn.BCELoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate,  weight_decay = 0.01)
#   optimizer = MADGRAD(model.parameters(), lr=lr_rate)

  # Define learning rate scheduler
  num_epochs = epochs
  num_training_steps = num_epochs * len(train_loader)
  lr_scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps)


  # Training loop
  best_val_accuracy = 0.0
  patience = 5
  early_stopping_counter = 0

  print(f"Start Training MAF on {dataset_name.upper()}")
  print("--------------------------------")
  print("Attention Heads#:",heads)
  print("Epochs#:",epochs)
  print("Learning Rate:",lr_rate )
  print("--------------------------------")
  for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_accuracy = 0

    # Wrap the train_loader with tqdm for the progress bar
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as t:
        for batch in t:
            images = batch['image'].to(device)
            # print(images)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].float().to(device)

            optimizer.zero_grad()
            outputs = model(images, input_ids, attention_mask)
            outputs = outputs.squeeze(dim=1)
            loss = criterion(outputs, labels)
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
    total_val_loss = 0  # Variable to accumulate validation loss
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation", unit="batch"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].float().to(device)

            outputs = model(images, input_ids, attention_mask)
            outputs = outputs.squeeze(dim=1)

            val_loss = criterion(outputs, labels)  # Calculate validation loss
            total_val_loss += val_loss.item()
            preds = (outputs > 0.5).float()  # binary

            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())  

    # Calculate validation accuracy and loss
    val_accuracy = accuracy_score(val_labels, val_preds)
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy * 100:.2f}%,  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy * 100:.2f}%")

    # Early stopping logic
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        early_stopping_counter = 0  # Reset the counter if validation improves
        torch.save(model.state_dict(), os.path.join(model_dir, f'maf_{dataset_name}_{lr_rate}.pth'))
        print("Model Saved.")
    else:
        early_stopping_counter += 1  # Increment the counter if validation does not improve
        print(f"No improvement in validation accuracy. Early stopping counter: {early_stopping_counter}/{patience}")

    # Stop training if early stopping criteria is met
    if early_stopping_counter >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break
  

    lr_scheduler.step()  # Update learning rate

  print(f"Best Validation Accuracy: {best_val_accuracy * 100:.2f}%")
  print("--------------------------------")
  print("Traininig is Done.")

def predict(model_path,test_loader, heads):
  # Create an instance of the model
  num_classes = 1  # Number of output classes
  num_heads = heads  # Number of attention heads for multihead attention
  model = MAF(clip_model, num_classes, num_heads)
  model = model.to(device)  

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
          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          labels = batch['label'].float().to(device)
          outputs = model(images, input_ids, attention_mask).squeeze().cpu().numpy()
          preds = (outputs > 0.5).astype(int)
          test_labels.extend(labels.cpu().numpy())
          test_preds.extend(preds)

  print('Evaluation Done.')
  print("--------------------------------")
  print(classification_report(test_labels, test_preds, digits = 3))


def load_model(model_path, heads):
  # Create an instance of the model
  num_classes = 1  # Number of output classes
  num_heads = heads  # Number of attention heads for multihead attention
  model = MAF(clip_model, num_classes, num_heads)
  model = model.to(device)  
  model.load_state_dict(torch.load(os.path.join(model_dir, model_path)))
  model.eval()

  return model
 