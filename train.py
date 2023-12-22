from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from contract_cleaner_faster2 import SourceCodeCleanerAndFormatter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import torch
from torch import nn
import numpy as np
import re
import pandas as pd

cleaner = SourceCodeCleanerAndFormatter("SolidityLexer.g4")
cleaner.read_input_file()
cleaner.remove_comments()
file_content = cleaner.source_code
# Use regular expression to extract values in quotes
quoted_values = re.findall(r"'([^']*)'", file_content)
filtered_list = [element for element in quoted_values if '\n' not in element]
unique_list = list(set(filtered_list))
df = pd.read_csv("output2.csv")
df = df.dropna()
texts = df["Text"].tolist()
labels = df["label"].astype(int).tolist()
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label)
        }

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes, new_vocab):
        super(BERTClassifier, self).__init__()

        # Load the BERT model and tokenizer
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        # Extend the vocabulary of the tokenizer with new_vocab
        self.tokenizer.add_tokens(new_vocab)

        # Resize the token embeddings matrix of the model
        self.bert.resize_token_embeddings(len(self.tokenizer))

        # Rest of the model setup
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits
def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    for batch in tqdm(data_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
def predict_sentiment(text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      _, preds = torch.max(outputs, dim=1)
    return "positive" if preds.item() == 1 else "negative"
def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
# Set up parameters
bert_model_name = 'bert-base-uncased'
num_classes = 2
max_length = 128
batch_size = 32
num_epochs = 100
learning_rate = 2e-5

tokenizer = BertTokenizer.from_pretrained(bert_model_name)
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier(bert_model_name, num_classes, unique_list).to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
import time
all_acc = []

for epoch in range(num_epochs):
    start = time.time()
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Train the model on the training dataset
    train(model, train_dataloader, optimizer, scheduler, device)

    # Evaluate the model on the validation dataset
    accuracy, report = evaluate(model, val_dataloader, device)

    # Print validation accuracy and evaluation report
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(report)
    all_acc.append((report, accuracy, time.time() - start))

model_save_path = f"model_epoch.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved at: {model_save_path}")
max_accuracy_element = max(all_acc, key=lambda x: x[1])

print("Maximum accuracy element:")
[print(x) for x in max_accuracy_element]

