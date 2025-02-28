# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchinfo import summary
from sklearn.metrics import classification_report, confusion_matrix
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import io
import contextlib
from sklearn.model_selection import train_test_split
import numpy as np
import json
import requests

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Sample text
# Step 1: Download the dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
text = response.text  # This is the entire text data


def text_preparation(text, max_length, hidden_size):

    chars = sorted(list(set(text)))
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}

    # Encode the text into integers
    encoded_text = [char_to_int[ch] for ch in text]

    sequences = []
    targets = []
    for i in range(0, len(encoded_text) - max_length):
        seq = encoded_text[i:i+max_length]
        target = encoded_text[i+max_length]
        sequences.append(seq)
        targets.append(target)

    sequences = torch.tensor(sequences, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    class CharDataset(Dataset):
        def __init__(self, sequences, targets):
            self.sequences = sequences
            self.targets = targets

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, index):
            return self.sequences[index], self.targets[index]

    dataset = CharDataset(sequences, targets)

    batch_size = hidden_size
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)

    return train_loader, test_loader, chars


# Defining the LSTM model
class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm  = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output[:, -1, :])
        return output, hidden

    def init_hidden(self, batch_size):
        # Initialize both hidden state and cell state with zeros
        return (torch.zeros(1, batch_size, self.hidden_size, device=device), 
                torch.zeros(1, batch_size, self.hidden_size, device=device))

# Defining the GRU model
class CharGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharGRU, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        output = self.fc(output[:, -1, :])
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


# Function to store results in a JSON file
def store_results(results, filename="results.json"):
    """ Stores the training results into a JSON file for future comparison. """

    for model_name, data in results.items():
        data['true_labels'] = data['true_labels'].tolist()  # Convert to list
        data['predicted_labels'] = data['predicted_labels'].tolist()  # Convert to list

    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results successfully stored in {filename}")

# Function to visualize training and validation loss & accuracy
def plot_results(filename="results.json"):
    """ Reads results and generates training vs validation plots. """
    if not os.path.exists(filename):
        print(f"No results file found: {filename}")
        return

    with open(filename, "r") as f:
        results = json.load(f)

    for model_name, data in results.items():
        epochs = list(range(1, len(data["training_losses"]) + 1))

        plt.figure(figsize=(12, 5))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, data["training_losses"], label="Training Loss")
        plt.plot(epochs, data["validation_losses"], label="Validation Loss", linestyle="dashed")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Loss Over Epochs - {model_name}")
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, data["training_accuracies"], label="Training Accuracy")
        plt.plot(epochs, data["validation_accuracies"], label="Validation Accuracy", linestyle="dashed")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Accuracy Over Epochs - {model_name}")
        plt.legend()

        plt.tight_layout()
        plt.show()

# Function to compare results across models
def compare_results(filename="results.json"):
    """ Loads and compares model performance across different architectures and sequence lengths. """
    if not os.path.exists(filename):
        print(f"No results file found: {filename}")
        return

    with open(filename, "r") as f:
        results = json.load(f)

    print("\nModel Performance Comparison:\n")
    print("{:<20} {:<12} {:<15} {:<15} {:<12} {:<12} {:<18} {:<15} {:<15}".format(
        "Model", "Max Length", "Train Loss", "Val Accuracy", "Exec Time", "Model Size", "Comp Complexity", "Inf Time", "Perplexity"
    ))

    for model_name, data in results.items():
        train_loss = np.mean(data["training_losses"])
        val_accuracy = np.max(data["validation_accuracies"])  # Best validation accuracy
        exec_time = data["execution_time"]
        model_size = data["model_size"]
        comp_complexity = data["computational_complexity"]
        inf_time = data["inference_time"]
        perplexity = data["perplexity"]

        max_length = model_name.split("_")[-1]  # Extract max length from model name
        print("{:<20} {:<12} {:<15.4f} {:<15.2f} {:<12.2f} {:<12d} {:<18d} {:<15.6f} {:<15.4f}".format(
            model_name, max_length, train_loss, val_accuracy, exec_time, model_size, comp_complexity, inf_time, perplexity
        ))

# Function to calculate model size (number of parameters)
def calculate_model_size(model):
    """ Computes the total number of trainable parameters in a model. """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_complexity(model_name, seq_length, hidden_size):
    if "LSTM" in model_name or "GRU" in model_name:
        # approximate: seq_length * hidden_size^2
        complexity = seq_length * (hidden_size ** 2)
    else:
        # for FC, let's approximate: seq_length * hidden_size
        complexity = seq_length * hidden_size
    return complexity

def measure_inference_time(model, seq_length, device='cpu'):
    """
    Measures how long a single forward pass takes on a random batch of size 1.
    Returns the time in seconds.
    """
    model.eval()
    with torch.no_grad():
        # Construct a random input
        x = torch.randint(0, 50, (1, seq_length), device=device)  # assume <= 50 unique tokens for test
        hidden = model.init_hidden(1) if hasattr(model, 'init_hidden') else None
        start = time.time()
        if hidden is not None:
            _ = model(x, hidden)
        else:
            _ = model(x, None)
        end = time.time()
    return end - start

def train_text_network(net, train_loader, validation_loader, hidden_size, criterion, optimizer, epochs, model_name=""):
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []

    start_time = time.time()

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # ------------------ Training Loop ------------------ #
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()   
            hidden = net.init_hidden(data.size(0))
            output, hidden = net(data, hidden)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()        

            running_loss += loss.item()

            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

            # #CODE TO BE ADJUSTED START

        # Compute training accuracy
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100.0 * correct / total 
        
        
        training_losses.append(avg_train_loss)
        training_accuracies.append(train_accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {running_loss / len(train_loader)}")

        # ------------------ Validation Loop ------------------ #
        # Validation
        net.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predicted = []
        true_labels = []

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                hidden = net.init_hidden(inputs.size(0))
                outputs, _ = net(inputs, hidden)
                loss = criterion(outputs, labels)

                running_loss += loss.item()

                _, predicted  = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += inputs.size(0)

                # Collect predictions for the final epoch
                all_predicted.extend(inputs.cpu().numpy())
                true_labels.extend(predicted.cpu().numpy())

        val_loss = running_loss / len(validation_loader)
        val_accuracy = 100.0 * correct / total

        validation_losses.append(val_loss)
        validation_accuracies.append(val_accuracy)

        if (epoch + 1) % 1 == 0:
            print(f'Epoch {epoch+1}/{epochs} -- '
                  f'Training Loss: {avg_train_loss:.4f}, '
                  f"Training Acc: {train_accuracy:.2f}%, "
                  f'Validation Loss: {val_loss:.4f}, '
                  f'Validation Accuracy: {val_accuracy:.2f}%')

    execution_time = time.time() - start_time
    print(f"Execution Time: {execution_time} seconds")

    model_size = calculate_model_size(net)
    print(f"Model size: {model_size} parameters")

    try:
        seq_length_str = model_name.split("_")[-1]
        seq_length = int(seq_length_str)
    except:
        seq_length = 20  # fallback

    comp_complexity = compute_complexity(model_name, seq_length, hidden_size)

    #===CHANGE=== 3) measure inference time
    inf_time = measure_inference_time(net, seq_length, device=device)

    # We'll also compute a naive "perplexity" from final validation loss
    # perplexity = exp(val_loss_avg). For demonstration only.
    perplexity = np.exp(val_loss)



    # Store results
    results = {
        'training_losses': training_losses,
        'training_accuracies': training_accuracies,
        'validation_losses': validation_losses,
        'validation_accuracies': validation_accuracies,
        'true_labels': np.array(true_labels),
        'predicted_labels': np.array(all_predicted),
        'execution_time': execution_time,
        'model_size': model_size,
        'computational_complexity': comp_complexity,
        'inference_time': inf_time,
        'perplexity': perplexity
    }

    return results

    #CODE TO BE ADJUSTED END

def compare_models(models):

  results_dict = {}

  for model_name, model_params in models.items():
    print(f"Training {model_params['name']} with: {model_params['hidden_size']}...")

    train_loader, test_loader, chars = text_preparation(text, 
                                                        model_params["max_length"], 
                                                        model_params["hidden_size"])

    # Hyperparameters
    input_size = len(chars)
    hidden_size = model_params["hidden_size"]
    output_size = len(chars)

    #Model Class
    model_class = model_params['model_class']
    model = model_class(input_size, hidden_size, output_size).to(device)

    criterion = model_params['criterion']()
    optimizer = model_params['optimizer'](model.parameters(), lr=model_params['learning_rate'])

    results = train_text_network(model, 
                                 train_loader, 
                                 test_loader, 
                                 hidden_size, 
                                 criterion, 
                                 optimizer, 
                                 model_params['epochs'],
                                 model_name=model_params["name"])

    results_dict[model_params['name']] = results

  store_results(results_dict)
  return results_dict

def main():
    """
    Main function with:
      - LSTM multiple layers, different hidden sizes, sequence lengths
      - GRU with lengths 20, 30, 50
    """
    expanded_models = {}

     # LSTM and GRU with multiple layers, different hidden sizes
    for model_class, model_name in [(CharLSTM, "LSTM"), (CharGRU, "GRU")]:
        for hidden_dim in [128]:
            for length in [20, 30, 50]:
                expanded_models[f"{model_name}_{length}"] = {
                "name": f"{model_name}_{length}",
                "model_class": model_class,
                "max_length": length,
                "hidden_size": hidden_dim,
                "criterion": nn.CrossEntropyLoss,
                "optimizer": optim.Adam,
                "learning_rate": 0.001,
                "epochs": 20
            }

    # Train
    results = compare_models(expanded_models)
    # Compare
    compare_results("results.json")
    plot_results("results.json")


if __name__ == '__main__':
    main()