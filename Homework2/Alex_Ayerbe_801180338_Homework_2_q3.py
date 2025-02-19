# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
from sklearn.metrics import classification_report, confusion_matrix
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import io
import contextlib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Load CIFAR-10 dataset to calculate mean and std
train_dataset_cifar10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
imgs10 = torch.stack([img_t for img_t, _ in train_dataset_cifar10], dim=3)
mean10 = imgs10.view(3, -1).mean(dim=1)
std10 = imgs10.view(3, -1).std(dim=1)

# Define transformation with calculated mean and std
transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean10, std10)
])

# Load CIFAR-10 dataset with normalization
train_dataset_10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar10)
train_loader_10 = DataLoader(train_dataset_10, batch_size=32, shuffle=True)

val_dataset_10  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar10)
validation_loader_10  = DataLoader(val_dataset_10, batch_size=32, shuffle=False)

classes_cifar10 = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Load CIFAR-100 training set (to compute mean and std)
train_dataset_cifar100 = datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())
imgs100 = torch.stack([img for img, _ in train_dataset_cifar100], dim=3)
mean100 = imgs100.view(3, -1).mean(dim=1)
std100 = imgs100.view(3, -1).std(dim=1)

# Define transformation for CIFAR-100 using computed mean and std
transform_cifar100 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean100, std100)
])

# CIFAR-100 Train and Validation datasets & loaders
train_dataset_100 = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_cifar100)
train_loader_100 = DataLoader(train_dataset_100, batch_size=32, shuffle=True)

val_dataset_100 = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_cifar100)
validation_loader_100 = DataLoader(val_dataset_100, batch_size=32, shuffle=False)

# CIFAR-100 class labels (list of 100 class names)
classes_cifar100 = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]




# Define ResNet-18 architecture
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = x  # Save input for the skip connection
        # Forward pass through the first convolution, batch norm, and ReLU activation
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        # Forward pass through the second convolution and batch norm
        out = self.bn2(self.conv2(out))
        # Adding the shortcut connection's output to the main path's output
        out += self.shortcut(x)
        # Final ReLU activation after adding the shortcut
        out = nn.ReLU()(out)
        return out

class BasicBlockDropout(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlockDropout, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(0.5)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        #Not used yet
        #identity = x  # Save input for the skip connection
        # Forward pass through the first convolution, batch norm, and ReLU activation
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        # Forward pass through the second convolution and batch norm
        out = self.bn2(self.conv2(out))
        # Adding the shortcut connection's output to the main path's output
        out += self.shortcut(x)
        # Final ReLU activation after adding the shortcut
        out = nn.ReLU()(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(512, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.AvgPool2d(4)(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNetDropout(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetDropout, self).__init__()
        self.in_channels = 64

        # For CIFAR, use a 3x3 initial conv (instead of 7x7) and no initial max pool.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_dropout = nn.Dropout(0.5)

        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            layers.append(nn.Dropout(0.5))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = nn.AvgPool2d(4)(out)
        out = out.view(out.size(0), -1)

        out = self.fc_dropout(out)
        out = self.linear(out)
        return out

# -------------------------------
# Model Definitions
# -------------------------------

# Convenience functions for instantiating ResNet-11 and ResNet-18.
def ResNet11(num_classes=10):
    # Here, layers = [1, 1, 1, 1] gives a shallower network (~11 layers when counting the initial conv and FC)
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)

def ResNet11Dropout(num_classes=10):
    # Here, layers = [1, 1, 1, 1] gives a shallower network (~11 layers when counting the initial conv and FC)
    return ResNetDropout(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)

def ResNet18(num_classes=10):
    # Standard ResNet-18 configuration: layers = [2, 2, 2, 2]
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def ResNet18Dropout(num_classes=10):
    # Standard ResNet-18 configuration: layers = [2, 2, 2, 2]
    return ResNetDropout(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)



def train_network(net, train_loader, validation_loader, criterion, optimizer, num_epochs, device):
    """
    Trains a network and returns a results dictionary containing training/validation losses,
    accuracies, and the final true/predicted labels.
    """
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []
    
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        # Training Loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        training_losses.append(train_loss)
        train_accuracy = 100 * correct / total
        training_accuracies.append(train_accuracy)
        
        # Evaluation Loop on Validation Data
        net.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predicted = []
        true_labels = []
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_predicted.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        val_loss = running_loss / len(validation_loader)
        validation_losses.append(val_loss)
        val_accuracy = 100 * correct / total
        validation_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs} -- Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
    
    # Store results in a dictionary (compatible for further analysis)
    results = {
        'training_losses': training_losses,
        'training_accuracies': training_accuracies,
        'validation_losses': validation_losses,
        'validation_accuracies': validation_accuracies,
        'true_labels': true_labels,           # From the final epoch evaluation
        'predicted_labels': all_predicted      # From the final epoch evaluation
    }
    return results

# -------------------------------
# Data Analysis Function
# -------------------------------

def analyze_results(results, directory, model_info=None, class_labels=None, show_figures=False, save_figures=True, dataset_name="UnknownDataset", num_epochs=None):
    """
    Given a results dictionary (from train_network), this function plots training and validation
    losses and accuracies, prints the classification report, and displays the confusion matrix.
    """

    final_train_loss = results['training_losses'][-1]
    final_val_loss = results['validation_losses'][-1]
    final_train_acc = results['training_accuracies'][-1]
    final_val_acc = results['validation_accuracies'][-1]

    print(f"Final Training Loss: {results['training_losses'][-1]}")
    print(f"Final Validation Loss: {results['validation_losses'][-1]}")
    print(f"Final Training Accuracy : {results['training_accuracies'][-1]} %")
    print(f"Final Validation Accuracy : {results['validation_accuracies'][-1]} %")

    epochs_range = range(1, len(results['training_losses']) + 1)
    
    # Generate a suffix for figure filenames if model_info is provided
    # and we want to differentiate images by model/optimizer/etc.
    if model_info is not None:
        # We can store num_epochs in model_info if we like:
        # model_info['num_epochs'] = (some integer), set in compare_models
        suffix_parts = [
            f"{dataset_name}",
            f"{model_info.get('model_name', 'model')}",
            f"epochs{num_epochs}",
            f"{model_info.get('optimizer_type', 'opt')}",
            f"lr{model_info.get('lr', 'x')}"
        ]
        # Include momentum if it's not None
        if model_info.get('momentum') is not None:
            suffix_parts.append(f"mom{model_info.get('momentum')}")
        fig_suffix = "_".join(str(s) for s in suffix_parts)
    else:
        fig_suffix = "results"

    # Plot Loss Curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, results['training_losses'], label='Training Loss')
    plt.plot(epochs_range, results['validation_losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    
    # Plot Accuracy Curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, results['training_accuracies'], label='Training Accuracy')
    plt.plot(epochs_range, results['validation_accuracies'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend()
    
    plt.tight_layout()
    
    if show_figures == True:
        plt.show()
    if save_figures == True:
        plt.savefig(os.path.join(directory, f"{fig_suffix}_loss_accuracy_.png"), dpi=200)
        plt.close()

    # Print Classification Report and Plot Confusion Matrix (if labels are provided)
    if 'true_labels' in results and 'predicted_labels' in results:
        print("Classification Report:")
        print(classification_report(results['true_labels'], results['predicted_labels']))
        
        conf_matrix = confusion_matrix(results['true_labels'], results['predicted_labels'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        if show_figures == True:
            plt.show()
        if save_figures == True:
            plt.savefig(os.path.join(directory, f"{fig_suffix}_confusion_matrix.png"), dpi=200)
            plt.close()

def check_overfitting(train_acc, val_acc, threshold=5.0):
    """
    Returns True if the difference between training accuracy and validation accuracy
    is greater than the given threshold (default 5%). 
    """
    return (train_acc - val_acc) > threshold

def check_if_more_epochs_needed(results, improvement_threshold=0.2):
    """
    A simple heuristic to see if validation accuracy is still improving
    by at least 'improvement_threshold' percentage points in the last epoch
    compared to the second-to-last epoch. If so, we suspect more epochs might help.
    """
    val_accs = results['validation_accuracies']
    # If we have fewer than 2 epochs, we can't compare
    if len(val_accs) < 2:
        return False
    # Compare the difference between the last two epochs' validation accuracy
    diff = val_accs[-1] - val_accs[-2]
    return diff >= improvement_threshold

def compare_models(model_classes, optim_configs, train_loader, validation_loader, class_labels, num_epochs=10, lr=0.01, momentum=0.9, show_figures=False, save_figures=True, dataset_name="UnknownDataset", save_txt=True):
    criterion = nn.CrossEntropyLoss()
    model_results_list = []
    header = f"model_comparison_{dataset_name}_"
    
    # # Add Model information to header
    # if model_classes:  # Check if the list is not empty
    #     header += "_" + "_".join(str(model.__name__) for model in model_classes)

    # if model_classes:
    #     header += "_" + "_".join(model.__name__ if isinstance(model, type) else model.__class__.__name__ for model in model_classes)
    model_names = []
    for mc in model_classes:
        # If it's a function or class, we can use __name__; if it's an already-instantiated object, fall back to __class__.__name__
        if hasattr(mc, "__name__"):
            model_names.append(mc.__name__)
        else:
            model_names.append(mc.__class__.__name__)
    header += "_".join(model_names)

    #Check to see if path exits
    if not os.path.exists(header):
        os.makedirs(header)

    #current_time = datetime.now().strftime("%m-%d-%Y_%H:%M")
    txt_filename = os.path.join(header, f"{header}_epochs{num_epochs}_results.txt")

    # We will open the file once, and write all results after we gather them
    f = None
    if save_txt == True:
        f = open(txt_filename, "w")

    # Helper function to write lines to file if needed
    def write_line(line=""):
        print(line)        # Always print to console
        if f is not None:  # Optionally write to file
            f.write(line + "\n")

    for model_class in model_classes:
        for opt_conf in optim_configs:
            #print(f"\nTraining {model_class.__name__}")

            # Check if model_item is a class or an instance
            if isinstance(model_class, nn.Module):
                model = model_class.to(device)
                model_name = model.__class__.__name__
            else:
                model = model_class(num_classes=len(class_labels)).to(device)
                model_name = model_class.__name__


            #Build the Model
            model = model_class(num_classes=len(class_labels)).to(device)

            # Create the optimizer according to opt_conf
            opt_type = opt_conf.get("type", "SGD")
            lr_value = opt_conf.get("lr", 0.01)
            momentum_value = opt_conf.get("momentum", 0.0)


            if opt_type == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=lr_value, momentum=momentum_value)
            elif opt_type == "Adam":
                optimizer = optim.Adam(model.parameters(), lr=lr_value)
            else:
                # fallback to SGD if unknown
                write_line("Unknown optimizer; defaulting to SGD.")
                optimizer = optim.SGD(model.parameters(), lr=lr_value, momentum=momentum_value)
            
            
            # Train the Model
            write_line(f"\nTraining {model_class.__name__} on {dataset_name} with {opt_type}, LR={lr_value}, Momentum={momentum_value}, for Epochs={num_epochs}")
            results = train_network(model, train_loader, validation_loader, 
                                        criterion, optimizer, num_epochs, device)

            # Summarize model architecture
            # (You can suppress the console prints if desired)
            summary_str = []
            model.eval()
            # Instead of printing summary to console, store it in summary_str
            with contextlib.redirect_stdout(io.StringIO()) as buff:
                summary(model, (3, 32, 32))
            summary_str = buff.getvalue()

            # Print to console & optionally file
            write_line(summary_str.strip())

            # Analyze final metrics
            final_train_loss = results['training_losses'][-1]
            final_val_loss = results['validation_losses'][-1]
            final_train_acc  = results['training_accuracies'][-1]
            final_val_acc    = results['validation_accuracies'][-1]

            # Count total trainable parameters
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Store in a structured dict for comparison
            model_info = {
                'model_name': model_class.__name__,
                'optimizer_type': opt_type,
                'lr': lr_value,
                'momentum': momentum_value if opt_type == "SGD" else None,
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss,
                'final_train_acc': final_train_acc,
                'final_val_acc': final_val_acc,
                'num_params': total_params,
                'overfitting': check_overfitting(final_train_acc, final_val_acc),
                'needs_more_epochs': check_if_more_epochs_needed(results)
            }
            model_results_list.append((model_info, results))
    
                        # Now we can optionally analyze and save figures for each model
            analyze_results(results, 
                            directory=header,
                            class_labels=class_labels, 
                            model_info=model_info, 
                            save_figures=save_figures,
                            show_figures=show_figures,
                            dataset_name=dataset_name,
                            num_epochs=num_epochs
                            )


    # -------------------------
    # Compare Models All Together
    # -------------------------
    # Identify the best (lowest) training loss / validation loss; best (highest) training/validation accuracy
    best_train_loss = min(m[0]['final_train_loss'] for m in model_results_list)
    best_val_loss   = min(m[0]['final_val_loss']   for m in model_results_list)
    best_train_acc  = max(m[0]['final_train_acc']  for m in model_results_list)
    best_val_acc    = max(m[0]['final_val_acc']    for m in model_results_list)


    if f is not None:
        write_line("\n=== MODEL COMPARISON SUMMARY ===")
        # Print table header
        write_line("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        # Print a header with separators (|) between related columns
        write_line(
            f"{'Model':<25}| "
            f"{'Opt':<7}| "
            f"{'LR':<7}| "
            f"{'Momentum':<10}| "
            f"{'#Params':<12}| "
            f"{'Tr.Loss':<10}"
            f"{'vs.Best(%)':<12}| "
            f"{'Val.Loss':<10}"
            f"{'vs.Best(%)':<12}| "
            f"{'Tr.Acc(%)':<10}"
            f"{'vs.Best(%)':<12}| "
            f"{'Val.Acc(%)':<10}"
            f"{'vs.Best(%)':<12}| "
            f"{'Overfit?':<10}| "
            f"{'More?':<10}| "
        )
        write_line("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        
        for (model_info, _) in model_results_list:
            # Calculate % difference from the best for losses/accuracies
            # For losses (smaller is better): diff = ((this - best)/best)*100
            train_loss_diff = 0.0
            if best_train_loss != 0.0:
                train_loss_diff = 100.0 * (model_info['final_train_loss'] - best_train_loss) / best_train_loss
            
            val_loss_diff = 0.0
            if best_val_loss != 0.0:
                val_loss_diff = 100.0 * (model_info['final_val_loss'] - best_val_loss) / best_val_loss
            
            # For accuracies (larger is better): diff = ((best - this)/best)*100
            train_acc_diff = 0.0
            if best_train_acc != 0.0:
                train_acc_diff = 100.0 * (best_train_acc - model_info['final_train_acc']) / best_train_acc
            
            val_acc_diff = 0.0
            if best_val_acc != 0.0:
                val_acc_diff = 100.0 * (best_val_acc - model_info['final_val_acc']) / best_val_acc
            
            overfit_str = "Yes" if model_info['overfitting'] else "No"
            more_epochs_str = "Yes" if model_info['needs_more_epochs'] else "No"

            # If momentum is None (e.g. Adam), we'll print '-'
            momentum_str = f"{model_info['momentum']:.2f}" if model_info['momentum'] is not None else "-"
            
            write_line(
                f"{model_info['model_name']:<25}| "
                f"{model_info['optimizer_type']:<7}| "
                f"{model_info['lr']:<7.4g}| "
                f"{momentum_str:<10}| "
                f"{model_info['num_params']:<12}| "
                f"{model_info['final_train_loss']:<10.4f}"
                f"{train_loss_diff:<12.2f}| "
                f"{model_info['final_val_loss']:<10.4f}"
                f"{val_loss_diff:<12.2f}| "
                f"{model_info['final_train_acc']:<10.2f}"
                f"{train_acc_diff:<12.2f}| "
                f"{model_info['final_val_acc']:<10.2f}"
                f"{val_acc_diff:<12.2f}| "
                f"{overfit_str:<10}| "
                f"{more_epochs_str:<10}| "
            )
        write_line("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        write_line("Note: For losses, 'vs.Best(%)' = how much higher the model's loss is than the best loss.")
        write_line("      For accuracies, 'vs.Best(%)' = how much lower the model's accuracy is than the best accuracy.")
        write_line("      'Overfit?' checks if Train Acc exceeds Val Acc by >5%.")
        write_line("      'More?' indicates if the model could still improve with more epochs (last Val Acc improved by >=0.2%).")

    if f is not None:
        f.close()

optim_configs = [
        #{"type": "SGD",  "lr": 0.01, "momentum": 0.9},
        #{"type": "SGD",  "lr": 0.001, "momentum": 0.9}
        {"type": "Adam", "lr": 0.001}
]


# compare_models([ResNet11, ResNet11Dropout, ResNet18, ResNet18Dropout], 
#                 optim_configs, 
#                 train_loader_10, 
#                 validation_loader_10, 
#                 classes_cifar10, 
#                 num_epochs=10, 
#                 dataset_name="cifar10")


# ---- Call the function for CIFAR-100 ----
compare_models([ResNet11, ResNet11Dropout, ResNet18, ResNet18Dropout], 
               optim_configs, 
               train_loader_100, 
               validation_loader_100, 
               classes_cifar100, 
               num_epochs=10, 
               dataset_name="cifar100")