import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import os
from PIL import Image
import json
from datetime import datetime

def evaluate_model(model, dataloader, device, classes=["NORMAL", "PNEUMONIA"]):
    """
    Comprehensive model evaluation with metrics and visualizations
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader for evaluation
        device: torch.device
        classes: List of class names
    
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = np.mean(y_true == y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    
    # ROC curve for binary classification
    if len(classes) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
        avg_precision = average_precision_score(y_true, y_prob[:, 1])
    else:
        fpr, tpr, roc_auc = None, None, None
        precision, recall, avg_precision = None, None, None
    
    results = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred,
        'true_labels': y_true,
        'probabilities': y_prob,
        'roc_data': {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc},
        'pr_data': {'precision': precision, 'recall': recall, 'avg_precision': avg_precision}
    }
    
    return results

def plot_confusion_matrix(cm, classes, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curve(fpr, tpr, roc_auc, save_path=None):
    """Plot ROC curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """Plot training history"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'bo-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'bo-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'ro-', label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_evaluation_report(results, model_info, save_dir="reports"):
    """Save comprehensive evaluation report"""
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(save_dir, f"evaluation_report_{timestamp}.json")
    
    # Prepare serializable results
    serializable_results = {
        'timestamp': timestamp,
        'model_info': model_info,
        'accuracy': float(results['accuracy']),
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'classification_report': results['classification_report'],
        'roc_auc': float(results['roc_data']['auc']) if results['roc_data']['auc'] else None,
        'average_precision': float(results['pr_data']['avg_precision']) if results['pr_data']['avg_precision'] else None
    }
    
    with open(report_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"📄 Evaluation report saved to: {report_path}")
    return report_path

def visualize_sample_predictions(model, dataloader, device, classes, num_samples=8):
    """Visualize sample predictions"""
    model.eval()
    
    # Get a batch of images
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)
    
    # Plot samples
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        # Denormalize image for display
        img = images[i].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        # Convert to numpy and transpose
        img_np = img.permute(1, 2, 0).numpy()
        
        # Plot
        axes[i].imshow(img_np)
        axes[i].axis('off')
        
        true_label = classes[labels[i].item()]
        pred_label = classes[predictions[i].item()]
        confidence = probabilities[i].max().item()
        
        color = 'green' if true_label == pred_label else 'red'
        title = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}"
        axes[i].set_title(title, color=color, fontsize=10)
    
    plt.tight_layout()
    plt.suptitle('Sample Predictions', fontsize=16, y=1.02)
    plt.show()

def check_dataset_balance(data_dir):
    """Check and report dataset class balance"""
    results = {}
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(data_dir, split)
        if os.path.exists(split_path):
            split_counts = {}
            total = 0
            
            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)
                if os.path.isdir(class_path):
                    count = len([f for f in os.listdir(class_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                    split_counts[class_name] = count
                    total += count
            
            # Calculate percentages
            split_percentages = {k: (v/total)*100 for k, v in split_counts.items()}
            
            results[split] = {
                'counts': split_counts,
                'percentages': split_percentages,
                'total': total
            }
    
    return results

def print_dataset_summary(data_dir="data"):
    """Print a comprehensive dataset summary"""
    balance = check_dataset_balance(data_dir)
    
    print("📊 Dataset Summary")
    print("=" * 50)
    
    for split, data in balance.items():
        print(f"\n{split.upper()} SET:")
        for class_name, count in data['counts'].items():
            percentage = data['percentages'][class_name]
            print(f"  {class_name}: {count} images ({percentage:.1f}%)")
        print(f"  Total: {data['total']} images")
    
    # Check for imbalance
    for split, data in balance.items():
        if len(data['counts']) == 2:
            percentages = list(data['percentages'].values())
            imbalance_ratio = max(percentages) / min(percentages)
            if imbalance_ratio > 2:
                print(f"\n⚠️  {split.upper()} set is imbalanced (ratio: {imbalance_ratio:.1f}:1)")

def create_directories():
    """Create necessary directories for the project"""
    dirs = ["models", "reports", "logs", "exports"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"📁 Created directory: {dir_name}")

def log_experiment(config, results, log_dir="logs"):
    """Log experiment configuration and results"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"experiment_{timestamp}.json")
    
    log_data = {
        'timestamp': timestamp,
        'config': config,
        'results': {
            'accuracy': float(results['accuracy']),
            'roc_auc': float(results['roc_data']['auc']) if results['roc_data']['auc'] else None,
            'avg_precision': float(results['pr_data']['avg_precision']) if results['pr_data']['avg_precision'] else None
        }
    }
    
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"📝 Experiment logged to: {log_path}")
    return log_path