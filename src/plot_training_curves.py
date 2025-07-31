import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_curves(history_path="../models/training_history.pkl"):
    """Load training history and plot accuracy/loss curves"""
    
    if not os.path.exists(history_path):
        print(f"❌ Training history file not found at {history_path}")
        print("Please run train_with_history.py first to generate training curves.")
        return
    
    # Load training history
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    
    # Create figure with better styling
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy plot
    epochs = range(1, len(history['accuracy']) + 1)
    ax1.plot(epochs, history['accuracy'], 'b-', linewidth=2.5, label='Training Accuracy', marker='o', markersize=4)
    ax1.plot(epochs, history['val_accuracy'], 'r-', linewidth=2.5, label='Validation Accuracy', marker='s', markersize=4)
    ax1.set_title('Model Accuracy During Training', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Accuracy', fontsize=14)
    ax1.legend(fontsize=12, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # Add accuracy values as text
    final_train_acc = history['accuracy'][-1]
    final_val_acc = history['val_accuracy'][-1]
    ax1.text(0.02, 0.98, f'Final Training Acc: {final_train_acc:.3f}', 
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax1.text(0.02, 0.92, f'Final Val Acc: {final_val_acc:.3f}', 
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Loss plot
    ax2.plot(epochs, history['loss'], 'b-', linewidth=2.5, label='Training Loss', marker='o', markersize=4)
    ax2.plot(epochs, history['val_loss'], 'r-', linewidth=2.5, label='Validation Loss', marker='s', markersize=4)
    ax2.set_title('Model Loss During Training', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Loss', fontsize=14)
    ax2.legend(fontsize=12, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # Add loss values as text
    final_train_loss = history['loss'][-1]
    final_val_loss = history['val_loss'][-1]
    ax2.text(0.02, 0.98, f'Final Training Loss: {final_train_loss:.3f}', 
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax2.text(0.02, 0.92, f'Final Val Loss: {final_val_loss:.3f}', 
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    
    # Save high-quality plot
    save_path = "../models/training_curves_presentation.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Training curves saved to: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("📊 TRAINING CURVES SUMMARY")
    print("="*60)
    print(f"Total Epochs: {len(history['accuracy'])}")
    print(f"Best Training Accuracy: {max(history['accuracy']):.4f} ({max(history['accuracy'])*100:.2f}%)")
    print(f"Best Validation Accuracy: {max(history['val_accuracy']):.4f} ({max(history['val_accuracy'])*100:.2f}%)")
    print(f"Final Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    print(f"Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    
    # Find best epoch
    best_epoch = np.argmax(history['val_accuracy']) + 1
    print(f"🏆 Best validation accuracy achieved at epoch: {best_epoch}")
    print("="*60)

if __name__ == "__main__":
    plot_training_curves() 