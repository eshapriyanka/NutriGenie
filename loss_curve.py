import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. SIMULATED TRAINING DATA ---
# Realistic values for GPT-2 Fine-Tuning
epochs = [1, 2, 3, 4, 5]
train_loss = [4.2, 2.8, 1.5, 0.9, 0.65]  # The model memorizing the data
val_loss =   [4.4, 3.1, 1.9, 1.2, 0.95]  # The model understanding new data (Generalization)

# --- 2. PLOT SETUP ---
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Plotting the lines
plt.plot(epochs, train_loss, marker='o', linestyle='-', linewidth=3, color='#2ecc71', label='Training Loss')
plt.plot(epochs, val_loss, marker='s', linestyle='--', linewidth=3, color='#e74c3c', label='Validation Loss')

# --- 3. ANNOTATIONS (To make you look smart) ---
plt.annotate('Model "Pre-trained" State\n(High Error)', xy=(1, 4.4), xytext=(1.5, 4.5),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=11)

plt.annotate('Convergence Reached\n(Optimal Performance)', xy=(5, 0.95), xytext=(3.5, 1.5),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=11)

# --- 4. FORMATTING ---
plt.title('GPT-2 Fine-Tuning Performance: Loss Reduction', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Epochs (Training Cycles)', fontsize=12, fontweight='bold')
plt.ylabel('Cross-Entropy Loss (Error Rate)', fontsize=12, fontweight='bold')
plt.legend(fontsize=12)
plt.xticks(epochs) # Show only integers 1, 2, 3...

# Save and Show
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=300)
plt.show()

print("Graph saved as 'loss_curve.png'")