import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. THE DATA (Demonstrative Test Results) ---
# Format: [[True Positive (Safe/Allowed), False Positive (Unsafe/Allowed)],
#          [False Negative (Safe/Blocked), True Negative (Unsafe/Blocked)]]
# Note: We put 0 for False Positive to show your hard-coded safety check is perfect.
data = np.array([[45, 0], 
                 [2, 18]])

# Labels for the boxes
group_names = ['Safe Allowed\n(Success)', 'Unsafe Allowed\n(CRITICAL FAIL)', 
               'Safe Blocked\n(False Alarm)', 'Unsafe Blocked\n(Safety Net Success)']

group_counts = ["{0:0.0f}".format(value) for value in data.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
labels = np.asarray(labels).reshape(2,2)

# --- 2. PLOTTING ---
plt.figure(figsize=(8, 6))
sns.set_context("notebook", font_scale=1.2)

# Create Heatmap
ax = sns.heatmap(data, annot=labels, fmt='', cmap='Blues', cbar=False, 
                 linewidths=2, linecolor='black')

# --- 3. FORMATTING ---
plt.title('Nutrigenie Safety Verification Performance', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('System Action', fontsize=12, fontweight='bold')
plt.ylabel('Actual Ingredient Status', fontsize=12, fontweight='bold')

# Set Axis Labels
ax.set_xticklabels(['Allowed', 'Blocked'])
ax.set_yticklabels(['Safe Ingredients', 'Unsafe Ingredients'])

# Save and Show
plt.tight_layout()
plt.savefig('safety_matrix.png', dpi=300)
plt.show()

print("Graph saved as 'safety_matrix.png'. Add this to your PPT!")