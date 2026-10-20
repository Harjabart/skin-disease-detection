import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Confusion Matrix Data from your evaluation run
cm = np.array([
    [10,  6,  0,  11,  3, 19,  0],
    [ 0, 29,  0,  29,  6, 13,  0],
    [ 0,  0,  0,   8,  2,  7,  0],
    [ 1,  8,  0, 928, 33, 33,  3],
    [ 3,  4,  0,  70, 61, 27,  2],
    [ 3,  9,  0,  40, 17, 96,  0],
    [ 0,  1,  0,   9,  1,  1, 10]
])

classes = [
    'Actinic keratosis', 'Basal cell carcinoma', 'Dermatofibroma',
    'Melanocytic nevus', 'Melanoma', 'Seborrheic keratosis', 'Vascular lesion'
]

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix - MobileNetV2 Skin Lesion Classification', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=10, fontweight='bold')
plt.ylabel('True Label', fontsize=10, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
print("Saved 'confusion_matrix.png' successfully.")