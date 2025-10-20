"""
Calculate chance level for COCO multi-label classification task.
"""

import torch
import numpy as np
from dataset_handler import COCODatasetHandler

# Load validation dataset
COCO_ROOT = r"C:\Users\julia\Documents\KEMAI Datasets\CoCo\2017"

coco_handler_val = COCODatasetHandler(COCO_ROOT, "val2017")
coco = coco_handler_val.coco

# Get all image IDs (limit to 200 like in training)
img_ids = coco.getImgIds()[:200]
cat_ids = coco.getCatIds()
cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}

# Collect all labels
all_labels = []
for img_id in img_ids:
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    label = torch.zeros(len(cat_ids), dtype=torch.float32)
    for ann in anns:
        cat_id = ann['category_id']
        if cat_id in cat_id_to_idx:
            label[cat_id_to_idx[cat_id]] = 1.0

    all_labels.append(label)

all_labels = torch.stack(all_labels, dim=0)  # Shape: (200, 80)

print("="*80)
print("CHANCE LEVEL ANALYSIS FOR COCO MULTI-LABEL CLASSIFICATION")
print("="*80)

# Calculate label statistics
n_samples = all_labels.shape[0]
n_classes = all_labels.shape[1]
positive_per_class = all_labels.sum(dim=0)
positive_rate_per_class = positive_per_class / n_samples

print(f"\n1. Dataset Statistics:")
print(f"   Number of samples: {n_samples}")
print(f"   Number of classes: {n_classes}")
print(f"   Average labels per image: {all_labels.sum(dim=1).mean():.2f}")
print(f"   Min labels per image: {all_labels.sum(dim=1).min():.0f}")
print(f"   Max labels per image: {all_labels.sum(dim=1).max():.0f}")

# Calculate overall positive rate (class imbalance)
overall_positive_rate = all_labels.sum() / (n_samples * n_classes)
print(f"\n2. Label Distribution:")
print(f"   Overall positive rate: {overall_positive_rate:.4f} ({overall_positive_rate*100:.2f}%)")
print(f"   Overall negative rate: {1-overall_positive_rate:.4f} ({(1-overall_positive_rate)*100:.2f}%)")

# Hamming Accuracy Chance Level
# If we always predict the majority class (negative) for all labels
hamming_chance_always_negative = 1 - overall_positive_rate
print(f"\n3. Hamming Accuracy Chance Levels:")
print(f"   Always predict NEGATIVE (baseline): {hamming_chance_always_negative:.4f} ({hamming_chance_always_negative*100:.2f}%)")
print(f"   Always predict POSITIVE (worst): {overall_positive_rate:.4f} ({overall_positive_rate*100:.2f}%)")

# Random guess chance level
random_hamming_acc = 0.5  # 50% if we flip a coin for each label
print(f"   Random guess (50/50): {random_hamming_acc:.4f} ({random_hamming_acc*100:.2f}%)")

# Exact Match Chance Level
# Probability of guessing all 80 labels correctly by random
exact_match_random = (0.5) ** n_classes
print(f"\n4. Exact Match Chance Levels:")
print(f"   Random guess: {exact_match_random:.2e} (~0%)")
print(f"   Always predict all negative: {((all_labels.sum(dim=1) == 0).float().mean()):.4f}")

# Model Performance (from notebook)
print(f"\n5. Model Performance (from training):")
print(f"   Hamming Accuracy: ~96.66%")
print(f"   Exact Match: ~0.50-1.50%")

# Comparison
print(f"\n6. Performance vs Chance:")
print(f"   Hamming Accuracy:")
print(f"      Model: 96.66%")
print(f"      Chance (always negative): {hamming_chance_always_negative*100:.2f}%")
print(f"      Improvement: {(0.9666 - hamming_chance_always_negative)*100:.2f} percentage points")
print(f"      Relative improvement: {((0.9666 / hamming_chance_always_negative) - 1)*100:.2f}%")
print(f"\n   Exact Match:")
print(f"      Model: ~1.00%")
print(f"      Chance (random): ~0%")
print(f"      Much better than random, but still very low")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
print("""
The high Hamming Accuracy (~96.66%) is misleading because:
1. COCO has severe class imbalance - most labels are NEGATIVE
2. The "always negative" baseline achieves ~{:.2f}% Hamming Accuracy
3. The model only improves by ~{:.2f} percentage points over this trivial baseline
4. This suggests the model is mostly predicting "negative" for most classes

The Exact Match Accuracy (~1%) is very low, meaning:
1. The model rarely predicts ALL 80 labels correctly for an image
2. With ~{:.1f} labels per image on average, this is challenging
3. Better metrics would be: precision, recall, F1 per class, or mAP

Recommendation: Focus on per-class metrics and consider:
- Precision/Recall/F1 for each class
- Mean Average Precision (mAP)
- Top-k accuracy
- Class-wise analysis to see which classes are learned
""".format(hamming_chance_always_negative*100,
           (0.9666 - hamming_chance_always_negative)*100,
           all_labels.sum(dim=1).mean()))
print("="*80)
