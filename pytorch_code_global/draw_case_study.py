import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Load the counts for SR-GNN and SAGE predictions
res_sage_cnt = np.load('sage_cnt.npy', allow_pickle=True).item()
res_cnt = np.load('gt_cnt.npy', allow_pickle=True).item()

# Prepare the data for plotting
item_ids_sage = list(res_sage_cnt.keys())
item_ids_srgnn = list(res_cnt.keys())

occurrences_sage = [res_sage_cnt[item] for item in item_ids_sage]
occurrences_srgnn = [res_cnt[item] for item in item_ids_srgnn]

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(item_ids_sage, occurrences_sage, label="SAGE(SR-GNN)", color='r')
plt.plot(item_ids_srgnn, occurrences_srgnn, label="SR-GNN", color='g')
plt.xlabel('Ranking')
plt.ylabel('Occurrence')
plt.title('Predicted results of SAGE(SR-GNN) and SR-GNN on sessions with the same last-item')
plt.legend(loc='upper right')
# plt.grid(True)
plt.tight_layout()
plt.savefig("sage_vs_srgnn.png")  # Save the figure if needed
plt.show()

plt.savefig(os.path.join("figures", f"last_item_confusion_uniform_spacing.png"))