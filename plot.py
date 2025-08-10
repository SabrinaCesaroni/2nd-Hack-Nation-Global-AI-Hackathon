import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def plot_colored_line_by_class(logits):
    """
    Plot a line of uncertainty/confidence across residues,
    coloring each segment by predicted class.

    Args:
        logits: Tensor of shape (seq_len, num_classes) for one sequence
    """
    probs = F.softmax(logits, dim=-1)
    max_probs, predicted_classes = torch.max(probs, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)

    seq_len = logits.size(0)
    x = np.arange(seq_len)

    y = max_probs.numpy()


    # Prepare line segments for coloring
    points = np.array([x, y]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Map classes to colors
    class_colors = {0: 'red', 1: 'green', 2: 'blue'}
    colors = [class_colors[int(c)] for c in predicted_classes[:-1]]  # one color per segment

    lc = LineCollection(segments, colors=colors, linewidth=2)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.add_collection(lc)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min() - 0.1*(y.max()-y.min()), y.max() + 0.1*(y.max()-y.min()))

    ax.set_xlabel("Residue Position")
    ax.set_ylabel("Entropy (Uncertainty)")
    ax.set_title("Residue-level Uncertainty Colored by Predicted Class")

    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=3, label='Helix'),
        Line2D([0], [0], color='green', lw=3, label='Strand'),
        Line2D([0], [0], color='blue', lw=3, label='Coil'),
    ]
    ax.legend(handles=legend_elements, title="Predicted Class")

    plt.show()

# Example usage:
seq_len = 50
num_classes = 3
dummy_logits = torch.randn(seq_len, num_classes)
plot_colored_line_by_class(dummy_logits)
