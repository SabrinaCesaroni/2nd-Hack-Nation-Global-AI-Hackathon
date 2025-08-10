import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def plot_colored_line_by_class(logits, sequence_str=None):
    """
    Plot a line of uncertainty/confidence across residues,
    coloring each segment by predicted class.
    Optionally trim trailing 'X' padding from the plot.
    """
    probs = F.softmax(logits, dim=-1)
    max_probs, predicted_classes = torch.max(probs, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)

    seq_len = logits.size(0)

    # If sequence_str is provided, trim to before first 'X'
    if sequence_str is not None:
        first_x_idx = sequence_str.find("X")
        if first_x_idx != -1:  # found an X
            seq_len = first_x_idx
            logits = logits[:seq_len]
            entropy = entropy[:seq_len]
            predicted_classes = predicted_classes[:seq_len]

    x = np.arange(seq_len)
    y = entropy.cpu().numpy()

    # Prepare line segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    class_colors = {0: 'red', 1: 'green', 2: 'blue'}
    colors = [class_colors[int(c)] for c in predicted_classes[:-1]]

    lc = LineCollection(segments, colors=colors, linewidth=2)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.add_collection(lc)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min() - 0.1 * (y.max() - y.min()),
                y.max() + 0.1 * (y.max() - y.min()))

    ax.set_xlabel("Residue Position")
    ax.set_ylabel("Entropy (Uncertainty)")
    ax.set_title("Residue-level Uncertainty Colored by Predicted Class")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=3, label='Helix'),
        Line2D([0], [0], color='green', lw=3, label='Strand'),
        Line2D([0], [0], color='blue', lw=3, label='Coil'),
    ]
    ax.legend(handles=legend_elements, title="Predicted Class")

    plt.show()
# The actual amino acid string for this sequence
sequence_str = sample_sequence[0]  # assuming list of strings

plot_colored_line_by_class(output.squeeze(0), sequence_str=sequence_str)
