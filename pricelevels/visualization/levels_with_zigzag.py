import matplotlib.pyplot as plt
import numpy as np
from ..zigzag_utils import peak_valley_pivots

from ._helpers import _plot_levels


def plot_with_pivots(X, levels, zigzag_percent=1, only_good=False, path=None):
    # Convert to numpy array for consistent operations
    X_np = np.asarray(X)
    
    # Calculate pivots once
    pivots = peak_valley_pivots(X_np, zigzag_percent / 100, -zigzag_percent / 100)
    
    # Pre-calculate indices and masks for better performance
    indices = np.arange(len(X_np))
    non_zero_mask = pivots != 0
    peak_mask = pivots == 1
    valley_mask = pivots == -1
    
    # Calculate plot limits once
    x_min, x_max = 0, len(X_np)
    y_min, y_max = X_np.min() * 0.995, X_np.max() * 1.005
    
    # Set up the figure once
    fig = plt.figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    # Set limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Plot main price line
    ax.plot(indices, X_np, 'k-', alpha=0.9, linewidth=1.0)
    
    # Plot zigzag lines more efficiently
    if np.any(non_zero_mask):
        ax.plot(indices[non_zero_mask], X_np[non_zero_mask], 'k:', alpha=0.5, linewidth=0.8)
    
    # Plot peaks and valleys more efficiently
    if np.any(peak_mask):
        ax.scatter(indices[peak_mask], X_np[peak_mask], color='g', s=30, zorder=5)
    
    if np.any(valley_mask):
        ax.scatter(indices[valley_mask], X_np[valley_mask], color='r', s=30, zorder=5)
    
    # Plot levels
    _plot_levels(ax, levels, only_good)
    
    # Tight layout for better appearance
    plt.tight_layout()
    
    # Save or show
    if path:
        plt.savefig(path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    # Close to free memory
    plt.close(fig)
