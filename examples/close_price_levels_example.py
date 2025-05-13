import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from pricelevels.close_levels import ClosePriceClusterLevels
from pricelevels.visualization._helpers import _plot_levels

# Start timing
start_time = time.time()

# Load data
df = pd.read_csv('sample.txt')

# Extract close prices
close_prices = df['Close'].values

# Parameters
merge_percent = 2.0  # Merge levels within 2% of each other
window_size = 30     # Window size for finding local extrema
percentile = 95      # Percentile threshold for significant changes

# Create the ClosePriceClusterLevels finder
cl = ClosePriceClusterLevels(
    merge_distance=None,
    merge_percent=merge_percent,
    level_selector='median',
    window_size=window_size,
    percentile_threshold=percentile
)

# Fit the model using only close prices
cl.fit(close_prices)

# Print timing information
processing_time = time.time() - start_time
print(f"Processing completed in {processing_time:.4f} seconds")
print(f"Found {len(cl.levels) if cl.levels else 0} support/resistance levels")
print(f"Window size: {window_size}, Percentile threshold: {percentile}%")

# Create a visualization of the close prices with support/resistance levels
def plot_with_close_levels(close_prices, levels, only_good=False, path=None):
    """
    Plot close prices with support/resistance levels.
    
    Parameters
    ----------
    close_prices : numpy.ndarray
        Array of close prices.
    levels : list
        List of level dictionaries.
    only_good : bool, default=False
        Whether to only show levels with positive scores.
    path : str, optional
        Path to save the figure. If None, the figure is displayed.
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot close prices
    ax.plot(close_prices, color='black', linewidth=1, alpha=0.8)
    
    # Plot support/resistance levels
    _plot_levels(ax, levels, only_good)
    
    # Add title and labels
    plt.title('Close Price Support and Resistance Levels', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    
    # Improve layout
    plt.tight_layout()
    
    # Save or show
    if path:
        plt.savefig(path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    # Close to free memory
    plt.close(fig)

# Display chart with close price levels
plot_with_close_levels(close_prices, cl.levels)

# Uncomment to save chart to image
# plot_with_close_levels(close_prices, cl.levels, path='close_price_levels.png')
