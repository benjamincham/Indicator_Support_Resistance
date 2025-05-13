import pandas as pd
import numpy as np
import time

from pricelevels.cluster import ZigZagClusterLevels
from pricelevels.visualization.levels_with_zigzag import plot_with_pivots

# Start timing
start_time = time.time()

# Load data
df = pd.read_csv('sample.txt')
close_values = df['Close'].values

# EXTREMELY restrictive parameters to get minimal pivot points
zig_zag_percent = 8.0      # Very high percentage change required (8%)
merge_percent = 2.0        # Merge levels that are within 2% of each other
min_bars_between = 120     # Require at least 120 bars between pivot points

# Create the ZigZag level finder with extremely restrictive parameters
zl = ZigZagClusterLevels(
    peak_percent_delta=zig_zag_percent,
    merge_distance=None,
    merge_percent=merge_percent,
    min_bars_between_peaks=min_bars_between,
    peaks='All',           # You can also try 'High' or 'Low' only
    level_selector='median'
)

# Fit the model
zl.fit(close_values)

# Print timing and results
processing_time = time.time() - start_time
print(f"Processing completed in {processing_time:.4f} seconds")
print(f"Found {len(zl.levels) if zl.levels else 0} support/resistance levels")
print(f"ZigZag percent: {zig_zag_percent}%, Min bars between peaks: {min_bars_between}")

# Display chart
plot_with_pivots(close_values, zl.levels, zig_zag_percent)
