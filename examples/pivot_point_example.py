import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import mplfinance as mpf

from pricelevels.pivot_points import PivotPointClusterLevels
from pricelevels.visualization._helpers import _plot_levels

# Start timing
start_time = time.time()

# Load data with OHLC columns
df = pd.read_csv('sample.txt')

# Ensure we have a datetime index for proper pivot point calculation
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

# Verify we have all required OHLC columns
print(f"Available columns: {df.columns.tolist()}")
required_columns = ['Open', 'High', 'Low', 'Close']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Parameters
merge_percent = 10.0  # Merge levels within 0.5% of each other

# Create the PivotPoint level finder
pp = PivotPointClusterLevels(
    merge_distance=None,
    merge_percent=merge_percent,
    level_selector='median',
    pivot_period='daily',
    include_mid_points=True  # Include mid-point levels for more granularity
)

# Fit the model
pp.fit(df)

# Print timing information
processing_time = time.time() - start_time
print(f"Processing completed in {processing_time:.4f} seconds")
print(f"Found {len(pp.levels) if pp.levels else 0} support/resistance levels")

# Create a visualization of the pivot points with candlestick chart
def plot_with_pivot_points(ohlc_df, levels, only_good=False, path=None):
    """
    Plot candlestick chart with pivot-based support/resistance levels.
    
    Parameters
    ----------
    ohlc_df : pandas.DataFrame
        DataFrame with OHLC data and DatetimeIndex.
    levels : list
        List of level dictionaries.
    only_good : bool, default=False
        Whether to only show levels with positive scores.
    path : str, optional
        Path to save the figure. If None, the figure is displayed.
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot candlestick chart
    mpf.plot(
        ohlc_df, 
        type='candle', 
        style='yahoo', 
        ax=ax,
        volume=False,
        show_nontrading=False
    )
    
    # Plot support/resistance levels
    _plot_levels(ax, levels, only_good)
    
    # Add title and labels
    plt.title('Pivot Point Support and Resistance Levels', fontsize=14)
    plt.xlabel('Date', fontsize=12)
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

# Display chart with pivot point levels
plot_with_pivot_points(df, pp.levels)

# Uncomment to save chart to image
# plot_with_pivot_points(df, pp.levels, path='pivot_points.png')
