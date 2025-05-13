import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from ._abstract import BaseZigZagLevels, BaseLevelFinder


def _cluster_prices_to_levels(prices, distance, level_selector='mean'):
    # Return early if there are no prices to cluster
    if len(prices) == 0:
        return None
        
    # Reshape prices once for efficiency
    prices_reshaped = prices.reshape(-1, 1)
    
    # Use try/except for handling edge cases
    try:
        # Create clustering object with appropriate parameters
        clustering = AgglomerativeClustering(
            distance_threshold=distance, 
            n_clusters=None
        )
        clustering.fit(prices_reshaped)
    except ValueError:
        return None

    # Use NumPy operations instead of DataFrame where possible for better performance
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    result = []
    
    # Process each cluster without creating intermediate DataFrames
    for label in unique_labels:
        mask = labels == label
        cluster_prices = prices[mask]
        
        # Calculate the selected statistic directly
        if level_selector == 'mean':
            price_value = np.mean(cluster_prices)
        elif level_selector == 'median':
            price_value = np.median(cluster_prices)
        elif level_selector == 'min':
            price_value = np.min(cluster_prices)
        elif level_selector == 'max':
            price_value = np.max(cluster_prices)
        else:
            # Fall back to pandas for other aggregation methods
            df = pd.DataFrame(data=cluster_prices, columns=('price',))
            price_value = df['price'].agg(level_selector).iloc[0]
        
        result.append({
            'cluster': label,
            'price': price_value,
            'peak_count': len(cluster_prices)
        })
    
    return result


class ZigZagClusterLevels(BaseZigZagLevels):

    def _aggregate_prices_to_levels(self, prices, distance):
        return _cluster_prices_to_levels(prices, distance, self._level_selector)


class RawPriceClusterLevels(BaseLevelFinder):
    def __init__(self, merge_distance, merge_percent=None, level_selector='median', use_maximums=True,
                 bars_for_peak=21):

        self._use_max = use_maximums
        self._bars_for_peak = bars_for_peak
        super().__init__(merge_distance, merge_percent, level_selector)

    def _validate_init_args(self):
        super()._validate_init_args()
        if self._bars_for_peak % 2 == 0:
            raise Exception('N bars to define peak should be odd number')

    def _find_potential_level_prices(self, X):
        # Convert to numpy array for faster operations
        X_array = np.asarray(X)
        bars_to_shift = int((self._bars_for_peak - 1) / 2)
        
        # Pre-allocate array for results
        result = np.full(len(X_array), np.nan)
        
        # Use NumPy's stride_tricks for faster rolling window operations
        # This avoids pandas overhead for simple max/min operations
        from numpy.lib.stride_tricks import sliding_window_view
        
        # Handle edge cases
        if len(X_array) < self._bars_for_peak:
            return np.array([])
            
        # Create rolling window view
        windows = sliding_window_view(X_array, self._bars_for_peak)
        
        # Calculate max or min for each window
        if self._use_max:
            window_stats = np.max(windows, axis=1)
        else:
            window_stats = np.min(windows, axis=1)
        
        # Pad results to match original array length
        result[bars_to_shift:bars_to_shift+len(window_stats)] = window_stats
        
        # Find indices where the original value equals the window statistic
        # This is equivalent to the pandas equality check but much faster
        potential_peaks = []
        for i in range(bars_to_shift, len(X_array) - bars_to_shift):
            if X_array[i] == result[i]:
                potential_peaks.append(X_array[i])
        
        # Use numpy's unique function instead of pandas
        return np.unique(np.array(potential_peaks))

    def _aggregate_prices_to_levels(self, prices, distance):
        return _cluster_prices_to_levels(prices, distance, self._level_selector)
