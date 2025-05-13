import numpy as np
import pandas as pd
from ._abstract import BaseLevelFinder
from .cluster import _cluster_prices_to_levels


class ClosePriceClusterLevels(BaseLevelFinder):
    """
    A class that finds support and resistance levels using only close prices.
    
    This implementation identifies potential support and resistance levels
    by analyzing patterns in close prices without requiring OHLC data.
    
    Parameters
    ----------
    merge_distance : float or None
        The absolute distance to merge close levels. If None, merge_percent must be provided.
    merge_percent : float or None
        The percentage of mean price to use as merge distance. If merge_distance is provided,
        this parameter is ignored.
    level_selector : str, default='median'
        The method to select the level price from a cluster. Options: 'mean', 'median', 'min', 'max'.
    window_size : int, default=20
        The size of the rolling window to identify local extrema.
    percentile_threshold : float, default=90.0
        The percentile threshold (0-100) to identify significant price levels.
    """
    
    def __init__(self, merge_distance, merge_percent=None, level_selector='median', 
                 window_size=20, percentile_threshold=90.0):
        self._window_size = window_size
        self._percentile_threshold = percentile_threshold
        super().__init__(merge_distance, merge_percent, level_selector)
    
    def _validate_init_args(self):
        """Validate initialization arguments."""
        super()._validate_init_args()
        
        if self._window_size < 2:
            raise ValueError("window_size must be at least 2")
        
        if not 0 <= self._percentile_threshold <= 100:
            raise ValueError("percentile_threshold must be between 0 and 100")
    
    def _find_potential_level_prices(self, X):
        """
        Find potential price levels using close price analysis.
        
        Parameters
        ----------
        X : numpy.ndarray
            Array of close prices.
            
        Returns
        -------
        numpy.ndarray
            Array of potential price levels.
        """
        # Ensure X is a numpy array
        X = np.asarray(X)
        
        # Find local maxima and minima
        levels = []
        
        # Method 1: Find local maxima and minima using rolling window
        for i in range(self._window_size, len(X) - self._window_size):
            window = X[i - self._window_size:i + self._window_size + 1]
            if X[i] == np.max(window):
                levels.append(X[i])  # Local maximum
            elif X[i] == np.min(window):
                levels.append(X[i])  # Local minimum
        
        # Method 2: Add price levels at significant percentiles
        percentiles = np.linspace(0, 100, 11)  # 0, 10, 20, ..., 100
        for p in percentiles:
            levels.append(np.percentile(X, p))
        
        # Method 3: Add levels where price changes direction significantly
        price_changes = np.diff(X)
        significant_changes = np.percentile(np.abs(price_changes), self._percentile_threshold)
        for i in range(1, len(price_changes)):
            if (price_changes[i-1] > 0 and price_changes[i] < 0) or \
               (price_changes[i-1] < 0 and price_changes[i] > 0):
                if abs(price_changes[i]) >= significant_changes:
                    levels.append(X[i])
        
        # Return unique values
        return np.unique(np.array(levels))
    
    def _aggregate_prices_to_levels(self, prices, distance):
        """
        Aggregate close prices into distinct levels using clustering.
        
        Parameters
        ----------
        prices : numpy.ndarray
            Array of potential price levels.
        distance : float
            Distance threshold for clustering.
            
        Returns
        -------
        list
            List of dictionaries containing level information.
        """
        return _cluster_prices_to_levels(prices, distance, self._level_selector)
