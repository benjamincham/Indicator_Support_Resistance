import numpy as np

def peak_valley_pivots(x, up_thresh, down_thresh):
    """
    Finds the peaks and valleys of a series.
    
    Parameters
    ----------
    x : array-like
        The input array containing the data.
    up_thresh : float
        The threshold for detecting a peak. A point is considered a peak if it is
        at least up_thresh larger than the previous peak or valley.
    down_thresh : float
        The threshold for detecting a valley. A point is considered a valley if it is
        at least down_thresh smaller than the previous peak or valley.
        
    Returns
    -------
    numpy.ndarray
        An array of the same size as x with values 1 (for peaks), -1 (for valleys),
        and 0 (for neither).
    """
    # Early return for small arrays
    if len(x) <= 1:
        return np.zeros(len(x), dtype=int)
    
    # Convert to numpy array once for better performance
    x_np = np.asarray(x, dtype=np.float64)  # Ensure consistent dtype for calculations
    length = len(x_np)
    
    # Pre-allocate the result array
    pivots = np.zeros(length, dtype=np.int8)  # Use int8 instead of int for memory efficiency
    
    # Initialize variables
    trend = 0  # 0 for no trend, 1 for uptrend, -1 for downtrend
    last_pivot_idx = 0
    last_pivot_value = x_np[0]
    
    # Use local variables for thresholds to avoid repeated lookups
    up_t = up_thresh
    down_t = -down_thresh
    
    # Main loop - optimized for speed
    for i in range(1, length):
        current_value = x_np[i]
        diff = current_value - last_pivot_value
        
        if trend == 0:  # No trend
            if diff >= up_t:
                trend = 1
            elif -diff >= down_t:
                trend = -1
        elif trend == 1:  # Uptrend - looking for peaks
            if current_value > last_pivot_value:
                # New high, update the last pivot
                last_pivot_idx = i
                last_pivot_value = current_value
            elif -diff >= down_t:
                # Trend reversal - mark the last pivot as a peak
                pivots[last_pivot_idx] = 1
                # Start a new downtrend
                trend = -1
                last_pivot_idx = i
                last_pivot_value = current_value
        else:  # trend == -1, Downtrend - looking for valleys
            if current_value < last_pivot_value:
                # New low, update the last pivot
                last_pivot_idx = i
                last_pivot_value = current_value
            elif diff >= up_t:
                # Trend reversal - mark the last pivot as a valley
                pivots[last_pivot_idx] = -1
                # Start a new uptrend
                trend = 1
                last_pivot_idx = i
                last_pivot_value = current_value
    
    # Mark the last pivot if it's the end of a trend
    if trend == 1:
        pivots[last_pivot_idx] = 1
    elif trend == -1:
        pivots[last_pivot_idx] = -1
    
    return pivots
