import numpy as np
import pandas as pd
from ._abstract import BaseLevelFinder
from .cluster import _cluster_prices_to_levels


class PivotPointClusterLevels(BaseLevelFinder):
    """
    A class that finds support and resistance levels using traditional pivot point calculations.
    
    This implementation calculates pivot points based on high, low, and close prices
    and generates multiple support and resistance levels from these pivot points.
    
    Parameters
    ----------
    merge_distance : float or None
        The absolute distance to merge close levels. If None, merge_percent must be provided.
    merge_percent : float or None
        The percentage of mean price to use as merge distance. If merge_distance is provided,
        this parameter is ignored.
    level_selector : str, default='median'
        The method to select the level price from a cluster. Options: 'mean', 'median', 'min', 'max'.
    pivot_period : str, default='daily'
        The period to use for pivot point calculations. Options: 'daily', 'weekly', 'monthly'.
    include_mid_points : bool, default=True
        Whether to include mid-point calculations (M1, M2, etc.).
    """
    
    def __init__(self, merge_distance, merge_percent=None, level_selector='median', 
                 pivot_period='daily', include_mid_points=True):
        self._pivot_period = pivot_period
        self._include_mid_points = include_mid_points
        self._ohlc_df = None
        super().__init__(merge_distance, merge_percent, level_selector)
    
    def _validate_init_args(self):
        """Validate initialization arguments."""
        super()._validate_init_args()
        
        valid_periods = ['daily', 'weekly', 'monthly']
        if self._pivot_period not in valid_periods:
            raise ValueError(f"pivot_period must be one of {valid_periods}")
    
    def fit(self, data):
        """
        Override the base fit method to handle OHLC data properly.
        
        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame with OHLC data.
        """
        # Validate input data
        if not isinstance(data, pd.DataFrame):
            raise ValueError("PivotPointClusterLevels requires a pandas DataFrame with OHLC data")
        
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Store the OHLC data
        self._ohlc_df = data.copy()
        
        # Get mean close price for distance calculation
        X = data['Close'].values
        
        # Calculate pivot point levels
        prices = self._find_potential_level_prices()
        
        # Cluster the levels
        levels = self._aggregate_prices_to_levels(prices, self._get_distance(X))
        
        self._levels = levels
    
    def _find_potential_level_prices(self):
        """
        Calculate potential price levels using pivot point analysis.
        
        Returns
        -------
        numpy.ndarray
            Array of potential price levels.
        """
        ohlc_df = self._ohlc_df
        
        # Resample data based on pivot_period if needed
        if self._pivot_period == 'weekly':
            # Ensure we have a datetime index
            if not isinstance(ohlc_df.index, pd.DatetimeIndex):
                raise ValueError("Weekly pivot points require a DatetimeIndex")
            
            # Resample to weekly data
            ohlc_df = ohlc_df.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            })
        elif self._pivot_period == 'monthly':
            # Ensure we have a datetime index
            if not isinstance(ohlc_df.index, pd.DatetimeIndex):
                raise ValueError("Monthly pivot points require a DatetimeIndex")
            
            # Resample to monthly data
            ohlc_df = ohlc_df.resample('M').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            })
        
        # Calculate pivot points
        pivot_levels = self._calculate_pivot_levels(ohlc_df)
        
        # Return unique values as numpy array
        return np.unique(pivot_levels)
    
    def _calculate_pivot_levels(self, ohlc_df):
        """
        Calculate pivot points and support/resistance levels.
        
        Parameters
        ----------
        ohlc_df : pandas.DataFrame
            DataFrame with OHLC data.
            
        Returns
        -------
        numpy.ndarray
            Array of calculated pivot levels.
        """
        # Extract high, low, close
        high = ohlc_df['High'].values
        low = ohlc_df['Low'].values
        close = ohlc_df['Close'].values
        
        # Initialize list to store all levels
        all_levels = []
        
        # Calculate for each period
        for i in range(1, len(close)):
            # Previous period values
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            
            # Calculate pivot point (PP)
            pp = (prev_high + prev_low + prev_close) / 3
            
            # Calculate support levels
            s1 = (2 * pp) - prev_high
            s2 = pp - (prev_high - prev_low)
            s3 = s1 - (prev_high - prev_low)
            
            # Calculate resistance levels
            r1 = (2 * pp) - prev_low
            r2 = pp + (prev_high - prev_low)
            r3 = r1 + (prev_high - prev_low)
            
            # Add basic levels
            period_levels = [pp, s1, s2, s3, r1, r2, r3]
            
            # Add mid-point levels if requested
            if self._include_mid_points:
                m1 = (s1 + pp) / 2  # Midpoint between S1 and PP
                m2 = (s2 + s1) / 2  # Midpoint between S2 and S1
                m3 = (s3 + s2) / 2  # Midpoint between S3 and S2
                m4 = (pp + r1) / 2  # Midpoint between PP and R1
                m5 = (r1 + r2) / 2  # Midpoint between R1 and R2
                m6 = (r2 + r3) / 2  # Midpoint between R2 and R3
                
                period_levels.extend([m1, m2, m3, m4, m5, m6])
            
            # Add to all levels
            all_levels.extend(period_levels)
        
        return np.array(all_levels)
    
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
