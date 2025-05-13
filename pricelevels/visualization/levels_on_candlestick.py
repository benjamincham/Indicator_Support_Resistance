import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import mplfinance as mpf

from ._helpers import _plot_levels


def plot_levels_on_candlestick(df, levels, only_good=False, path=None,
                               formatter=mdates.DateFormatter('%y-%m-%d %H:%M:%S')):
    ohlc = df[['Datetime', 'Open', 'High', 'Low', 'Close']].copy()
    ohlc["Datetime"] = pd.to_datetime(ohlc['Datetime'])
    ohlc.set_index('Datetime', inplace=True)
    
    f1, ax = plt.subplots(figsize=(10, 5))
    
    mpf.plot(ohlc, type='candle', style='yahoo', ax=ax, 
             colorup='green', colordown='red',
             show_nontrading=False)
    
    _plot_levels(ax, levels, only_good)
    
    ax.xaxis.set_major_formatter(formatter)
    
    if path:
        plt.savefig(path)
    else:
        plt.show()
    plt.close()
