# src/data_loader.py
import pandas as pd
from typing import List, Dict

def load_price_series(files: List[str]) -> Dict[str, pd.Series]:
    """
    Load price series from CSV files.

    Parameters
    ----------
    stock_files : list[str]
        List of file paths (relative to project root) pointing to CSVs
        containing a 'Dates' column and a 'PX_LAST' column.

    Returns
    -------
    dict[str, pd.Series]
        Mapping from ticker symbol (derived from filename) to a pandas Series
        of closing prices, indexed by datetime and sorted ascending.

    Raises
    ------
    FileNotFoundError
        If any CSV file cannot be opened.
    ValueError
        If the required 'PX_LAST' column is missing.
    """
    series = {}
    for f in files:
        df = pd.read_csv(f, parse_dates=['Dates'], index_col='Dates')
        symbol = f.split('/')[-1].replace('-bloomberg.csv','')
        series[symbol] = df['PX_LAST'].sort_index()
    return series
