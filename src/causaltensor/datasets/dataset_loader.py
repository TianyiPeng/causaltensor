"""
Dataset Loader for CausalTensor

This module provides a unified interface for loading various datasets used in causal inference.
Each dataset returns Y_df (outcome matrix), and optionally Z_df (treatment matrix) and X_df (covariates matrix).
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple, Callable

try:
    import pyreadr  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pyreadr = None


def _require_pyreadr(dataset_name: str) -> None:
    if pyreadr is None:
        raise ImportError(
            f"Loading dataset '{dataset_name}' requires the optional dependency 'pyreadr'. "
            "Install it with `pip install pyreadr`."
        )


def create_y_dataframe(df: pd.DataFrame, index_col: str, column_col: str, value_col: str) -> pd.DataFrame:
    """
    Create Y dataframe (entity x time matrix) from long-form data.
    Removes rows or columns with NaN values before pivoting, choosing which to remove
    based on which has a higher percentage of NaNs.
    
    Args:
        df: Long-form dataframe
        index_col: Column name for entities/rows
        column_col: Column name for time periods/columns
        value_col: Column name for values
    
    Returns:
        Y_df: Wide-form dataframe with entities as index and time as columns
    """
    df = df.copy()  # Avoid SettingWithCopyWarning
    df[column_col] = df[column_col].astype(int)
    
    # Check for NaN values in value_col
    if df[value_col].isna().any():
        # Calculate how many data points would be lost by removing entities vs times
        entities_with_nan = df[df[value_col].isna()][index_col].unique()
        times_with_nan = df[df[value_col].isna()][column_col].unique()
        
        total_entities = df[index_col].nunique()
        total_times = df[column_col].nunique()
        total_rows = len(df)
        
        # Count how many rows would be removed by each strategy
        rows_if_remove_entities = len(df[df[index_col].isin(entities_with_nan)])
        rows_if_remove_times = len(df[df[column_col].isin(times_with_nan)])
        
        pct_entities_with_nan = len(entities_with_nan) / total_entities if total_entities > 0 else 0
        pct_times_with_nan = len(times_with_nan) / total_times if total_times > 0 else 0
        
        # Remove whichever loses fewer total data points
        if rows_if_remove_entities <= rows_if_remove_times:
            # Remove entities (rows) - loses fewer data points
            df = df[~df[index_col].isin(entities_with_nan)]
            print(f"Removed {len(entities_with_nan)} entities ({pct_entities_with_nan:.1%}) with NaN values, losing {rows_if_remove_entities}/{total_rows} data points ({rows_if_remove_entities/total_rows:.1%})")
        else:
            # Remove time periods (columns) - loses fewer data points
            df = df[~df[column_col].isin(times_with_nan)]
            print(f"Removed {len(times_with_nan)} time periods ({pct_times_with_nan:.1%}) with NaN values, losing {rows_if_remove_times}/{total_rows} data points ({rows_if_remove_times/total_rows:.1%})")
    
    Y_df = df.pivot(index=index_col, columns=column_col, values=value_col)
    return Y_df


def create_z_dataframe(Y_df: pd.DataFrame, treated_entity: str, treatment_start_year: int) -> pd.DataFrame:
    """
    Create Z dataframe (treatment indicator matrix).
    
    Args:
        Y_df: Entity x time matrix
        treated_entity: Name of the treated entity
        treatment_start_year: Year when treatment starts
    
    Returns:
        Z_df: Treatment indicator matrix (1 for treated entity after treatment, 0 otherwise)
    """
    Z_df = pd.DataFrame(0, index=Y_df.index, columns=Y_df.columns)
    treated_mask = Z_df.index == treated_entity
    post_treatment_mask = Z_df.columns >= treatment_start_year
    Z_df.loc[treated_mask, post_treatment_mask] = 1
    return Z_df


def create_x_dataframe(df: pd.DataFrame, Y_df: pd.DataFrame, index_col: str, time_col: str, 
                      covariate_cols: list, avg_start_year: int, avg_end_year: int, 
                      additional_cols: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """
    Create X dataframe (covariates matrix) by averaging over pre-treatment period.
    
    Args:
        df: Long-form dataframe
        Y_df: Entity x time matrix
        index_col: Column name for entities
        time_col: Column name for time periods
        covariate_cols: List of covariate columns to average
        avg_start_year: Start year for averaging period
        avg_end_year: End year for averaging period
        additional_cols: Dict mapping new column names to years to add from Y_df
                        e.g., {'Smoking 1988': 1988, 'Smoking 1980': 1980}
    
    Returns:
        X_df: Covariates matrix with entities as rows
    """
    # Filter to averaging period and only entities present in Y_df
    mask = (df[time_col] >= avg_start_year) & (df[time_col] <= avg_end_year)
    mask = mask & df[index_col].isin(Y_df.index) & df[time_col].isin(Y_df.columns)
    
    # Average covariates over period
    X_df = (
        df.loc[mask, [index_col] + covariate_cols]
          .groupby(index_col, as_index=False)
          .mean()
    )
    
    # Set index to match Y_df for proper alignment
    X_df = X_df.set_index(index_col)
    
    # Add additional columns from Y_df if specified
    if additional_cols is not None:
        for col_name, year in additional_cols.items():
            if year in Y_df.columns:
                # Use loc to ensure proper alignment with Y_df index
                X_df[col_name] = Y_df.loc[X_df.index, year]
    
    return X_df


def load_dataset(dataset_name: str, datasets_path: str = "src/causaltensor/datasets/raw/") -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load a dataset by name and return Y_df, Z_df (optional), and X_df (optional).
    
    Args:
        dataset_name: Name of the dataset to load
        datasets_path: Path to the datasets directory
    
    Returns:
        Tuple of (Y_df, Z_df, X_df) where Z_df and X_df may be None
    
    Available datasets:
        - smoking: California smoking ban (1988)
        - basque: Basque Country GDP (1975)
        - german_reunification: West Germany reunification (1990)
        - texas: Texas prison reform (1993)
        - pwt_spain_eu: Spain joining EU (1986)
        - pwt_chile_trade: Chile trade liberalization (1976)
        - pwt_korea_democracy: Republic of Korea democratization (1988)
        - pwt_norway_oil: Norway oil discovery (1971)
        - retailrocket: Retailrocket ecommerce dataset
        - dunnhumby: Dunnhumby retail dataset
        - truus: Truus retail dataset
        - movielens: MovieLens recommendation dataset
    """
    try:
        loader = DATASET_BUILDERS[dataset_name]
    except KeyError as exc:
        available = ", ".join(DATASET_BUILDERS.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available datasets: {available}") from exc
    return loader(datasets_path)


def _load_smoking_dataset(datasets_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load smoking dataset - California smoking ban (1988)"""
    _require_pyreadr("smoking")
    result = pyreadr.read_r(f'{datasets_path}smoking.rda')
    df = list(result.values())[0]
    
    # State mapping from documentation
    state_mapping = {
        1: 'Alabama', 2: 'Arkansas', 3: 'California', 4: 'Colorado',
        5: 'Connecticut', 6: 'Delaware', 7: 'Georgia', 8: 'Idaho',
        9: 'Illinois', 10: 'Indiana', 11: 'Iowa', 12: 'Kansas',
        13: 'Kentucky', 14: 'Louisiana', 15: 'Maine', 16: 'Minnesota',
        17: 'Mississippi', 18: 'Missouri', 19: 'Montana', 20: 'Nebraska',
        21: 'Nevada', 22: 'New Hampshire', 23: 'New Mexico', 24: 'North Carolina',
        25: 'North Dakota', 26: 'Ohio', 27: 'Oklahoma', 28: 'Pennsylvania',
        29: 'Rhode Island', 30: 'South Carolina', 31: 'South Dakota', 32: 'Tennessee',
        33: 'Texas', 34: 'Utah', 35: 'Vermont', 36: 'Virginia',
        37: 'West Virginia', 38: 'Wisconsin', 39: 'Wyoming'
    }
    df["state"] = df["state"].map(state_mapping)
    
    # Create Y_df, Z_df, and X_df
    Y_df = create_y_dataframe(df, index_col="state", column_col="year", value_col="cigsale")
    Z_df = create_z_dataframe(Y_df, treated_entity='California', treatment_start_year=1988)
    
    cols_to_avg = [col for col in df.columns if col not in ['state', 'year', 'cigsale']]
    X_df = create_x_dataframe(
        df, Y_df, 
        index_col='state', 
        time_col='year',
        covariate_cols=cols_to_avg,
        avg_start_year=1980, 
        avg_end_year=1988,
        additional_cols={'Smoking 1988': 1988, 'Smoking 1980': 1980, 'Smoking 1975': 1975}
    )
    
    return Y_df, Z_df, X_df


def _load_basque_dataset(datasets_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Basque dataset - Basque Country GDP (1975)"""
    _require_pyreadr("basque")
    result = pyreadr.read_r(f'{datasets_path}basque.rda')
    df = list(result.values())[0]
    
    # Create Y_df, Z_df, and X_df
    Y_df = create_y_dataframe(df, index_col="regionname", column_col="year", value_col="gdpcap")
    Z_df = create_z_dataframe(Y_df, treated_entity='Basque Country (Pais Vasco)', treatment_start_year=1975)
    
    cols_to_avg = ['invest', 'secagriculture', 'secenergy', 'secindustry', 'secconstruction', 
                   'secservicesventa', 'secservicesnonventa', 'schoolillit', 'schoolprim', 
                   'schoolmed', 'schoolhigh', 'schoolposthigh', 'popdens']
    
    X_df = create_x_dataframe(
        df, Y_df,
        index_col='regionname',
        time_col='year',
        covariate_cols=cols_to_avg,
        avg_start_year=1964,
        avg_end_year=1969,
        additional_cols={'gdpcap1960': 1960, 'gdpcap1965': 1965, 'gdpcap1970': 1970}
    )
    
    return Y_df, Z_df, X_df


def _load_german_reunification_dataset(datasets_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load German reunification dataset - West Germany reunification (1990)"""
    df = pd.read_csv(f'{datasets_path}german_reunification.csv')
    
    # Create Y_df, Z_df, and X_df
    Y_df = create_y_dataframe(df, index_col="country", column_col="year", value_col="gdp")
    Z_df = create_z_dataframe(Y_df, treated_entity='West Germany', treatment_start_year=1990)
    
    cols_to_avg = ['infrate', 'trade', 'schooling', 'industry']
    
    X_df = create_x_dataframe(
        df, Y_df,
        index_col='country',
        time_col='year',
        covariate_cols=cols_to_avg,
        avg_start_year=1980,
        avg_end_year=1985,
        additional_cols={'gdp1960': 1960, 'gdp1970': 1970, 'gdp1980': 1980, 'gdp1985': 1985}
    )
    
    # Add investment columns (these are pre-existing in the original data)
    X_df['invest60'] = df[~df['invest60'].isna()]['invest60'].values
    X_df['invest70'] = df[~df['invest70'].isna()]['invest70'].values
    X_df['invest80'] = df[~df['invest80'].isna()]['invest80'].values
    
    return Y_df, Z_df, X_df


def _load_texas_dataset(datasets_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Texas dataset - Texas prison reform (1993)"""
    _require_pyreadr("texas")
    result = pyreadr.read_r(f'{datasets_path}texas.rda')
    df = list(result.values())[0]
    
    fips_map = {
        1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas", 6: "California",
        8: "Colorado", 9: "Connecticut", 10: "Delaware", 11: "District of Columbia",
        12: "Florida", 13: "Georgia", 15: "Hawaii", 16: "Idaho", 17: "Illinois",
        18: "Indiana", 19: "Iowa", 20: "Kansas", 21: "Kentucky", 22: "Louisiana",
        23: "Maine", 24: "Maryland", 25: "Massachusetts", 26: "Michigan",
        27: "Minnesota", 28: "Mississippi", 29: "Missouri", 30: "Montana",
        31: "Nebraska", 32: "Nevada", 33: "New Hampshire", 34: "New Jersey",
        35: "New Mexico", 36: "New York", 37: "North Carolina", 38: "North Dakota",
        39: "Ohio", 40: "Oklahoma", 41: "Oregon", 42: "Pennsylvania", 44: "Rhode Island",
        45: "South Carolina", 46: "South Dakota", 47: "Tennessee", 48: "Texas",
        49: "Utah", 50: "Vermont", 51: "Virginia", 53: "Washington", 54: "West Virginia",
        55: "Wisconsin", 56: "Wyoming"
    }
    
    df["state"] = df["statefip"].map(fips_map)
    
    # Create Y_df, Z_df, and X_df
    Y_df = create_y_dataframe(df, index_col="state", column_col="year", value_col="bmprate")
    Z_df = create_z_dataframe(Y_df, treated_entity='Texas', treatment_start_year=1993)
    
    cols_to_avg = ["income", "ur", "poverty", "black", "perc1519", "aidscapita", 
                   "crack", "alcohol", "parole", "probation", "capacity_operational"]
    
    X_df = create_x_dataframe(
        df, Y_df,
        index_col='state',
        time_col='year',
        covariate_cols=cols_to_avg,
        avg_start_year=1985,
        avg_end_year=1993,
        additional_cols={'bmprate1985': 1985, 'bmprate1990': 1990, 'bmprate1993': 1993}
    )
    
    return Y_df, Z_df, X_df


def _load_pwt_spain_eu_dataset(datasets_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load PWT dataset - Spain joining EU (1986)"""
    df = pd.read_csv(f'{datasets_path}PWT.csv')
    df['openness'] = df['csh_x'] + df['csh_m']
    
    df1 = df[(df['year'] >= 1970) & (df['year'] <= 2000)]
    Y_df = create_y_dataframe(df1, index_col="country", column_col="year", value_col="rgdpe")
    Z_df = create_z_dataframe(Y_df, treated_entity='Spain', treatment_start_year=1986)
    
    cols_to_avg = ["hc", "csh_i", "csh_c", "csh_g", "openness", "pl_gdpo", "pop"]
    
    X_df = create_x_dataframe(
        df1, Y_df,
        index_col='country',
        time_col='year',
        covariate_cols=cols_to_avg,
        avg_start_year=1970,
        avg_end_year=1980,
        additional_cols={'rgdpe1970': 1970, 'rgdpe1980': 1980, 'rgdpe1985': 1985}
    )
    
    return Y_df, Z_df, X_df


def _load_pwt_chile_trade_dataset(datasets_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load PWT dataset - Chile trade liberalization (1976)"""
    df = pd.read_csv(f'{datasets_path}PWT.csv')
    df['openness'] = df['csh_x'] + df['csh_m']
    
    df1 = df[(df['year'] >= 1960) & (df['year'] <= 1995)]
    Y_df = create_y_dataframe(df1, index_col="country", column_col="year", value_col="rgdpo")
    Z_df = create_z_dataframe(Y_df, treated_entity='Chile', treatment_start_year=1976)
    
    cols_to_avg = ["hc", "csh_i", "csh_c", "csh_g", "openness", "pl_gdpo", "rkna"]
    
    X_df = create_x_dataframe(
        df1, Y_df,
        index_col='country',
        time_col='year',
        covariate_cols=cols_to_avg,
        avg_start_year=1970,
        avg_end_year=1975,
        additional_cols={'rgdpo1970': 1970, 'rgdpo1975': 1975}
    )
    
    return Y_df, Z_df, X_df


def _load_pwt_korea_democracy_dataset(datasets_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load PWT dataset - Republic of Korea democratization (1988)"""
    df = pd.read_csv(f'{datasets_path}PWT.csv')
    df['openness'] = df['csh_x'] + df['csh_m']
    
    df1 = df[(df['year'] >= 1970) & (df['year'] <= 2000)]
    Y_df = create_y_dataframe(df1, index_col="country", column_col="year", value_col="rgdpe")
    Z_df = create_z_dataframe(Y_df, treated_entity='Republic of Korea', treatment_start_year=1988)
    
    cols_to_avg = ["hc", "csh_i", "csh_g", "openness", "ctfp"]
    
    X_df = create_x_dataframe(
        df1, Y_df,
        index_col='country',
        time_col='year',
        covariate_cols=cols_to_avg,
        avg_start_year=1980,
        avg_end_year=1987,
        additional_cols={'rgdpe1980': 1980, 'rgdpe1988': 1985}
    )
    
    return Y_df, Z_df, X_df


def _load_pwt_norway_oil_dataset(datasets_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load PWT dataset - Norway oil discovery (1971)"""
    df = pd.read_csv(f'{datasets_path}PWT.csv')
    df['openness'] = df['csh_x'] + df['csh_m']
    
    df1 = df[(df['year'] >= 1960) & (df['year'] <= 1980)]
    Y_df = create_y_dataframe(df1, index_col="country", column_col="year", value_col="rgdpe")
    Z_df = create_z_dataframe(Y_df, treated_entity='Norway', treatment_start_year=1971)
    
    cols_to_avg = ["hc", "csh_i", "csh_g", "openness", "rkna", "pl_gdpo"]
    
    X_df = create_x_dataframe(
        df1, Y_df,
        index_col='country',
        time_col='year',
        covariate_cols=cols_to_avg,
        avg_start_year=1960,
        avg_end_year=1970,
        additional_cols={'rgdpe1965': 1965, 'rgdpe1970': 1970}
    )
    
    return Y_df, Z_df, X_df


def _load_retailrocket_dataset(datasets_path: str) -> Tuple[pd.DataFrame, None, None]:
    """Load Retailrocket recommendation dataset"""
    df = pd.read_csv(f'{datasets_path}retailrocket_filtered.csv', engine="python", sep=None)
    df = df.groupby(['itemid', 'day']).size().reset_index(name='count')
    Y_df = create_y_dataframe(df, index_col="itemid", column_col="day", value_col="count").fillna(0)
    
    return Y_df, None, None


def _load_dunnhumby_dataset(datasets_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, None]:
    """Load Dunnhumby retail dataset"""
    df = pd.read_csv(f'{datasets_path}dunnhumby_filtered.csv', engine="python", sep=None)
    Y_df = create_y_dataframe(df, index_col="PRODUCT_ID", column_col="DAY", value_col="SALES_VALUE").fillna(0)
    Z_df = create_y_dataframe(df, index_col="PRODUCT_ID", column_col="DAY", value_col="PROMO").fillna(0)
    
    return Y_df, Z_df, None


def _load_truus_dataset(datasets_path: str) -> Tuple[pd.DataFrame, None, None]:
    """Load Truus recommendation dataset"""
    df = pd.read_csv(f'{datasets_path}truus.csv', engine="python", sep=None)
    df = df.groupby(['sku_id', 'day']).size().reset_index(name='count')
    Y_df = create_y_dataframe(df, index_col="sku_id", column_col="day", value_col="count").fillna(0)
    
    return Y_df, None, None


def _load_movielens_dataset(datasets_path: str) -> Tuple[pd.DataFrame, None, None]:
    """Load MovieLens recommendation dataset"""
    df = pd.read_csv(f'{datasets_path}movielens.data', sep='\t', header=None, 
                     names=['user_id', 'movie_id', 'rating', 'timestamp'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
    df['day'] = pd.to_numeric(df['date'].astype('category').cat.codes)
    df = df.groupby(['movie_id', 'day']).size().reset_index(name='count')
    Y_df = create_y_dataframe(df, index_col="movie_id", column_col="day", value_col="count").fillna(0)
    
    return Y_df, None, None


DATASET_BUILDERS: Dict[str, Callable[[str], Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]]] = {
    "smoking": _load_smoking_dataset,
    "basque": _load_basque_dataset,
    "german_reunification": _load_german_reunification_dataset,
    "texas": _load_texas_dataset,
    "pwt_spain_eu": _load_pwt_spain_eu_dataset,
    "pwt_chile_trade": _load_pwt_chile_trade_dataset,
    "pwt_korea_democracy": _load_pwt_korea_democracy_dataset,
    "pwt_norway_oil": _load_pwt_norway_oil_dataset,
    "retailrocket": _load_retailrocket_dataset,
    "dunnhumby": _load_dunnhumby_dataset,
    "truus": _load_truus_dataset,
    "movielens": _load_movielens_dataset,
}


def available_datasets() -> Tuple[str, ...]:
    """Return the supported dataset names."""
    return tuple(DATASET_BUILDERS.keys())


# Example usage
if __name__ == "__main__":
    # Example usage
    try:
        # Load smoking dataset
        Y_df, Z_df, X_df = load_dataset("smoking")
        print("Smoking dataset loaded successfully!")
        print(f"Y_df shape: {Y_df.shape}")
        print(f"Z_df shape: {Z_df.shape}")
        print(f"X_df shape: {X_df.shape}")
        
        # Load a recommendation dataset (no Z_df or X_df)
        Y_df, Z_df, X_df = load_dataset("movielens")
        print("\nMovieLens dataset loaded successfully!")
        print(f"Y_df shape: {Y_df.shape}")
        print(f"Z_df: {Z_df}")
        print(f"X_df: {X_df}")
        
    except Exception as e:
        print(f"Error: {e}")
