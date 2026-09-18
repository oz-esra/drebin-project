import json
import pandas as pd

def load_drebin_dataset(features_path: str, metadata_path: str, max_date: str = "2023-12-31"):
    """
    Loads Hypercube DREBIN features and metadata.
    Filters by VTT=2 criteria, date range, and SHA-256 alignment.
    """
    print("Loading metadata...")
    df_meta = pd.read_json(metadata_path)
    
    # Apply VTT=2 malware labeling (vt_detection >= 2 is malware)
    df_meta['label'] = (df_meta['vt_detection'] >= 2).astype(int)
    df_meta['gp_date'] = pd.to_datetime(df_meta['gp_date'])
    
    # Filter by 2021-2023 timeframe for VTT=2 target set
    if max_date:
        df_meta = df_meta[df_meta['gp_date'] <= max_date].copy()
        
    print("Loading DREBIN features...")
    with open(features_path, 'r') as f:
        all_features = json.load(f)
        
    # SHA-256 key alignment
    common_shas = set(df_meta['sha256']).intersection(set(all_features.keys()))
    df_meta = df_meta[df_meta['sha256'].isin(common_shas)].copy()
    
    # Sort chronologically for TESSERACT compliance
    df_sorted = df_meta.sort_values('gp_date').reset_index(drop=True)
    
    return df_sorted, all_features