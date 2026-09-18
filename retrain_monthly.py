FEATURES_PATH = r"C:\Users\esrao\Downloads\hypercube_drebin.json"
METADATA_PATH = r"C:\Users\esrao\Downloads\metadata.json"

def main():
    print("1. Loading metadata and extraction records...")
    df_meta = pd.read_json(METADATA_PATH)
    df_meta['gp_date'] = pd.to_datetime(df_meta['gp_date'])
    df_meta = df_meta[df_meta['gp_date'] <= "2023-12-31"].copy()
    df_meta['label'] = (df_meta['vt_detection'] >= 2).astype(int)