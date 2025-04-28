import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader


def process_bitcoin_data(ts_file_path, news_file_path, start_date, end_date, text_col='full_article', channel_id='price'):
    # Read the files
    df_daily = pd.read_csv(ts_file_path)
    df_daily['TIME'] = pd.to_datetime(df_daily['TIME']).dt.date
    with open(news_file_path, 'r') as f:
        news_data = json.load(f)
    df_news = pd.DataFrame(news_data)
    df_news['publication_date'] = pd.to_datetime(df_news['publication_time']).dt.date

    # Group text by date and concatenate
    df_news_grouped = df_news.groupby('publication_date')[text_col].apply(lambda x: ' '.join(x)).reset_index()
    # Merge with Text and TS
    df_combined = pd.merge(df_daily, df_news_grouped, left_on='TIME', right_on='publication_date', how='left')
    df_combined = df_combined.rename(columns={'TIME': 'date'})

    # Filter by date
    df_filtered = df_combined[(df_combined['date'] >= pd.to_datetime(start_date).date()) & (df_combined['date'] <= pd.to_datetime(end_date).date())]
    
    # Filter by ID (todo: multivariates)
    df_filtered = df_filtered[df_filtered["ID"] == channel_id]
    df_filtered = df_filtered[["date", text_col, "VALUE"]].reset_index(drop=True)

    print(f"There are {len(df_filtered)} rows in the filtered bitcoin dataframe")

    return df_filtered

def split_series(df, lookback, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return df[:train_end], df[train_end-lookback:val_end], df[val_end-lookback:]


class TimeSeriesDataset(Dataset):
    def __init__(self, df, lookback_window, prediction_window, 
                 feature_cols = ['VALUE'], target_cols=['VALUE'], text_col='full_article'):
        self.df = df
        self.lookback = lookback_window
        self.predict = prediction_window
        self.target_cols = target_cols
        self.text_col = text_col
        self.feature_cols = feature_cols
        
        # Create sequences with both text and value for X, only value for Y
        self.x_text, self.x_value, self.y = self.create_xy_pairs(df, lookback_window, prediction_window)

    def create_xy_pairs(self, df, lookback, predict):
        x_text, x_value, y = [], [], []
        
        for i in range(len(df) - lookback - predict + 1):
            # Input sequence (both text and value)
            x_text_seq = df[self.text_col].iloc[i:i+lookback].values
            x_value_seq = df[self.feature_cols].iloc[i:i+lookback].values
            
            # Target sequence (value only)
            y_seq = df[self.target_cols].iloc[i+lookback:i+lookback+predict].values
            
            x_text.append(x_text_seq)
            x_value.append(x_value_seq)
            y.append(y_seq)
            
        return x_text, torch.tensor(x_value, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x_value)

    def __getitem__(self, idx):
        return {
            'text': self.x_text[idx],  # List of articles
            'value': self.x_value[idx],  # Tensor of values
            'target': self.y[idx]  # Tensor of target values
        }

# Create custom collate function for the dataloader

# After collate_fn:
# {
#     'text': [['article1', 'article2'], ['article3', 'article4']],  # List of lists
#     'value': tensor([[1.0, 2.0], [5.0, 6.0]]),  # Stacked tensor
#     'target': tensor([[3.0, 4.0], [7.0, 8.0]])  # Stacked tensor
# }
def collate_fn(batch):
    return {
        'text': [item['text'] for item in batch],
        'value': torch.stack([item['value'] for item in batch]),
        'target': torch.stack([item['target'] for item in batch])
    }


