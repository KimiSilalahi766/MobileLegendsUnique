import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
import os

class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_columns = []
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        try:
            # Load datasets
            pemain_df = self._load_pemain_data()
            match_df = self._load_match_data()
            
            # Merge datasets
            merged_df = self._merge_datasets(pemain_df, match_df)
            
            # Feature engineering
            processed_df = self._engineer_features(merged_df)
            
            # Preprocessing
            final_df = self._preprocess_features(processed_df)
            
            logging.info(f"Data preprocessing completed. Shape: {final_df.shape}")
            return final_df
            
        except Exception as e:
            logging.error(f"Error in data preprocessing: {str(e)}")
            raise
    
    def _load_pemain_data(self):
        """Load pemain_medan.csv"""
        file_path = 'data/pemain_medan.csv'
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            logging.info(f"Successfully loaded pemain_medan.csv with {len(df)} players")
            logging.info(f"Player ID range: {df['player_id'].min()} to {df['player_id'].max()}")
            logging.info(f"Columns: {df.columns.tolist()}")
            return df
        else:
            logging.error("pemain_medan.csv file not found!")
            raise FileNotFoundError("Original CSV file is required")
    
    def _load_match_data(self):
        """Load match_history_sample.csv"""
        file_path = 'data/match_history_sample.csv'
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            logging.info(f"Successfully loaded match_history_sample.csv with {len(df)} matches")
            logging.info(f"Unique players in match history: {df['player_id'].nunique()}")
            return df
        else:
            logging.error("match_history_sample.csv file not found!")
            raise FileNotFoundError("Original CSV file is required")
    
    def _generate_sample_pemain_data(self):
        """Generate sample pemain data"""
        np.random.seed(42)
        
        roles = ['Fighter', 'Mage', 'Marksman', 'Tank', 'Assassin', 'Support']
        
        data = []
        for i in range(1, 251):  # 250 players
            data.append({
                'player_id': i,
                'win_rate': np.random.normal(0.55, 0.15),  # Win rate around 55%
                'kills': np.random.poisson(8),  # Average 8 kills
                'deaths': np.random.poisson(5),  # Average 5 deaths
                'assists': np.random.poisson(6),  # Average 6 assists
                'main_role': np.random.choice(roles)
            })
        
        df = pd.DataFrame(data)
        df['win_rate'] = np.clip(df['win_rate'], 0.1, 0.9)  # Ensure reasonable win rates
        
        return df
    
    def _generate_sample_match_data(self):
        """Generate sample match history data"""
        np.random.seed(42)
        
        heroes = ['Layla', 'Tigreal', 'Alucard', 'Miya', 'Alice', 'Nana', 'Zilong', 'Eudora', 'Gord', 'Cyclops']
        roles = ['Fighter', 'Mage', 'Marksman', 'Tank', 'Assassin', 'Support']
        results = ['Win', 'Loss']
        ranks = ['Warrior', 'Elite', 'Master', 'Grandmaster', 'Epic', 'Legend', 'Mythic']
        
        data = []
        for player_id in range(1, 251):  # For each player
            num_matches = np.random.randint(10, 50)  # 10-50 matches per player
            
            for match_num in range(1, num_matches + 1):
                kills = np.random.poisson(6)
                deaths = max(1, np.random.poisson(4))  # Ensure at least 1 death
                assists = np.random.poisson(5)
                
                data.append({
                    'player_id': player_id,
                    'match_number': match_num,
                    'hero': np.random.choice(heroes),
                    'role': np.random.choice(roles),
                    'kills': kills,
                    'deaths': deaths,
                    'assists': assists,
                    'result': np.random.choice(results, p=[0.55, 0.45]),  # 55% win rate
                    'duration_minutes': np.random.normal(15, 5),  # 15 minutes average
                    'rank_tier': np.random.choice(ranks),
                    'kda_ratio': (kills + assists) / deaths
                })
        
        return pd.DataFrame(data)
    
    def _merge_datasets(self, pemain_df, match_df):
        """Merge the two datasets"""
        # Aggregate match data per player to get additional statistics
        match_agg = match_df.groupby('player_id').agg({
            'duration_minutes': 'mean',
            'match_number': 'count'  # Total matches
        }).reset_index()
        
        # Rename to avoid conflicts
        match_agg.rename(columns={
            'duration_minutes': 'avg_duration_minutes',
            'match_number': 'total_matches'
        }, inplace=True)
        
        # Merge with pemain data (use left join to keep all players)
        merged_df = pd.merge(pemain_df, match_agg, on='player_id', how='left')
        
        # Fill NaN values for players without match history
        merged_df['avg_duration_minutes'] = merged_df['avg_duration_minutes'].fillna(15.0)  # Default 15 minutes
        merged_df['total_matches'] = merged_df['total_matches'].fillna(20)  # Default 20 matches
        
        return merged_df
    
    def _engineer_features(self, df):
        """Engineer new features"""
        # Calculate avg_kda using kills, deaths, assists from pemain_medan.csv
        df['avg_kda'] = np.where(
            df['deaths'] > 0,
            (df['kills'] + df['assists']) / df['deaths'],
            df['kills'] + df['assists']  # If no deaths, use kills + assists
        )
        
        # Calculate match_frequency (matches per day, assuming 30 days period)
        df['match_frequency'] = df['total_matches'] / 30.0
        
        return df
    
    def _preprocess_features(self, df):
        """Preprocess features with encoding and normalization"""
        # One-hot encode main_role
        role_dummies = pd.get_dummies(df['main_role'], prefix='role')
        df = pd.concat([df, role_dummies], axis=1)
        
        # Define numerical features to normalize
        numerical_features = ['win_rate', 'avg_kda', 'match_frequency']
        
        # Normalize numerical features
        df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
        
        # Store feature columns for clustering
        self.feature_columns = numerical_features + list(role_dummies.columns)
        
        return df
    
    def get_feature_columns(self):
        """Get the list of feature columns for clustering"""
        return self.feature_columns
