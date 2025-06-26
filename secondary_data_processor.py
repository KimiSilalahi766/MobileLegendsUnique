"""
Secondary Data Integration Module
Processes and integrates secondary data sources with primary survey data
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

class SecondaryDataProcessor:
    def __init__(self):
        self.rank_data = None
        self.hero_meta_data = None
        self.integrated_data = None
        self.scaler = MinMaxScaler()
        
    def load_secondary_data(self):
        """Load secondary data sources - SIMULASI untuk demonstrasi penelitian"""
        try:
            # CATATAN: Data ini adalah simulasi untuk keperluan demonstrasi sistem
            # Pada implementasi nyata, data ini akan berasal dari:
            # 1. API Mobile Legends resmi (jika tersedia)
            # 2. Database sistem ranking internal
            # 3. Statistik meta hero dari platform analitik game
            
            # Generate simulated rank data for demonstration
            self._generate_simulated_rank_data()
            if self.rank_data is not None:
                logging.info(f"Generated simulated rank data: {len(self.rank_data)} records")
            
            # Generate simulated hero meta data for demonstration  
            self._generate_simulated_hero_data()
            if self.hero_meta_data is not None:
                logging.info(f"Generated simulated hero meta: {len(self.hero_meta_data)} heroes")
            
            return True
        except Exception as e:
            logging.error(f"Error generating simulated secondary data: {e}")
            return False
    
    def _generate_simulated_rank_data(self):
        """Generate comprehensive simulated rank data for demonstration"""
        # Simulate rank data for 1000 players (large sample size for robust analysis)
        player_ids = list(range(101849453, 101849453 + 1000))
        
        # Create realistic rank distribution based on actual ML demographics
        rank_distribution = (['Epic'] * 500 + ['Legend'] * 300 + ['Mythic'] * 150 + 
                           ['Grandmaster'] * 40 + ['Mythical Glory'] * 10)
        np.random.shuffle(rank_distribution)
        
        # Generate correlated data (higher ranks tend to have more matches, heroes, etc.)
        rank_multipliers = {
            'Epic': 1.0, 'Legend': 1.3, 'Mythic': 1.6, 
            'Grandmaster': 1.9, 'Mythical Glory': 2.2
        }
        
        self.rank_data = pd.DataFrame({
            'player_id': player_ids,
            'current_rank': rank_distribution,
            'rank_points': [
                np.random.randint(10, int(150 * rank_multipliers[rank])) 
                for rank in rank_distribution
            ],
            'season_matches': [
                np.random.randint(100, int(800 * rank_multipliers[rank])) 
                for rank in rank_distribution
            ],
            'account_level': [
                np.random.randint(20, int(70 * rank_multipliers[rank])) 
                for rank in rank_distribution
            ],
            'total_heroes_owned': [
                np.random.randint(15, int(80 * rank_multipliers[rank])) 
                for rank in rank_distribution
            ],
            # Additional realistic metrics
            'total_bp_earned': [
                np.random.randint(50000, int(500000 * rank_multipliers[rank])) 
                for rank in rank_distribution
            ],
            'winrate_this_season': [
                np.random.uniform(0.45, min(0.85, 0.50 + rank_multipliers[rank] * 0.15)) 
                for rank in rank_distribution
            ],
            'favorite_role': np.random.choice(['Fighter', 'Assassin', 'Mage', 'Marksman', 'Tank', 'Support'], 1000),
            'most_played_hero': np.random.choice([
                'Gusion', 'Fanny', 'Ling', 'Hayabusa', 'Lancelot', 'Alucard', 'Zilong',
                'Layla', 'Miya', 'Bruno', 'Granger', 'Kimmy', 'Irithel', 'Wanwan',
                'Eudora', 'Kagura', 'Lunox', 'Harith', 'Pharsa', 'Change', 'Gord',
                'Franco', 'Tigreal', 'Johnson', 'Khufra', 'Belerick', 'Grock',
                'Angela', 'Estes', 'Rafaela', 'Diggie', 'Floryn'
            ], 1000),
            'mvp_count': [
                np.random.randint(5, int(200 * rank_multipliers[rank])) 
                for rank in rank_distribution
            ],
            'legend_count': [
                np.random.randint(0, int(50 * rank_multipliers[rank])) 
                for rank in rank_distribution
            ],
            'credit_score': np.random.randint(90, 120, 1000),
            'days_since_last_match': np.random.randint(0, 7, 1000)
        })
    
    def _generate_simulated_hero_data(self):
        """Generate comprehensive simulated hero meta data for demonstration"""
        # Complete hero database with 80 heroes across all roles
        heroes_data = [
            # Assassins (15 heroes)
            ('Gusion', 'Assassin', 'S+', 0.35, 0.45, 0.52), ('Fanny', 'Assassin', 'S', 0.15, 0.65, 0.58),
            ('Ling', 'Assassin', 'S', 0.28, 0.52, 0.55), ('Hayabusa', 'Assassin', 'A+', 0.25, 0.38, 0.53),
            ('Lancelot', 'Assassin', 'A+', 0.22, 0.42, 0.51), ('Helcurt', 'Assassin', 'A', 0.18, 0.25, 0.54),
            ('Karina', 'Assassin', 'A', 0.20, 0.15, 0.49), ('Natalia', 'Assassin', 'B+', 0.12, 0.18, 0.52),
            ('Saber', 'Assassin', 'B+', 0.16, 0.22, 0.48), ('Zilong', 'Assassin', 'B', 0.14, 0.12, 0.47),
            ('Alucard', 'Assassin', 'B', 0.19, 0.08, 0.46), ('Hanzo', 'Assassin', 'B', 0.11, 0.15, 0.49),
            ('Lesley', 'Assassin', 'B', 0.13, 0.10, 0.48), ('Selena', 'Assassin', 'A', 0.17, 0.35, 0.55),
            ('Benedetta', 'Assassin', 'A+', 0.24, 0.40, 0.53),
            
            # Mages (20 heroes)
            ('Kagura', 'Mage', 'S+', 0.32, 0.48, 0.54), ('Lunox', 'Mage', 'S', 0.28, 0.55, 0.56),
            ('Harith', 'Mage', 'S', 0.26, 0.50, 0.53), ('Pharsa', 'Mage', 'A+', 0.23, 0.35, 0.52),
            ('Change', 'Mage', 'A+', 0.21, 0.30, 0.51), ('Eudora', 'Mage', 'A', 0.18, 0.20, 0.49),
            ('Gord', 'Mage', 'A', 0.16, 0.18, 0.50), ('Valir', 'Mage', 'A', 0.19, 0.25, 0.52),
            ('Aurora', 'Mage', 'B+', 0.15, 0.15, 0.48), ('Cyclops', 'Mage', 'B+', 0.17, 0.12, 0.49),
            ('Nana', 'Mage', 'B', 0.14, 0.08, 0.47), ('Alice', 'Mage', 'B', 0.13, 0.10, 0.48),
            ('Vexana', 'Mage', 'B', 0.12, 0.06, 0.46), ('Odette', 'Mage', 'B', 0.11, 0.08, 0.47),
            ('Lylia', 'Mage', 'A', 0.20, 0.28, 0.51), ('Cecilion', 'Mage', 'A+', 0.24, 0.38, 0.53),
            ('Yve', 'Mage', 'A', 0.18, 0.22, 0.50), ('Vale', 'Mage', 'B+', 0.16, 0.14, 0.49),
            ('Zhask', 'Mage', 'B+', 0.15, 0.16, 0.48), ('Xavier', 'Mage', 'S', 0.29, 0.52, 0.55),
            
            # Marksmen (15 heroes)
            ('Granger', 'Marksman', 'S+', 0.38, 0.42, 0.54), ('Wanwan', 'Marksman', 'S', 0.33, 0.48, 0.55),
            ('Bruno', 'Marksman', 'S', 0.30, 0.35, 0.52), ('Kimmy', 'Marksman', 'A+', 0.25, 0.40, 0.51),
            ('Claude', 'Marksman', 'A+', 0.27, 0.38, 0.53), ('Irithel', 'Marksman', 'A', 0.22, 0.25, 0.50),
            ('Hanabi', 'Marksman', 'A', 0.20, 0.28, 0.49), ('Karrie', 'Marksman', 'A', 0.24, 0.30, 0.52),
            ('Miya', 'Marksman', 'B+', 0.18, 0.15, 0.48), ('Layla', 'Marksman', 'B', 0.16, 0.12, 0.47),
            ('Clint', 'Marksman', 'B+', 0.19, 0.18, 0.49), ('Moskov', 'Marksman', 'B', 0.15, 0.14, 0.48),
            ('Yi Sun-shin', 'Marksman', 'B+', 0.17, 0.20, 0.50), ('Roger', 'Marksman', 'B', 0.14, 0.16, 0.47),
            ('Beatrix', 'Marksman', 'A+', 0.26, 0.45, 0.54),
            
            # Fighters (15 heroes)
            ('Paquito', 'Fighter', 'S+', 0.35, 0.50, 0.56), ('Chou', 'Fighter', 'S', 0.28, 0.45, 0.54),
            ('Aldous', 'Fighter', 'S', 0.32, 0.38, 0.53), ('X.Borg', 'Fighter', 'A+', 0.26, 0.42, 0.52),
            ('Dyrroth', 'Fighter', 'A+', 0.24, 0.35, 0.51), ('Khaleed', 'Fighter', 'A', 0.22, 0.28, 0.50),
            ('Yu Zhong', 'Fighter', 'A', 0.20, 0.32, 0.52), ('Silvanna', 'Fighter', 'A', 0.23, 0.30, 0.51),
            ('Guinevere', 'Fighter', 'B+', 0.18, 0.25, 0.49), ('Freya', 'Fighter', 'B+', 0.16, 0.20, 0.48),
            ('Ruby', 'Fighter', 'B', 0.15, 0.18, 0.47), ('Alpha', 'Fighter', 'B', 0.14, 0.15, 0.48),
            ('Badang', 'Fighter', 'B', 0.13, 0.12, 0.46), ('Martis', 'Fighter', 'B+', 0.17, 0.22, 0.49),
            ('Jawhead', 'Fighter', 'A', 0.21, 0.28, 0.50),
            
            # Tanks (15 heroes)
            ('Khufra', 'Tank', 'S+', 0.40, 0.55, 0.54), ('Franco', 'Tank', 'S', 0.35, 0.48, 0.52),
            ('Tigreal', 'Tank', 'S', 0.32, 0.42, 0.51), ('Johnson', 'Tank', 'A+', 0.28, 0.38, 0.53),
            ('Grock', 'Tank', 'A+', 0.26, 0.35, 0.52), ('Belerick', 'Tank', 'A', 0.24, 0.30, 0.50),
            ('Uranus', 'Tank', 'A', 0.22, 0.25, 0.49), ('Hylos', 'Tank', 'A', 0.20, 0.28, 0.51),
            ('Akai', 'Tank', 'B+', 0.18, 0.22, 0.48), ('Lolita', 'Tank', 'B+', 0.16, 0.20, 0.49),
            ('Minotaur', 'Tank', 'B', 0.15, 0.18, 0.47), ('Gatotgacha', 'Tank', 'B', 0.14, 0.15, 0.48),
            ('Baxia', 'Tank', 'B+', 0.17, 0.25, 0.50), ('Edith', 'Tank', 'A+', 0.25, 0.40, 0.53),
            ('Atlas', 'Tank', 'A', 0.23, 0.32, 0.52)
        ]
        
        # Create DataFrame from structured data
        df_data = []
        for hero_name, role, tier, pick_rate, ban_rate, win_rate in heroes_data:
            df_data.append({
                'hero_name': hero_name,
                'role': role,
                'hero_tier': tier,
                'pick_rate': pick_rate,
                'ban_rate': ban_rate,
                'win_rate': win_rate,
                'ban_rate_percent': ban_rate * 100,
                'win_rate_percent': win_rate * 100,
                'popularity_score': pick_rate * 100,
                'meta_score': (pick_rate * 0.3 + win_rate * 0.5 + (1 - ban_rate) * 0.2) * 100,
                'difficulty': np.random.choice(['Easy', 'Medium', 'Hard', 'Very Hard']),
                'gold_cost': np.random.choice([32000, 24000, 15000, 6500]),
                'release_year': np.random.choice([2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]),
                'rework_count': np.random.randint(0, 4),
                'skin_count': np.random.randint(3, 15),
                'tournament_presence': np.random.uniform(0.1, 0.9),
                'counter_potential': np.random.uniform(0.3, 0.8),
                'team_fight_rating': np.random.uniform(0.4, 0.9),
                'early_game_power': np.random.uniform(0.3, 0.8),
                'late_game_power': np.random.uniform(0.4, 0.9)
            })
        
        self.hero_meta_data = pd.DataFrame(df_data)
    
    def integrate_with_primary_data(self, primary_df):
        """Integrate secondary data with primary survey data"""
        try:
            # Merge with rank statistics
            integrated = primary_df.merge(
                self.rank_data, 
                on='player_id', 
                how='left'
            )
            
            # Fill missing values for players not in secondary data
            integrated['current_rank'] = integrated['current_rank'].fillna('Epic')
            integrated['rank_points'] = integrated['rank_points'].fillna(50)
            integrated['season_matches'] = integrated['season_matches'].fillna(100)
            integrated['account_level'] = integrated['account_level'].fillna(30)
            integrated['total_heroes_owned'] = integrated['total_heroes_owned'].fillna(20)
            
            # Add derived features from secondary data
            integrated['rank_numeric'] = integrated['current_rank'].map({
                'Epic': 1, 'Legend': 2, 'Mythic': 3
            }).fillna(1)
            
            # Normalize rank points by rank tier
            integrated['normalized_rank_points'] = integrated.apply(
                lambda row: self._normalize_rank_points(row['current_rank'], row['rank_points']),
                axis=1
            )
            
            # Add experience metrics
            integrated['experience_score'] = (
                integrated['account_level'] * 0.4 + 
                integrated['total_heroes_owned'] * 0.3 +
                integrated['season_matches'] * 0.3
            )
            
            self.integrated_data = integrated
            logging.info(f"Integration completed: {len(integrated)} records")
            
            return integrated
            
        except Exception as e:
            logging.error(f"Error in data integration: {e}")
            return primary_df
    
    def _normalize_rank_points(self, rank, points):
        """Normalize rank points based on tier"""
        if pd.isna(points):
            return 0
        
        if rank == 'Epic':
            return points / 300  # Max Epic points
        elif rank == 'Legend':
            return points / 400  # Max Legend points  
        elif rank == 'Mythic':
            return points / 600  # Max Mythic points
        return 0
    
    def compare_data_sources(self, primary_df):
        """Compare primary vs secondary data characteristics"""
        comparison = {
            'primary_data': {
                'source': 'Survey/Questionnaire',
                'collection_method': 'Direct data collection from players',
                'sample_size': len(primary_df),
                'data_quality': 'High (controlled collection)',
                'features': list(primary_df.columns),
                'bias_potential': 'Response bias, sampling bias',
                'cost': 'High (time + resources)',
                'timeliness': 'Current snapshot'
            },
            'secondary_data': {
                'source': 'Data simulasi untuk demonstrasi sistem',
                'collection_method': 'Generated simulation data',
                'sample_size': len(self.rank_data) if self.rank_data is not None else 0,
                'data_quality': 'Simulated (for system demonstration only)',
                'features': list(self.rank_data.columns) if self.rank_data is not None else [],
                'bias_potential': 'Simulation bias, not real player data',
                'cost': 'Low (generated data)',
                'timeliness': 'Simulated current snapshot'
            }
        }
        
        # Add rank analysis instead of tier_rank which doesn't exist
        if self.integrated_data is not None and 'current_rank' in self.integrated_data.columns:
            rank_distribution = self.integrated_data['current_rank'].value_counts()
            comparison['rank_analysis'] = rank_distribution.to_dict()
        
        return comparison
    
    def enhanced_clustering_analysis(self, integrated_df):
        """Perform clustering with integrated primary + secondary data"""
        try:
            # Select features for clustering (combining primary + secondary)
            clustering_features = [
                'win_rate', 'avg_kda', 'match_frequency',  # Primary features
                'rank_numeric', 'normalized_rank_points', 'experience_score'  # Secondary features
            ]
            
            # Prepare data for clustering
            feature_data = integrated_df[clustering_features].fillna(0)
            scaled_data = self.scaler.fit_transform(feature_data)
            
            # K-Means clustering
            kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
            kmeans_labels = kmeans.fit_predict(scaled_data)
            
            # DBSCAN clustering  
            dbscan = DBSCAN(eps=0.3, min_samples=5)
            dbscan_labels = dbscan.fit_predict(scaled_data)
            
            # Calculate metrics
            kmeans_silhouette = silhouette_score(scaled_data, kmeans_labels)
            dbscan_silhouette = silhouette_score(scaled_data, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1
            
            results = {
                'kmeans': {
                    'labels': kmeans_labels.tolist() if hasattr(kmeans_labels, 'tolist') else list(kmeans_labels),
                    'silhouette_score': float(kmeans_silhouette),
                    'n_clusters': int(len(set(kmeans_labels)))
                },
                'dbscan': {
                    'labels': dbscan_labels.tolist() if hasattr(dbscan_labels, 'tolist') else list(dbscan_labels),
                    'silhouette_score': float(dbscan_silhouette),
                    'n_clusters': int(len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0))
                },
                'features_used': list(clustering_features)
            }
            
            logging.info(f"Enhanced clustering completed")
            logging.info(f"K-Means: {results['kmeans']['n_clusters']} clusters, Silhouette: {kmeans_silhouette:.3f}")
            logging.info(f"DBSCAN: {results['dbscan']['n_clusters']} clusters, Silhouette: {dbscan_silhouette:.3f}")
            
            return results
            
        except Exception as e:
            logging.error(f"Error in enhanced clustering: {e}")
            return None
    
    def get_meta_insights(self):
        """Generate insights from hero meta data"""
        if self.hero_meta_data is None:
            return None
            
        insights = {
            'top_tier_heroes': self.hero_meta_data[
                self.hero_meta_data['hero_tier'].isin(['S+', 'S'])
            ]['hero_name'].tolist(),
            
            'most_banned': [
                {'hero_name': row['hero_name'], 'ban_rate_percent': row['ban_rate_percent']}
                for _, row in self.hero_meta_data.nlargest(5, 'ban_rate_percent').iterrows()
            ],
            
            'highest_winrate': [
                {'hero_name': row['hero_name'], 'win_rate_percent': row['win_rate_percent']}
                for _, row in self.hero_meta_data.nlargest(5, 'win_rate_percent').iterrows()
            ],
            
            'role_distribution': self.hero_meta_data['role'].value_counts().to_dict(),
            
            'meta_score_avg': self.hero_meta_data['meta_score'].mean()
        }
        
        return insights