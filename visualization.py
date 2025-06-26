import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import logging

class VisualizationEngine:
    def __init__(self):
        # Ultra high-contrast color palette for maximum visibility
        self.color_palette = [
            '#E74C3C',  # Strong Red
            '#2ECC71',  # Strong Green
            '#3498DB',  # Strong Blue
            '#F39C12',  # Strong Orange
            '#9B59B6',  # Strong Purple
            '#1ABC9C',  # Strong Turquoise
            '#E67E22',  # Strong Dark Orange
            '#34495E',  # Strong Dark Blue Grey
            '#FF1744',  # Neon Red
            '#00E676',  # Neon Green
            '#2196F3',  # Material Blue
            '#FF9800',  # Material Orange
            '#9C27B0',  # Material Purple
            '#00BCD4',  # Material Cyan
            '#795548',  # Brown
            '#607D8B'   # Blue Grey
        ]
    
    def create_scatter_plot(self, df, labels):
        """Create interactive scatter plot for clustering results"""
        try:
            logging.info(f"Creating scatter plot with {len(df)} data points and {len(set(labels))} unique labels")
            
            # Debug: Check data structure
            if 'win_rate' not in df.columns or 'avg_kda' not in df.columns:
                logging.error(f"Missing required columns. Available: {df.columns.tolist()}")
                return self._create_error_plot("Missing required columns for visualization")
            
            # Create scatter plot
            fig = go.Figure()
            
            unique_labels = sorted(set(labels))
            logging.info(f"Unique labels: {unique_labels}")
            
            for i, label in enumerate(unique_labels):
                # Handle both array and scalar labels properly
                if isinstance(labels, (list, np.ndarray)):
                    mask = np.array(labels) == label
                    mask_indices = np.where(mask)[0]
                else:
                    # Handle scalar case
                    mask_indices = [0] if labels == label else []
                
                if len(mask_indices) > 0:
                    cluster_data = df.iloc[mask_indices]
                else:
                    continue
                
                logging.info(f"Cluster {label}: {len(cluster_data)} points")
                
                # Handle noise points in DBSCAN (label = -1)
                if label == -1:
                    name = 'Noise'
                    color = 'gray'
                else:
                    name = f'Cluster {label + 1}'  # Start from 1 instead of 0
                    color = self.color_palette[i % len(self.color_palette)]
                
                # Extract data safely
                x_data = cluster_data['win_rate'].values
                y_data = cluster_data['avg_kda'].values
                
                logging.info(f"Cluster {label} - X range: {x_data.min():.3f} to {x_data.max():.3f}")
                logging.info(f"Cluster {label} - Y range: {y_data.min():.3f} to {y_data.max():.3f}")
                
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode='markers',
                    name=name,
                    marker=dict(
                        color=color,
                        size=10,
                        opacity=0.8,
                        line=dict(width=1, color='white')
                    ),
                    text=[f"Player {int(pid)}<br>Role: {role}<br>Freq: {freq:.2f}" 
                          for pid, role, freq in zip(
                              cluster_data['player_id'],
                              cluster_data['main_role'],
                              cluster_data['match_frequency']
                          )],
                    hovertemplate='<b>%{text}</b><br>' +
                                 'Win Rate: %{x:.3f}<br>' +
                                 'Avg KDA: %{y:.3f}<br>' +
                                 '<extra></extra>'
                ))
            
            fig.update_layout(
                title=dict(
                    text='Player Clustering Results',
                    x=0.5,
                    font=dict(size=16, color='white')
                ),
                xaxis=dict(
                    title='Win Rate (Normalized)',
                    range=[-0.1, 1.1],
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.3)',
                    color='white'
                ),
                yaxis=dict(
                    title='Average KDA (Normalized)',
                    range=[-0.1, 1.1],
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.3)',
                    color='white'
                ),
                hovermode='closest',
                template='plotly_dark',
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.01
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=60, r=80, t=60, b=60)
            )
            
            return fig.to_json()
            
        except Exception as e:
            logging.error(f"Error creating scatter plot: {str(e)}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return self._create_error_plot(str(e))
    
    def get_cluster_profiles(self, df, labels):
        """Get detailed profiles for each cluster"""
        try:
            profiles = []
            unique_labels = sorted(set(labels))
            
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                    
                mask = labels == label
                cluster_data = df[mask]
                
                if len(cluster_data) == 0:
                    continue
                
                # Calculate cluster statistics
                profile = {
                    'cluster_id': int(label) + 1,  # Start from 1 instead of 0
                    'size': len(cluster_data),
                    'percentage': round(len(cluster_data) / len(df) * 100, 1),
                    'avg_win_rate': round(cluster_data['win_rate'].mean(), 3),
                    'avg_kda': round(cluster_data['avg_kda'].mean(), 3),
                    'avg_match_frequency': round(cluster_data['match_frequency'].mean(), 3),
                    'dominant_role': self._get_dominant_role(cluster_data),
                    'representative_players': self._get_representative_players(cluster_data),
                    'radar_data': self._create_radar_data(cluster_data),
                    'role_distribution': self._get_role_distribution(cluster_data)
                }
                
                # Generate interpretive label
                profile['label'] = self._generate_cluster_label(profile)
                
                profiles.append(profile)
            
            return profiles
            
        except Exception as e:
            logging.error(f"Error creating cluster profiles: {str(e)}")
            return []
    
    def _get_dominant_role(self, cluster_data):
        """Get the dominant role in a cluster"""
        return cluster_data['main_role'].mode().iloc[0] if not cluster_data['main_role'].empty else 'Unknown'
    
    def _get_representative_players(self, cluster_data, n=3):
        """Get representative players from a cluster"""
        # Sort by win rate and get top players
        top_players = cluster_data.nlargest(n, 'win_rate')
        
        return [
            {
                'player_id': int(row['player_id']),
                'main_role': row['main_role'],
                'win_rate': round(row['win_rate'], 3),
                'avg_kda': round(row['avg_kda'], 3)
            }
            for _, row in top_players.iterrows()
        ]
    
    def _create_radar_data(self, cluster_data):
        """Create radar chart data for cluster characteristics"""
        try:
            # Calculate normalized averages for radar chart
            radar_data = {
                'win_rate': float(cluster_data['win_rate'].mean()),
                'avg_kda': float(cluster_data['avg_kda'].mean()),
                'match_frequency': float(cluster_data['match_frequency'].mean())
            }
            
            return radar_data
            
        except Exception as e:
            logging.error(f"Error creating radar data: {str(e)}")
            return {'win_rate': 0, 'avg_kda': 0, 'match_frequency': 0}
    
    def _get_role_distribution(self, cluster_data):
        """Get role distribution for a cluster"""
        try:
            role_counts = cluster_data['main_role'].value_counts()
            
            return [
                {
                    'role': role,
                    'count': int(count),
                    'percentage': round(count / len(cluster_data) * 100, 1)
                }
                for role, count in role_counts.items()
            ]
            
        except Exception as e:
            logging.error(f"Error getting role distribution: {str(e)}")
            return []
    
    def _generate_cluster_label(self, profile):
        """Generate interpretive label for cluster"""
        dominant_role = profile['dominant_role']
        avg_win_rate = profile['avg_win_rate']
        avg_kda = profile['avg_kda']
        
        # Determine performance level
        if avg_win_rate > 0.7:
            performance = "High WR"
        elif avg_win_rate > 0.5:
            performance = "Mid WR"
        else:
            performance = "Low WR"
        
        # Determine KDA level
        if avg_kda > 0.7:
            kda_level = "High KDA"
        elif avg_kda > 0.5:
            kda_level = "Mid KDA"
        else:
            kda_level = "Low KDA"
        
        return f"{performance} {dominant_role} Players ({kda_level})"
    
    def get_cluster_stats(self, df, labels, cluster_id):
        """Get statistics for a specific cluster"""
        try:
            mask = labels == cluster_id
            cluster_data = df[mask]
            
            if len(cluster_data) == 0:
                return {}
            
            stats = {
                'cluster_size': len(cluster_data),
                'avg_win_rate': round(cluster_data['win_rate'].mean(), 3),
                'avg_kda': round(cluster_data['avg_kda'].mean(), 3),
                'avg_match_frequency': round(cluster_data['match_frequency'].mean(), 3),
                'dominant_role': self._get_dominant_role(cluster_data)
            }
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting cluster stats: {str(e)}")
            return {}
    
    def _create_error_plot(self, error_message):
        """Create error plot when visualization fails"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Visualization Error: {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Visualization Error",
            template='plotly_dark',
            height=400
        )
        return fig.to_json()
