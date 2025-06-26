import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import logging

class ClusteringEngine:
    def __init__(self):
        self.models = {}
    
    def perform_clustering(self, data, algorithm, params):
        """Perform clustering with specified algorithm and parameters"""
        try:
            logging.info(f"Starting clustering with {algorithm}, data shape: {data.shape}")
            # Handle both DataFrame and numpy array inputs
            if hasattr(data, 'columns'):
                logging.info(f"Data columns: {data.columns.tolist()}")
            else:
                logging.info(f"Data is numpy array with {data.shape[1]} features")
            logging.info(f"Parameters: {params}")
            
            if algorithm == 'kmeans':
                return self._perform_kmeans(data, params)
            elif algorithm == 'dbscan':
                return self._perform_dbscan(data, params)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
                
        except Exception as e:
            logging.error(f"Error in clustering: {str(e)}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _perform_kmeans(self, data, params):
        """Perform K-Means clustering"""
        n_clusters = params.get('n_clusters', 4)
        
        # Validate parameters
        if n_clusters < 2 or n_clusters > 10:
            raise ValueError("n_clusters must be between 2 and 10")
        
        # Perform clustering
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = model.fit_predict(data)
        
        logging.info(f"KMeans completed: {len(labels)} data points, {len(set(labels))} clusters")
        
        # Calculate metrics
        metrics = self._calculate_metrics(data, labels)
        
        # Store model
        self.models['kmeans'] = model
        
        return {
            'labels': labels,
            'metrics': metrics,
            'n_clusters': n_clusters,
            'algorithm': 'K-Means'
        }
    
    def _perform_dbscan(self, data, params):
        """Perform DBSCAN clustering"""
        eps = params.get('eps', 0.5)
        min_samples = params.get('min_samples', 5)
        
        # Validate parameters with more flexible ranges for different dataset types
        if eps < 0.1 or eps > 2.0:
            raise ValueError("eps must be between 0.1 and 2.0")
        if min_samples < 2 or min_samples > 20:
            raise ValueError("min_samples must be between 2 and 20")
        
        # Adjust parameters based on dataset size
        if len(data) < 50:  # Small dataset adjustments
            eps = max(eps, 1.0)  # Increase eps for small datasets
            min_samples = max(2, min(min_samples, len(data) // 5))  # Adjust min_samples
        
        # Perform clustering
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(data)
        
        # Handle case where no clusters are found - be more permissive
        unique_labels = set(labels)
        if len(unique_labels) <= 1 and -1 in unique_labels:
            # All points are noise, try with more permissive parameters
            model_fallback = DBSCAN(eps=eps * 1.5, min_samples=max(2, min_samples - 1))
            labels = model_fallback.fit_predict(data)
            unique_labels = set(labels)
            
        if len(unique_labels) <= 1 and -1 in unique_labels:
            raise ValueError("DBSCAN found no clusters or only noise. Try adjusting parameters.")
        
        # Calculate metrics
        metrics = self._calculate_metrics(data, labels)
        
        # Store the successful model
        self.models['dbscan'] = model
        
        # Count actual clusters (excluding noise points labeled as -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        return {
            'labels': labels,
            'metrics': metrics,
            'n_clusters': n_clusters,
            'algorithm': 'DBSCAN'
        }
    
    def _calculate_metrics(self, data, labels):
        """Calculate clustering evaluation metrics"""
        metrics = {}
        
        try:
            # Only calculate metrics if we have more than one cluster
            unique_labels = set(labels)
            if len(unique_labels) > 1 and len(unique_labels) < len(data):
                # Silhouette Score
                if -1 in labels:  # DBSCAN with noise points
                    # Only calculate for non-noise points
                    mask = labels != -1
                    if mask.sum() > 1 and len(set(labels[mask])) > 1:
                        metrics['silhouette_score'] = round(
                            silhouette_score(data[mask], labels[mask]), 3
                        )
                    else:
                        metrics['silhouette_score'] = 0.0
                else:
                    metrics['silhouette_score'] = round(
                        silhouette_score(data, labels), 3
                    )
                
                # Davies-Bouldin Index
                if -1 in labels:
                    mask = labels != -1
                    if mask.sum() > 1 and len(set(labels[mask])) > 1:
                        metrics['davies_bouldin_score'] = round(
                            davies_bouldin_score(data[mask], labels[mask]), 3
                        )
                    else:
                        metrics['davies_bouldin_score'] = float('inf')
                else:
                    metrics['davies_bouldin_score'] = round(
                        davies_bouldin_score(data, labels), 3
                    )
            else:
                metrics['silhouette_score'] = 0.0
                metrics['davies_bouldin_score'] = float('inf')
                
        except Exception as e:
            logging.warning(f"Error calculating metrics: {str(e)}")
            metrics['silhouette_score'] = 0.0
            metrics['davies_bouldin_score'] = float('inf')
        
        return metrics
    
    def get_model(self, algorithm):
        """Get the trained model for specified algorithm"""
        return self.models.get(algorithm)
