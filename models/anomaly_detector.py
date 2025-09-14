"""
Anomaly Detection for Manufacturing Batches
Detects suspicious batches using statistical and ML methods
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from typing import Dict, List, Tuple, Optional
import pickle
import os
import json
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Detect anomalous manufacturing batches"""
    
    def __init__(self, contamination: float = 0.08):
        """
        Initialize anomaly detector
        
        Args:
            contamination: Expected proportion of anomalies (0.08 = 8%)
        """
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200, # More estimators can improve accuracy
            max_features=1.0
        )
        self.feature_columns = []
        self.is_trained = False
        self.feature_stats = {}
        
    def prepare_features(self, batches_df: pd.DataFrame, parts_df: pd.DataFrame, suppliers_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for anomaly detection with enhanced feature engineering."""
        logger.info("Preparing features for anomaly detection...")

        features_df = batches_df.copy()
        features_df = features_df.merge(parts_df[['part_id', 'spec_weight_g', 'complexity_level', 'is_critical', 'category']], on='part_id')
        features_df = features_df.merge(suppliers_df[['supplier_id', 'quality_rating', 'tier']], on='supplier_id')

        # --- NEW: Calculate supplier's average performance as a baseline ---
        # This is a safe way to create a baseline without data leakage from the target variable.
        supplier_avg_qc = features_df.groupby('supplier_id')['qc_pass_rate'].transform('mean')

        # --- ENHANCED FEATURE ENGINEERING ---
        features_df['weight_deviation_pct'] = ((features_df['weight_mean_g'] - features_df['spec_weight_g']) / features_df['spec_weight_g'])
        features_df['qc_shortfall'] = 1.0 - features_df['qc_pass_rate']

        # NEW: How does this batch compare to the supplier's usual performance?
        features_df['supplier_consistency_delta'] = features_df['qc_pass_rate'] - supplier_avg_qc

        # NEW: Interaction between part complexity and weight deviation. A small error on a complex part is more significant.
        features_df['complexity_x_deviation'] = features_df['complexity_level'] * features_df['weight_deviation_pct'].abs()

        # NEW: Interaction between supplier quality and this batch's quality. A good supplier having a bad batch is a strong signal.
        features_df['supplier_quality_interaction'] = features_df['quality_rating'] * features_df['qc_shortfall']

        # Time-based features
        features_df['manufacture_date'] = pd.to_datetime(features_df['manufacture_date'])
        features_df['days_since_manufacture'] = (datetime.now() - features_df['manufacture_date']).dt.days

        # Categorical encoding
        features_df['is_critical_int'] = features_df['is_critical'].astype(int)

        # Define the final feature set for the model
        self.feature_columns = [
            'qc_pass_rate',
            'weight_deviation_pct',
            'qc_shortfall',
            'quality_rating',
            'lead_time_days',
            'complexity_level',
            'tier',
            'is_critical_int',
            'supplier_consistency_delta',   # New feature
            'complexity_x_deviation',     # New feature
            'supplier_quality_interaction'  # New feature
        ]

        # Store feature statistics for later explanation
        self.feature_stats = {
            col: {
                'mean': features_df[col].mean(),
                'std': features_df[col].std(),
                'percentiles': {
                    'p75': features_df[col].quantile(0.75),
                    'p95': features_df[col].quantile(0.95),
                    'p99': features_df[col].quantile(0.99)
                }
            }
            for col in self.feature_columns
        }

        logger.info(f"✓ Prepared {len(self.feature_columns)} features for {len(features_df)} batches")
        return features_df
    
    def train(self, features_df: pd.DataFrame) -> Dict:
        """Train anomaly detection model"""
        logger.info("Training anomaly detection model...")
        
        # Prepare training data
        X = features_df[self.feature_columns].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train isolation forest
        self.isolation_forest.fit(X_scaled)
        self.is_trained = True
        
        # Get training predictions for evaluation
        predictions = self.isolation_forest.predict(X_scaled)
        anomaly_scores = self.isolation_forest.decision_function(X_scaled)
        
        # Convert predictions (-1 = anomaly, 1 = normal) to (1 = anomaly, 0 = normal)
        predictions_binary = (predictions == -1).astype(int)
        
        # Calculate training metrics if ground truth available
        metrics = {}
        if 'is_anomalous' in features_df.columns:
            ground_truth = features_df['is_anomalous'].astype(int)
            
            metrics = {
                'accuracy': (predictions_binary == ground_truth).mean(),
                'precision': np.sum((predictions_binary == 1) & (ground_truth == 1)) / max(np.sum(predictions_binary == 1), 1),
                'recall': np.sum((predictions_binary == 1) & (ground_truth == 1)) / max(np.sum(ground_truth == 1), 1),
                'auc_roc': roc_auc_score(ground_truth, -anomaly_scores) if len(np.unique(ground_truth)) > 1 else 0,
                'confusion_matrix': confusion_matrix(ground_truth, predictions_binary).tolist()
            }
            
            logger.info(f"✓ Training complete - Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, AUC: {metrics['auc_roc']:.3f}")
        else:
            logger.info("✓ Training complete (no ground truth available)")
        
        return {
            'model_trained': True,
            'n_features': len(self.feature_columns),
            'n_samples': len(X),
            'contamination': self.contamination,
            'metrics': metrics
        }
    
    def detect_anomalies(self, features_df: pd.DataFrame, return_scores: bool = True) -> pd.DataFrame:
        """Detect anomalies in batch data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before detecting anomalies")
        
        # Prepare features
        X = features_df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions and scores
        predictions = self.isolation_forest.predict(X_scaled)
        anomaly_scores = self.isolation_forest.decision_function(X_scaled)
        
        # Convert predictions
        is_anomalous = (predictions == -1)
        
        # Create results dataframe
        results_df = features_df.copy()
        results_df['anomaly_predicted'] = is_anomalous
        results_df['anomaly_score'] = anomaly_scores
        results_df['anomaly_confidence'] = self._calculate_confidence(anomaly_scores)
        
        if return_scores:
            # Add individual feature contributions
            results_df = self._add_feature_explanations(results_df, X)
        
        return results_df
    
    def _calculate_confidence(self, scores: np.ndarray) -> np.ndarray:
        """Calculate confidence scores (0-1) from anomaly scores"""
        # Normalize scores to 0-1 range
        min_score, max_score = scores.min(), scores.max()
        if max_score == min_score:
            return np.ones_like(scores) * 0.5
        
        normalized = (scores - min_score) / (max_score - min_score)
        
        # Invert so that lower scores (more anomalous) give higher confidence
        confidence = 1 - normalized
        return confidence
    
    def _add_feature_explanations(self, results_df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        """Add feature-level explanations for anomalies"""
        explanations = []
        
        for idx, row in results_df.iterrows():
            if row['anomaly_predicted']:
                explanation = self._explain_anomaly(X.iloc[idx])
                explanations.append(explanation)
            else:
                explanations.append([])
        
        results_df['explanation'] = explanations
        return results_df
    
    def _explain_anomaly(self, feature_row: pd.Series) -> List[Dict]:
        """Explain why a specific batch is anomalous."""
        explanations = []

        for feature in self.feature_columns:
            value = feature_row[feature]
            stats = self.feature_stats[feature]

            # Define "bad" features where a high value is an anomaly indicator
            high_value_anomaly_features = [
                'qc_shortfall', 'weight_deviation_pct', 'complexity_x_deviation',
                'supplier_quality_interaction', 'lead_time_days'
            ]

            # Check if the feature's value is in an extreme percentile
            if value > stats['percentiles']['p95'] and feature in high_value_anomaly_features:
                severity = "HIGH" if value > stats['percentiles']['p99'] else "MEDIUM"
                explanations.append({
                    'feature': feature,
                    'value': float(value),
                    'severity': severity,
                    'description': f"{feature} is unusually high ({value:.3f} > 95th percentile: {stats['percentiles']['p95']:.3f})"
                })

        # Sort by severity
        severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        explanations.sort(key=lambda x: severity_order.get(x['severity'], 3))

        return explanations
    
    def get_anomaly_summary(self, results_df: pd.DataFrame) -> Dict:
        """Generate summary statistics for detected anomalies"""
        anomalies = results_df[results_df['anomaly_predicted']]
        
        if len(anomalies) == 0:
            return {'total_anomalies': 0, 'message': 'No anomalies detected'}
        
        # Group by suspected root causes
        explanation_counts = {}
        for explanations in anomalies['explanation']:
            for exp in explanations:
                feature = exp['feature']
                severity = exp['severity']
                key = f"{feature}_{severity}"
                explanation_counts[key] = explanation_counts.get(key, 0) + 1
        
        # Top issues
        top_issues = sorted(explanation_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Supplier breakdown
        supplier_anomalies = anomalies.groupby('supplier_id').size().sort_values(ascending=False)
        
        # Part category breakdown
        category_anomalies = anomalies.groupby('category').size().sort_values(ascending=False)
        
        return {
            'total_anomalies': len(anomalies),
            'anomaly_rate': len(anomalies) / len(results_df),
            'avg_confidence': anomalies['anomaly_confidence'].mean(),
            'top_issues': [{'issue': issue, 'count': count} for issue, count in top_issues],
            'by_supplier': supplier_anomalies.head(5).to_dict(),
            'by_category': category_anomalies.to_dict(),
            'high_confidence_anomalies': len(anomalies[anomalies['anomaly_confidence'] > 0.8])
        }
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'scaler': self.scaler,
            'isolation_forest': self.isolation_forest,
            'feature_columns': self.feature_columns,
            'feature_stats': self.feature_stats,
            'contamination': self.contamination,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.scaler = model_data['scaler']
        self.isolation_forest = model_data['isolation_forest']
        self.feature_columns = model_data['feature_columns']
        self.feature_stats = model_data['feature_stats']
        self.contamination = model_data['contamination']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"✓ Model loaded from {filepath}")
    
    def batch_quality_report(self, features_df: pd.DataFrame, results_df: pd.DataFrame) -> Dict:
        """Generate comprehensive batch quality report"""
        
        # Overall statistics
        total_batches = len(features_df)
        anomalous_batches = len(results_df[results_df['anomaly_predicted']])
        
        # Quality metrics by supplier
        supplier_stats = features_df.groupby(['supplier_id', 'name']).agg({
            'qc_pass_rate': ['mean', 'std', 'min'],
            'weight_deviation_pct': lambda x: abs(x).mean(),
            'batch_id': 'count'
        }).round(4)
        
        supplier_stats.columns = ['qc_avg', 'qc_std', 'qc_min', 'weight_dev_avg', 'batch_count']
        supplier_stats = supplier_stats.reset_index()
        supplier_stats['anomaly_count'] = supplier_stats['supplier_id'].map(
            results_df[results_df['anomaly_predicted']].groupby('supplier_id').size()
        ).fillna(0).astype(int)
        supplier_stats['anomaly_rate'] = supplier_stats['anomaly_count'] / supplier_stats['batch_count']
        
        # Critical parts analysis
        critical_part_anomalies = results_df[
            (results_df['anomaly_predicted']) & (results_df['is_critical'])
        ]
        
        return {
            'summary': {
                'total_batches': total_batches,
                'anomalous_batches': anomalous_batches,
                'anomaly_rate': round(anomalous_batches / total_batches, 4),
                'critical_part_anomalies': len(critical_part_anomalies)
            },
            'supplier_performance': supplier_stats.to_dict('records'),
            'critical_alerts': critical_part_anomalies[['batch_id', 'part_id', 'name', 'supplier_id', 'anomaly_confidence']].to_dict('records'),
            'recommendations': self._generate_recommendations(results_df)
        }
    
    def _generate_recommendations(self, results_df: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations based on anomaly analysis"""
        recommendations = []
        
        anomalies = results_df[results_df['anomaly_predicted']]
        
        if len(anomalies) == 0:
            return ["No significant anomalies detected. Continue standard monitoring."]
        
        # High confidence anomalies
        high_conf = len(anomalies[anomalies['anomaly_confidence'] > 0.8])
        if high_conf > 0:
            recommendations.append(f"URGENT: {high_conf} high-confidence anomalous batches require immediate investigation")
        
        # Critical part anomalies
        critical_anomalies = len(anomalies[anomalies['is_critical']])
        if critical_anomalies > 0:
            recommendations.append(f"WARNING: {critical_anomalies} anomalous batches involve safety-critical parts")
        
        # Supplier-specific issues
        supplier_issues = anomalies.groupby('supplier_id').size()
        worst_supplier = supplier_issues.idxmax() if len(supplier_issues) > 0 else None
        if worst_supplier and supplier_issues[worst_supplier] > 2:
            recommendations.append(f"SUPPLIER REVIEW: Supplier {worst_supplier} has {supplier_issues[worst_supplier]} anomalous batches")
        
        # General recommendations
        anomaly_rate = len(anomalies) / len(results_df)
        if anomaly_rate > 0.15:
            recommendations.append("High overall anomaly rate suggests systemic quality issues")
        
        return recommendations

def main():
    """Test anomaly detection with sample data"""
    # This would normally be called from the main pipeline
    logger.info("Anomaly detector module ready")

if __name__ == "__main__":
    main()