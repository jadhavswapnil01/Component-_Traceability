"""
Data Loader Utilities
Handles loading and validation of manufacturing data
"""

import pandas as pd
import os
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Load and validate manufacturing data from CSV files"""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        self.required_files = {
            'suppliers': ['supplier_id', 'name', 'country', 'quality_rating'],
            'parts': ['part_id', 'name', 'category', 'spec_weight_g'],
            'stations': ['station_id', 'name', 'category'], # <-- ADD THIS LINE
            'batches': ['batch_id', 'supplier_id', 'part_id', 'quantity'],
            'vehicles': ['vin', 'model', 'assembly_date'],
            'assembly_events': ['vin', 'batch_id', 'part_id', 'assembly_timestamp'],
            'qc_inspections': ['vin', 'inspector_id', 'passed'],
            'failures': ['vin', 'reported_date', 'failure_mode']
        }
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all manufacturing data from CSV files"""
        logger.info(f"Loading data from {self.data_dir}...")
        
        datasets = {}
        
        for dataset_name, required_columns in self.required_files.items():
            filepath = os.path.join(self.data_dir, f"{dataset_name}.csv")
            
            if not os.path.exists(filepath):
                logger.error(f"Required file not found: {filepath}")
                raise FileNotFoundError(f"Missing required data file: {filepath}")
            
            try:
                df = pd.read_csv(filepath)
                
                # Validate required columns
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    logger.error(f"Missing columns in {dataset_name}: {missing_cols}")
                    raise ValueError(f"Missing required columns in {dataset_name}: {missing_cols}")
                
                # Basic data validation
                if len(df) == 0:
                    logger.warning(f"Empty dataset: {dataset_name}")
                else:
                    logger.info(f"✓ Loaded {len(df):,} {dataset_name}")
                
                datasets[dataset_name] = df
                
            except Exception as e:
                logger.error(f"Error loading {dataset_name}: {str(e)}")
                raise
        
        # Perform cross-dataset validation
        self._validate_data_integrity(datasets)
        
        logger.info("✓ All data loaded successfully")
        return datasets
    
    def _validate_data_integrity(self, datasets: Dict[str, pd.DataFrame]):
        """Validate relationships between datasets"""
        logger.info("Validating data integrity...")
        
        # Check supplier references
        supplier_ids = set(datasets['suppliers']['supplier_id'])
        batch_suppliers = set(datasets['batches']['supplier_id'])
        missing_suppliers = batch_suppliers - supplier_ids
        if missing_suppliers:
            logger.warning(f"Batches reference missing suppliers: {missing_suppliers}")
        
        # Check part references
        part_ids = set(datasets['parts']['part_id'])
        batch_parts = set(datasets['batches']['part_id'])
        missing_parts = batch_parts - part_ids
        if missing_parts:
            logger.warning(f"Batches reference missing parts: {missing_parts}")
        
        # Check vehicle references
        vehicle_vins = set(datasets['vehicles']['vin'])
        assembly_vins = set(datasets['assembly_events']['vin'])
        missing_vehicles = assembly_vins - vehicle_vins
        if missing_vehicles:
            logger.warning(f"Assembly events reference missing vehicles: {len(missing_vehicles)} VINs")
        
        # Check batch references in assembly
        batch_ids = set(datasets['batches']['batch_id'])
        assembly_batches = set(datasets['assembly_events']['batch_id'])
        missing_batches = assembly_batches - batch_ids
        if missing_batches:
            logger.warning(f"Assembly events reference missing batches: {len(missing_batches)} batches")
        
        logger.info("✓ Data integrity validation complete")
    
    def get_data_summary(self, datasets: Dict[str, pd.DataFrame]) -> Dict:
        """Generate summary statistics for loaded datasets"""
        summary = {
            'datasets': {},
            'relationships': {},
            'date_ranges': {}
        }
        
        # Dataset sizes
        for name, df in datasets.items():
            summary['datasets'][name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            }
        
        # Key relationships
        if 'batches' in datasets:
            summary['relationships']['suppliers_per_part'] = (
                datasets['batches'].groupby('part_id')['supplier_id'].nunique().mean()
            )
            summary['relationships']['batches_per_supplier'] = (
                datasets['batches'].groupby('supplier_id').size().mean()
            )
        
        if 'assembly_events' in datasets:
            summary['relationships']['parts_per_vehicle'] = (
                datasets['assembly_events'].groupby('vin').size().mean()
            )
        
        # Date ranges
        if 'batches' in datasets and 'manufacture_date' in datasets['batches'].columns:
            dates = pd.to_datetime(datasets['batches']['manufacture_date'])
            summary['date_ranges']['batches'] = {
                'start': dates.min().isoformat() if not dates.empty else None,
                'end': dates.max().isoformat() if not dates.empty else None,
                'span_days': (dates.max() - dates.min()).days if not dates.empty else None
            }
        
        if 'vehicles' in datasets and 'assembly_date' in datasets['vehicles'].columns:
            dates = pd.to_datetime(datasets['vehicles']['assembly_date'])
            summary['date_ranges']['vehicles'] = {
                'start': dates.min().isoformat() if not dates.empty else None,
                'end': dates.max().isoformat() if not dates.empty else None,
                'span_days': (dates.max() - dates.min()).days if not dates.empty else None
            }
        
        return summary
    
    def export_data_sample(self, datasets: Dict[str, pd.DataFrame], output_dir: str, sample_size: int = 100):
        """Export sample of data for inspection"""
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in datasets.items():
            if len(df) > sample_size:
                sample_df = df.sample(n=sample_size, random_state=42)
            else:
                sample_df = df.copy()
            
            sample_path = os.path.join(output_dir, f"{name}_sample.csv")
            sample_df.to_csv(sample_path, index=False)
            logger.info(f"✓ Exported {len(sample_df)} {name} samples to {sample_path}")
    
    def validate_data_quality(self, datasets: Dict[str, pd.DataFrame]) -> Dict:
        """Perform comprehensive data quality assessment"""
        quality_report = {
            'overall_score': 0,
            'issues': [],
            'dataset_scores': {}
        }
        
        total_score = 0
        dataset_count = 0
        
        for name, df in datasets.items():
            dataset_score = 100  # Start with perfect score
            dataset_issues = []
            
            # Check for missing values
            missing_pct = (df.isnull().sum() / len(df) * 100)
            critical_missing = missing_pct[missing_pct > 10]  # More than 10% missing
            if not critical_missing.empty:
                dataset_score -= min(30, len(critical_missing) * 10)
                dataset_issues.append(f"High missing values: {critical_missing.to_dict()}")
            
            # Check for duplicate records
            if df.duplicated().any():
                dup_count = df.duplicated().sum()
                dataset_score -= min(20, dup_count / len(df) * 100)
                dataset_issues.append(f"Duplicate records: {dup_count}")
            
            # Dataset-specific validations
            if name == 'batches':
                # Check QC pass rates
                if 'qc_pass_rate' in df.columns:
                    invalid_qc = ((df['qc_pass_rate'] < 0) | (df['qc_pass_rate'] > 1)).sum()
                    if invalid_qc > 0:
                        dataset_score -= 15
                        dataset_issues.append(f"Invalid QC pass rates: {invalid_qc} batches")
                
                # Check weight deviations
                if 'weight_mean_g' in df.columns and 'spec_weight_g' in datasets.get('parts', pd.DataFrame()).columns:
                    # This would require merging, simplified for now
                    pass
            
            elif name == 'vehicles':
                # Check VIN format (simplified)
                if 'vin' in df.columns:
                    invalid_vins = df['vin'].str.len() != 10  # Assuming 10-char VINs
                    if invalid_vins.any():
                        dataset_score -= 10
                        dataset_issues.append(f"Invalid VIN formats: {invalid_vins.sum()}")
            
            quality_report['dataset_scores'][name] = {
                'score': max(0, dataset_score),
                'issues': dataset_issues
            }
            
            total_score += max(0, dataset_score)
            dataset_count += 1
        
        quality_report['overall_score'] = total_score / dataset_count if dataset_count > 0 else 0
        
        # Collect all issues
        for dataset_info in quality_report['dataset_scores'].values():
            quality_report['issues'].extend(dataset_info['issues'])
        
        return quality_report

class DataPreprocessor:
    """Preprocess raw data for analysis and modeling"""
    
    def __init__(self):
        self.date_columns = ['manufacture_date', 'assembly_date', 'reported_date', 'inspection_date']
    
    def preprocess_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply preprocessing to all datasets"""
        logger.info("Preprocessing datasets...")
        
        processed = {}
        
        for name, df in datasets.items():
            processed_df = df.copy()
            
            # Convert date columns
            for date_col in self.date_columns:
                if date_col in processed_df.columns:
                    processed_df[date_col] = pd.to_datetime(processed_df[date_col], errors='coerce')
            
            # Dataset-specific preprocessing
            if name == 'batches':
                processed_df = self._preprocess_batches(processed_df)
            elif name == 'vehicles':
                processed_df = self._preprocess_vehicles(processed_df)
            elif name == 'suppliers':
                processed_df = self._preprocess_suppliers(processed_df)
            
            processed[name] = processed_df
            logger.info(f"✓ Preprocessed {name}")
        
        return processed
    
    def _preprocess_batches(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess batch data"""
        # Ensure numeric columns are properly typed
        numeric_columns = ['quantity', 'qc_pass_rate', 'weight_mean_g', 'weight_std_g', 'lead_time_days']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Cap extreme values
        if 'qc_pass_rate' in df.columns:
            df['qc_pass_rate'] = df['qc_pass_rate'].clip(0, 1)
        
        # Add derived columns
        if 'manufacture_date' in df.columns:
            df['manufacture_year'] = df['manufacture_date'].dt.year
            df['manufacture_month'] = df['manufacture_date'].dt.month
            df['manufacture_quarter'] = df['manufacture_date'].dt.quarter
        
        return df
    
    def _preprocess_vehicles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess vehicle data"""
        # Standardize categorical values
        if 'model' in df.columns:
            df['model'] = df['model'].str.strip().str.title()
        
        if 'color' in df.columns:
            df['color'] = df['color'].str.strip().str.title()
        
        # Add derived date features
        if 'assembly_date' in df.columns:
            df['assembly_year'] = df['assembly_date'].dt.year
            df['assembly_month'] = df['assembly_date'].dt.month
            df['assembly_quarter'] = df['assembly_date'].dt.quarter
            df['assembly_day_of_week'] = df['assembly_date'].dt.dayofweek
        
        return df
    
    def _preprocess_suppliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess supplier data"""
        # Standardize country names
        if 'country' in df.columns:
            df['country'] = df['country'].str.strip().str.title()
        
        # Ensure quality rating is in valid range
        if 'quality_rating' in df.columns:
            df['quality_rating'] = df['quality_rating'].clip(0, 1)
        
        # Add supplier age if establishment year available
        if 'established_year' in df.columns:
            current_year = pd.Timestamp.now().year
            df['supplier_age_years'] = current_year - df['established_year']
        
        return df

def main():
    """Test data loading functionality"""
    loader = DataLoader()
    
    try:
        datasets = loader.load_all_data()
        summary = loader.get_data_summary(datasets)
        quality_report = loader.validate_data_quality(datasets)
        
        print("Data Summary:", summary)
        print("Quality Report:", quality_report)
        
        preprocessor = DataPreprocessor()
        processed_datasets = preprocessor.preprocess_datasets(datasets)
        
        logger.info("Data loading and preprocessing test complete")
        
    except Exception as e:
        logger.error(f"Data loading test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()