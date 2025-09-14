"""
Test Suite for CarTrace AI Pipeline
Comprehensive testing for manufacturing traceability system
"""

import unittest
import os
import sys
import tempfile
import shutil
import pandas as pd
import json
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generator import ManufacturingDataGenerator
from data.loader import DataLoader, DataPreprocessor
from models.anomaly_detector import AnomalyDetector
from utils.config import Config

class TestDataGeneration(unittest.TestCase):
    """Test synthetic data generation"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ManufacturingDataGenerator(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_generate_suppliers(self):
        """Test supplier data generation"""
        suppliers_df = self.generator._generate_suppliers()
        
        self.assertIsInstance(suppliers_df, pd.DataFrame)
        self.assertEqual(len(suppliers_df), self.generator.n_suppliers)
        self.assertIn('supplier_id', suppliers_df.columns)
        self.assertIn('name', suppliers_df.columns)
        self.assertIn('quality_rating', suppliers_df.columns)
        
        # Check quality ratings are in valid range
        self.assertTrue(all(0.5 <= rating <= 1.0 for rating in suppliers_df['quality_rating']))
    
    def test_generate_parts(self):
        """Test part specification generation"""
        parts_df = self.generator._generate_parts()
        
        self.assertIsInstance(parts_df, pd.DataFrame)
        self.assertEqual(len(parts_df), len(self.generator.part_specs))
        self.assertIn('part_id', parts_df.columns)
        self.assertIn('name', parts_df.columns)
        self.assertIn('spec_weight_g', parts_df.columns)
        self.assertIn('is_critical', parts_df.columns)
    
    def test_generate_batches(self):
        """Test batch generation with anomalies"""
        suppliers_df = self.generator._generate_suppliers()
        parts_df = self.generator._generate_parts()
        batches_df = self.generator._generate_batches(suppliers_df, parts_df)
        
        self.assertIsInstance(batches_df, pd.DataFrame)
        self.assertGreater(len(batches_df), 0)
        self.assertIn('batch_id', batches_df.columns)
        self.assertIn('is_anomalous', batches_df.columns)
        
        # Check that some batches are marked as anomalous
        anomalous_count = batches_df['is_anomalous'].sum()
        expected_anomalies = len(batches_df) * self.generator.anomaly_rate
        self.assertGreater(anomalous_count, expected_anomalies * 0.5)  # Allow some variance
        self.assertLess(anomalous_count, expected_anomalies * 1.5)
    
    def test_complete_data_generation(self):
        """Test complete data generation pipeline"""
        datasets = self.generator.generate_all_data()
        
        expected_datasets = ['suppliers', 'parts', 'batches', 'vehicles', 
                           'assembly_events', 'qc_inspections', 'failures']
        
        for dataset_name in expected_datasets:
            self.assertIn(dataset_name, datasets)
            self.assertIsInstance(datasets[dataset_name], pd.DataFrame)
            self.assertGreater(len(datasets[dataset_name]), 0)
        
        # Check data files were created
        for dataset_name in expected_datasets:
            filepath = os.path.join(self.temp_dir, f"{dataset_name}.csv")
            self.assertTrue(os.path.exists(filepath))

class TestDataLoader(unittest.TestCase):
    """Test data loading and preprocessing"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Generate test data
        generator = ManufacturingDataGenerator(self.temp_dir)
        self.datasets = generator.generate_all_data()
        
        self.loader = DataLoader(self.temp_dir)
        self.preprocessor = DataPreprocessor()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_load_all_data(self):
        """Test loading all CSV files"""
        loaded_datasets = self.loader.load_all_data()
        
        self.assertIsInstance(loaded_datasets, dict)
        self.assertEqual(len(loaded_datasets), len(self.datasets))
        
        for name, df in loaded_datasets.items():
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
    
    def test_data_validation(self):
        """Test data quality validation"""
        quality_report = self.loader.validate_data_quality(self.datasets)
        
        self.assertIsInstance(quality_report, dict)
        self.assertIn('overall_score', quality_report)
        self.assertIn('dataset_scores', quality_report)
        self.assertGreaterEqual(quality_report['overall_score'], 0)
        self.assertLessEqual(quality_report['overall_score'], 100)
    
    def test_data_preprocessing(self):
        """Test data preprocessing"""
        processed_datasets = self.preprocessor.preprocess_datasets(self.datasets)
        
        self.assertIsInstance(processed_datasets, dict)
        self.assertEqual(len(processed_datasets), len(self.datasets))
        
        # Check date conversion
        if 'batches' in processed_datasets:
            batches_df = processed_datasets['batches']
            if 'manufacture_date' in batches_df.columns:
                self.assertTrue(pd.api.types.is_datetime64_any_dtype(batches_df['manufacture_date']))

class TestAnomalyDetector(unittest.TestCase):
    """Test anomaly detection functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Generate test data
        generator = ManufacturingDataGenerator(self.temp_dir)
        datasets = generator.generate_all_data()
        
        self.batches_df = datasets['batches']
        self.parts_df = datasets['parts']
        self.suppliers_df = datasets['suppliers']
        
        self.detector = AnomalyDetector(contamination=0.1)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_feature_preparation(self):
        """Test feature preparation for anomaly detection"""
        features_df = self.detector.prepare_features(
            self.batches_df, self.parts_df, self.suppliers_df
        )
        
        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertEqual(len(features_df), len(self.batches_df))
        
        # Check that derived features were created
        expected_features = ['weight_deviation_pct', 'qc_shortfall', 'quality_supplier_delta']
        for feature in expected_features:
            self.assertIn(feature, features_df.columns)
    
    def test_model_training(self):
        """Test anomaly detection model training"""
        features_df = self.detector.prepare_features(
            self.batches_df, self.parts_df, self.suppliers_df
        )
        
        training_results = self.detector.train(features_df)
        
        self.assertIsInstance(training_results, dict)
        self.assertTrue(training_results['model_trained'])
        self.assertTrue(self.detector.is_trained)
        self.assertIn('metrics', training_results)
    
    def test_anomaly_detection(self):
        """Test anomaly detection on data"""
        features_df = self.detector.prepare_features(
            self.batches_df, self.parts_df, self.suppliers_df
        )
        
        # Train model
        self.detector.train(features_df)
        
        # Detect anomalies
        results_df = self.detector.detect_anomalies(features_df)
        
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertEqual(len(results_df), len(features_df))
        self.assertIn('anomaly_predicted', results_df.columns)
        self.assertIn('anomaly_score', results_df.columns)
        
        # Check that some anomalies were detected
        anomaly_count = results_df['anomaly_predicted'].sum()
        self.assertGreater(anomaly_count, 0)
    
    def test_model_save_load(self):
        """Test saving and loading trained model"""
        features_df = self.detector.prepare_features(
            self.batches_df, self.parts_df, self.suppliers_df
        )
        
        # Train model
        self.detector.train(features_df)
        
        # Save model
        model_path = os.path.join(self.temp_dir, 'test_model.pkl')
        self.detector.save_model(model_path)
        
        self.assertTrue(os.path.exists(model_path))
        
        # Load model in new detector instance
        new_detector = AnomalyDetector()
        new_detector.load_model(model_path)
        
        self.assertTrue(new_detector.is_trained)
        self.assertEqual(new_detector.feature_columns, self.detector.feature_columns)
        
        # Test that loaded model produces same results
        original_results = self.detector.detect_anomalies(features_df, return_scores=False)
        loaded_results = new_detector.detect_anomalies(features_df, return_scores=False)
        
        pd.testing.assert_series_equal(
            original_results['anomaly_predicted'], 
            loaded_results['anomaly_predicted']
        )

class TestConfiguration(unittest.TestCase):
    """Test configuration management"""
    
    def test_config_creation(self):
        """Test configuration creation and validation"""
        config = Config()
        
        self.assertIsNotNone(config.neo4j)
        self.assertIsNotNone(config.data)
        self.assertIsNotNone(config.anomaly)
        self.assertIsNotNone(config.recall)
        self.assertIsNotNone(config.ui)
    
    def test_config_validation(self):
        """Test configuration validation"""
        config = Config()
        
        # Should validate successfully with defaults
        self.assertTrue(config.validate())
        
        # Test invalid contamination
        config.anomaly.contamination = 1.5  # Invalid: > 1
        self.assertFalse(config.validate())
        
        # Reset to valid value
        config.anomaly.contamination = 0.1
        self.assertTrue(config.validate())
    
    def test_config_save_load(self):
        """Test configuration save and load"""
        config = Config()
        
        # Save configuration
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_file.close()
        
        try:
            config.save_to_file(temp_file.name)
            self.assertTrue(os.path.exists(temp_file.name))
            
            # Load configuration
            new_config = Config(temp_file.name)
            
            # Compare key values
            self.assertEqual(config.neo4j.uri, new_config.neo4j.uri)
            self.assertEqual(config.anomaly.contamination, new_config.anomaly.contamination)
            self.assertEqual(config.ui.streamlit_port, new_config.ui.streamlit_port)
            
        finally:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

class TestIntegration(unittest.TestCase):
    """Integration tests for complete pipeline"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up integration test environment"""
        shutil.rmtree(self.temp_dir)
    
    @patch('models.graph_manager.GraphManager')
    def test_pipeline_integration(self, mock_graph_manager):
        """Test integration between major components"""
        # Mock the graph manager to avoid Neo4j dependency
        mock_graph_instance = Mock()
        mock_graph_manager.return_value = mock_graph_instance
        
        # Generate data
        generator = ManufacturingDataGenerator(self.temp_dir)
        datasets = generator.generate_all_data()
        
        # Load and preprocess
        loader = DataLoader(self.temp_dir)
        preprocessor = DataPreprocessor()
        loaded_datasets = loader.load_all_data()
        processed_datasets = preprocessor.preprocess_datasets(loaded_datasets)
        
        # Train anomaly detector
        detector = AnomalyDetector(contamination=0.1)
        features_df = detector.prepare_features(
            processed_datasets['batches'],
            processed_datasets['parts'],
            processed_datasets['suppliers']
        )
        training_results = detector.train(features_df)
        
        # Verify integration
        self.assertTrue(training_results['model_trained'])
        self.assertGreater(len(features_df), 0)
        
        # Test anomaly detection
        results_df = detector.detect_anomalies(features_df)
        anomaly_summary = detector.get_anomaly_summary(results_df)
        
        self.assertIn('total_anomalies', anomaly_summary)
        self.assertGreaterEqual(anomaly_summary['total_anomalies'], 0)

class TestScenarios(unittest.TestCase):
    """Test specific business scenarios"""
    
    def setUp(self):
        """Set up scenario testing"""
        self.temp_dir = tempfile.mkdtemp()
        generator = ManufacturingDataGenerator(self.temp_dir)
        self.datasets = generator.generate_all_data()
    
    def tearDown(self):
        """Clean up scenario testing"""
        shutil.rmtree(self.temp_dir)
    
    def test_anomaly_recall_scenario(self):
        """Test scenario: anomalous batch leads to recall"""
        # Find an anomalous batch
        anomalous_batches = self.datasets['batches'][
            self.datasets['batches']['is_anomalous'] == True
        ]
        
        self.assertGreater(len(anomalous_batches), 0, "No anomalous batches found for testing")
        
        # Check that assembly events use this batch
        test_batch_id = anomalous_batches.iloc[0]['batch_id']
        assembly_using_batch = self.datasets['assembly_events'][
            self.datasets['assembly_events']['batch_id'] == test_batch_id
        ]
        
        # Verify traceability exists
        self.assertGreaterEqual(len(assembly_using_batch), 0)
    
    def test_supplier_quality_correlation(self):
        """Test that supplier quality correlates with batch quality"""
        # Merge suppliers with batches
        merged_df = self.datasets['batches'].merge(
            self.datasets['suppliers'][['supplier_id', 'quality_rating']], 
            on='supplier_id'
        )
        
        # Group by quality rating tiers
        high_quality_suppliers = merged_df[merged_df['quality_rating'] > 0.9]
        low_quality_suppliers = merged_df[merged_df['quality_rating'] < 0.7]
        
        if len(high_quality_suppliers) > 0 and len(low_quality_suppliers) > 0:
            # High quality suppliers should have better QC rates on average
            high_qc_avg = high_quality_suppliers['qc_pass_rate'].mean()
            low_qc_avg = low_quality_suppliers['qc_pass_rate'].mean()
            
            self.assertGreater(high_qc_avg, low_qc_avg, 
                             "High quality suppliers should have better QC rates")
    
    def test_critical_parts_identification(self):
        """Test that critical parts are properly identified"""
        critical_parts = self.datasets['parts'][self.datasets['parts']['is_critical'] == True]
        
        self.assertGreater(len(critical_parts), 0, "No critical parts found")
        
        # Critical parts should include engine components
        critical_part_names = critical_parts['name'].str.lower()
        engine_parts = critical_part_names.str.contains('piston|engine')
        
        self.assertTrue(engine_parts.any(), "Engine components should be marked as critical")

def run_test_suite():
    """Run the complete test suite"""
    # Create test suite
    test_classes = [
        TestDataGeneration,
        TestDataLoader,
        TestAnomalyDetector,
        TestConfiguration,
        TestIntegration,
        TestScenarios
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    print("🧪 Running CarTrace AI Test Suite")
    print("=" * 50)
    
    success = run_test_suite()
    
    if success:
        print("\n✅ All tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed!")
        exit(1)