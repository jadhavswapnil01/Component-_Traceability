"""
Configuration Management
Centralized configuration for the CarTrace AI system
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Neo4jConfig:
    """Neo4j database configuration"""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "cartrace123"
    database: str = "neo4j"
    connection_timeout: int = 30
    max_retry_time: int = 30

@dataclass
class DataConfig:
    """Data processing configuration"""
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    model_dir: str = "models/saved"
    anomaly_contamination: float = 0.08
    test_split_ratio: float = 0.2
    random_seed: int = 42

@dataclass
class AnomalyDetectionConfig:
    """Anomaly detection model configuration"""
    contamination: float = 0.08
    n_estimators: int = 150
    max_features: float = 1.0
    confidence_threshold: float = 0.7
    critical_part_threshold: float = 0.5
    
@dataclass
class RecallConfig:
    """Recall simulation configuration"""
    base_recall_cost_per_vehicle: float = 500.0
    critical_part_multiplier: float = 2.0
    high_severity_multiplier: float = 1.5
    communication_cost_base: float = 10000.0
    logistics_cost_per_vehicle: float = 150.0

@dataclass
class UIConfig:
    """User interface configuration"""
    streamlit_port: int = 8501
    page_title: str = "CarTrace AI - Manufacturing Traceability"
    page_icon: str = "🚗"
    layout: str = "wide"
    sidebar_state: str = "expanded"

class Config:
    """Main configuration class"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.neo4j = Neo4jConfig()
        self.data = DataConfig()
        self.anomaly = AnomalyDetectionConfig()
        self.recall = RecallConfig()
        self.ui = UIConfig()
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        
        # Override with environment variables
        self._load_from_env()
        
        # Ensure directories exist
        self._create_directories()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        
        # Neo4j configuration
        if os.getenv('NEO4J_URI'):
            self.neo4j.uri = os.getenv('NEO4J_URI')
        if os.getenv('NEO4J_USERNAME'):
            self.neo4j.username = os.getenv('NEO4J_USERNAME')
        if os.getenv('NEO4J_PASSWORD'):
            self.neo4j.password = os.getenv('NEO4J_PASSWORD')
        
        # Data configuration
        if os.getenv('DATA_DIR'):
            self.data.raw_data_dir = os.path.join(os.getenv('DATA_DIR'), 'raw')
            self.data.processed_data_dir = os.path.join(os.getenv('DATA_DIR'), 'processed')
        
        # Anomaly detection
        if os.getenv('ANOMALY_CONTAMINATION'):
            self.anomaly.contamination = float(os.getenv('ANOMALY_CONTAMINATION'))
        
        # UI configuration
        if os.getenv('STREAMLIT_PORT'):
            self.ui.streamlit_port = int(os.getenv('STREAMLIT_PORT'))
            
        logger.info("✓ Configuration loaded from environment variables")
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.data.raw_data_dir,
            self.data.processed_data_dir,
            self.data.model_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        logger.info("✓ Required directories created/verified")
    
    def load_from_file(self, config_file: str):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configurations
            if 'neo4j' in config_data:
                for key, value in config_data['neo4j'].items():
                    if hasattr(self.neo4j, key):
                        setattr(self.neo4j, key, value)
            
            if 'data' in config_data:
                for key, value in config_data['data'].items():
                    if hasattr(self.data, key):
                        setattr(self.data, key, value)
            
            if 'anomaly' in config_data:
                for key, value in config_data['anomaly'].items():
                    if hasattr(self.anomaly, key):
                        setattr(self.anomaly, key, value)
            
            if 'recall' in config_data:
                for key, value in config_data['recall'].items():
                    if hasattr(self.recall, key):
                        setattr(self.recall, key, value)
            
            if 'ui' in config_data:
                for key, value in config_data['ui'].items():
                    if hasattr(self.ui, key):
                        setattr(self.ui, key, value)
            
            logger.info(f"✓ Configuration loaded from {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_file}: {str(e)}")
            raise
    
    def save_to_file(self, config_file: str):
        """Save current configuration to JSON file"""
        config_data = {
            'neo4j': {
                'uri': self.neo4j.uri,
                'username': self.neo4j.username,
                'password': self.neo4j.password,
                'database': self.neo4j.database,
                'connection_timeout': self.neo4j.connection_timeout,
                'max_retry_time': self.neo4j.max_retry_time
            },
            'data': {
                'raw_data_dir': self.data.raw_data_dir,
                'processed_data_dir': self.data.processed_data_dir,
                'model_dir': self.data.model_dir,
                'anomaly_contamination': self.data.anomaly_contamination,
                'test_split_ratio': self.data.test_split_ratio,
                'random_seed': self.data.random_seed
            },
            'anomaly': {
                'contamination': self.anomaly.contamination,
                'n_estimators': self.anomaly.n_estimators,
                'max_features': self.anomaly.max_features,
                'confidence_threshold': self.anomaly.confidence_threshold,
                'critical_part_threshold': self.anomaly.critical_part_threshold
            },
            'recall': {
                'base_recall_cost_per_vehicle': self.recall.base_recall_cost_per_vehicle,
                'critical_part_multiplier': self.recall.critical_part_multiplier,
                'high_severity_multiplier': self.recall.high_severity_multiplier,
                'communication_cost_base': self.recall.communication_cost_base,
                'logistics_cost_per_vehicle': self.recall.logistics_cost_per_vehicle
            },
            'ui': {
                'streamlit_port': self.ui.streamlit_port,
                'page_title': self.ui.page_title,
                'page_icon': self.ui.page_icon,
                'layout': self.ui.layout,
                'sidebar_state': self.ui.sidebar_state
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"✓ Configuration saved to {config_file}")
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        issues = []
        
        # Validate Neo4j configuration
        if not self.neo4j.uri:
            issues.append("Neo4j URI is required")
        if not self.neo4j.username:
            issues.append("Neo4j username is required")
        if not self.neo4j.password:
            issues.append("Neo4j password is required")
        
        # Validate data configuration
        if not self.data.raw_data_dir:
            issues.append("Raw data directory is required")
        
        # Validate anomaly detection configuration
        if not 0 < self.anomaly.contamination < 1:
            issues.append("Anomaly contamination must be between 0 and 1")
        if self.anomaly.n_estimators < 10:
            issues.append("Number of estimators should be at least 10")
        
        # Validate recall configuration
        if self.recall.base_recall_cost_per_vehicle < 0:
            issues.append("Base recall cost cannot be negative")
        
        # Validate UI configuration
        if not 1000 <= self.ui.streamlit_port <= 65535:
            issues.append("Streamlit port must be between 1000 and 65535")
        
        if issues:
            for issue in issues:
                logger.error(f"Configuration issue: {issue}")
            return False
        
        logger.info("✓ Configuration validation passed")
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            'neo4j': {
                'uri': self.neo4j.uri,
                'username': self.neo4j.username,
                'database': self.neo4j.database
            },
            'data': {
                'raw_data_dir': self.data.raw_data_dir,
                'model_dir': self.data.model_dir,
                'anomaly_contamination': self.anomaly.contamination
            },
            'anomaly_detection': {
                'contamination': self.anomaly.contamination,
                'n_estimators': self.anomaly.n_estimators,
                'confidence_threshold': self.anomaly.confidence_threshold
            },
            'recall_simulation': {
                'base_cost_per_vehicle': self.recall.base_recall_cost_per_vehicle,
                'critical_part_multiplier': self.recall.critical_part_multiplier
            },
            'ui': {
                'port': self.ui.streamlit_port,
                'title': self.ui.page_title
            }
        }

# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get the global configuration instance"""
    return config

def load_config(config_file: Optional[str] = None) -> Config:
    """Load configuration from file"""
    global config
    config = Config(config_file)
    return config

def validate_config() -> bool:
    """Validate the current configuration"""
    return config.validate()

# Environment detection utilities
def is_development() -> bool:
    """Check if running in development environment"""
    return os.getenv('ENVIRONMENT', 'development').lower() == 'development'

def is_production() -> bool:
    """Check if running in production environment"""
    return os.getenv('ENVIRONMENT', 'development').lower() == 'production'

def get_log_level() -> str:
    """Get appropriate log level based on environment"""
    if is_development():
        return os.getenv('LOG_LEVEL', 'INFO')
    else:
        return os.getenv('LOG_LEVEL', 'WARNING')

def setup_logging():
    """Setup logging configuration"""
    log_level = getattr(logging, get_log_level().upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('cartrace_ai.log') if is_production() else logging.NullHandler()
        ]
    )
    
    logger.info(f"✓ Logging configured (level: {get_log_level()})")

if __name__ == "__main__":
    # Test configuration
    config = Config()
    print("Configuration Summary:")
    summary = config.get_summary()
    for section, settings in summary.items():
        print(f"\n{section.upper()}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
    
    # Test validation
    is_valid = config.validate()
    print(f"\nConfiguration valid: {is_valid}")
    
    # Test save/load
    config.save_to_file("test_config.json")
    new_config = Config("test_config.json")
    print("✓ Configuration save/load test passed")
    
    # Clean up
    if os.path.exists("test_config.json"):
        os.remove("test_config.json")