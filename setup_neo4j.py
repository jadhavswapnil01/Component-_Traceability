"""
Neo4j Database Setup Script
Initialize and configure Neo4j for CarTrace AI system
"""

import os
import sys
import subprocess
import time
import requests
from py2neo import Graph
from utils.config import get_config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jSetup:
    """Setup and configure Neo4j database"""
    
    def __init__(self):
        self.config = get_config()
        self.neo4j_home = os.getenv('NEO4J_HOME', r'C:\neo4j')
        self.neo4j_bin = os.path.join(self.neo4j_home, 'bin')
        self.neo4j_conf_dir = os.path.join(self.neo4j_home, 'conf')
        self.neo4j_data_dir = os.path.join(self.neo4j_home, 'data')
        
        # Create a clean environment for subprocesses
        self.env = os.environ.copy()
        java_home = self.env.get('JAVA_HOME')
        if java_home:
            self.env['PATH'] = os.path.join(java_home, 'bin') + os.pathsep + self.env['PATH']
        
    def check_neo4j_installation(self) -> bool:
        """Check if Neo4j is properly installed"""
        logger.info("Checking Neo4j installation...")
        
        if not os.path.exists(self.neo4j_home):
            logger.error(f"Neo4j not found at {self.neo4j_home}")
            logger.error("Please install Neo4j Community Edition:")
            logger.error("1. Download from: https://neo4j.com/download-center/#community")
            logger.error(f"2. Extract to: {self.neo4j_home}")
            logger.error("3. Set NEO4J_HOME environment variable")
            return False
        
        neo4j_admin = os.path.join(self.neo4j_bin, 'neo4j-admin.bat')
        if not os.path.exists(neo4j_admin):
            logger.error(f"Neo4j admin tool not found: {neo4j_admin}")
            return False
        
        logger.info("✓ Neo4j installation found")
        return True
    
    def check_java_installation(self) -> bool:
        """Check if Java is installed and accessible"""
        logger.info("Checking Java installation...")
        
        # --- START of CHANGES ---
        # Force the script to use the java.exe from JAVA_HOME
        java_home = self.env.get('JAVA_HOME')
        if not java_home:
            logger.error("JAVA_HOME is not set. Please set it to your Java 21 directory.")
            return False
        
        # Construct the explicit path to the java executable
        java_exe = os.path.join(java_home, 'bin', 'java.exe')
        
        if not os.path.exists(java_exe):
            logger.error(f"Java executable not found at: {java_exe}")
            return False
        # --- END of CHANGES ---

        try:
            # Use the explicit path instead of just 'java'
            result = subprocess.run([java_exe, '-version'], 
                                    capture_output=True, 
                                    text=True, 
                                    timeout=10,
                                    env=self.env)
            
            if result.returncode == 0:
                java_version = result.stderr.split('\n')[0] if result.stderr else result.stdout.split('\n')[0]
                logger.info(f"✓ Java found: {java_version}")
                return True
            else:
                logger.error("Java not found or not accessible")
                return False
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.error("Java not found in PATH")
            logger.error("Please install Java 11+ and add to PATH")
            return False
    
    def setup_neo4j_config(self):
        """Configure Neo4j settings"""
        logger.info("Configuring Neo4j...")
        
        config_file = os.path.join(self.neo4j_conf_dir, 'neo4j.conf')
        
        # Default configuration for development
        neo4j_config = f"""
# CarTrace AI Neo4j Configuration
# Server configuration
server.default_listen_address=0.0.0.0
server.bolt.listen_address=:7687
server.http.listen_address=:7474
server.https.listen_address=:7473

# Memory settings (adjust based on available RAM)
server.memory.heap.initial_size=2G
server.memory.heap.max_size=5G
server.memory.pagecache.size=3G

dbms.memory.transaction.total.max=0

# Security settings
dbms.security.auth_enabled=true
dbms.security.procedures.unrestricted=gds.*,apoc.*

# Performance settings (Corrected from dbms.* to db.*)
db.checkpoint.interval.time=15m
db.checkpoint.interval.tx=100000

# Logging (Corrected from dbms.* to db.*)
db.logs.query.enabled=INFO
db.logs.query.threshold=1s

# Data directory
server.directories.data={self.neo4j_data_dir.replace(os.sep, '/')}

# Import directory for CSV loading
server.directories.import=import

# Allow CSV import from any location (development only)
dbms.security.allow_csv_import_from_file_urls=true
"""
        
        try:
            with open(config_file, 'w') as f:
                f.write(neo4j_config)
            logger.info(f"✓ Neo4j configuration written to {config_file}")
        except Exception as e:
            logger.error(f"Failed to write Neo4j configuration: {str(e)}")
            raise
    
    def set_initial_password(self):
        """Set initial password for Neo4j"""
        logger.info("Setting Neo4j initial password...")
        
        neo4j_admin = os.path.join(self.neo4j_bin, 'neo4j-admin.bat')
        
        try:
            # Stop Neo4j if running
            self.stop_neo4j()
            time.sleep(2)
            
            # Set initial password
            cmd = [neo4j_admin, 'dbms', 'set-initial-password', self.config.neo4j.password]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=self.env) # Add this
            
            if result.returncode == 0:
                logger.info("✓ Neo4j initial password set")
            else:
                # Password might already be set
                if "already exists" in result.stderr or "already exists" in result.stdout:
                    logger.info("✓ Neo4j password already configured")
                else:
                    logger.warning(f"Password setting result: {result.stderr}")
                    
        except subprocess.TimeoutExpired:
            logger.error("Timeout setting Neo4j password")
        except Exception as e:
            logger.error(f"Error setting Neo4j password: {str(e)}")
    
    def start_neo4j(self) -> bool:
        """Start Neo4j service"""
        logger.info("Starting Neo4j...")
        
        neo4j_cmd = os.path.join(self.neo4j_bin, 'neo4j.bat')
        
        try:
            # Start Neo4j in background
            subprocess.Popen([neo4j_cmd, 'console'], 
                 creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0,
                 env=self.env) # Change os.environ to self.env
            
            # Wait for Neo4j to start
            max_attempts = 30
            for attempt in range(max_attempts):
                try:
                    response = requests.get('http://localhost:7474', timeout=5)
                    if response.status_code == 200:
                        logger.info("✓ Neo4j started successfully")
                        return True
                except requests.RequestException:
                    pass
                
                time.sleep(2)
                logger.info(f"Waiting for Neo4j to start... ({attempt + 1}/{max_attempts})")
            
            logger.error("Neo4j failed to start within timeout")
            return False
            
        except Exception as e:
            logger.error(f"Error starting Neo4j: {str(e)}")
            return False
    
    def stop_neo4j(self):
        """Stop Neo4j service"""
        neo4j_cmd = os.path.join(self.neo4j_bin, 'neo4j.bat')
        
        try:
            subprocess.run([neo4j_cmd, 'stop'], capture_output=True, timeout=30, env=self.env) # Add this
            logger.info("Neo4j stopped")
        except subprocess.TimeoutExpired:
            logger.warning("Timeout stopping Neo4j")
        except Exception as e:
            logger.warning(f"Error stopping Neo4j: {str(e)}")
    
    def test_connection(self) -> bool:
        """Test connection to Neo4j database"""
        logger.info("Testing Neo4j connection...")
        
        try:
            graph = Graph(self.config.neo4j.uri, 
                         auth=(self.config.neo4j.username, self.config.neo4j.password))
            
            # Test basic query
            result = graph.run("RETURN 'Connection successful' as message").data()
            if result and result[0]['message'] == 'Connection successful':
                logger.info("✓ Neo4j connection test successful")
                
                # Get Neo4j version
                version_result = graph.run("CALL dbms.components() YIELD name, versions").data()
                if version_result:
                    neo4j_info = next((comp for comp in version_result if comp['name'] == 'Neo4j Kernel'), None)
                    if neo4j_info:
                        version = neo4j_info['versions'][0]
                        logger.info(f"✓ Neo4j version: {version}")
                
                return True
            else:
                logger.error("Neo4j connection test failed")
                return False
                
        except Exception as e:
            logger.error(f"Neo4j connection failed: {str(e)}")
            logger.error("Please check:")
            logger.error("1. Neo4j is running (http://localhost:7474)")
            logger.error(f"2. Username/password: {self.config.neo4j.username}/{self.config.neo4j.password}")
            logger.error("3. Firewall settings allow connections to port 7687")
            return False
    
    def create_database_schema(self):
        """Create indexes and constraints for optimal performance"""
        logger.info("Creating database schema...")
        
        try:
            graph = Graph(self.config.neo4j.uri, 
                         auth=(self.config.neo4j.username, self.config.neo4j.password))
            
            # Create constraints and indexes
            schema_queries = [
                # Unique constraints
                "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Supplier) REQUIRE s.supplier_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Part) REQUIRE p.part_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (b:Batch) REQUIRE b.batch_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (v:Vehicle) REQUIRE v.vin IS UNIQUE",
                
                # Indexes for performance
                "CREATE INDEX IF NOT EXISTS FOR (b:Batch) ON (b.manufacture_date)",
                "CREATE INDEX IF NOT EXISTS FOR (b:Batch) ON (b.is_anomalous)",
                "CREATE INDEX IF NOT EXISTS FOR (v:Vehicle) ON (v.assembly_date)",
                "CREATE INDEX IF NOT EXISTS FOR (f:Failure) ON (f.reported_date)",
                "CREATE INDEX IF NOT EXISTS FOR (qc:QCInspection) ON (qc.inspection_date)",
                
                # Composite indexes for common queries
                "CREATE INDEX IF NOT EXISTS FOR (b:Batch) ON (b.supplier_id, b.part_id)",
                "CREATE INDEX IF NOT EXISTS FOR (b:Batch) ON (b.is_anomalous, b.anomaly_type)"
            ]
            
            for query in schema_queries:
                try:
                    graph.run(query)
                except Exception as e:
                    logger.warning(f"Schema query failed (may already exist): {query[:50]}... - {str(e)}")
            
            logger.info("✓ Database schema created/updated")
            
        except Exception as e:
            logger.error(f"Error creating database schema: {str(e)}")
            raise
    
    def run_setup(self) -> bool:
        """Run complete Neo4j setup process"""
        logger.info("=== Starting Neo4j Setup ===")
        
        # Check prerequisites
        if not self.check_java_installation():
            return False
        
        if not self.check_neo4j_installation():
            return False
        
        try:
            # Configure Neo4j
            self.setup_neo4j_config()
            
            # Set initial password
            self.set_initial_password()
            
            # Start Neo4j
            if not self.start_neo4j():
                return False
            
            # Test connection
            if not self.test_connection():
                return False
            
            # Create schema
            self.create_database_schema()
            
            logger.info("=== Neo4j Setup Complete ===")
            logger.info(f"Neo4j Browser: http://localhost:7474")
            logger.info(f"Bolt URI: {self.config.neo4j.uri}")
            logger.info(f"Username: {self.config.neo4j.username}")
            logger.info(f"Password: {self.config.neo4j.password}")
            logger.info("Ready for CarTrace AI system!")
            
            return True
            
        except Exception as e:
            logger.error(f"Neo4j setup failed: {str(e)}")
            return False

def main():
    """Main setup function"""
    print("CarTrace AI - Neo4j Database Setup")
    print("=" * 40)
    
    setup = Neo4jSetup()
    
    if setup.run_setup():
        print("\n✓ Neo4j setup completed successfully!")
        print("\nNext steps:")
        print("1. Run: python run_pipeline.py")
        print("2. Launch dashboard: streamlit run app.py")
        return 0
    else:
        print("\n✗ Neo4j setup failed!")
        print("\nPlease check the error messages above and:")
        print("1. Ensure Java 11+ is installed")
        print("2. Download Neo4j Community Edition")
        print("3. Set NEO4J_HOME environment variable")
        print("4. Check firewall/antivirus settings")
        return 1

if __name__ == "__main__":
    sys.exit(main())