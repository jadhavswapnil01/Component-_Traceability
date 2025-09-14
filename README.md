# 🚗 CarTrace AI - Manufacturing Traceability & Anomaly Detection

A comprehensive manufacturing traceability system built with graph databases, machine learning, and modern web interfaces. CarTrace AI helps automotive manufacturers track components from supplier to final vehicle, detect quality anomalies, and simulate recalls with explainable AI.

## 🎯 Key Features

### 🔗 **Advanced Component Traceability**
- **End-to-end tracking**: From raw materials through multi-tier suppliers to final vehicles
- **Bidirectional queries**: Find all components in a vehicle OR all vehicles using a specific batch
- **Neo4j graph database**: Optimized for complex relationship queries with sub-second response
- **Visual representation**: Interactive graph views of component relationships and supply chains

### 🔍 **AI-Powered Anomaly Detection**
- **Machine Learning**: Isolation Forest algorithm for multivariate anomaly detection
- **Statistical analysis**: Z-score based outlier detection with 6 different anomaly patterns
- **Predictive analytics**: AI-powered failure prediction and quality forecasting
- **Explainable AI**: Clear explanations for why batches are flagged as anomalous

### 🚨 **Intelligent Recall Management**
- **Comprehensive impact assessment**: Calculate affected vehicles, costs, and regulatory timelines
- **Priority scoring**: Automatic risk assessment based on part criticality and failure patterns
- **Cost optimization**: Complete recall cost analysis with material, labor, and logistics factors
- **Export capabilities**: Generate recall lists, notices, and communication materials

### 📊 **Professional Dashboard**
- **Streamlit web interface**: Modern, responsive design with real-time monitoring
- **Multi-dimensional search**: Advanced search across VINs, batches, suppliers, and parts
- **Interactive analytics**: Comprehensive charts showing trends, patterns, and correlations
- **Live monitoring**: System health scores and automated alert management

### 🏭 **Enterprise Manufacturing Features**
- **Supply chain intelligence**: Multi-tier supplier risk analysis and geographic assessment
- **Regulatory compliance**: Automated compliance tracking and audit readiness
- **Cost analysis**: Material, labor, overhead, quality, and transport cost tracking
- **Component failure investigation**: Trace failures back through entire supply chain

## 🏗️ System Architecture

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Data Sources      │───▶│    CarTrace AI       │───▶│    Outputs          │
│                     │    │    Pipeline          │    │                     │
│ • 45 Suppliers      │    │                      │    │ • Component Trace   │
│ • 25 Part Types     │    │ ┌──────────────────┐ │    │ • Anomaly Reports   │
│ • 15,000 Vehicles   │    │ │ Enhanced Data    │ │    │ • Recall Analysis   │
│ • Multi-tier Supply │    │ │ Generator        │ │    │ • Supply Chain Risk │
│ • QC & Compliance   │    │ └──────────────────┘ │    │ • Predictive Alerts │
│ • Failure Reports   │    │                      │    │                     │
└─────────────────────┘    │ ┌──────────────────┐ │    └─────────────────────┘
                           │ │ Neo4j Graph DB   │ │             │
┌─────────────────────┐    │ │ (Enterprise Scale│ │             ▼
│   Professional      │◀───│ │  Complex Relations│ │    ┌─────────────────────┐
│   Dashboard         │    │ └──────────────────┘ │    │   Decision Support  │
│                     │    │                      │    │                     │
│ • Real-time Monitor │    │ ┌──────────────────┐ │    │ • Quality Assurance │
│ • Advanced Search   │    │ │ AI/ML Anomaly    │ │    │ • Supply Chain Mgmt │
│ • Predictive Analytics│  │ │ Detection        │ │    │ • Recall Management │
│ • Interactive Viz   │    │ │ (6 Pattern Types)│ │    │ • Regulatory Teams  │
│ • Export Tools      │    │ └──────────────────┘ │    │ • Executive Reports │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

## 🚀 Complete Setup Guide

### ⚙️ Prerequisites & System Requirements

#### **Java Version Check (CRITICAL)**
CarTrace AI requires **Java 17 or 21** for Neo4j to function properly.

```powershell
# Check your current Java version
java -version
```

**Expected Output for Java 21:**
```
openjdk version "21.0.1" 2023-10-17
OpenJDK Runtime Environment (build 21.0.1+12-29)
OpenJDK 64-Bit Server VM (build 21.0.1+12-29, mixed mode, sharing)
```

#### **If You Don't Have Java 17/21:**

1. **Download Java 21 (Recommended)**:
   - Go to: https://www.oracle.com/java/technologies/downloads/#java21
   - Download the **ZIP archive** (not the installer) for Windows
   - Example: `jdk-21_windows-x64_bin.zip`

2. **Extract to Standard Location**:
   ```
   Extract to: C:\Java\jdk-21\
   ```
   
   **Important**: Use the ZIP version, not the installer, for isolated project use.

3. **Verify Installation**:
   ```
   C:\Java\jdk-21\bin\java -version
   ```

#### **Other Requirements**
- **Windows 10/11**
- **Python 3.9+** 
- **Git** (optional, for cloning)
- **16GB+ RAM** (recommended for full dataset)

### 📁 Project Setup

#### **1. Clone and Create Project Structure**

```bash
# Clone the repository
git clone https://github.com/yourusername/cartrace-ai.git
cd cartrace-ai

# Or create manually if not using git
mkdir cartrace_ai
cd cartrace_ai
```

#### **2. Python Virtual Environment**

```bash
# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### **3. Download and Setup Neo4j**

1. **Download Neo4j Community Edition**:
   - Go to: https://neo4j.com/download-center/#community
   - Download for Windows
   - Extract to: `C:\neo4j`

2. **Set Neo4j Environment Variables** (Optional):
   ```bash
   set NEO4J_HOME=C:\neo4j
   # Add C:\neo4j\bin to PATH if desired
   ```

### 🚀 Launch Development Environment

#### **Use the Development Launcher (Recommended)**

We've created a PowerShell script that sets up everything correctly:

```powershell
# Run the development environment launcher
.\start-dev.ps1
```

**This script automatically:**
- ✅ Sets Java 21 for this session only
- ✅ Enables Python UTF-8 mode
- ✅ Activates your virtual environment
- ✅ Configures optimal environment variables

#### **Manual Setup (Alternative)**

If you prefer manual setup:

```bash
# Set Java for this session
set JAVA_HOME=C:\Java\jdk-21
set Path=C:\Java\jdk-21\bin;%Path%

# Set Python UTF-8 mode
set PYTHONUTF8=1

# Activate virtual environment
venv\Scripts\activate
```

### 🗄️ Database Initialization

```bash
# Initialize and configure Neo4j database
python setup_neo4j.py
```

**This process takes ~2 minutes and will:**
- ✅ Configure Neo4j settings optimally
- ✅ Set initial password (`cartrace123`)
- ✅ Start Neo4j service automatically
- ✅ Create database schema and indexes
- ✅ Verify connection and health

### 📊 Generate Data and Run Pipeline

```bash
# Run the complete data pipeline
python run_pipeline.py
```

**This comprehensive process takes ~20 minutes and includes:**

**Phase 1: Data Generation (8 minutes)**
- 📊 Generate 15,000 vehicles with realistic manufacturing data
- 🏭 Create 45 suppliers across 3 tiers with geographic distribution
- 🔩 Generate 25 part types with varying complexity and criticality
- 📦 Create realistic batches with seasonal variations
- 🔧 Simulate assembly processes across multiple stations

**Phase 2: Database Loading (5 minutes)**
- 🔄 Load all data into Neo4j with optimized relationships
- 🔗 Create complex traceability links
- 📈 Build supplier hierarchy connections
- ⚡ Create performance indexes

**Phase 3: AI Model Training (4 minutes)**
- 🤖 Train Isolation Forest anomaly detection model
- 📊 Generate realistic anomaly patterns (6 types)
- 🎯 Calibrate detection thresholds
- 💾 Save trained models for inference

**Phase 4: Validation & Reports (3 minutes)**
- ✅ Comprehensive system validation
- 📋 Generate sample traceability reports
- 🚨 Create initial anomaly alerts
- 📈 Build performance benchmarks

### 🌐 Launch Dashboard

```bash
# Start the professional web interface
streamlit run app.py
```

**Access your dashboard at: http://localhost:8501**

## 📁 Project Structure

```
cartrace_ai/
├── README.md                 # This comprehensive guide
├── requirements.txt          # Python dependencies
├── start-dev.ps1            # Development environment launcher
├── setup_neo4j.py           # Database setup and configuration
├── run_pipeline.py          # Complete pipeline orchestrator
├── app.py                   # Professional Streamlit dashboard
├── cartrace_ai.log          # System logs
│
├── data/                    # Data handling modules
│   ├── __init__.py
│   ├── generator.py         # Enhanced synthetic data generation
│   ├── loader.py           # Optimized data loading with validation
│   ├── raw/                # Generated CSV files
│   └── processed/          # Processed datasets
│
├── models/                  # AI/ML models and graph operations
│   ├── __init__.py
│   ├── anomaly_detector.py # Advanced anomaly detection with ML
│   ├── graph_manager.py    # Neo4j graph operations and queries
│   └── saved/              # Trained model artifacts and weights
│
├── utils/                   # Utility modules
│   ├── __init__.py
│   └── config.py           # Comprehensive configuration management
│
└── tests/                   # Test suite
    ├── __init__.py
    └── test_pipeline.py    # Comprehensive integration tests
```

## 🗃️ Enhanced Data Model

### **Core Entities**
- **Suppliers**: 45 companies across 3 tiers with quality ratings, geographic locations
- **Parts**: 25 component types with weight, tolerances, criticality, complexity levels
- **Batches**: Supplier shipments with comprehensive QC data, environmental conditions
- **Vehicles**: 15,000 final products with VIN, model, assembly station history
- **Assembly Events**: Detailed component installation with timestamps and operators
- **Failures**: Field failure reports with root cause analysis
- **Compliance Records**: Regulatory compliance and certification tracking

### **Complex Relationships**
- `(:Supplier)-[:SUPPLIES {tier: 1-3}]->(:Batch)`
- `(:Batch)-[:CONTAINS {quantity, quality_score}]->(:Part)`
- `(:Vehicle)-[:USES {installation_date, operator}]->(:Batch)`
- `(:Vehicle)-[:ASSEMBLED_AT {start_time, end_time}]->(:Station)`
- `(:Vehicle)-[:FAILED {failure_date, severity}]->(:Failure)`
- `(:Supplier)-[:TIER_RELATIONSHIP]->(:Supplier)` (supply chain hierarchy)

## 🎛️ Professional Dashboard Features

### **1. Executive Overview Dashboard**
- **System health**: Real-time database status and pipeline performance
- **Key metrics**: Production volume, quality trends, supplier performance
- **Live monitoring**: Active alerts, system utilization, processing queues
- **Performance charts**: Production trends, anomaly detection accuracy

### **2. Advanced Component Traceability**
- **Multi-dimensional search**: VIN, Batch ID, Supplier, Part Type, Date Range
- **Vehicle deep-dive**: Complete component history with supplier traceability
- **Batch impact analysis**: All vehicles and components affected by specific batch
- **Visual traceability**: Interactive graph showing relationships and flow
- **Export capabilities**: Comprehensive traceability reports with audit trails

### **3. AI-Powered Anomaly Detection**
- **Real-time detection**: Live monitoring of current production anomalies
- **Detailed explanations**: Feature-level analysis showing why batches are flagged
- **Pattern recognition**: Six different anomaly types with correlation analysis
- **Supplier insights**: Anomaly patterns by supplier with risk scoring
- **Model performance**: Precision, recall, accuracy metrics with confidence intervals

### **4. Intelligent Recall Simulation**
- **Comprehensive impact**: Affected vehicles, customers, geographic distribution
- **Advanced cost modeling**: Material, labor, logistics, communication, regulatory costs
- **Risk-based priority**: Automatic severity classification with compliance timelines
- **Scenario analysis**: Compare different recall strategies and their impacts
- **Communication tools**: Generate customer notices, dealer alerts, regulatory reports

### **5. Supply Chain Analytics**
- **Multi-tier analysis**: Supplier performance across all tiers
- **Geographic risk**: Regional supplier distribution and risk factors
- **Quality trends**: Historical supplier performance with predictive insights
- **Cost analysis**: Comprehensive cost breakdown by supplier and component
- **Compliance tracking**: Regulatory compliance status and certification monitoring

### **6. System Administration**
- **Database health**: Connection status, query performance, storage utilization
- **Pipeline monitoring**: Data processing status, model accuracy, system alerts
- **Configuration management**: System settings, thresholds, alert configurations
- **Export tools**: System reports, data exports, backup management

## 🔧 Configuration Management

CarTrace AI uses a sophisticated configuration system supporting multiple environments:

### **Environment Variables**
```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=cartrace123

# Data Configuration
DATA_DIR=data
ANOMALY_CONTAMINATION=0.08

# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO
```

### **Configuration File (config.json)**
```json
{
  "neo4j": {
    "uri": "bolt://localhost:7687",
    "username": "neo4j",
    "password": "cartrace123",
    "connection_timeout": 30
  },
  "data": {
    "anomaly_contamination": 0.08,
    "random_seed": 42
  },
  "anomaly": {
    "contamination": 0.08,
    "n_estimators": 150,
    "confidence_threshold": 0.7
  },
  "recall": {
    "base_recall_cost_per_vehicle": 500.0,
    "critical_part_multiplier": 2.0
  },
  "ui": {
    "streamlit_port": 8501,
    "page_title": "CarTrace AI - Manufacturing Traceability"
  }
}
```

## 🧪 Comprehensive Testing

Run the full test suite to validate system functionality:

```bash
# Run all integration tests
python -m pytest tests/ -v

# Or use the built-in comprehensive test runner
python tests/test_pipeline.py
```

**Test Coverage:**
- ✅ Data generation with realistic patterns and constraints
- ✅ Anomaly detection accuracy across 6 different pattern types  
- ✅ Graph database operations and complex relationship queries
- ✅ Configuration management and environment validation
- ✅ End-to-end integration scenarios with real-world workflows
- ✅ Business logic validation for manufacturing processes
- ✅ Performance benchmarks and scalability testing

## 📈 Performance Benchmarks

**System Specifications:** Windows 11, 16GB RAM, Intel i7-10700K, SSD

| Operation | Dataset Size | Performance | Details |
|-----------|-------------|-------------|---------|
| **Data Generation** | 15K vehicles, 150K batches | ~8 minutes | Multi-threaded with progress tracking |
| **Neo4j Loading** | Complete dataset + relationships | ~5 minutes | Optimized batch loading with indexes |
| **AI Model Training** | 150K batch records | ~4 minutes | Isolation Forest with hyperparameter tuning |
| **Vehicle Traceability** | Single VIN lookup | <200ms | Optimized Cypher queries with caching |
| **Batch Impact Analysis** | 1K affected vehicles | <500ms | Graph traversal with relationship scoring |
| **Recall Simulation** | 5K affected vehicles | <1 second | Comprehensive cost and impact calculation |
| **Anomaly Detection** | Real-time batch scoring | <50ms | Pre-trained model with feature preprocessing |
| **Dashboard Loading** | Full interface | <3 seconds | Streamlit with caching and lazy loading |

## 🎯 Real Manufacturing Use Cases

### **1. Quality Assurance Teams**
- **Daily anomaly monitoring**: Review AI-flagged batches with detailed explanations
- **Root cause investigation**: Trace failures through entire supply chain
- **Preventive action**: Stop problematic batches before vehicle assembly
- **Supplier performance**: Monitor quality trends and compliance status

### **2. Supply Chain Managers**
- **Multi-tier supplier analysis**: Monitor performance across all supplier tiers
- **Risk assessment**: Geographic and supplier diversification analysis
- **Contract negotiations**: Data-driven supplier discussions with quality metrics
- **Cost optimization**: Comprehensive cost analysis with predictive insights

### **3. Recall Management Teams**
- **Rapid impact assessment**: Instant recall scope and cost calculation
- **Regulatory compliance**: Automated timeline and requirement tracking
- **Communication coordination**: Generate customer notices and dealer alerts
- **Decision support**: Priority-based recall recommendations with risk scoring

### **4. Manufacturing Engineers**
- **Process optimization**: Identify systemic quality patterns and improvements
- **Component tracking**: Monitor parts through entire production process
- **Cost analysis**: Material, labor, and overhead cost optimization
- **Compliance management**: Regulatory tracking and audit preparation

### **5. Executive Management**
- **Strategic insights**: High-level quality trends and supplier performance
- **Risk management**: Supply chain risk assessment with mitigation strategies
- **Cost control**: Comprehensive cost tracking and recall impact analysis
- **Regulatory readiness**: Compliance status and audit preparation

## 🔒 Security & Compliance

### **Data Security**
- **Local deployment**: All data remains within your secure network
- **Database encryption**: Neo4j authentication with encrypted connections
- **Access control**: Role-based access with audit logging
- **Data integrity**: Comprehensive validation and backup procedures

### **Regulatory Compliance**
- **Traceability standards**: Meets automotive industry traceability requirements
- **Audit trails**: Complete tracking of all system interactions and changes
- **Data retention**: Configurable retention policies for compliance needs
- **Export capabilities**: Generate reports for regulatory submissions

## 🚀 Deployment Options

### **Development Environment (This Guide)**
- **Local setup**: Single machine with full functionality
- **Synthetic data**: Realistic test data for development and training
- **Development tools**: Debugging, testing, and configuration management

### **Production Deployment**
- **Docker containers**: Scalable containerized deployment
- **Neo4j cluster**: High-availability database with automatic failover
- **Load balancing**: Multiple application instances for high availability
- **Monitoring**: Comprehensive system monitoring and alerting

### **Cloud Integration**
- **Neo4j Aura**: Managed graph database service with global availability
- **Streamlit Cloud**: Hosted dashboard deployment with automatic scaling
- **AWS/Azure/GCP**: Full cloud infrastructure with enterprise features
- **Data integration**: APIs for connecting to existing manufacturing systems

## 🛟 Troubleshooting Guide

### **Java Issues**
```bash
# Problem: "Java version not supported" 
# Solution: Ensure Java 17 or 21 is active
java -version
# Should show version 17.x.x or 21.x.x

# Problem: Java not found
# Solution: Check JAVA_HOME and PATH
echo $env:JAVA_HOME  # PowerShell
set | findstr JAVA   # Command Prompt

# Solution: Use development launcher
.\start-dev.ps1
```

### **Neo4j Connection Issues**
```bash
# Check Neo4j status
neo4j status

# Restart Neo4j service
neo4j restart

# Check if port 7687 is available
netstat -an | findstr :7687

# Test connection manually
python -c "from neo4j import GraphDatabase; print('Neo4j connection OK')"
```

### **Python Environment Issues**
```bash
# Update pip and dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt --upgrade

# Clear cache and reinstall
pip cache purge
pip install -r requirements.txt --force-reinstall

# Check virtual environment
where python
# Should point to venv\Scripts\python.exe
```

### **Dashboard Issues**
```bash
# Check if port 8501 is available
netstat -an | findstr :8501

# Try alternative port
streamlit run app.py --server.port 8502

# Clear Streamlit cache
streamlit cache clear

# Check logs for errors
tail -f cartrace_ai.log
```

### **Performance Issues**
```bash
# Check available memory
wmic OS get TotalVisibleMemorySize,FreePhysicalMemory

# Monitor Neo4j performance
# Access Neo4j Browser at http://localhost:7474

# Check disk space
dir C:\ | findstr "free"

# Optimize Java memory for Neo4j
# Edit C:\neo4j\conf\neo4j.conf
# dbms.memory.heap.initial_size=2g
# dbms.memory.heap.max_size=4g
```

## 🛣️ Future Roadmap

### **Phase 1: Current Features ✅**
- [x] Graph-based traceability with enterprise scale
- [x] AI-powered anomaly detection with explainability
- [x] Intelligent recall simulation with cost optimization
- [x] Professional dashboard with real-time monitoring

### **Phase 2: Advanced Analytics (Q2 2024)**
- [ ] Predictive failure modeling with deep learning
- [ ] Time-series anomaly detection for production trends
- [ ] Advanced supplier risk scoring with external data
- [ ] Automated quality report generation and distribution

### **Phase 3: Integration & Automation (Q3 2024)**
- [ ] Real-time data ingestion from manufacturing systems
- [ ] REST API endpoints for external system integration
- [ ] Advanced 3D visualization of supply chain networks
- [ ] Mobile dashboard for field quality inspectors

### **Phase 4: AI Enhancement (Q4 2024)**
- [ ] Computer vision for image-based quality control
- [ ] Natural language query interface for traceability
- [ ] Automated root cause analysis with ML explanations
- [ ] Predictive maintenance integration with IoT sensors

### **Phase 5: Enterprise Features (2025)**
- [ ] Multi-tenant architecture for global manufacturing
- [ ] Advanced workflow automation and approval processes
- [ ] Integration with ERP and PLM systems
- [ ] Blockchain integration for supply chain verification

## 📊 Success Metrics

After completing this setup, you will have:

✅ **Enterprise-scale database** with 15,000 vehicles and complex relationships  
✅ **AI-powered anomaly detection** with 6 pattern types and explanations  
✅ **Professional dashboard** with real-time monitoring and analytics  
✅ **Complete traceability** from raw materials to final vehicles  
✅ **Intelligent recall simulation** with comprehensive cost analysis  
✅ **Supply chain intelligence** with multi-tier supplier analysis  
✅ **Regulatory compliance** tracking and audit readiness  
✅ **Predictive insights** for quality and supplier performance  

## 🚀 Quick Start Summary

1. **Check Java version**: `java -version` (need 17 or 21)
2. **Download Java 21** if needed from Oracle (ZIP version)
3. **Extract to**: `C:\Java\jdk-21\`
4. **Run development launcher**: `.\start-dev.ps1`
5. **Initialize database**: `python setup_neo4j.py` (~2 minutes)
6. **Run pipeline**: `python run_pipeline.py` (~20 minutes)
7. **Launch dashboard**: `streamlit run app.py`
8. **Open browser to**: `http://localhost:8501`

## 🎉 You're Ready for Manufacturing Excellence!

**Next Steps:**
1. 🎮 **Explore the dashboard** - Try vehicle lookups and batch tracing
2. 🔍 **Review anomalies** - Examine AI-flagged batches with explanations
3. 🚨 **Simulate recalls** - Test recall scenarios for anomalous batches
4. 📊 **Analyze suppliers** - Review supplier performance and risk metrics
5. 🔧 **Customize configuration** - Adjust thresholds and parameters for your needs
6. 📈 **Monitor trends** - Track quality patterns and manufacturing performance

**Happy Manufacturing!** 🏭✨

---

## 📄 License & Support

- **License**: MIT License - see LICENSE file for details
- **Documentation**: Comprehensive code comments and this README
- **Issues**: Open GitHub issues for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Contributions**: Follow standard fork/pull request workflow

**Built with ❤️ for manufacturing excellence and supply chain transparency.**