import pandas as pd
import numpy as np
import json
import uuid
import os
from datetime import datetime, timedelta
from faker import Faker
from typing import Tuple, List, Dict, Set
import warnings
import string
import random
import networkx as nx
from tqdm import tqdm
warnings.filterwarnings('ignore')

fake = Faker()
np.random.seed(42)

class EnhancedManufacturingDataGenerator:
    """Generate comprehensive synthetic automotive manufacturing data"""
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Expanded Configuration
        self.n_suppliers = 45  # More realistic supplier count
        self.n_parts = 25      # More comprehensive part catalog
        self.n_vehicles = 15000 # Larger vehicle population
        self.batches_per_supplier_part = 25  # More batch history
        self.anomaly_rate = 0.06  # 6% anomaly rate
        
        # Enhanced part specifications with hierarchical structure
        self.part_specs = [
            # Engine Components
            {"name": "Engine Block", "category": "Engine", "subcategory": "Core", "weight": 85000.0, "tolerance": 0.5, "critical": True, "complexity": 5},
            {"name": "Cylinder Head", "category": "Engine", "subcategory": "Core", "weight": 25000.0, "tolerance": 0.8, "critical": True, "complexity": 5},
            {"name": "Piston Assembly", "category": "Engine", "subcategory": "Internal", "weight": 420.0, "tolerance": 2.0, "critical": True, "complexity": 4},
            {"name": "Crankshaft", "category": "Engine", "subcategory": "Internal", "weight": 22000.0, "tolerance": 0.3, "critical": True, "complexity": 5},
            {"name": "Timing Chain", "category": "Engine", "subcategory": "Timing", "weight": 1200.0, "tolerance": 1.5, "critical": True, "complexity": 3},
            {"name": "Oil Pump", "category": "Engine", "subcategory": "Lubrication", "weight": 2800.0, "tolerance": 2.0, "critical": True, "complexity": 3},
            
            # Transmission Components
            {"name": "Transmission Case", "category": "Transmission", "subcategory": "Housing", "weight": 35000.0, "tolerance": 0.8, "critical": True, "complexity": 4},
            {"name": "Gear Set", "category": "Transmission", "subcategory": "Internal", "weight": 8500.0, "tolerance": 0.5, "critical": True, "complexity": 5},
            {"name": "Torque Converter", "category": "Transmission", "subcategory": "Coupling", "weight": 12000.0, "tolerance": 1.0, "critical": True, "complexity": 4},
            
            # Braking System
            {"name": "Brake Caliper", "category": "Brake", "subcategory": "Hydraulic", "weight": 3200.0, "tolerance": 1.5, "critical": True, "complexity": 3},
            {"name": "Brake Disc", "category": "Brake", "subcategory": "Friction", "weight": 8500.0, "tolerance": 2.0, "critical": True, "complexity": 2},
            {"name": "Brake Pad Set", "category": "Brake", "subcategory": "Friction", "weight": 850.0, "tolerance": 3.0, "critical": True, "complexity": 2},
            {"name": "ABS Module", "category": "Brake", "subcategory": "Electronic", "weight": 2100.0, "tolerance": 1.0, "critical": True, "complexity": 5},
            
            # Suspension Components
            {"name": "MacPherson Strut", "category": "Suspension", "subcategory": "Damping", "weight": 4200.0, "tolerance": 2.5, "critical": True, "complexity": 3},
            {"name": "Control Arm", "category": "Suspension", "subcategory": "Linkage", "weight": 2800.0, "tolerance": 1.8, "critical": True, "complexity": 2},
            {"name": "Wheel Hub Assembly", "category": "Suspension", "subcategory": "Bearing", "weight": 3100.0, "tolerance": 1.0, "critical": True, "complexity": 3},
            
            # Electrical Components
            {"name": "Engine Control Unit", "category": "Electrical", "subcategory": "Control", "weight": 950.0, "tolerance": 0.5, "critical": True, "complexity": 5},
            {"name": "Wiring Harness", "category": "Electrical", "subcategory": "Distribution", "weight": 4500.0, "tolerance": 5.0, "critical": True, "complexity": 3},
            {"name": "Battery", "category": "Electrical", "subcategory": "Power", "weight": 18000.0, "tolerance": 2.0, "critical": True, "complexity": 2},
            {"name": "Alternator", "category": "Electrical", "subcategory": "Charging", "weight": 5200.0, "tolerance": 1.5, "critical": True, "complexity": 4},
            
            # Body Components
            {"name": "Door Panel", "category": "Body", "subcategory": "Closure", "weight": 12000.0, "tolerance": 5.0, "critical": False, "complexity": 2},
            {"name": "Hood Assembly", "category": "Body", "subcategory": "Closure", "weight": 15000.0, "tolerance": 3.0, "critical": False, "complexity": 2},
            {"name": "Windshield", "category": "Body", "subcategory": "Glass", "weight": 22000.0, "tolerance": 1.0, "critical": True, "complexity": 3},
            {"name": "Side Mirror", "category": "Body", "subcategory": "Vision", "weight": 850.0, "tolerance": 8.0, "critical": False, "complexity": 2},
            {"name": "Headlight Assembly", "category": "Body", "subcategory": "Lighting", "weight": 1200.0, "tolerance": 2.5, "critical": False, "complexity": 3}
        ]
        
        # Global supplier base with specializations
        self.supplier_profiles = [
            # Tier 1 Suppliers (Premium, specialized)
            {"tier": 1, "specialization": "Engine", "countries": ["Germany", "Japan"], "quality_range": (0.92, 0.99), "cost_multiplier": 1.3},
            {"tier": 1, "specialization": "Transmission", "countries": ["Germany", "USA"], "quality_range": (0.90, 0.98), "cost_multiplier": 1.25},
            {"tier": 1, "specialization": "Brake", "countries": ["Germany", "Italy", "Japan"], "quality_range": (0.91, 0.99), "cost_multiplier": 1.2},
            {"tier": 1, "specialization": "Electrical", "countries": ["Germany", "Japan", "South Korea"], "quality_range": (0.89, 0.97), "cost_multiplier": 1.4},
            
            # Tier 2 Suppliers (Good quality, moderate cost)
            {"tier": 2, "specialization": "Suspension", "countries": ["USA", "Mexico", "Brazil"], "quality_range": (0.85, 0.93), "cost_multiplier": 1.0},
            {"tier": 2, "specialization": "Body", "countries": ["China", "India", "Mexico"], "quality_range": (0.83, 0.91), "cost_multiplier": 0.8},
            {"tier": 2, "specialization": "Engine", "countries": ["China", "India", "Turkey"], "quality_range": (0.82, 0.90), "cost_multiplier": 0.7},
            
            # Tier 3 Suppliers (Cost-focused)
            {"tier": 3, "specialization": "Body", "countries": ["China", "India", "Vietnam"], "quality_range": (0.75, 0.88), "cost_multiplier": 0.6},
            {"tier": 3, "specialization": "Electrical", "countries": ["China", "Malaysia", "Philippines"], "quality_range": (0.78, 0.85), "cost_multiplier": 0.65}
        ]
        
        # Manufacturing stations with realistic assembly flow
        self.manufacturing_stations = [
            {"id": "STN-ENG-001", "name": "Engine Assembly Line 1", "category": "Engine", "capacity": 120, "shifts": ["day", "night"]},
            {"id": "STN-ENG-002", "name": "Engine Assembly Line 2", "category": "Engine", "capacity": 110, "shifts": ["day", "night"]},
            {"id": "STN-TRN-001", "name": "Transmission Assembly", "category": "Transmission", "capacity": 150, "shifts": ["day", "night"]},
            {"id": "STN-CHS-001", "name": "Chassis Assembly", "category": "Chassis", "capacity": 180, "shifts": ["day", "evening", "night"]},
            {"id": "STN-BDY-001", "name": "Body Shop Line 1", "category": "Body", "capacity": 200, "shifts": ["day", "evening", "night"]},
            {"id": "STN-BDY-002", "name": "Body Shop Line 2", "category": "Body", "capacity": 190, "shifts": ["day", "evening", "night"]},
            {"id": "STN-PNT-001", "name": "Paint Shop", "category": "Finishing", "capacity": 160, "shifts": ["day", "night"]},
            {"id": "STN-ASM-001", "name": "Final Assembly Line 1", "category": "Assembly", "capacity": 220, "shifts": ["day", "evening", "night"]},
            {"id": "STN-ASM-002", "name": "Final Assembly Line 2", "category": "Assembly", "capacity": 210, "shifts": ["day", "evening", "night"]},
            {"id": "STN-QC-001", "name": "Quality Control Station 1", "category": "QC", "capacity": 100, "shifts": ["day", "evening"]},
            {"id": "STN-QC-002", "name": "Quality Control Station 2", "category": "QC", "capacity": 95, "shifts": ["day", "evening"]},
            {"id": "STN-TST-001", "name": "Road Test", "category": "Testing", "capacity": 50, "shifts": ["day"]},
        ]
        
        # Vehicle models with realistic configurations
        self.vehicle_models = [
            {"model": "EcoSedan-X1", "segment": "Compact", "complexity": 3, "target_market": ["Domestic", "Export"], "annual_volume": 45000},
            {"model": "SportSedan-S2", "segment": "Mid-size", "complexity": 4, "target_market": ["Domestic", "Premium Export"], "annual_volume": 32000},
            {"model": "FamilySUV-M3", "segment": "SUV", "complexity": 4, "target_market": ["Domestic", "Export"], "annual_volume": 38000},
            {"model": "LuxurySUV-L4", "segment": "Premium SUV", "complexity": 5, "target_market": ["Premium Export"], "annual_volume": 18000},
            {"model": "CityHatch-C5", "segment": "Compact", "complexity": 2, "target_market": ["Domestic"], "annual_volume": 55000},
            {"model": "WorkTruck-T6", "segment": "Commercial", "complexity": 3, "target_market": ["Domestic", "Commercial Export"], "annual_volume": 25000}
        ]
        
        # Quality standards and certifications
        self.certifications = [
            "ISO9001", "TS16949", "ISO14001", "OHSAS18001", "VDA6.3", "AIAG", "PPAP"
        ]
        
        # Failure modes with realistic patterns
        self.failure_patterns = {
            "Engine": ["oil_leak", "overheating", "misfire", "timing_failure", "bearing_wear"],
            "Transmission": ["gear_slip", "fluid_leak", "shift_delay", "torque_converter_failure"],
            "Brake": ["pad_wear", "disc_warping", "caliper_seizure", "abs_malfunction"],
            "Suspension": ["strut_leak", "bushing_wear", "spring_fatigue", "alignment_drift"],
            "Electrical": ["ecu_failure", "sensor_malfunction", "harness_corrosion", "battery_drain"],
            "Body": ["paint_defect", "rust_formation", "seal_failure", "trim_detachment"]
        }

    def generate_all_data(self) -> Dict[str, pd.DataFrame]:
        """Generate complete enhanced dataset"""
        print("Generating enhanced synthetic manufacturing data...")
        print(f"Scale: {self.n_vehicles:,} vehicles, {self.n_suppliers} suppliers, {self.n_parts} parts")
        
        # Generate core entities in dependency order
        suppliers_df = self._generate_suppliers()
        parts_df = self._generate_parts()
        stations_df = self._generate_stations()
        batches_df = self._generate_batches(suppliers_df, parts_df)
        vehicles_df = self._generate_vehicles()
        assembly_df = self._generate_assembly_events(vehicles_df, batches_df, stations_df, parts_df)
        qc_df = self._generate_qc_inspections(vehicles_df, assembly_df)
        failures_df = self._generate_failures(vehicles_df, assembly_df, batches_df)
        
        # Generate advanced datasets
        supplier_audits_df = self._generate_supplier_audits(suppliers_df)
        compliance_df = self._generate_compliance_records(batches_df)
        maintenance_df = self._generate_maintenance_records()
        cost_analysis_df = self._generate_cost_analysis(batches_df, suppliers_df, parts_df)
        
        # Save all datasets
        datasets = {
            'suppliers': suppliers_df,
            'parts': parts_df,
            'stations': stations_df,
            'batches': batches_df,
            'vehicles': vehicles_df,
            'assembly_events': assembly_df,
            'qc_inspections': qc_df,
            'failures': failures_df,
            'supplier_audits': supplier_audits_df,
            'compliance_records': compliance_df,
            'maintenance_records': maintenance_df,
            'cost_analysis': cost_analysis_df
        }
        
        for name, df in datasets.items():
            filepath = os.path.join(self.output_dir, f"{name}.csv")
            df.to_csv(filepath, index=False)
            print(f"✓ Generated {len(df):,} {name} records -> {filepath}")
        
        # Generate comprehensive reports
        self._generate_data_quality_report(datasets)
        self._generate_anomaly_report(batches_df)
        self._generate_supply_chain_map(suppliers_df, parts_df)
        
        print(f"\n✓ Enhanced dataset generation complete!")
        print(f"  Total records: {sum(len(df) for df in datasets.values()):,}")
        print(f"  Data directory: {self.output_dir}")
        
        return datasets
    
    def _generate_suppliers(self) -> pd.DataFrame:
        """Generate realistic supplier base with tiers and specializations"""
        suppliers = []
        supplier_counter = 1
        
        for profile in self.supplier_profiles:
            # Number of suppliers per profile
            suppliers_in_profile = max(1, self.n_suppliers // len(self.supplier_profiles))
            
            for _ in range(suppliers_in_profile):
                if supplier_counter > self.n_suppliers:
                    break
                    
                # Generate realistic supplier characteristics
                country = np.random.choice(profile["countries"])
                quality_rating = np.random.uniform(*profile["quality_range"])
                
                # History issues inversely related to quality and tier
                base_issues = max(0, int(np.random.poisson((1 - quality_rating) * 10)))
                tier_modifier = {1: 0.5, 2: 1.0, 3: 1.8}[profile["tier"]]
                history_issues = int(base_issues * tier_modifier)
                
                # Financial metrics
                annual_revenue = np.random.uniform(10, 500) * (1000000 if profile["tier"] == 1 else 
                                                               500000 if profile["tier"] == 2 else 100000)
                
                # Certification probability based on tier
                cert_count = np.random.randint(1, len(self.certifications) + 1)
                cert_prob = {1: 0.9, 2: 0.7, 3: 0.4}[profile["tier"]]
                
                # --- FIX 1: Corrected Certification Probability Logic ---
                # The original logic was flawed and did not create a valid probability distribution.
                # This corrected logic creates unnormalized weights and then normalizes them.
                p_unnormalized = [cert_prob if i < 3 else (1 - cert_prob) / 4 for i in range(len(self.certifications))]
                p_normalized = np.array(p_unnormalized) / np.sum(p_unnormalized)

                certifications = np.random.choice(
                    self.certifications,
                    size=min(cert_count, len(self.certifications)),
                    replace=False,
                    p=p_normalized
                )
                
                suppliers.append({
                    'supplier_id': f'SUPP-{supplier_counter:03d}',
                    'name': f"{fake.company()} {np.random.choice(['Ltd', 'Inc', 'Corp', 'GmbH', 'S.A.'])}",
                    'country': country,
                    'region': self._get_region(country),
                    'tier': profile["tier"],
                    'specialization': profile["specialization"],
                    'quality_rating': round(quality_rating, 4),
                    'history_issues': history_issues,
                    'certifications': '|'.join(certifications),
                    'established_year': np.random.randint(1970, 2020),
                    'annual_revenue_usd': int(annual_revenue),
                    'employee_count': int(annual_revenue / 50000),  # Rough estimate
                    'lead_time_avg_days': np.random.randint(7, 60),
                    'payment_terms': np.random.choice(['NET30', 'NET45', 'NET60']),
                    'risk_score': round((1 - quality_rating) * 100 + np.random.uniform(-10, 10), 1)
                })
                
                supplier_counter += 1
        
        return pd.DataFrame(suppliers)
    
    def _generate_parts(self) -> pd.DataFrame:
        """Generate comprehensive parts catalog"""
        parts = []
        
        for i, spec in enumerate(self.part_specs, 1):
            # Base cost calculation with complexity factor
            base_cost = np.random.uniform(10, 1000) * (1 + spec['complexity'] * 0.3)
            if spec['critical']:
                base_cost *= 1.5
                
            # Material and process information
            materials = {
                "Engine": ["Cast Iron", "Aluminum", "Steel", "Titanium"],
                "Transmission": ["Steel", "Aluminum", "Carbon Steel"],
                "Brake": ["Cast Iron", "Carbon Composite", "Steel"],
                "Suspension": ["Steel", "Aluminum", "Composite"],
                "Electrical": ["Copper", "Plastic", "Silicon", "Aluminum"],
                "Body": ["Steel", "Aluminum", "Plastic", "Glass"]
            }
            
            primary_material = np.random.choice(materials.get(spec['category'], ["Steel"]))
            
            # Manufacturing processes
            processes = {
                5: ["Precision Machining", "Investment Casting", "Multi-stage Assembly"],
                4: ["CNC Machining", "Die Casting", "Precision Assembly"],
                3: ["Machining", "Stamping", "Standard Assembly"],
                2: ["Stamping", "Molding", "Basic Assembly"],
                1: ["Cutting", "Forming", "Simple Assembly"]
            }
            
            manufacturing_process = np.random.choice(processes[spec['complexity']])
            
            parts.append({
                'part_id': f'PART-{i:03d}',
                'name': spec['name'],
                'category': spec['category'],
                'subcategory': spec['subcategory'],
                'spec_weight_g': spec['weight'],
                'spec_tolerance_pct': spec['tolerance'],
                'is_critical': spec['critical'],
                'complexity_level': spec['complexity'],
                'unit_cost_base': round(base_cost, 2),
                'primary_material': primary_material,
                'manufacturing_process': manufacturing_process,
                'lifecycle_months': np.random.randint(60, 120),
                'replacement_interval_km': np.random.randint(50000, 200000) if not spec['critical'] else 0,
                'regulatory_compliance': '|'.join(np.random.choice(['FMVSS', 'ECE', 'DOT', 'SAE'], size=2, replace=False)),
                'environmental_rating': np.random.choice(['A', 'B', 'C'], p=[0.3, 0.5, 0.2])
            })
            
        return pd.DataFrame(parts)

    def _generate_stations(self) -> pd.DataFrame:
        """Generate manufacturing stations data"""
        stations = []
        
        for station in self.manufacturing_stations:
            stations.append({
                'station_id': station['id'],
                'name': station['name'],
                'category': station['category'],
                'daily_capacity': station['capacity'],
                'shift_pattern': '|'.join(station['shifts']),
                'efficiency_rating': np.random.uniform(0.85, 0.98),
                'maintenance_interval_hours': np.random.randint(160, 480),
                'last_maintenance': fake.date_between(start_date='-30d', end_date='-1d').isoformat(),
                'operator_count_per_shift': np.random.randint(3, 12),
                'automated_level': np.random.choice(['Manual', 'Semi-Auto', 'Automated'], p=[0.3, 0.5, 0.2])
            })
            
        return pd.DataFrame(stations)

    def _generate_batches(self, suppliers_df: pd.DataFrame, parts_df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive batch data with enhanced anomaly patterns"""
        batches = []
        batch_counter = 1
        anomaly_tracking = []
        
        # Create supplier-part compatibility matrix
        supplier_part_matrix = self._create_supplier_part_matrix(suppliers_df, parts_df)
        
        for _, supplier in suppliers_df.iterrows():
            compatible_parts = supplier_part_matrix[supplier_part_matrix['supplier_id'] == supplier['supplier_id']]
            
            for _, part_match in compatible_parts.iterrows():
                part = parts_df[parts_df['part_id'] == part_match['part_id']].iloc[0]
                
                # Generate multiple batches with seasonal variation
                n_batches = self.batches_per_supplier_part + np.random.randint(-8, 12)
                
                for batch_idx in range(n_batches):
                    batch_id = f"BATCH-{batch_counter:06d}"
                    
                    # Seasonal manufacturing date (higher volume in certain months)
                    season_weights = [0.8, 0.9, 1.2, 1.1, 1.0, 0.9, 0.7, 0.8, 1.3, 1.2, 1.1, 1.0]
                    month = np.random.choice(range(1, 13), p=np.array(season_weights)/np.sum(season_weights))
                    year = np.random.choice([2023, 2024], p=[0.3, 0.7])
                    day = np.random.randint(1, 29)
                    manufacture_date = datetime(year, month, day) - timedelta(days=np.random.randint(0, 180))
                    
                    # Base characteristics
                    base_quantity = int(np.random.uniform(1500, 5000))
                    
                    # Supplier quality influence
                    supplier_quality_factor = supplier['quality_rating']
                    base_qc_rate = 0.88 + (supplier_quality_factor - 0.7) * 0.3
                    base_qc_rate = np.clip(base_qc_rate, 0.75, 0.999)
                    
                    # Part complexity influence
                    complexity_factor = 1 - (part['complexity_level'] - 1) * 0.03
                    base_qc_rate *= complexity_factor
                    
                    # Determine anomaly status with enhanced logic
                    is_anomalous = self._determine_anomaly_status(supplier, part, manufacture_date)
                    anomaly_type = "none"
                    
                    if is_anomalous:
                        anomaly_type, quantity, qc_pass_rate, weight_mean = self._generate_anomaly(
                            base_quantity, base_qc_rate, part['spec_weight_g'], 
                            supplier, part, manufacture_date
                        )
                        
                        anomaly_tracking.append({
                            'batch_id': batch_id,
                            'anomaly_type': anomaly_type,
                            'supplier_id': supplier['supplier_id'],
                            'part_id': part['part_id'],
                            'severity': self._calculate_anomaly_severity(anomaly_type, qc_pass_rate, weight_mean, part)
                        })
                    else:
                        # Normal batch generation
                        quantity = base_quantity + np.random.randint(-200, 300)
                        qc_pass_rate = base_qc_rate + np.random.uniform(-0.02, 0.04)
                        weight_mean = part['spec_weight_g'] * (1 + np.random.normal(0, 0.005))
                    
                    # Additional batch characteristics
                    weight_std = np.random.uniform(0.5, 4.0) * (1.5 if is_anomalous else 1.0)
                    lead_time = supplier['lead_time_avg_days'] + np.random.randint(-7, 14)
                    
                    # Cost calculation
                    batch_cost = quantity * part['unit_cost_base'] * np.random.uniform(0.95, 1.05)
                    if supplier['tier'] == 1:
                        batch_cost *= 1.3
                    elif supplier['tier'] == 3:
                        batch_cost *= 0.7
                    
                    # Generate supplier-specific labeling
                    supplier_batch_label = self._generate_supplier_label(supplier, batch_counter, batch_idx)
                    
                    batches.append({
                        'batch_id': batch_id,
                        'supplier_batch_label': supplier_batch_label,
                        'supplier_id': supplier['supplier_id'],
                        'part_id': part['part_id'],
                        'quantity': int(quantity),
                        'manufacture_date': manufacture_date.isoformat(),
                        'qc_pass_rate': round(np.clip(qc_pass_rate, 0.65, 0.999), 4),
                        'weight_mean_g': round(weight_mean, 3),
                        'weight_std_g': round(weight_std, 3),
                        'lead_time_days': int(max(1, lead_time)),
                        'batch_cost_usd': round(batch_cost, 2),
                        'is_anomalous': is_anomalous,
                        'anomaly_type': anomaly_type,
                        'production_shift': np.random.choice(['day', 'night', 'evening']),
                        'operator_id': f"OP-{np.random.randint(1, 100):03d}",
                        'temperature_c': np.random.uniform(18, 25),
                        'humidity_pct': np.random.uniform(40, 70),
                        'material_lot': f"MAT-{np.random.randint(1000, 9999)}",
                        'inspection_level': np.random.choice(['Standard', 'Enhanced', 'Reduced'], p=[0.7, 0.25, 0.05]),
                        'storage_location': f"WH-{np.random.choice(['A', 'B', 'C'])}-{np.random.randint(1, 50):02d}"
                    })
                    
                    batch_counter += 1
        
        # Save anomaly ground truth
        if anomaly_tracking:
            anomaly_df = pd.DataFrame(anomaly_tracking)
            anomaly_df.to_csv(os.path.join(self.output_dir, 'anomaly_ground_truth.csv'), index=False)
        
        return pd.DataFrame(batches)

    def _create_supplier_part_matrix(self, suppliers_df: pd.DataFrame, parts_df: pd.DataFrame) -> pd.DataFrame:
        """Create realistic supplier-part compatibility matrix"""
        matrix = []
        
        for _, supplier in suppliers_df.iterrows():
            # Suppliers specialize in their category but can supply some others
            primary_parts = parts_df[parts_df['category'] == supplier['specialization']]
            secondary_parts = parts_df[parts_df['category'] != supplier['specialization']]
            
            # All parts in specialization
            for _, part in primary_parts.iterrows():
                matrix.append({
                    'supplier_id': supplier['supplier_id'],
                    'part_id': part['part_id'],
                    'is_primary': True,
                    'capability_score': np.random.uniform(0.8, 1.0)
                })
            
            # Some parts outside specialization (tier 1 more versatile)
            if supplier['tier'] <= 2:
                n_secondary = min(len(secondary_parts), np.random.randint(2, 8))
                # MODULO ADDED TO CONSTRAIN THE SEED TO THE VALID 32-BIT RANGE
                seed = (42 + abs(hash(supplier['supplier_id']))) % (2**32)
                secondary_sample = secondary_parts.sample(n=n_secondary, random_state=seed)
                for _, part in secondary_sample.iterrows():
                    matrix.append({
                        'supplier_id': supplier['supplier_id'],
                        'part_id': part['part_id'],
                        'is_primary': False,
                        'capability_score': np.random.uniform(0.5, 0.8)
                    })
        
        return pd.DataFrame(matrix)
    
    def _determine_anomaly_status(self, supplier: Dict, part: Dict, manufacture_date: datetime) -> bool:
        """Enhanced anomaly determination logic"""
        base_anomaly_prob = self.anomaly_rate
        
        # Supplier tier influences anomaly probability
        tier_modifier = {1: 0.5, 2: 1.0, 3: 2.0}[supplier['tier']]
        
        # Part complexity increases anomaly risk
        complexity_modifier = 1 + (part['complexity_level'] - 3) * 0.3
        
        # Time-based factors (end of quarter rush, new year startup issues)
        month = manufacture_date.month
        time_modifier = 1.0
        if month in [3, 6, 9, 12]:  # Quarter ends
            time_modifier = 1.4
        elif month in [1, 2]:  # Year start
            time_modifier = 1.2
        
        # Historical quality factor
        quality_modifier = 2.0 - supplier['quality_rating']  # Lower quality = higher anomaly risk
        
        adjusted_prob = base_anomaly_prob * tier_modifier * complexity_modifier * time_modifier * quality_modifier
        adjusted_prob = min(0.25, adjusted_prob)  # Cap at 25%
        
        return np.random.random() < adjusted_prob
    
    def _generate_anomaly(self, base_quantity: int, base_qc_rate: float, spec_weight: float, 
                          supplier: Dict, part: Dict, manufacture_date: datetime) -> Tuple[str, int, float, float]:
        """Generate specific anomaly patterns"""
        anomaly_types = ['quality_degradation', 'process_deviation', 'material_defect', 'timing_issue', 'quantity_variance', 'environmental_factor']
        
        # --- FIX 2: Corrected Anomaly Probability Logic ---
        # The original weight lists did not sum to 1.0. Normalizing them fixes this.
        weights = [0.3, 0.25, 0.2, 0.1, 0.1, 0.05]
        if part['is_critical']:
            weights = [0.4, 0.3, 0.2, 0.05, 0.03, 0.02]  # Critical parts more likely to have quality issues
        
        normalized_weights = np.array(weights) / np.sum(weights)
        anomaly_type = np.random.choice(anomaly_types, p=normalized_weights)
        
        if anomaly_type == 'quality_degradation':
            # Significant drop in QC pass rate
            qc_pass_rate = base_qc_rate - np.random.uniform(0.15, 0.35)
            weight_mean = spec_weight * (1 + np.random.normal(0, 0.02))
            quantity = base_quantity + np.random.randint(-100, 100)
            
        elif anomaly_type == 'process_deviation':
            # Process parameter drift affecting weight and quality
            weight_deviation = np.random.choice([-1, 1]) * np.random.uniform(0.05, 0.15)
            weight_mean = spec_weight * (1 + weight_deviation)
            qc_pass_rate = base_qc_rate - abs(weight_deviation) * 2
            quantity = base_quantity
            
        elif anomaly_type == 'material_defect':
            # Raw material batch issue
            weight_mean = spec_weight * (1 + np.random.uniform(-0.08, 0.08))
            qc_pass_rate = max(0.65, base_qc_rate - np.random.uniform(0.20, 0.40))
            quantity = base_quantity
            
        elif anomaly_type == 'timing_issue':
            # Rush production or delayed processing
            weight_mean = spec_weight * (1 + np.random.normal(0, 0.03))
            qc_pass_rate = base_qc_rate - np.random.uniform(0.08, 0.20)
            quantity = int(base_quantity * np.random.uniform(0.7, 1.4))
            
        elif anomaly_type == 'quantity_variance':
            # Significant quantity deviation
            quantity = int(base_quantity * np.random.choice([0.4, 0.6, 1.6, 2.2]))
            weight_mean = spec_weight * (1 + np.random.normal(0, 0.015))
            qc_pass_rate = base_qc_rate - np.random.uniform(0.05, 0.15)
            
        else:  # environmental_factor
            # Environmental conditions affecting production
            weight_mean = spec_weight * (1 + np.random.normal(0, 0.025))
            qc_pass_rate = base_qc_rate - np.random.uniform(0.10, 0.25)
            quantity = base_quantity
        
        qc_pass_rate = max(0.60, qc_pass_rate)
        return anomaly_type, quantity, qc_pass_rate, weight_mean

    def _calculate_anomaly_severity(self, anomaly_type: str, qc_rate: float, weight: float, part: Dict) -> int:
        """Calculate anomaly severity score (1-5)"""
        severity = 1
        
        # QC rate impact
        if qc_rate < 0.75:
            severity += 2
        elif qc_rate < 0.85:
            severity += 1
        
        # Critical part impact
        if part['is_critical']:
            severity += 1
        
        # Anomaly type impact
        type_severity = {
            'material_defect': 2,
            'quality_degradation': 2,
            'process_deviation': 1,
            'timing_issue': 1,
            'quantity_variance': 0,
            'environmental_factor': 1
        }
        severity += type_severity.get(anomaly_type, 0)
        
        return min(5, severity)
    
    def _generate_supplier_label(self, supplier: Dict, batch_counter: int, batch_idx: int) -> str:
        """Generate realistic supplier-specific batch labels"""
        country = supplier['country']
        
        if country in ['Germany', 'Japan']:
            # Structured European/Japanese format
            return f"LOT-{np.random.randint(1000,9999)}-{batch_idx+1:02d}"
        elif country in ['USA']:
            # US format
            return f"BATCH{np.random.randint(100,999)}{chr(65+np.random.randint(0,26))}"
        elif country in ['China', 'India']:
            # Asian format
            return f"B{np.random.randint(10,99)}{np.random.randint(1000,9999)}"
        else:
            # Generic format
            return f"{''.join(random.choices(string.ascii_uppercase, k=2))}{np.random.randint(1000,9999)}"
    
    def _generate_vehicles(self) -> pd.DataFrame:
        """Generate realistic vehicle population"""
        vehicles = []
        
        # Calculate production schedule based on model volumes
        total_planned = sum(model['annual_volume'] for model in self.vehicle_models)
        production_ratios = {model['model']: model['annual_volume'] / total_planned for model in self.vehicle_models}
        
        for i in range(1, self.n_vehicles + 1):
            # Select model based on production ratios
            model_choice = np.random.choice(
                [m['model'] for m in self.vehicle_models],
                p=list(production_ratios.values())
            )
            
            model_info = next(m for m in self.vehicle_models if m['model'] == model_choice)
            
            # Generate production date with seasonal patterns
            base_date = datetime(2024, 1, 1)
            days_offset = np.random.randint(0, 365)
            assembly_date = base_date + timedelta(days=days_offset)
            
            # Color selection based on market preferences
            color_preferences = {
                'White': 0.28, 'Black': 0.19, 'Silver': 0.16, 'Gray': 0.11,
                'Red': 0.10, 'Blue': 0.08, 'Brown': 0.04, 'Green': 0.02, 'Other': 0.02
            }
            color = np.random.choice(list(color_preferences.keys()), p=list(color_preferences.values()))
            
            # Market destination
            target_markets = model_info['target_market']
            target_market = np.random.choice(target_markets)
            
            # VIN generation (simplified but consistent)
            vin = f"VIN{100000 + i}"
            
            # Options and configuration
            trim_level = np.random.choice(['Base', 'Sport', 'Luxury', 'Premium'], p=[0.4, 0.3, 0.2, 0.1])
            engine_type = np.random.choice(['ICE', 'Hybrid', 'Electric'], p=[0.7, 0.25, 0.05])
            
            vehicles.append({
                'vin': vin,
                'model': model_choice,
                'segment': model_info['segment'],
                'color': color,
                'trim_level': trim_level,
                'engine_type': engine_type,
                'assembly_date': assembly_date.isoformat(),
                'assembly_shift': np.random.choice(['day', 'night', 'evening']),
                'assembly_line': np.random.choice(['Line_1', 'Line_2']),
                'target_market': target_market,
                'planned_delivery_date': (assembly_date + timedelta(days=np.random.randint(14, 45))).isoformat(),
                'msrp_usd': int(np.random.uniform(25000, 65000) * (1 + model_info['complexity'] * 0.2)),
                'production_sequence': i,
                'quality_gate_passed': np.random.choice([True, False], p=[0.97, 0.03])
            })
        
        return pd.DataFrame(vehicles)

    def _generate_assembly_events(self, vehicles_df: pd.DataFrame, batches_df: pd.DataFrame, stations_df: pd.DataFrame, parts_df: pd.DataFrame) -> pd.DataFrame:
        """Generate detailed assembly events with realistic workflow"""
        assembly_events = []

        # --- OPTIMIZATION: Create a fast part category lookup dictionary ONCE ---
        part_id_to_category = parts_df.set_index('part_id')['category'].to_dict()

        part_batches = {}
        for _, batch in batches_df.iterrows():
            part_id = batch['part_id']
            if part_id not in part_batches:
                part_batches[part_id] = []
            part_batches[part_id].append({
                'batch_id': batch['batch_id'],
                'manufacture_date': batch['manufacture_date'],
            })

        for part_id in part_batches:
            part_batches[part_id].sort(key=lambda x: x['manufacture_date'])

        batch_usage_tracker = {batch['batch_id']: 0 for _, batch in batches_df.iterrows()}
        batch_quantities = dict(zip(batches_df['batch_id'], batches_df['quantity']))

        station_mapping = {
            'Engine': ['STN-ENG-001', 'STN-ENG-002'],
            'Transmission': ['STN-TRN-001'],
            'Brake': ['STN-CHS-001'],
            'Suspension': ['STN-CHS-001'],
            'Electrical': ['STN-ASM-001', 'STN-ASM-002'],
            'Body': ['STN-BDY-001', 'STN-BDY-002']
        }

        for _, vehicle in tqdm(vehicles_df.iterrows(), total=self.n_vehicles, desc="Generating Assembly Events"):
            vehicle_assembly_events = []
            assembly_datetime = datetime.fromisoformat(vehicle['assembly_date'])

            for part_id in part_batches.keys():
                available_batches = [
                    b for b in part_batches[part_id]
                    if batch_usage_tracker[b['batch_id']] < batch_quantities[b['batch_id']] and
                    datetime.fromisoformat(b['manufacture_date']) <= assembly_datetime
                ]

                if not available_batches:
                    continue

                # --- FIX: Restored original weighted batch selection logic ---
                if len(available_batches) > 1:
                    weights = [1.0 / (i + 1) for i in range(len(available_batches))]
                    weights = np.array(weights) / np.sum(weights)
                    selected_batch_info = np.random.choice(available_batches, p=weights)
                else:
                    selected_batch_info = available_batches[0]

                selected_batch_id = selected_batch_info['batch_id']
                batch_usage_tracker[selected_batch_id] += 1
                # --- End of Fix ---

                part_category = part_id_to_category.get(part_id)
                possible_stations = station_mapping.get(part_category, ['STN-ASM-001', 'STN-ASM-002'])
                station_id = random.choice(possible_stations)

                assembly_timestamp = assembly_datetime.replace(
                    hour=np.random.randint(6, 22), 
                    minute=np.random.randint(0, 60)
                )

                vehicle_assembly_events.append({
                    'assembly_id': str(uuid.uuid4()),
                    'vin': vehicle['vin'],
                    'batch_id': selected_batch_id, # Use the correctly selected batch ID
                    'part_id': part_id,
                    'station_id': station_id,
                    'operator_id': f"OP-{np.random.randint(1, 150):03d}",
                    'assembly_timestamp': assembly_timestamp.isoformat(),
                    'shift': vehicle['assembly_shift'],
                    'sequence_number': len(vehicle_assembly_events) + 1,
                    'cycle_time_seconds': np.random.randint(45, 300),
                    'tool_used': np.random.choice(['Manual', 'Pneumatic', 'Electric', 'Hydraulic']),
                    'torque_spec_nm': np.random.randint(10, 200) if np.random.random() > 0.3 else None,
                    'installation_verified': np.random.choice([True, False], p=[0.98, 0.02])
                })

            assembly_events.extend(vehicle_assembly_events)

        return pd.DataFrame(assembly_events)
    
    def _generate_qc_inspections(self, vehicles_df: pd.DataFrame, assembly_df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive quality control inspection data."""
        qc_inspections = []

        inspection_types = [
            {'type': 'Visual', 'probability': 1.0, 'duration_min': 15},
            {'type': 'Dimensional', 'probability': 0.8, 'duration_min': 30},
            {'type': 'Functional', 'probability': 0.9, 'duration_min': 45},
            {'type': 'Road_Test', 'probability': 0.3, 'duration_min': 60},
            {'type': 'Final_Audit', 'probability': 1.0, 'duration_min': 90}
        ]

        issue_codes = {
            'ENG-001': {'description': 'Engine noise excessive', 'severity_range': [2, 4], 'probability': 0.02},
            'BRK-002': {'description': 'Brake pedal feel abnormal', 'severity_range': [3, 5], 'probability': 0.015},
            'ELC-003': {'description': 'Electrical system malfunction', 'severity_range': [1, 4], 'probability': 0.025},
            'BDY-004': {'description': 'Body panel misalignment', 'severity_range': [1, 3], 'probability': 0.03},
            'INT-005': {'description': 'Interior trim defect', 'severity_range': [1, 2], 'probability': 0.035},
            'SUS-006': {'description': 'Suspension noise', 'severity_range': [2, 3], 'probability': 0.01},
            'TRN-007': {'description': 'Transmission shift quality', 'severity_range': [2, 4], 'probability': 0.008}
        }

        # --- OPTIMIZATION: Determine anomalous vehicles once before the loop ---
        anomalous_batches_df = pd.read_csv(os.path.join(self.output_dir, 'anomaly_ground_truth.csv'))
        anomalous_batch_ids = set(anomalous_batches_df['batch_id'])
        vins_with_anomalous_parts = set(assembly_df[assembly_df['batch_id'].isin(anomalous_batch_ids)]['vin'])

        for _, vehicle in tqdm(vehicles_df.iterrows(), total=self.n_vehicles, desc="Generating QC Inspections"):
            has_anomalous_components = vehicle['vin'] in vins_with_anomalous_parts

            for inspection in inspection_types:
                if np.random.random() > inspection['probability']:
                    continue

                inspection_date = datetime.fromisoformat(vehicle['assembly_date']) + timedelta(hours=np.random.randint(1, 48))

                base_issue_prob = 0.05
                if has_anomalous_components:
                    base_issue_prob *= 3

                found_issue = np.random.random() < base_issue_prob

                if found_issue:
                    issue_probs = [info['probability'] for info in issue_codes.values()]
                    issue_probs = np.array(issue_probs) / np.sum(issue_probs)
                    selected_issue = np.random.choice(list(issue_codes.keys()), p=issue_probs)

                    issue_info = issue_codes[selected_issue]
                    severity = np.random.randint(*issue_info['severity_range'])
                    notes = issue_info['description']
                    passed = severity <= 2
                else:
                    selected_issue = None
                    severity = 0
                    notes = "Passed all checks"
                    passed = True

                qc_inspections.append({
                    'qc_id': str(uuid.uuid4()), 'vin': vehicle['vin'], 'inspection_type': inspection['type'],
                    'inspection_date': inspection_date.isoformat(), 'inspector_id': f'INS-{np.random.randint(1, 25):02d}',
                    'station_id': np.random.choice(['STN-QC-001', 'STN-QC-002']), 'issue_code': selected_issue,
                    'severity': severity, 'notes': notes, 'passed': passed,
                    'inspection_duration_min': inspection['duration_min'] + np.random.randint(-10, 20),
                    'corrective_action': 'Rework required' if not passed else 'None',
                    'reinspection_required': severity >= 3
                })

        return pd.DataFrame(qc_inspections)

    def _generate_failures(self, vehicles_df: pd.DataFrame, assembly_df: pd.DataFrame, batches_df: pd.DataFrame) -> pd.DataFrame:
        """Generate realistic field failures with traceability to batches"""
        failures = []
        
        # Create mapping of vehicles to their batches
        vehicle_batch_map = {}
        for _, assembly in assembly_df.iterrows():
            vin = assembly['vin']
            if vin not in vehicle_batch_map:
                vehicle_batch_map[vin] = []
            vehicle_batch_map[vin].append(assembly['batch_id'])
        
        # Get anomalous batches for higher failure correlation
        anomalous_batches = set(batches_df[batches_df['is_anomalous']]['batch_id'].tolist())
        
        # Calculate failure probability for each vehicle
        for _, vehicle in tqdm(vehicles_df.iterrows(), total=self.n_vehicles, desc="Generating Field Failures "):
            vin = vehicle['vin']
            vehicle_batches = vehicle_batch_map.get(vin, [])
            
            # Base failure probability
            base_failure_prob = 0.03  # 3% base failure rate
            
            # Increase probability if vehicle uses anomalous batches
            anomalous_batch_count = sum(1 for batch_id in vehicle_batches if batch_id in anomalous_batches)
            if anomalous_batch_count > 0:
                base_failure_prob *= (1 + anomalous_batch_count * 1.5)
            
            # Vehicle age factor (older vehicles more likely to fail)
            assembly_date = datetime.fromisoformat(vehicle['assembly_date'])
            vehicle_age_days = (datetime.now() - assembly_date).days
            age_factor = 1 + (vehicle_age_days / 365) * 0.5  # 50% increase per year
            
            adjusted_failure_prob = min(0.25, base_failure_prob * age_factor)
            
            if np.random.random() < adjusted_failure_prob:
                # Generate failure
                failure_date = assembly_date + timedelta(
                    days=np.random.randint(30, min(vehicle_age_days + 1, 365))
                )
                
                # Select failure category based on vehicle's components
                category_probs = {
                    'Engine': 0.25, 'Transmission': 0.15, 'Brake': 0.20,
                    'Electrical': 0.20, 'Suspension': 0.10, 'Body': 0.10
                }
                
                failure_category = np.random.choice(list(category_probs.keys()), 
                                                    p=list(category_probs.values()))
                
                failure_mode = np.random.choice(self.failure_patterns[failure_category])
                
                # Severity correlation with anomalous batches
                if anomalous_batch_count > 0:
                    severity = np.random.randint(3, 6)  # Higher severity
                else:
                    severity = np.random.randint(1, 4)  # Lower severity
                
                # Cost correlation with severity and category
                base_cost = {
                    'Engine': 1500, 'Transmission': 2000, 'Brake': 800,
                    'Electrical': 600, 'Suspension': 900, 'Body': 400
                }[failure_category]
                
                repair_cost = base_cost * (1 + severity * 0.3) * np.random.uniform(0.7, 1.4)
                
                failures.append({
                    'failure_id': str(uuid.uuid4()),
                    'vin': vin,
                    'reported_date': failure_date.isoformat(),
                    'failure_mode': failure_mode,
                    'component_category': failure_category,
                    'severity': severity,
                    'mileage_at_failure': int(np.random.uniform(5000, 80000)),
                    'warranty_claim': np.random.choice([True, False], p=[0.75, 0.25]),
                    'repair_cost': round(repair_cost, 2),
                    'downtime_days': np.random.randint(1, 14),
                    'dealer_code': f"DLR-{np.random.randint(1, 200):03d}",
                    'customer_complaint': self._generate_customer_complaint(failure_mode),
                    'root_cause_identified': np.random.choice([True, False], p=[0.7, 0.3]),
                    'recall_related': anomalous_batch_count > 0 and np.random.random() < 0.3
                })
        
        return pd.DataFrame(failures)

    def _generate_supplier_audits(self, suppliers_df: pd.DataFrame) -> pd.DataFrame:
        """Generate supplier audit records"""
        audits = []
        
        for _, supplier in suppliers_df.iterrows():
            # Number of audits based on supplier tier and risk
            n_audits = {1: 2, 2: 1, 3: 1}[supplier['tier']] + (1 if supplier['risk_score'] > 50 else 0)
            
            for audit_idx in range(n_audits):
                audit_date = fake.date_between(start_date='-365d', end_date='-30d')
                
                # Audit score based on supplier quality
                base_score = supplier['quality_rating'] * 100
                audit_score = base_score + np.random.uniform(-15, 10)
                audit_score = np.clip(audit_score, 60, 100)
                
                # Findings based on score
                if audit_score < 75:
                    findings = "Multiple non-conformances identified"
                    corrective_actions = "Immediate corrective action plan required"
                elif audit_score < 85:
                    findings = "Minor non-conformances noted"
                    corrective_actions = "Continuous improvement recommended"
                else:
                    findings = "No major issues identified"
                    corrective_actions = "Maintain current standards"
                
                audits.append({
                    'audit_id': str(uuid.uuid4()),
                    'supplier_id': supplier['supplier_id'],
                    'audit_date': audit_date.isoformat(),
                    'audit_type': np.random.choice(['ISO9001', 'TS16949', 'Process', 'System']),
                    'auditor_name': fake.name(),
                    'audit_score': round(audit_score, 1),
                    'findings': findings,
                    'corrective_actions_required': corrective_actions,
                    'follow_up_date': (audit_date + timedelta(days=30)).isoformat(),
                    'certification_status': 'Valid' if audit_score >= 80 else 'Conditional'
                })
        
        return pd.DataFrame(audits)

    def _generate_compliance_records(self, batches_df: pd.DataFrame) -> pd.DataFrame:
        """Generate regulatory compliance records"""
        compliance_records = []
        
        regulations = ['FMVSS', 'ECE', 'DOT', 'EPA', 'CARB', 'NHTSA']
        
        # Sample subset of batches for compliance testing
        sampled_batches = batches_df.sample(n=min(len(batches_df), 500))
        
        for _, batch in sampled_batches.iterrows():
            regulation = np.random.choice(regulations)
            test_date = datetime.fromisoformat(batch['manufacture_date']) + timedelta(days=np.random.randint(1, 30))
            
            # Compliance based on batch quality
            compliance_prob = 0.95 if not batch['is_anomalous'] else 0.75
            compliant = np.random.random() < compliance_prob
            
            compliance_records.append({
                'compliance_id': str(uuid.uuid4()),
                'batch_id': batch['batch_id'],
                'regulation': regulation,
                'test_date': test_date.isoformat(),
                'test_standard': f"{regulation}-{np.random.randint(100, 999)}",
                'compliant': compliant,
                'test_result': 'Pass' if compliant else 'Fail',
                'certificate_number': f"CERT-{np.random.randint(10000, 99999)}" if compliant else None,
                'expiry_date': (test_date + timedelta(days=365)).isoformat() if compliant else None,
                'testing_lab': np.random.choice(['Lab A', 'Lab B', 'Lab C', 'Internal'])
            })
        
        return pd.DataFrame(compliance_records)

    def _generate_maintenance_records(self) -> pd.DataFrame:
        """Generate manufacturing equipment maintenance records"""
        maintenance_records = []
        
        # Equipment list
        equipment_list = [
            'Robot_ARM_001', 'Robot_ARM_002', 'Conveyor_Belt_A', 'Conveyor_Belt_B',
            'Welding_Station_1', 'Welding_Station_2', 'Paint_Booth_1', 'Press_Machine_1',
            'CNC_Machine_1', 'Assembly_Jig_A', 'Quality_Scanner_1', 'Torque_Gun_001'
        ]
        
        for equipment in equipment_list:
            n_maintenance = np.random.randint(4, 12)  # 4-12 maintenance events per year
            
            for _ in range(n_maintenance):
                maintenance_date = fake.date_between(start_date='-365d', end_date='today')
                
                maintenance_type = np.random.choice(['Preventive', 'Corrective', 'Emergency'], p=[0.7, 0.25, 0.05])
                
                # Downtime based on maintenance type
                if maintenance_type == 'Emergency':
                    downtime_hours = np.random.randint(8, 48)
                elif maintenance_type == 'Corrective':
                    downtime_hours = np.random.randint(2, 12)
                else:
                    downtime_hours = np.random.randint(1, 4)
                
                maintenance_records.append({
                    'maintenance_id': str(uuid.uuid4()),
                    'equipment_id': equipment,
                    'maintenance_date': maintenance_date.isoformat(),
                    'maintenance_type': maintenance_type,
                    'technician_id': f"TECH-{np.random.randint(1, 20):02d}",
                    'downtime_hours': downtime_hours,
                    'cost_usd': np.random.randint(200, 5000),
                    'parts_replaced': np.random.choice(['Filter', 'Belt', 'Sensor', 'Motor', 'None'], p=[0.3, 0.2, 0.2, 0.1, 0.2]),
                    'next_maintenance_due': (maintenance_date + timedelta(days=np.random.randint(30, 90))).isoformat()
                })
        
        return pd.DataFrame(maintenance_records)

    def _generate_cost_analysis(self, batches_df: pd.DataFrame, suppliers_df: pd.DataFrame, parts_df: pd.DataFrame) -> pd.DataFrame:
        """Generate cost analysis data"""
        cost_records = []
        
        # Sample batches for cost analysis
        sampled_batches = batches_df.sample(n=min(len(batches_df), 1000))
        
        for _, batch in sampled_batches.iterrows():
            supplier = suppliers_df[suppliers_df['supplier_id'] == batch['supplier_id']].iloc[0]
            part = parts_df[parts_df['part_id'] == batch['part_id']].iloc[0]
            
            # Calculate various cost components
            material_cost = batch['batch_cost_usd'] * 0.6
            labor_cost = batch['batch_cost_usd'] * 0.25
            overhead_cost = batch['batch_cost_usd'] * 0.15
            
            # Quality cost (higher for anomalous batches)
            quality_cost = material_cost * 0.02
            if batch['is_anomalous']:
                quality_cost *= 3
            
            # Transportation cost based on supplier location
            transport_multiplier = {'Germany': 1.2, 'Japan': 1.3, 'China': 0.8, 'India': 0.7, 'USA': 1.0}
            transport_cost = material_cost * 0.05 * transport_multiplier.get(supplier['country'], 1.0)
            
            total_cost = material_cost + labor_cost + overhead_cost + quality_cost + transport_cost
            
            cost_records.append({
                'cost_id': str(uuid.uuid4()),
                'batch_id': batch['batch_id'],
                'supplier_id': batch['supplier_id'],
                'part_id': batch['part_id'],
                'material_cost_usd': round(material_cost, 2),
                'labor_cost_usd': round(labor_cost, 2),
                'overhead_cost_usd': round(overhead_cost, 2),
                'quality_cost_usd': round(quality_cost, 2),
                'transport_cost_usd': round(transport_cost, 2),
                'total_cost_usd': round(total_cost, 2),
                'cost_per_unit_usd': round(total_cost / batch['quantity'], 4),
                'currency': 'USD',
                'cost_date': batch['manufacture_date']
            })
        
        return pd.DataFrame(cost_records)
    
    def _generate_customer_complaint(self, failure_mode: str) -> str:
        """Generate realistic customer complaint text"""
        complaints = {
            'oil_leak': "Oil spots under vehicle, engine oil level dropping",
            'overheating': "Temperature gauge shows high, steam from engine",
            'misfire': "Engine runs rough, lacks power, check engine light on",
            'timing_failure': "Strange noise from engine, won't start properly",
            'bearing_wear': "Grinding noise from engine, metal particles in oil",
            'gear_slip': "Transmission slipping, won't stay in gear",
            'fluid_leak': "Red fluid leaking, transmission not shifting properly",
            'shift_delay': "Delayed engagement when shifting, harsh shifts",
            'torque_converter_failure': "Shuddering during acceleration, poor fuel economy",
            'pad_wear': "Squealing noise when braking, longer stopping distance",
            'disc_warping': "Vibration in steering wheel when braking",
            'caliper_seizure': "Vehicle pulls to one side when braking",
            'abs_malfunction': "ABS light on, strange pedal feel",
            'strut_leak': "Bouncing ride, oil on strut, poor handling",
            'bushing_wear': "Clunking noise over bumps, steering feels loose",
            'spring_fatigue': "Vehicle sits low on one side, harsh ride",
            'alignment_drift': "Vehicle pulls to one side, uneven tire wear",
            'ecu_failure': "Check engine light, poor performance, won't start",
            'sensor_malfunction': "Warning lights on dashboard, erratic behavior",
            'harness_corrosion': "Electrical issues, intermittent problems",
            'battery_drain': "Battery dies overnight, electrical components failing",
            'paint_defect': "Paint peeling, color mismatch, surface roughness",
            'rust_formation': "Rust spots appearing, metal corrosion",
            'seal_failure': "Water leaking into cabin, wind noise",
            'trim_detachment': "Interior trim pieces falling off, rattling"
        }
        return complaints.get(failure_mode, "Vehicle performance issues reported")
    
    def _get_region(self, country: str) -> str:
        """Map country to region"""
        region_map = {
            'Germany': 'Europe', 'Japan': 'Asia', 'USA': 'North America',
            'South Korea': 'Asia', 'India': 'Asia', 'China': 'Asia',
            'Mexico': 'North America', 'Brazil': 'South America',
            'Italy': 'Europe', 'Turkey': 'Europe', 'Vietnam': 'Asia',
            'Malaysia': 'Asia', 'Philippines': 'Asia'
        }
        return region_map.get(country, 'Other')
    
    def _generate_data_quality_report(self, datasets: Dict[str, pd.DataFrame]):
        """Generate comprehensive data quality report"""
        report_path = os.path.join(self.output_dir, 'data_quality_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=== ENHANCED MANUFACTURING DATA QUALITY REPORT ===\n\n")
            f.write(f"Generation Date: {datetime.now().isoformat()}\n")
            f.write(f"Total Datasets: {len(datasets)}\n\n")
            
            total_records = sum(len(df) for df in datasets.values())
            f.write(f"Total Records Generated: {total_records:,}\n\n")
            
            f.write("Dataset Summary:\n")
            for name, df in datasets.items():
                f.write(f"  {name}: {len(df):,} records, {len(df.columns)} columns\n")
            
            f.write(f"\nKey Metrics:\n")
            f.write(f"  Suppliers: {len(datasets['suppliers'])} ({len(datasets['suppliers'][datasets['suppliers']['tier'] == 1])} Tier 1)\n")
            f.write(f"  Parts Catalog: {len(datasets['parts'])} parts across {datasets['parts']['category'].nunique()} categories\n")
            f.write(f"  Manufacturing Batches: {len(datasets['batches']):,}\n")
            f.write(f"  Vehicles Produced: {len(datasets['vehicles']):,}\n")
            f.write(f"  Assembly Events: {len(datasets['assembly_events']):,}\n")
            f.write(f"  Quality Inspections: {len(datasets['qc_inspections']):,}\n")
            f.write(f"  Field Failures: {len(datasets['failures']):,} ({len(datasets['failures'])/len(datasets['vehicles']):.1%} failure rate)\n")
            
            # Anomaly statistics
            anomaly_count = len(datasets['batches'][datasets['batches']['is_anomalous']])
            f.write(f"\nAnomaly Statistics:\n")
            f.write(f"  Anomalous Batches: {anomaly_count} ({anomaly_count/len(datasets['batches']):.1%})\n")
            
            anomaly_types = datasets['batches'][datasets['batches']['is_anomalous']]['anomaly_type'].value_counts()
            for anomaly_type, count in anomaly_types.items():
                f.write(f"    {anomaly_type}: {count} batches\n")
            
            f.write(f"\nSupplier Distribution:\n")
            for tier in [1, 2, 3]:
                tier_suppliers = len(datasets['suppliers'][datasets['suppliers']['tier'] == tier])
                f.write(f"  Tier {tier}: {tier_suppliers} suppliers\n")
            
            f.write(f"\nGeographic Distribution:\n")
            country_dist = datasets['suppliers']['country'].value_counts()
            for country, count in country_dist.head(10).items():
                f.write(f"  {country}: {count} suppliers\n")
        
        print(f"✓ Data quality report -> {report_path}")
    
    def _generate_anomaly_report(self, batches_df: pd.DataFrame):
        """Generate detailed anomaly injection report"""
        anomalous_batches = batches_df[batches_df['is_anomalous']]
        
        report_path = os.path.join(self.output_dir, 'anomaly_injection_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=== ENHANCED ANOMALY INJECTION REPORT ===\n\n")
            f.write(f"Total Batches: {len(batches_df):,}\n")
            f.write(f"Anomalous Batches: {len(anomalous_batches):,}\n")
            f.write(f"Anomaly Rate: {len(anomalous_batches)/len(batches_df):.2%}\n\n")
            
            # Anomaly type distribution
            f.write("Anomaly Type Distribution:\n")
            anomaly_types = anomalous_batches['anomaly_type'].value_counts()
            for anomaly_type, count in anomaly_types.items():
                percentage = count / len(anomalous_batches) * 100
                f.write(f"  {anomaly_type}: {count} batches ({percentage:.1f}%)\n")
            
            # Supplier impact
            f.write(f"\nSupplier Impact:\n")
            supplier_anomalies = anomalous_batches.groupby('supplier_id').size().sort_values(ascending=False)
            for supplier_id, count in supplier_anomalies.head(10).items():
                f.write(f"  {supplier_id}: {count} anomalous batches\n")
            
            # Part category impact
            f.write(f"\nPart Category Impact:\n")
            # This would require joining with parts data
            f.write("  (Part category analysis would be generated with full dataset)\n")
            
            # Quality impact analysis
            f.write(f"\nQuality Impact:\n")
            normal_qc = batches_df[~batches_df['is_anomalous']]['qc_pass_rate'].mean()
            anomaly_qc = anomalous_batches['qc_pass_rate'].mean()
            f.write(f"  Average QC Rate (Normal): {normal_qc:.1%}\n")
            f.write(f"  Average QC Rate (Anomalous): {anomaly_qc:.1%}\n")
            f.write(f"  Quality Impact: {(normal_qc - anomaly_qc):.1%} reduction\n")
        
        print(f"✓ Anomaly injection report -> {report_path}")

    def _generate_supply_chain_map(self, suppliers_df: pd.DataFrame, parts_df: pd.DataFrame):
        """Generate supply chain mapping data"""
        map_data = {
            'suppliers': [],
            'parts': [],
            'relationships': []
        }
        
        # Supplier nodes
        for _, supplier in suppliers_df.iterrows():
            map_data['suppliers'].append({
                'id': supplier['supplier_id'],
                'name': supplier['name'],
                'country': supplier['country'],
                'tier': supplier['tier'],
                'specialization': supplier['specialization'],
                'quality_rating': supplier['quality_rating'],
                'risk_score': supplier['risk_score']
            })
        
        # Part nodes
        for _, part in parts_df.iterrows():
            map_data['parts'].append({
                'id': part['part_id'],
                'name': part['name'],
                'category': part['category'],
                'is_critical': part['is_critical'],
                'complexity_level': part['complexity_level']
            })
        
        # Save supply chain map
        map_path = os.path.join(self.output_dir, 'supply_chain_map.json')
        with open(map_path, 'w') as f:
            json.dump(map_data, f, indent=2)
        
        print(f"✓ Supply chain map -> {map_path}")

def main():
    """Generate all enhanced synthetic data"""
    print("=== ENHANCED CARTRACE AI DATA GENERATOR ===")
    print("Generating comprehensive manufacturing dataset...")
    
    generator = EnhancedManufacturingDataGenerator()
    datasets = generator.generate_all_data()
    
    print(f"\n=== ENHANCED DATA GENERATION COMPLETE ===")
    print(f"Total datasets created: {len(datasets)}")
    print(f"Total records: {sum(len(df) for df in datasets.values()):,}")
    print(f"Data directory: {generator.output_dir}")
    print(f"\nDatasets:")
    for name, df in datasets.items():
        print(f"  {name}: {len(df):,} records")
    
    print(f"\nReady for enhanced pipeline execution:")
    print(f"  1. Run: python run_pipeline.py")
    print(f"  2. Launch dashboard: streamlit run app.py")
    
    return datasets

if __name__ == "__main__":
    main()