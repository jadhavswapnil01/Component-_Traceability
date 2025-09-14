"""
Enhanced Neo4j Graph Manager for Manufacturing Traceability
Comprehensive graph operations with advanced traceability and analytics
"""

import pandas as pd
from py2neo import Graph, Node, Relationship, NodeMatcher
from typing import List, Dict, Optional, Tuple, Set, Any
import json
from datetime import datetime, timedelta
import logging
import networkx as nx
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedGraphManager:
    """Enhanced graph database manager for manufacturing traceability"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "cartrace123"):
        """Initialize connection to Neo4j with enhanced capabilities"""
        try:
            self.graph = Graph(uri, auth=(user, password))
            self.matcher = NodeMatcher(self.graph)
            logger.info("✓ Connected to Neo4j database")
            
            # Test connection
            self.graph.run("RETURN 1")
            
        except Exception as e:
            logger.error(f"✗ Failed to connect to Neo4j: {e}")
            raise
    
    def clear_database(self):
        """Clear all nodes, relationships, indexes, and constraints (compatible with Neo4j 4.x and 5.x)."""
        logger.info("Clearing the Neo4j database schema and data...")

        # Get and drop all constraints
        try:
            constraint_names = [r['name'] for r in self.graph.run("SHOW CONSTRAINTS YIELD name")]
            for name in constraint_names:
                self.graph.run(f"DROP CONSTRAINT {name}")
            if constraint_names:
                logger.info(f"✓ Dropped {len(constraint_names)} constraints")
        except Exception as e:
            logger.warning(f"Could not drop constraints (may not exist or new DB): {e}")

        # Get and drop all remaining indexes
        try:
            index_names = [r['name'] for r in self.graph.run("SHOW INDEXES YIELD name")]
            for name in index_names:
                try:
                    self.graph.run(f"DROP INDEX {name}")
                except Exception:
                    pass  # Ignore errors for indexes linked to constraints
            if index_names:
                logger.info(f"✓ Cleared {len(index_names)} indexes")
        except Exception as e:
            logger.warning(f"Could not drop indexes (may not exist or new DB): {e}")

        # Delete all nodes and relationships in batches
        logger.info("Clearing all nodes and relationships in batches...")
        while True:
            result = self.graph.run("""
                MATCH (n)
                WITH n LIMIT 50000
                DETACH DELETE n
                RETURN count(n) as deleted_count
            """).data()

            deleted_count = result[0]['deleted_count']
            if deleted_count > 0:
                logger.info(f"  ...deleted {deleted_count} nodes in this batch.")
            else:
                break # Exit loop when no more nodes are deleted

        logger.info("✓ Cleared all nodes and relationships")
    
    def create_comprehensive_indexes(self):
        """Create comprehensive indexes for optimal performance"""
        indexes_and_constraints = [
            # Unique constraints for primary keys
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Supplier) REQUIRE s.supplier_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Part) REQUIRE p.part_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (b:Batch) REQUIRE b.batch_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (v:Vehicle) REQUIRE v.vin IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (st:Station) REQUIRE st.station_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Failure) REQUIRE f.failure_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (qc:QCInspection) REQUIRE qc.qc_id IS UNIQUE",
            
            # Performance indexes for common queries
            "CREATE INDEX IF NOT EXISTS FOR (b:Batch) ON (b.manufacture_date)",
            "CREATE INDEX IF NOT EXISTS FOR (b:Batch) ON (b.is_anomalous)",
            "CREATE INDEX IF NOT EXISTS FOR (b:Batch) ON (b.supplier_id)",
            "CREATE INDEX IF NOT EXISTS FOR (b:Batch) ON (b.part_id)",
            "CREATE INDEX IF NOT EXISTS FOR (v:Vehicle) ON (v.assembly_date)",
            "CREATE INDEX IF NOT EXISTS FOR (v:Vehicle) ON (v.model)",
            "CREATE INDEX IF NOT EXISTS FOR (f:Failure) ON (f.reported_date)",
            "CREATE INDEX IF NOT EXISTS FOR (f:Failure) ON (f.component_category)",
            "CREATE INDEX IF NOT EXISTS FOR (qc:QCInspection) ON (qc.inspection_date)",
            "CREATE INDEX IF NOT EXISTS FOR (qc:QCInspection) ON (qc.passed)",
            "CREATE INDEX IF NOT EXISTS FOR (s:Supplier) ON (s.tier)",
            "CREATE INDEX IF NOT EXISTS FOR (s:Supplier) ON (s.quality_rating)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Part) ON (p.category)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Part) ON (p.is_critical)",
            
            # Composite indexes for complex queries
            "CREATE INDEX IF NOT EXISTS FOR (b:Batch) ON (b.is_anomalous, b.anomaly_type)",
            "CREATE INDEX IF NOT EXISTS FOR (b:Batch) ON (b.supplier_id, b.manufacture_date)",
            "CREATE INDEX IF NOT EXISTS FOR (v:Vehicle) ON (v.model, v.assembly_date)",
            
            # Full-text indexes for search functionality
            # Full-text indexes for search functionality
            "CREATE FULLTEXT INDEX vehicleSearch FOR (v:Vehicle) ON EACH [v.vin, v.model]",
            "CREATE FULLTEXT INDEX supplierSearch FOR (s:Supplier) ON EACH [s.name, s.supplier_id]",
            "CREATE FULLTEXT INDEX partSearch FOR (p:Part) ON EACH [p.name, p.part_id]"
        ]
        
        for index in indexes_and_constraints:
            try:
                self.graph.run(index)
            except Exception as e:
                logger.warning(f"Index creation issue (may already exist): {e}")
        
        logger.info("✓ Comprehensive database indexes created")
    
    def load_enhanced_manufacturing_data(self, datasets: Dict[str, pd.DataFrame]):
        """Load all manufacturing data with enhanced relationships"""
        logger.info("Loading enhanced manufacturing data into Neo4j...")
        
        # Clear and prepare database
        self.clear_database()
        self.create_comprehensive_indexes()
        
        # Load in dependency order with enhanced data
        self._load_suppliers(datasets['suppliers'])
        self._load_parts(datasets['parts'])
        self._load_stations(datasets['stations'])
        self._load_batches(datasets['batches'])
        self._load_vehicles(datasets['vehicles'])
        self._load_assembly_events(datasets['assembly_events'])
        self._load_qc_inspections(datasets['qc_inspections'])
        self._load_failures(datasets['failures'])
        
        # Load additional datasets if available
        if 'supplier_audits' in datasets:
            self._load_supplier_audits(datasets['supplier_audits'])
        if 'compliance_records' in datasets:
            self._load_compliance_records(datasets['compliance_records'])
        if 'maintenance_records' in datasets:
            self._load_maintenance_records(datasets['maintenance_records'])
        if 'cost_analysis' in datasets:
            self._load_cost_analysis(datasets['cost_analysis'])
        
        # Create advanced relationships
        self._create_advanced_relationships()
        
        logger.info("✓ Enhanced manufacturing data loaded into Neo4j")
    
    def _load_suppliers(self, suppliers_df: pd.DataFrame):
        """Load enhanced supplier data"""
        if suppliers_df.empty:
            return
        
        query = """
        UNWIND $rows AS s
        CREATE (sup:Supplier {
            supplier_id: s.supplier_id,
            name: s.name,
            country: s.country,
            region: s.region,
            tier: toInteger(s.tier),
            specialization: s.specialization,
            quality_rating: toFloat(s.quality_rating),
            history_issues: toInteger(s.history_issues),
            certifications: s.certifications,
            established_year: toInteger(s.established_year),
            annual_revenue_usd: toInteger(s.annual_revenue_usd),
            employee_count: toInteger(s.employee_count),
            lead_time_avg_days: toInteger(s.lead_time_avg_days),
            payment_terms: s.payment_terms,
            risk_score: toFloat(s.risk_score)
        })
        """
        rows = suppliers_df.fillna('').to_dict('records')
        self._run_chunked_unwind(query, 'rows', rows, chunk_size=100)
        logger.info(f"✓ Loaded {len(rows)} enhanced suppliers")
    
    def _load_parts(self, parts_df: pd.DataFrame):
        """Load enhanced parts data"""
        if parts_df.empty:
            return
        
        query = """
        UNWIND $rows AS p
        CREATE (part:Part {
            part_id: p.part_id,
            name: p.name,
            category: p.category,
            subcategory: p.subcategory,
            spec_weight_g: toFloat(p.spec_weight_g),
            spec_tolerance_pct: toFloat(p.spec_tolerance_pct),
            is_critical: toBoolean(p.is_critical),
            complexity_level: toInteger(p.complexity_level),
            unit_cost_base: toFloat(p.unit_cost_base),
            primary_material: p.primary_material,
            manufacturing_process: p.manufacturing_process,
            lifecycle_months: toInteger(p.lifecycle_months),
            replacement_interval_km: toInteger(p.replacement_interval_km),
            regulatory_compliance: p.regulatory_compliance,
            environmental_rating: p.environmental_rating
        })
        """
        rows = parts_df.fillna('').to_dict('records')
        self.graph.run(query, rows=rows)
        logger.info(f"✓ Loaded {len(rows)} enhanced parts")
    
    def _load_stations(self, stations_df: pd.DataFrame):
        """Load manufacturing stations"""
        if stations_df.empty:
            return
        
        query = """
        UNWIND $rows AS st
        CREATE (station:Station {
            station_id: st.station_id,
            name: st.name,
            category: st.category,
            daily_capacity: toInteger(st.daily_capacity),
            shift_pattern: st.shift_pattern,
            efficiency_rating: toFloat(st.efficiency_rating),
            maintenance_interval_hours: toInteger(st.maintenance_interval_hours),
            last_maintenance: st.last_maintenance,
            operator_count_per_shift: toInteger(st.operator_count_per_shift),
            automated_level: st.automated_level
        })
        """
        rows = stations_df.fillna('').to_dict('records')
        self.graph.run(query, rows=rows)
        logger.info(f"✓ Loaded {len(rows)} manufacturing stations")
    
    def _load_batches(self, batches_df: pd.DataFrame):
        """Load enhanced batch data with relationships"""
        if batches_df.empty:
            return
        
        query = """
        UNWIND $rows AS b
        MATCH (s:Supplier {supplier_id: b.supplier_id})
        MATCH (p:Part {part_id: b.part_id})
        CREATE (batch:Batch {
            batch_id: b.batch_id,
            supplier_batch_label: b.supplier_batch_label,
            quantity: toInteger(b.quantity),
            manufacture_date: b.manufacture_date,
            qc_pass_rate: toFloat(b.qc_pass_rate),
            weight_mean_g: toFloat(b.weight_mean_g),
            weight_std_g: toFloat(b.weight_std_g),
            lead_time_days: toInteger(b.lead_time_days),
            batch_cost_usd: toFloat(b.batch_cost_usd),
            is_anomalous: toBoolean(b.is_anomalous),
            anomaly_type: b.anomaly_type,
            production_shift: b.production_shift,
            operator_id: b.operator_id,
            temperature_c: toFloat(b.temperature_c),
            humidity_pct: toFloat(b.humidity_pct),
            material_lot: b.material_lot,
            inspection_level: b.inspection_level,
            storage_location: b.storage_location
        })
        CREATE (s)-[:SUPPLIES {
            lead_time: toInteger(b.lead_time_days),
            cost: toFloat(b.batch_cost_usd)
        }]->(batch)
        CREATE (batch)-[:CONTAINS {
            quantity: toInteger(b.quantity),
            specification_weight: toFloat(p.spec_weight_g)
        }]->(p)
        """
        rows = batches_df.fillna('').to_dict('records')
        self._run_chunked_unwind(query, 'rows', rows, chunk_size=200)
        logger.info(f"✓ Loaded {len(rows)} enhanced batches with relationships")
    
    def _load_vehicles(self, vehicles_df: pd.DataFrame):
        """Load enhanced vehicle data"""
        if vehicles_df.empty:
            return
        
        query = """
        UNWIND $rows AS v
        CREATE (vehicle:Vehicle {
            vin: v.vin,
            model: v.model,
            segment: v.segment,
            color: v.color,
            trim_level: v.trim_level,
            engine_type: v.engine_type,
            assembly_date: v.assembly_date,
            assembly_shift: v.assembly_shift,
            assembly_line: v.assembly_line,
            target_market: v.target_market,
            planned_delivery_date: v.planned_delivery_date,
            msrp_usd: toInteger(v.msrp_usd),
            production_sequence: toInteger(v.production_sequence),
            quality_gate_passed: toBoolean(v.quality_gate_passed)
        })
        """
        rows = vehicles_df.fillna('').to_dict('records')
        self._run_chunked_unwind(query, 'rows', rows, chunk_size=500)
        logger.info(f"✓ Loaded {len(rows)} enhanced vehicles")
    
    def _load_assembly_events(self, assembly_df: pd.DataFrame):
        """Load enhanced assembly events with detailed relationships"""
        if assembly_df.empty:
            return
        
        query = """
        UNWIND $events AS e
        MATCH (v:Vehicle {vin: e.vin})
        MATCH (b:Batch {batch_id: e.batch_id})
        MATCH (p:Part {part_id: e.part_id})
        OPTIONAL MATCH (st:Station {station_id: e.station_id})
        
        CREATE (assembly:AssemblyEvent {
            assembly_id: e.assembly_id,
            assembly_timestamp: e.assembly_timestamp,
            operator_id: e.operator_id,
            shift: e.shift,
            sequence_number: toInteger(e.sequence_number),
            cycle_time_seconds: toInteger(e.cycle_time_seconds),
            tool_used: e.tool_used,
            torque_spec_nm: CASE WHEN e.torque_spec_nm IS NULL THEN null ELSE toInteger(e.torque_spec_nm) END,
            installation_verified: toBoolean(e.installation_verified)
        })
        
        CREATE (v)-[:ASSEMBLED_WITH {
            timestamp: e.assembly_timestamp,
            sequence: toInteger(e.sequence_number),
            operator: e.operator_id
        }]->(assembly)
        
        CREATE (assembly)-[:USES_BATCH {
            batch_id: e.batch_id,
            part_id: e.part_id
        }]->(b)
        
        CREATE (assembly)-[:INSTALLS_PART]->(p)
        
        WITH assembly, st, e
        WHERE st IS NOT NULL
        CREATE (assembly)-[:AT_STATION {
            cycle_time: toInteger(e.cycle_time_seconds)
        }]->(st)
        """
        
        events = assembly_df.fillna('').to_dict('records')
        self._run_chunked_unwind(query, 'events', events, chunk_size=1000)
        logger.info(f"✓ Loaded {len(events)} enhanced assembly events")
    
    def _load_qc_inspections(self, qc_df: pd.DataFrame):
        """Load enhanced QC inspection data"""
        if qc_df.empty:
            return
        
        query = """
        UNWIND $rows AS qc
        MATCH (v:Vehicle {vin: qc.vin})
        OPTIONAL MATCH (st:Station {station_id: qc.station_id})
        
        CREATE (inspection:QCInspection {
            qc_id: qc.qc_id,
            inspection_type: qc.inspection_type,
            inspection_date: qc.inspection_date,
            inspector_id: qc.inspector_id,
            issue_code: qc.issue_code,
            severity: toInteger(qc.severity),
            notes: qc.notes,
            passed: toBoolean(qc.passed),
            inspection_duration_min: toInteger(qc.inspection_duration_min),
            corrective_action: qc.corrective_action,
            reinspection_required: toBoolean(qc.reinspection_required)
        })
        
        CREATE (v)-[:INSPECTED {
            inspection_type: qc.inspection_type,
            passed: toBoolean(qc.passed),
            severity: toInteger(qc.severity)
        }]->(inspection)
        
        WITH inspection, st
        WHERE st IS NOT NULL
        CREATE (inspection)-[:PERFORMED_AT]->(st)
        """
        rows = qc_df.fillna('').to_dict('records')
        self._run_chunked_unwind(query, 'rows', rows, chunk_size=500)
        logger.info(f"✓ Loaded {len(rows)} enhanced QC inspections")
    
    def _load_failures(self, failures_df: pd.DataFrame):
        """Load enhanced failure data with component traceability"""
        if failures_df.empty:
            return
        
        query = """
        UNWIND $rows AS f
        MATCH (v:Vehicle {vin: f.vin})
        CREATE (failure:Failure {
            failure_id: f.failure_id,
            reported_date: f.reported_date,
            failure_mode: f.failure_mode,
            component_category: f.component_category,
            severity: toInteger(f.severity),
            mileage_at_failure: toInteger(f.mileage_at_failure),
            warranty_claim: toBoolean(f.warranty_claim),
            repair_cost: toFloat(f.repair_cost),
            downtime_days: toInteger(f.downtime_days),
            dealer_code: f.dealer_code,
            customer_complaint: f.customer_complaint,
            root_cause_identified: toBoolean(f.root_cause_identified),
            recall_related: toBoolean(f.recall_related)
        })
        CREATE (v)-[:EXPERIENCED_FAILURE {
            reported_date: f.reported_date,
            severity: toInteger(f.severity),
            cost: toFloat(f.repair_cost)
        }]->(failure)
        """
        rows = failures_df.fillna('').to_dict('records')
        self.graph.run(query, rows=rows)
        logger.info(f"✓ Loaded {len(rows)} enhanced failures")
    
    def _load_supplier_audits(self, audits_df: pd.DataFrame):
        """Load supplier audit data"""
        if audits_df.empty:
            return
        
        query = """
        UNWIND $rows AS a
        MATCH (s:Supplier {supplier_id: a.supplier_id})
        CREATE (audit:SupplierAudit {
            audit_id: a.audit_id,
            audit_date: a.audit_date,
            audit_type: a.audit_type,
            auditor_name: a.auditor_name,
            audit_score: toFloat(a.audit_score),
            findings: a.findings,
            corrective_actions_required: a.corrective_actions_required,
            follow_up_date: a.follow_up_date,
            certification_status: a.certification_status
        })
        CREATE (s)-[:AUDITED {
            date: a.audit_date,
            score: toFloat(a.audit_score),
            status: a.certification_status
        }]->(audit)
        """
        rows = audits_df.fillna('').to_dict('records')
        self.graph.run(query, rows=rows)
        logger.info(f"✓ Loaded {len(rows)} supplier audits")
    
    def _load_compliance_records(self, compliance_df: pd.DataFrame):
        """Load compliance records"""
        if compliance_df.empty:
            return
        
        query = """
        UNWIND $rows AS c
        MATCH (b:Batch {batch_id: c.batch_id})
        CREATE (compliance:ComplianceRecord {
            compliance_id: c.compliance_id,
            regulation: c.regulation,
            test_date: c.test_date,
            test_standard: c.test_standard,
            compliant: toBoolean(c.compliant),
            test_result: c.test_result,
            certificate_number: c.certificate_number,
            expiry_date: c.expiry_date,
            testing_lab: c.testing_lab
        })
        CREATE (b)-[:COMPLIES_WITH {
            regulation: c.regulation,
            compliant: toBoolean(c.compliant),
            test_date: c.test_date
        }]->(compliance)
        """
        rows = compliance_df.fillna('').to_dict('records')
        self.graph.run(query, rows=rows)
        logger.info(f"✓ Loaded {len(rows)} compliance records")
    
    def _load_maintenance_records(self, maintenance_df: pd.DataFrame):
        """Load equipment maintenance records"""
        if maintenance_df.empty:
            return
        
        query = """
        UNWIND $rows AS m
        MERGE (equipment:Equipment {equipment_id: m.equipment_id})
        CREATE (maintenance:MaintenanceRecord {
            maintenance_id: m.maintenance_id,
            maintenance_date: m.maintenance_date,
            maintenance_type: m.maintenance_type,
            technician_id: m.technician_id,
            downtime_hours: toInteger(m.downtime_hours),
            cost_usd: toInteger(m.cost_usd),
            parts_replaced: m.parts_replaced,
            next_maintenance_due: m.next_maintenance_due
        })
        CREATE (equipment)-[:MAINTAINED {
            date: m.maintenance_date,
            type: m.maintenance_type,
            downtime: toInteger(m.downtime_hours)
        }]->(maintenance)
        """
        rows = maintenance_df.fillna('').to_dict('records')
        self.graph.run(query, rows=rows)
        logger.info(f"✓ Loaded {len(rows)} maintenance records")
    
    def _load_cost_analysis(self, cost_df: pd.DataFrame):
        """Load cost analysis data"""
        if cost_df.empty:
            return
        
        query = """
        UNWIND $rows AS c
        MATCH (b:Batch {batch_id: c.batch_id})
        CREATE (cost:CostAnalysis {
            cost_id: c.cost_id,
            material_cost_usd: toFloat(c.material_cost_usd),
            labor_cost_usd: toFloat(c.labor_cost_usd),
            overhead_cost_usd: toFloat(c.overhead_cost_usd),
            quality_cost_usd: toFloat(c.quality_cost_usd),
            transport_cost_usd: toFloat(c.transport_cost_usd),
            total_cost_usd: toFloat(c.total_cost_usd),
            cost_per_unit_usd: toFloat(c.cost_per_unit_usd),
            currency: c.currency,
            cost_date: c.cost_date
        })
        CREATE (b)-[:HAS_COST_ANALYSIS]->(cost)
        """
        rows = cost_df.fillna('').to_dict('records')
        self.graph.run(query, rows=rows)
        logger.info(f"✓ Loaded {len(rows)} cost analysis records")
    
    def _create_advanced_relationships(self):
        """Create advanced relationships for enhanced analytics"""

        logger.info("Creating advanced relationships...")

        # Link failures to components via assembly events
        failure_component_query = """
            MATCH (v:Vehicle)-[:EXPERIENCED_FAILURE]->(f:Failure)
            MATCH (v)-[:ASSEMBLED_WITH]->(ae:AssemblyEvent)-[:INSTALLS_PART]->(p:Part)
            WHERE p.category = f.component_category
            CREATE (f)-[:AFFECTS_COMPONENT {
                component_category: f.component_category,
                confidence: 0.8
            }]->(p)
        """
        self.graph.run(failure_component_query)

        # Create supplier performance relationships
        supplier_performance_query = """
            MATCH (s:Supplier)-[:SUPPLIES]->(b:Batch)
            WITH s,
                count(b) as total_batches,
                sum(CASE WHEN b.is_anomalous THEN 1 ELSE 0 END) as anomalous_batches,
                avg(b.qc_pass_rate) as avg_quality
            CREATE (s)-[:HAS_PERFORMANCE {
                total_batches: total_batches,
                anomalous_batches: anomalous_batches,
                anomaly_rate: toFloat(anomalous_batches) / total_batches,
                avg_quality: avg_quality,
                performance_score: avg_quality * (1 - toFloat(anomalous_batches) / total_batches)
            }]->(s)
        """
        self.graph.run(supplier_performance_query)

        # Create part reliability relationships
        part_reliability_query = """
            MATCH (p:Part)<-[:INSTALLS_PART]-(ae:AssemblyEvent)<-[:ASSEMBLED_WITH]-(v:Vehicle)
            OPTIONAL MATCH (v)-[:EXPERIENCED_FAILURE]->(f:Failure)
            WHERE f.component_category = p.category
            WITH p,
                count(DISTINCT v) as vehicles_with_part,
                count(DISTINCT f) as failures
            CREATE (p)-[:HAS_RELIABILITY {
                vehicles_using: vehicles_with_part,
                total_failures: failures,
                failure_rate: CASE WHEN vehicles_with_part > 0
                                THEN toFloat(failures) / vehicles_with_part
                                ELSE 0 END,
                reliability_score: CASE WHEN vehicles_with_part > 0
                                        THEN 1 - (toFloat(failures) / vehicles_with_part)
                                        ELSE 1 END
            }]->(p)
        """
        self.graph.run(part_reliability_query)

        logger.info("✓ Advanced relationships created")

    
    def _run_chunked_unwind(self, query: str, param_name: str, items: List[Dict], chunk_size: int = 200):
        """Helper: run an UNWIND query in chunks"""
        if not items:
            return
        
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            self.graph.run(query, **{param_name: chunk})
    
    def get_enhanced_vehicle_traceability(self, vin: str) -> Dict:
        """Get comprehensive vehicle traceability with enhanced details"""
        query = """
        MATCH (v:Vehicle {vin: $vin})
        OPTIONAL MATCH (v)-[:ASSEMBLED_WITH]->(ae:AssemblyEvent)-[:USES_BATCH]->(b:Batch)-[:CONTAINS]->(p:Part)
        OPTIONAL MATCH (b)<-[:SUPPLIES]-(s:Supplier)
        OPTIONAL MATCH (v)-[:INSPECTED]->(qc:QCInspection)
        OPTIONAL MATCH (v)-[:EXPERIENCED_FAILURE]->(f:Failure)
        OPTIONAL MATCH (ae)-[:AT_STATION]->(st:Station)
        
        RETURN v,
               collect(DISTINCT {
                   assembly_event: ae,
                   batch: b,
                   part: p,
                   supplier: s,
                   station: st,
                   assembly_timestamp: ae.assembly_timestamp,
                   operator: ae.operator_id,
                   verified: ae.installation_verified
               }) as components,
               collect(DISTINCT qc) as qc_inspections,
               collect(DISTINCT f) as failures
        """
        
        result = self.graph.run(query, vin=vin).data()
        if not result:
            return None
        
        data = result[0]
        
        # Process components with null filtering
        processed_components = []
        for comp in data['components']:
            if comp['batch'] is not None and comp['part'] is not None:
                processed_components.append({
                    'assembly_event': dict(comp['assembly_event']) if comp['assembly_event'] else None,
                    'batch': dict(comp['batch']),
                    'part': dict(comp['part']),
                    'supplier': dict(comp['supplier']) if comp['supplier'] else None,
                    'station': dict(comp['station']) if comp['station'] else None,
                    'assembly_details': {
                        'timestamp': comp['assembly_timestamp'],
                        'operator': comp['operator'],
                        'verified': comp['verified']
                    }
                })
        
        return {
            'vehicle': dict(data['v']),
            'components': processed_components,
            'qc_inspections': [dict(qc) for qc in data['qc_inspections'] if qc],
            'failures': [dict(f) for f in data['failures'] if f],
            'summary': {
                'total_components': len(processed_components),
                'anomalous_components': len([c for c in processed_components if c['batch'].get('is_anomalous', False)]),
                'qc_inspections_count': len([qc for qc in data['qc_inspections'] if qc]),
                'failures_count': len([f for f in data['failures'] if f]),
                'suppliers_involved': len(set(c['supplier']['supplier_id'] for c in processed_components if c['supplier']))
            }
        }
    
    def get_enhanced_batch_usage(self, batch_id: str) -> Dict:
        """Get enhanced batch usage information with full traceability"""
        query = """
        MATCH (b:Batch {batch_id: $batch_id})
        MATCH (b)-[:CONTAINS]->(p:Part)
        MATCH (s:Supplier)-[:SUPPLIES]->(b)
        OPTIONAL MATCH (ae:AssemblyEvent)-[:USES_BATCH]->(b)
        OPTIONAL MATCH (ae)<-[:ASSEMBLED_WITH]-(v:Vehicle)
        OPTIONAL MATCH (v)-[:EXPERIENCED_FAILURE]->(f:Failure)
        WHERE f.component_category = p.category
        OPTIONAL MATCH (ae)-[:AT_STATION]->(st:Station)
        
        WITH b, p, s, v, ae, st, collect(DISTINCT f) as vehicle_failures
        
        RETURN b, p, s,
               collect(DISTINCT {
                   vehicle: v,
                   assembly_event: ae,
                   station: st,
                   failures: vehicle_failures,
                   assembly_timestamp: ae.assembly_timestamp,
                   operator: ae.operator_id
               }) as usage_details
        """
        
        result = self.graph.run(query, batch_id=batch_id).data()
        if not result:
            return None
        
        data = result[0]
        
        # Process usage details with null filtering
        vehicles_data = []
        for usage in data['usage_details']:
            if usage['vehicle'] is not None:
                vehicles_data.append({
                    'vehicle': dict(usage['vehicle']),
                    'assembly_event': dict(usage['assembly_event']) if usage['assembly_event'] else None,
                    'station': dict(usage['station']) if usage['station'] else None,
                    'failures': [dict(f) for f in usage['failures'] if f],
                    'assembly_details': {
                        'timestamp': usage['assembly_timestamp'],
                        'operator': usage['operator']
                    }
                })
        
        batch_info = dict(data['b'])
        part_info = dict(data['p'])
        supplier_info = dict(data['s'])
        
        return {
            'batch': batch_info,
            'part': part_info,
            'supplier': supplier_info,
            'vehicles': vehicles_data,
            'summary': {
                'total_vehicles': len(vehicles_data),
                'vehicles_with_failures': len([v for v in vehicles_data if v['failures']]),
                'failure_rate': len([v for v in vehicles_data if v['failures']]) / max(len(vehicles_data), 1),
                'is_anomalous_batch': batch_info.get('is_anomalous', False),
                'anomaly_type': batch_info.get('anomaly_type', 'none')
            }
        }
    
    def simulate_comprehensive_recall(self, batch_id: str) -> Dict:
        """Enhanced recall simulation with comprehensive impact analysis"""
        # Get detailed batch usage
        batch_data = self.get_enhanced_batch_usage(batch_id)
        if not batch_data:
            return {'error': f'Batch {batch_id} not found'}
        
        vehicles = batch_data['vehicles']
        batch_info = batch_data['batch']
        part_info = batch_data['part']
        supplier_info = batch_data['supplier']
        
        if not vehicles:
            return {'error': f'No vehicles found using batch {batch_id}'}
        
        # Enhanced impact analysis
        total_vehicles = len(vehicles)
        vehicles_with_failures = len([v for v in vehicles if v['failures']])
        failure_rate = vehicles_with_failures / total_vehicles if total_vehicles > 0 else 0
        
        # Cost analysis
        base_recall_cost = 750  # Updated base cost
        notification_cost = total_vehicles * 25
        inspection_cost = total_vehicles * 150
        repair_costs = sum([
            sum([f['repair_cost'] for f in v['failures']]) 
            for v in vehicles if v['failures']
        ])
        
        # Logistics and administrative costs
        logistics_cost = total_vehicles * 50
        admin_cost = max(50000, total_vehicles * 30)  # Minimum administrative overhead
        
        estimated_total_cost = (
            notification_cost + inspection_cost + repair_costs + 
            logistics_cost + admin_cost
        )
        
        # Enhanced priority assessment
        is_critical_part = part_info.get('is_critical', False)
        is_anomalous = batch_info.get('is_anomalous', False)
        complexity_level = part_info.get('complexity_level', 1)
        
        high_severity_failures = any([
            any([f['severity'] >= 4 for f in v['failures']]) 
            for v in vehicles if v['failures']
        ])
        
        safety_critical_failures = any([
            any([f['component_category'] in ['Engine', 'Brake', 'Suspension'] and f['severity'] >= 3 
                 for f in v['failures']]) 
            for v in vehicles if v['failures']
        ])
        
        # Priority matrix
        if safety_critical_failures or (is_critical_part and high_severity_failures):
            priority = "CRITICAL"
            urgency_hours = 24
        elif is_critical_part and (failure_rate > 0.1 or is_anomalous):
            priority = "HIGH"
            urgency_hours = 72
        elif is_anomalous or failure_rate > 0.05 or high_severity_failures:
            priority = "MEDIUM"
            urgency_hours = 168  # 1 week
        else:
            priority = "LOW"
            urgency_hours = 720  # 30 days
        
        # Market segment analysis
        target_markets = list(set([v['vehicle'].get('target_market', 'Unknown') for v in vehicles]))
        vehicle_models = list(set([v['vehicle'].get('model', 'Unknown') for v in vehicles]))
        
        # Geographic impact (simplified)
        geographic_impact = self._analyze_geographic_impact(vehicles)
        
        # Supply chain impact
        supply_chain_impact = self._analyze_supply_chain_impact(supplier_info, batch_info)
        
        return {
            'batch_info': {
                'batch_id': batch_id,
                'part_name': part_info['name'],
                'part_category': part_info['category'],
                'supplier_name': supplier_info['name'],
                'supplier_tier': supplier_info.get('tier', 'Unknown'),
                'is_anomalous': is_anomalous,
                'anomaly_type': batch_info.get('anomaly_type', 'none'),
                'manufacture_date': batch_info.get('manufacture_date'),
                'is_critical_part': is_critical_part
            },
            'impact_analysis': {
                'total_vehicles': total_vehicles,
                'vehicles_with_failures': vehicles_with_failures,
                'failure_rate': round(failure_rate, 4),
                'affected_models': vehicle_models,
                'target_markets': target_markets,
                'geographic_scope': geographic_impact,
                'estimated_total_cost': round(estimated_total_cost, 2),
                'cost_breakdown': {
                    'notification': notification_cost,
                    'inspection': inspection_cost,
                    'repairs': round(repair_costs, 2),
                    'logistics': logistics_cost,
                    'administrative': admin_cost
                }
            },
            'priority_assessment': {
                'priority': priority,
                'urgency_hours': urgency_hours,
                'safety_critical': safety_critical_failures,
                'regulatory_impact': is_critical_part,
                'media_attention_risk': priority in ['CRITICAL', 'HIGH']
            },
            'affected_vehicles': [v['vehicle']['vin'] for v in vehicles],
            'supply_chain_impact': supply_chain_impact,
            'recommendations': self._generate_enhanced_recommendations(
                priority, batch_info, part_info, failure_rate, vehicles
            ),
            'timeline': self._generate_recall_timeline(priority, total_vehicles),
            'regulatory_requirements': self._get_regulatory_requirements(
                is_critical_part, safety_critical_failures, target_markets
            )
        }
    
    def _analyze_geographic_impact(self, vehicles: List[Dict]) -> Dict:
        """Analyze geographic distribution of affected vehicles"""
        # Simplified geographic analysis based on target markets
        markets = {}
        for vehicle in vehicles:
            market = vehicle['vehicle'].get('target_market', 'Unknown')
            markets[market] = markets.get(market, 0) + 1
        
        return {
            'markets': markets,
            'primary_market': max(markets.keys(), key=markets.get) if markets else 'Unknown',
            'international_scope': len([m for m in markets.keys() if 'Export' in m]) > 0
        }
    
    def _analyze_supply_chain_impact(self, supplier_info: Dict, batch_info: Dict) -> Dict:
        """Analyze broader supply chain implications"""
        return {
            'supplier_risk_increase': True if batch_info.get('is_anomalous') else False,
            'supplier_tier': supplier_info.get('tier'),
            'alternative_suppliers_needed': batch_info.get('is_anomalous', False),
            'production_line_impact': 'High' if batch_info.get('is_anomalous') else 'Medium',
            'inventory_review_needed': True
        }
    
    def _generate_enhanced_recommendations(self, priority: str, batch_info: Dict, 
                                        part_info: Dict, failure_rate: float, 
                                        vehicles: List[Dict]) -> List[str]:
        """Generate comprehensive recall recommendations"""
        recommendations = []
        
        # Priority-based recommendations
        if priority == "CRITICAL":
            recommendations.append("IMMEDIATE ACTION: Stop production and quarantine all related inventory")
            recommendations.append("Issue emergency safety notice to all affected customers within 24 hours")
            recommendations.append("Coordinate with regulatory authorities immediately")
        elif priority == "HIGH":
            recommendations.append("Urgent: Issue recall notice within 72 hours")
            recommendations.append("Implement enhanced inspection protocols for similar components")
        
        # Anomaly-based recommendations
        if batch_info.get('is_anomalous'):
            recommendations.append(f"Investigate root cause of {batch_info.get('anomaly_type')} anomaly")
            recommendations.append("Review all batches from same supplier and time period")
        
        # Failure pattern recommendations
        if failure_rate > 0.1:
            recommendations.append("High failure rate indicates systemic issue - expand investigation scope")
        
        # Part-specific recommendations
        if part_info.get('is_critical'):
            recommendations.append("Critical part affected - prioritize safety assessment")
        
        # Supplier recommendations
        recommendations.append("Conduct immediate supplier audit and corrective action plan")
        
        # Customer communication
        recommendations.append("Develop clear customer communication strategy")
        recommendations.append("Set up dedicated customer service hotline")
        
        return recommendations
    
    def _generate_recall_timeline(self, priority: str, vehicle_count: int) -> Dict:
        """Generate realistic recall timeline"""
        if priority == "CRITICAL":
            return {
                "notification": "0-24 hours",
                "customer_contact": "24-48 hours", 
                "repair_availability": "48-72 hours",
                "completion_target": f"{min(30, vehicle_count//50 + 7)} days"
            }
        elif priority == "HIGH":
            return {
                "notification": "0-72 hours",
                "customer_contact": "3-7 days",
                "repair_availability": "7-14 days", 
                "completion_target": f"{min(90, vehicle_count//30 + 14)} days"
            }
        else:
            return {
                "notification": "1-2 weeks",
                "customer_contact": "2-4 weeks",
                "repair_availability": "4-8 weeks",
                "completion_target": f"{min(180, vehicle_count//20 + 30)} days"
            }
    
    def _get_regulatory_requirements(self, is_critical: bool, safety_critical: bool, 
                                   markets: List[str]) -> List[str]:
        """Get applicable regulatory requirements"""
        requirements = []
        
        if safety_critical:
            requirements.append("NHTSA notification required within 5 days")
        
        if is_critical:
            requirements.append("DOT safety defect investigation may be required")
        
        if any('Export' in market for market in markets):
            requirements.append("International regulatory coordination required")
            requirements.append("ECE compliance verification needed")
        
        requirements.append("Maintain detailed records for regulatory audit")
        requirements.append("Customer notification per TREAD Act requirements")
        
        return requirements
    
    def get_supplier_risk_analysis(self, supplier_id: str = None) -> List[Dict]:
        """Enhanced supplier performance and risk analysis"""
        if supplier_id:
            query = """
            MATCH (s:Supplier {supplier_id: $supplier_id})
            OPTIONAL MATCH (s)-[:SUPPLIES]->(b:Batch)
            OPTIONAL MATCH (ae:AssemblyEvent)-[:USES_BATCH]->(b)
            OPTIONAL MATCH (ae)<-[:ASSEMBLED_WITH]-(v:Vehicle)
            OPTIONAL MATCH (v)-[:EXPERIENCED_FAILURE]->(f:Failure)
            OPTIONAL MATCH (s)-[:AUDITED]->(audit:SupplierAudit)
            
            RETURN s,
                   count(DISTINCT b) as total_batches,
                   count(DISTINCT CASE WHEN b.is_anomalous THEN b END) as anomalous_batches,
                   count(DISTINCT v) as total_vehicles,
                   count(DISTINCT f) as total_failures,
                   avg(b.qc_pass_rate) as avg_qc_rate,
                   avg(b.batch_cost_usd) as avg_batch_cost,
                   collect(DISTINCT audit.audit_score) as audit_scores,
                   max(audit.audit_date) as latest_audit
            """
            results = self.graph.run(query, supplier_id=supplier_id).data()
        else:
            query = """
            MATCH (s:Supplier)
            OPTIONAL MATCH (s)-[:SUPPLIES]->(b:Batch)
            OPTIONAL MATCH (ae:AssemblyEvent)-[:USES_BATCH]->(b)
            OPTIONAL MATCH (ae)<-[:ASSEMBLED_WITH]-(v:Vehicle)
            OPTIONAL MATCH (v)-[:EXPERIENCED_FAILURE]->(f:Failure)
            OPTIONAL MATCH (s)-[:AUDITED]->(audit:SupplierAudit)
            
            RETURN s,
                   count(DISTINCT b) as total_batches,
                   count(DISTINCT CASE WHEN b.is_anomalous THEN b END) as anomalous_batches,
                   count(DISTINCT v) as total_vehicles,
                   count(DISTINCT f) as total_failures,
                   avg(b.qc_pass_rate) as avg_qc_rate,
                   avg(b.batch_cost_usd) as avg_batch_cost,
                   collect(DISTINCT audit.audit_score) as audit_scores,
                   max(audit.audit_date) as latest_audit
            ORDER BY s.supplier_id
            """
            results = self.graph.run(query).data()
        
        analysis_data = []
        for result in results:
            supplier = dict(result['s'])
            
            # Calculate risk metrics
            total_batches = result['total_batches'] or 0
            anomalous_batches = result['anomalous_batches'] or 0
            total_vehicles = result['total_vehicles'] or 0
            total_failures = result['total_failures'] or 0
            
            anomaly_rate = anomalous_batches / max(total_batches, 1)
            failure_rate = total_failures / max(total_vehicles, 1)
            
            # Audit score analysis
            audit_scores = [score for score in result['audit_scores'] if score is not None]
            avg_audit_score = sum(audit_scores) / len(audit_scores) if audit_scores else 0
            
            # Risk score calculation
            base_risk = supplier.get('risk_score', 50)
            performance_risk = (anomaly_rate * 30) + (failure_rate * 40)
            audit_risk = max(0, (90 - avg_audit_score) / 2) if avg_audit_score > 0 else 20
            
            overall_risk = min(100, base_risk + performance_risk + audit_risk)
            
            # Risk category
            if overall_risk >= 80:
                risk_category = "HIGH"
            elif overall_risk >= 60:
                risk_category = "MEDIUM"
            elif overall_risk >= 40:
                risk_category = "LOW"
            else:
                risk_category = "MINIMAL"
            
            analysis_data.append({
                'supplier': supplier,
                'performance_metrics': {
                    'total_batches': total_batches,
                    'anomalous_batches': anomalous_batches,
                    'anomaly_rate': round(anomaly_rate, 4),
                    'total_vehicles': total_vehicles,
                    'total_failures': total_failures,
                    'failure_rate': round(failure_rate, 4),
                    'avg_qc_rate': round(result['avg_qc_rate'] or 0, 4),
                    'avg_batch_cost': round(result['avg_batch_cost'] or 0, 2)
                },
                'risk_analysis': {
                    'overall_risk_score': round(overall_risk, 1),
                    'risk_category': risk_category,
                    'base_risk': base_risk,
                    'performance_risk': round(performance_risk, 1),
                    'audit_risk': round(audit_risk, 1),
                    'avg_audit_score': round(avg_audit_score, 1) if avg_audit_score > 0 else None,
                    'latest_audit': result['latest_audit']
                },
                'recommendations': self._generate_supplier_recommendations(
                    risk_category, anomaly_rate, failure_rate, avg_audit_score
                )
            })
        
        return analysis_data
    
    def _generate_supplier_recommendations(self, risk_category: str, anomaly_rate: float, 
                                         failure_rate: float, audit_score: float) -> List[str]:
        """Generate supplier-specific recommendations"""
        recommendations = []
        
        if risk_category == "HIGH":
            recommendations.append("URGENT: Conduct immediate supplier audit")
            recommendations.append("Consider supplier probation or replacement")
            recommendations.append("Increase incoming inspection frequency to 100%")
        elif risk_category == "MEDIUM":
            recommendations.append("Schedule comprehensive supplier review within 30 days")
            recommendations.append("Implement enhanced quality monitoring")
        
        if anomaly_rate > 0.1:
            recommendations.append("High anomaly rate - investigate process control")
        
        if failure_rate > 0.05:
            recommendations.append("Elevated failure rate - review component design and manufacturing")
        
        if audit_score < 80 and audit_score > 0:
            recommendations.append("Poor audit performance - require corrective action plan")
        
        if not audit_score:
            recommendations.append("No recent audit data - schedule audit immediately")
        
        return recommendations
    
    def find_affected_vehicles_by_component_failure(self, failed_component_category: str, 
                                                   failure_mode: str = None) -> Dict:
        """Find all vehicles that could be affected by a specific component failure"""
        query = """
        // Find all batches for parts in the failed component category
        MATCH (p:Part {category: $component_category})
        MATCH (b:Batch)-[:CONTAINS]->(p)
        MATCH (s:Supplier)-[:SUPPLIES]->(b)
        
        // Find vehicles using these batches
        MATCH (ae:AssemblyEvent)-[:USES_BATCH]->(b)
        MATCH (v:Vehicle)-[:ASSEMBLED_WITH]->(ae)
        
        // Check for existing failures of this type
        OPTIONAL MATCH (v)-[:EXPERIENCED_FAILURE]->(f:Failure)
        WHERE f.component_category = $component_category
        AND ($failure_mode IS NULL OR f.failure_mode = $failure_mode)
        
        // Group by batch for analysis
        WITH b, p, s, 
             collect(DISTINCT {
                 vehicle: v,
                 existing_failure: f,
                 assembly_event: ae
             }) as vehicle_info
        
        RETURN b, p, s, vehicle_info,
               size([vi IN vehicle_info WHERE vi.existing_failure IS NOT NULL]) as confirmed_failures,
               size(vehicle_info) as total_vehicles_with_batch
        ORDER BY confirmed_failures DESC, b.is_anomalous DESC
        """
        
        results = self.graph.run(query, 
                               component_category=failed_component_category,
                               failure_mode=failure_mode).data()
        
        if not results:
            return {
                'component_category': failed_component_category,
                'failure_mode': failure_mode,
                'affected_batches': [],
                'total_at_risk_vehicles': 0,
                'confirmed_failures': 0
            }
        
        affected_batches = []
        total_at_risk = 0
        total_confirmed = 0
        
        for result in results:
            batch = dict(result['b'])
            part = dict(result['p'])
            supplier = dict(result['s'])
            confirmed_failures = result['confirmed_failures']
            total_vehicles = result['total_vehicles_with_batch']
            
            vehicle_details = []
            for vi in result['vehicle_info']:
                if vi['vehicle']:
                    vehicle_details.append({
                        'vehicle': dict(vi['vehicle']),
                        'has_failure': vi['existing_failure'] is not None,
                        'failure_details': dict(vi['existing_failure']) if vi['existing_failure'] else None,
                        'assembly_timestamp': vi['assembly_event']['assembly_timestamp']
                    })
            
            # Risk assessment for this batch
            risk_level = "HIGH" if batch.get('is_anomalous', False) else "MEDIUM"
            if confirmed_failures / max(total_vehicles, 1) > 0.1:
                risk_level = "CRITICAL"
            
            affected_batches.append({
                'batch': batch,
                'part': part,
                'supplier': supplier,
                'total_vehicles': total_vehicles,
                'confirmed_failures': confirmed_failures,
                'failure_rate': confirmed_failures / max(total_vehicles, 1),
                'risk_level': risk_level,
                'vehicles': vehicle_details
            })
            
            total_at_risk += total_vehicles
            total_confirmed += confirmed_failures
        
        return {
            'component_category': failed_component_category,
            'failure_mode': failure_mode,
            'affected_batches': affected_batches,
            'total_at_risk_vehicles': total_at_risk,
            'confirmed_failures': total_confirmed,
            'overall_failure_rate': total_confirmed / max(total_at_risk, 1),
            'high_risk_batches': len([b for b in affected_batches if b['risk_level'] in ['HIGH', 'CRITICAL']]),
            'recommendations': self._generate_component_failure_recommendations(
                failed_component_category, affected_batches, total_confirmed / max(total_at_risk, 1)
            )
        }
    
    def _generate_component_failure_recommendations(self, component_category: str, 
                                                  batches: List[Dict], overall_failure_rate: float) -> List[str]:
        """Generate recommendations for component failure analysis"""
        recommendations = []
        
        critical_batches = [b for b in batches if b['risk_level'] == 'CRITICAL']
        anomalous_batches = [b for b in batches if b['batch'].get('is_anomalous', False)]
        
        if critical_batches:
            recommendations.append(f"CRITICAL: {len(critical_batches)} batches show high failure rates - immediate recall consideration")
        
        if overall_failure_rate > 0.05:
            recommendations.append(f"Elevated failure rate ({overall_failure_rate:.1%}) indicates systematic issue")
        
        if anomalous_batches:
            recommendations.append(f"{len(anomalous_batches)} anomalous batches detected - prioritize investigation")
        
        if component_category in ['Engine', 'Brake', 'Suspension']:
            recommendations.append("Safety-critical component affected - regulatory notification may be required")
        
        recommendations.append("Implement enhanced monitoring for similar components")
        recommendations.append("Contact customers proactively for high-risk vehicles")
        
        return recommendations
    
    def get_supply_chain_analytics(self) -> Dict:
        """Get comprehensive supply chain analytics"""
        query = """
        // Supplier tier distribution and performance
        MATCH (s:Supplier)
        OPTIONAL MATCH (s)-[:SUPPLIES]->(b:Batch)
        OPTIONAL MATCH (ae:AssemblyEvent)-[:USES_BATCH]->(b)
        OPTIONAL MATCH (ae)<-[:ASSEMBLED_WITH]-(v:Vehicle)
        OPTIONAL MATCH (v)-[:EXPERIENCED_FAILURE]->(f:Failure)
        
        RETURN s.tier as tier,
               s.region as region,
               s.specialization as specialization,
               count(DISTINCT s) as supplier_count,
               count(DISTINCT b) as total_batches,
               count(DISTINCT CASE WHEN b.is_anomalous THEN b END) as anomalous_batches,
               count(DISTINCT v) as total_vehicles,
               count(DISTINCT f) as total_failures,
               avg(s.quality_rating) as avg_quality_rating,
               avg(s.risk_score) as avg_risk_score
        ORDER BY tier, region, specialization
        """
        
        results = self.graph.run(query).data()
        
        # Process results into analytics structure
        tier_analysis = defaultdict(lambda: {
            'supplier_count': 0,
            'total_batches': 0,
            'anomalous_batches': 0,
            'total_vehicles': 0,
            'total_failures': 0,
            'avg_quality_rating': 0,
            'avg_risk_score': 0
        })
        
        region_analysis = defaultdict(lambda: {
            'supplier_count': 0,
            'total_batches': 0,
            'anomaly_rate': 0,
            'failure_rate': 0
        })
        
        specialization_analysis = defaultdict(lambda: {
            'supplier_count': 0,
            'performance_score': 0,
            'risk_level': 'LOW'
        })
        
        for result in results:
            tier = result['tier']
            region = result['region']
            specialization = result['specialization']
            
            # Tier analysis
            tier_data = tier_analysis[tier]
            tier_data['supplier_count'] += result['supplier_count']
            tier_data['total_batches'] += result['total_batches'] or 0
            tier_data['anomalous_batches'] += result['anomalous_batches'] or 0
            tier_data['total_vehicles'] += result['total_vehicles'] or 0
            tier_data['total_failures'] += result['total_failures'] or 0
            tier_data['avg_quality_rating'] = result['avg_quality_rating'] or 0
            tier_data['avg_risk_score'] = result['avg_risk_score'] or 0
            
            # Region analysis
            region_data = region_analysis[region]
            region_data['supplier_count'] += result['supplier_count']
            region_data['total_batches'] += result['total_batches'] or 0
            if result['total_batches']:
                region_data['anomaly_rate'] = (result['anomalous_batches'] or 0) / result['total_batches']
            if result['total_vehicles']:
                region_data['failure_rate'] = (result['total_failures'] or 0) / result['total_vehicles']
            
            # Specialization analysis
            spec_data = specialization_analysis[specialization]
            spec_data['supplier_count'] += result['supplier_count']
            quality_score = result['avg_quality_rating'] or 0
            risk_score = result['avg_risk_score'] or 50
            spec_data['performance_score'] = quality_score * (100 - risk_score) / 100
            
            if risk_score > 70:
                spec_data['risk_level'] = 'HIGH'
            elif risk_score > 50:
                spec_data['risk_level'] = 'MEDIUM'
        
        # Calculate derived metrics
        for tier_data in tier_analysis.values():
            if tier_data['total_batches'] > 0:
                tier_data['anomaly_rate'] = tier_data['anomalous_batches'] / tier_data['total_batches']
            if tier_data['total_vehicles'] > 0:
                tier_data['failure_rate'] = tier_data['total_failures'] / tier_data['total_vehicles']
        
        return {
            'tier_analysis': dict(tier_analysis),
            'region_analysis': dict(region_analysis),
            'specialization_analysis': dict(specialization_analysis),
            'overall_metrics': self._calculate_overall_supply_chain_metrics(),
            'risk_assessment': self._assess_supply_chain_risks(tier_analysis, region_analysis),
            'recommendations': self._generate_supply_chain_recommendations(tier_analysis)
        }
    
    def _calculate_overall_supply_chain_metrics(self) -> Dict:
        """Calculate overall supply chain health metrics"""
        query = """
        MATCH (s:Supplier)
        OPTIONAL MATCH (s)-[:SUPPLIES]->(b:Batch)
        OPTIONAL MATCH (ae:AssemblyEvent)-[:USES_BATCH]->(b)
        OPTIONAL MATCH (ae)<-[:ASSEMBLED_WITH]-(v:Vehicle)
        OPTIONAL MATCH (v)-[:EXPERIENCED_FAILURE]->(f:Failure)
        
        RETURN count(DISTINCT s) as total_suppliers,
               count(DISTINCT b) as total_batches,
               count(DISTINCT CASE WHEN b.is_anomalous THEN b END) as anomalous_batches,
               count(DISTINCT v) as total_vehicles,
               count(DISTINCT f) as total_failures,
               avg(s.quality_rating) as avg_supplier_quality,
               avg(s.risk_score) as avg_supplier_risk
        """
        
        result = self.graph.run(query).data()[0]
        
        total_batches = result['total_batches'] or 0
        anomalous_batches = result['anomalous_batches'] or 0
        total_vehicles = result['total_vehicles'] or 0
        total_failures = result['total_failures'] or 0
        
        return {
            'total_suppliers': result['total_suppliers'],
            'total_batches': total_batches,
            'overall_anomaly_rate': anomalous_batches / max(total_batches, 1),
            'overall_failure_rate': total_failures / max(total_vehicles, 1),
            'avg_supplier_quality': result['avg_supplier_quality'] or 0,
            'avg_supplier_risk': result['avg_supplier_risk'] or 0,
            'supply_chain_health_score': self._calculate_health_score(
                result['avg_supplier_quality'] or 0,
                anomalous_batches / max(total_batches, 1),
                total_failures / max(total_vehicles, 1)
            )
        }
    
    def _calculate_health_score(self, avg_quality: float, anomaly_rate: float, failure_rate: float) -> float:
        """Calculate overall supply chain health score"""
        quality_component = avg_quality * 40  # 40% weight
        anomaly_component = (1 - anomaly_rate) * 30  # 30% weight
        reliability_component = (1 - failure_rate) * 30  # 30% weight
        
        return min(100, quality_component + anomaly_component + reliability_component)
    
    def _assess_supply_chain_risks(self, tier_analysis: Dict, region_analysis: Dict) -> Dict:
        """Assess supply chain risks"""
        risks = []
        
        # Tier concentration risk
        tier_1_suppliers = tier_analysis.get(1, {}).get('supplier_count', 0)
        total_suppliers = sum(data['supplier_count'] for data in tier_analysis.values())
        
        if tier_1_suppliers / max(total_suppliers, 1) < 0.3:
            risks.append({
                'type': 'TIER_CONCENTRATION',
                'level': 'MEDIUM',
                'description': 'Low proportion of Tier 1 suppliers may indicate quality risks'
            })
        
        # Geographic concentration risk
        max_region_suppliers = max(data['supplier_count'] for data in region_analysis.values()) if region_analysis else 0
        if max_region_suppliers / max(total_suppliers, 1) > 0.6:
            risks.append({
                'type': 'GEOGRAPHIC_CONCENTRATION',
                'level': 'HIGH',
                'description': 'High geographic concentration increases supply chain disruption risk'
            })
        
        # Performance risk
        for tier, data in tier_analysis.items():
            if data.get('anomaly_rate', 0) > 0.1:
                risks.append({
                    'type': 'QUALITY_PERFORMANCE',
                    'level': 'HIGH',
                    'description': f'Tier {tier} suppliers show elevated anomaly rate'
                })
        
        return {
            'identified_risks': risks,
            'overall_risk_level': 'HIGH' if any(r['level'] == 'HIGH' for r in risks) else 'MEDIUM' if risks else 'LOW'
        }
    
    def _generate_supply_chain_recommendations(self, tier_analysis: Dict) -> List[str]:
        """Generate supply chain improvement recommendations"""
        recommendations = []
        
        # Tier-based recommendations
        for tier, data in tier_analysis.items():
            anomaly_rate = data.get('anomaly_rate', 0)
            failure_rate = data.get('failure_rate', 0)
            
            if anomaly_rate > 0.1:
                recommendations.append(f"Tier {tier}: High anomaly rate ({anomaly_rate:.1%}) - implement enhanced controls")
            
            if failure_rate > 0.05:
                recommendations.append(f"Tier {tier}: Elevated failure rate - review supplier qualification criteria")
        
        # General recommendations
        recommendations.append("Implement real-time supplier performance monitoring")
        recommendations.append("Develop supplier diversification strategy to reduce concentration risk")
        recommendations.append("Establish supplier early warning system for quality deviations")
        
        return recommendations
    
    def get_comprehensive_database_stats(self) -> Dict:
        """Get comprehensive database statistics"""
        # Node counts by type
        node_stats_query = """
        CALL {
            MATCH (n:Supplier) RETURN 'Supplier' as type, count(n) as count
            UNION
            MATCH (n:Part) RETURN 'Part' as type, count(n) as count
            UNION
            MATCH (n:Batch) RETURN 'Batch' as type, count(n) as count
            UNION
            MATCH (n:Vehicle) RETURN 'Vehicle' as type, count(n) as count
            UNION
            MATCH (n:Station) RETURN 'Station' as type, count(n) as count
            UNION
            MATCH (n:AssemblyEvent) RETURN 'AssemblyEvent' as type, count(n) as count
            UNION
            MATCH (n:QCInspection) RETURN 'QCInspection' as type, count(n) as count
            UNION
            MATCH (n:Failure) RETURN 'Failure' as type, count(n) as count
            UNION
            MATCH (n:SupplierAudit) RETURN 'SupplierAudit' as type, count(n) as count
            UNION
            MATCH (n:ComplianceRecord) RETURN 'ComplianceRecord' as type, count(n) as count
            UNION
            MATCH (n:Equipment) RETURN 'Equipment' as type, count(n) as count
            UNION
            MATCH (n:MaintenanceRecord) RETURN 'MaintenanceRecord' as type, count(n) as count
            UNION
            MATCH (n:CostAnalysis) RETURN 'CostAnalysis' as type, count(n) as count
        }
        RETURN type, count
        ORDER BY count DESC
        """
        
        node_results = self.graph.run(node_stats_query).data()
        node_stats = {result['type']: result['count'] for result in node_results}
        
        # Relationship counts
        rel_stats_query = """
        MATCH ()-[r]->()
        RETURN type(r) as rel_type, count(r) as count
        ORDER BY count DESC
        """
        
        rel_results = self.graph.run(rel_stats_query).data()
        relationship_stats = {result['rel_type']: result['count'] for result in rel_results}
        
        # Data quality metrics
        quality_query = """
        MATCH (b:Batch)
        RETURN count(b) as total_batches,
               count(CASE WHEN b.is_anomalous THEN 1 END) as anomalous_batches,
               avg(b.qc_pass_rate) as avg_qc_rate,
               min(b.manufacture_date) as earliest_batch,
               max(b.manufacture_date) as latest_batch
        """
        
        quality_result = self.graph.run(quality_query).data()[0]
        
        return {
            'node_counts': node_stats,
            'relationship_counts': relationship_stats,
            'total_nodes': sum(node_stats.values()),
            'total_relationships': sum(relationship_stats.values()),
            'data_quality': {
                'total_batches': quality_result['total_batches'],
                'anomaly_rate': quality_result['anomalous_batches'] / max(quality_result['total_batches'], 1),
                'avg_quality': quality_result['avg_qc_rate'],
                'data_span': {
                    'earliest_batch': quality_result['earliest_batch'],
                    'latest_batch': quality_result['latest_batch']
                }
            },
            'database_health': {
                'connectivity_score': self._calculate_connectivity_score(node_stats, relationship_stats),
                'data_completeness': self._calculate_data_completeness(node_stats)
            }
        }
    
    def _calculate_connectivity_score(self, node_stats: Dict, rel_stats: Dict) -> float:
        """Calculate database connectivity score"""
        total_nodes = sum(node_stats.values())
        total_relationships = sum(rel_stats.values())
        
        if total_nodes == 0:
            return 0
        
        # Ideal ratio is approximately 3-5 relationships per node for manufacturing data
        connectivity_ratio = total_relationships / total_nodes
        optimal_ratio = 4.0
        
        # Score based on how close we are to optimal connectivity
        score = min(100, (connectivity_ratio / optimal_ratio) * 100)
        return round(score, 1)
    
    def _calculate_data_completeness(self, node_stats: Dict) -> float:
        """Calculate data completeness score"""
        # Expected minimum ratios for a complete dataset
        expected_ratios = {
            'AssemblyEvent': node_stats.get('Vehicle', 0) * 8,  # ~8 parts per vehicle
            'QCInspection': node_stats.get('Vehicle', 0) * 2,   # ~2 inspections per vehicle
            'Failure': node_stats.get('Vehicle', 0) * 0.03      # ~3% failure rate
        }
        
        completeness_scores = []
        for node_type, expected_count in expected_ratios.items():
            actual_count = node_stats.get(node_type, 0)
            if expected_count > 0:
                ratio = min(1.0, actual_count / expected_count)
                completeness_scores.append(ratio)
        
        if completeness_scores:
            return round(sum(completeness_scores) / len(completeness_scores) * 100, 1)
        else:
            return 0.0

def main():
    """Test enhanced graph manager functionality"""
    manager = EnhancedGraphManager()
    
    try:
        # Test connection and basic operations
        stats = manager.get_comprehensive_database_stats()
        logger.info(f"Database stats: {stats}")
        
        print("Enhanced Graph Manager ready for use!")
        
    except Exception as e:
        logger.error(f"Enhanced Graph Manager test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()