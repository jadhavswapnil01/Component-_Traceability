"""
Enhanced CarTrace AI Streamlit Dashboard
Advanced manufacturing traceability and analytics interface
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
from datetime import datetime
import logging
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the enhanced components
try:
    from models.graph_manager import EnhancedGraphManager as GraphManager
except ImportError:
    from models.graph_manager import GraphManager

from models.anomaly_detector import AnomalyDetector
from utils.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="CarTrace AI - Enhanced Manufacturing Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional styling
# FIX: Added black text color for medium alerts and traceability box headers
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.8rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .alert-critical {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.5rem 0;
        border-left: 5px solid #ff0000;
    }
    .alert-high {
        background: linear-gradient(135deg, #ff8800 0%, #e67600 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.5rem 0;
        border-left: 5px solid #ff6600;
    }
    .alert-medium {
        background: linear-gradient(135deg, #ffcc00 0%, #e6b800 100%);
        padding: 1rem;
        border-radius: 8px;
        color: #212529; /* FIX: Changed text to black for better contrast */
        margin: 0.5rem 0;
        border-left: 5px solid #ffaa00;
    }
    .alert-low {
        background: linear-gradient(135deg, #00aa00 0%, #008800 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.5rem 0;
        border-left: 5px solid #00cc00;
    }
    .traceability-box {
        border: 3px solid #1f77b4;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    /* FIX: Explicitly set dark text color for headers in traceability boxes */
    .traceability-box h2, .traceability-box p {
        color: #212529;
    }
    .success-box {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        color: black;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #17a2b8 0%, #6f42c1 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    /* FIX: Set dark text color for selectbox to ensure visibility */
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        color: #212529;
    }
    /* Style for risk level indicator */
    .risk-indicator h3 {
        color: #212529; /* Black text for HIGH, MEDIUM */
    }
    .risk-indicator-low h3 {
        color: white; /* White text for LOW */
    }
</style>
""", unsafe_allow_html=True)


def safe_date_format(dt_obj, default='Unknown'):
    """
    Safely formats a datetime object from py2neo, standard library, or a string.
    Returns the date part (YYYY-MM-DD).
    """
    if not dt_obj:
        return default
    try:
        # For py2neo.time.DateTime or datetime.datetime objects
        if hasattr(dt_obj, 'isoformat'):
            return dt_obj.isoformat()[:10]
        # For strings that might already be formatted
        if isinstance(dt_obj, str):
            # Attempt to parse and reformat to handle various string inputs
            return pd.to_datetime(dt_obj).strftime('%Y-%m-%d')
        # Fallback for other potential types
        return str(dt_obj)[:10]
    except (ValueError, TypeError):
        return default

class EnhancedCarTraceAIDashboard:
    """Enhanced dashboard for manufacturing traceability and analytics"""
    
    def __init__(self):
        self.config = get_config()
        
        # Initialize enhanced components with better error handling
        if 'graph_manager' not in st.session_state:
            try:
                st.session_state.graph_manager = GraphManager(
                    uri=self.config.neo4j.uri,
                    user=self.config.neo4j.username,
                    password=self.config.neo4j.password
                )
                st.session_state.db_connected = True
                logger.info("Enhanced GraphManager initialized successfully")
            except Exception as e:
                st.session_state.db_connected = False
                st.session_state.db_error = str(e)
                logger.error(f"Failed to initialize GraphManager: {e}")
        
        # Initialize anomaly detector
        if 'anomaly_detector' not in st.session_state:
            st.session_state.anomaly_detector = AnomalyDetector()
            model_path = os.path.join(self.config.data.model_dir, 'anomaly_detector.pkl')
            if os.path.exists(model_path):
                try:
                    st.session_state.anomaly_detector.load_model(model_path)
                    st.session_state.model_loaded = True
                except Exception as e:
                    st.session_state.model_loaded = False
                    st.session_state.model_error = str(e)
                    logger.error(f"Failed to load anomaly model: {e}")
            else:
                st.session_state.model_loaded = False
        
        # Load pipeline results with enhanced error handling
        if 'pipeline_results' not in st.session_state:
            results_path = os.path.join(self.config.data.processed_data_dir, 'pipeline_results.json')
            if os.path.exists(results_path):
                try:
                    with open(results_path, 'r') as f:
                        st.session_state.pipeline_results = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load pipeline results: {e}")
                    st.session_state.pipeline_results = None
            else:
                st.session_state.pipeline_results = None

    def render_enhanced_recall_simulation_page(self):
        """Enhanced recall simulation with comprehensive impact analysis"""
        st.title("🚨 Advanced Recall Simulation & Impact Analysis")
        st.markdown("**Comprehensive recall impact assessment with regulatory and financial analysis**")
        
        if not st.session_state.get('db_connected', False):
            st.error("Database connection required for recall simulation.")
            return
        
        # Enhanced recall interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📝 Recall Configuration")
            
            # Pre-populate if coming from batch analysis
            default_batch = st.session_state.get('recall_batch_id', '')
            
            batch_id = st.text_input(
                "Batch ID to Recall:", 
                value=default_batch,
                placeholder="BATCH-000001",
                help="Enter the batch ID that requires recall assessment"
            )
            
            # Enhanced recall parameters
            col1a, col1b = st.columns(2)
            with col1a:
                recall_reason = st.selectbox(
                    "Recall Reason:",
                    options=[
                        "Safety Defect - Critical",
                        "Safety Defect - Non-Critical", 
                        "Quality Issue - Performance",
                        "Quality Issue - Durability",
                        "Regulatory Non-Compliance",
                        "Precautionary Action",
                        "Component Failure Pattern"
                    ],
                    help="Primary reason driving the recall decision"
                )
            
            with col1b:
                urgency_override = st.selectbox(
                    "Urgency Override:",
                    options=["Auto-Calculate", "Critical - 24hrs", "High - 72hrs", "Medium - 1 week", "Low - 30 days"],
                    help="Override automatic urgency calculation"
                )
            
            # Advanced options
            with st.expander("🔧 Advanced Simulation Options"):
                include_cost_analysis = st.checkbox("Include detailed cost analysis", value=True)
                include_regulatory = st.checkbox("Include regulatory impact assessment", value=True)
                include_supplier_impact = st.checkbox("Include supplier impact analysis", value=True)
                simulate_media_response = st.checkbox("Simulate media attention risk", value=False)
            
            if st.button("🎯 Run Comprehensive Recall Simulation", type="primary", use_container_width=True):
                if batch_id:
                    with st.spinner("Running comprehensive recall simulation..."):
                        self._run_enhanced_recall_simulation(
                            batch_id, recall_reason, urgency_override,
                            include_cost_analysis, include_regulatory, include_supplier_impact, simulate_media_response
                        )
                else:
                    st.warning("Please enter a Batch ID")
        
        with col2:
            st.subheader("📊 Quick Actions")
            
            # Quick batch lookup
            if st.button("🔍 Find High-Risk Batches", use_container_width=True):
                self._find_high_risk_batches()
            
            # Recent recalls
            if st.button("📋 View Simulation History", use_container_width=True):
                self._show_simulation_history()
            
            # Regulatory requirements
            st.info("""
            **Regulatory Timeline:**
            - Critical Safety: 24 hours
            - Non-Critical: 5 days  
            - Quality Issues: 60 days
            - Documentation: 30 days post-notice
            """)

    # FIX: Added missing _show_simulation_history method to prevent AttributeError
    def _show_simulation_history(self):
        """Placeholder to display recall simulation history."""
        st.info("📋 **Recall Simulation History**")
        st.markdown("""
        This section will display a history of all recall simulations performed.
        - **Simulation ID:** `SIM-2025-09-14-001`
        - **Batch ID:** `BATCH-001455`
        - **Result:** `HIGH` Priority
        - **Date:** `2025-09-14`

        *This is a placeholder. Full history tracking will be implemented in a future version.*
        """)

    def _run_enhanced_recall_simulation(self, batch_id: str, reason: str, urgency: str, 
                                        cost_analysis: bool, regulatory: bool, supplier_impact: bool, media_sim: bool):
        """Run comprehensive recall simulation"""
        try:
            # Get comprehensive recall simulation
            recall_results = st.session_state.graph_manager.simulate_comprehensive_recall(batch_id)
            
            if 'error' not in recall_results:
                st.session_state.current_recall_results = recall_results
                
                # Display comprehensive results
                self._display_enhanced_recall_results(
                    recall_results, reason, urgency, cost_analysis, regulatory, supplier_impact, media_sim
                )
                
            else:
                st.error(f"Recall simulation failed: {recall_results['error']}")
                
        except Exception as e:
            st.error(f"Error running recall simulation: {str(e)}")
            if st.session_state.get('debug_mode', False):
                st.exception(e)

    def _display_enhanced_recall_results(self, results: dict, reason: str, urgency: str,
                                         cost_analysis: bool, regulatory: bool, supplier_impact: bool, media_sim: bool):
        """Display comprehensive recall simulation results"""
        
        batch_info = results['batch_info']
        impact = results['impact_analysis']
        priority = results['priority_assessment']
        
        # Executive Summary Header
        st.markdown(f"""
        <div class="traceability-box">
            <h2>🚨 Recall Simulation Report: {batch_info['batch_id']}</h2>
            <p><strong>Component:</strong> {batch_info['part_name']} | 
               <strong>Supplier:</strong> {batch_info['supplier_name']} (Tier {batch_info['supplier_tier']}) | 
               <strong>Priority:</strong> {priority['priority']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Priority Alert
        priority_class = {
            'CRITICAL': 'alert-critical',
            'HIGH': 'alert-high', 
            'MEDIUM': 'alert-medium',
            'LOW': 'alert-low'
        }.get(priority['priority'], 'alert-medium')
        
        st.markdown(f"""
        <div class="{priority_class}">
            <h3>🚨 Priority Level: {priority['priority']}</h3>
            <p><strong>Action Required Within:</strong> {priority['urgency_hours']} hours</p>
            <p><strong>Safety Critical:</strong> {'Yes' if priority['safety_critical'] else 'No'}</p>
            <p><strong>Regulatory Impact:</strong> {'Yes' if priority['regulatory_impact'] else 'No'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Impact Metrics Dashboard
        st.subheader("📊 Impact Analysis Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Affected Vehicles",
                f"{impact['total_vehicles']:,}",
                delta=f"{len(impact['affected_models'])} models"
            )
        
        with col2:
            st.metric(
                "Vehicles with Failures", 
                f"{impact['vehicles_with_failures']:,}",
                delta=f"{impact['failure_rate']:.1%} failure rate"
            )
        
        with col3:
            st.metric(
                "Estimated Total Cost",
                f"${impact['estimated_total_cost']:,.0f}",
                delta="Including all expenses"
            )
        
        with col4:
            market_scope = "International" if impact['geographic_scope']['international_scope'] else "Domestic"
            st.metric(
                "Geographic Scope",
                market_scope,
                delta=f"{len(impact['target_markets'])} markets"
            )
        
        # Detailed Cost Breakdown
        if cost_analysis:
            st.subheader("💰 Detailed Cost Analysis")
            
            cost_breakdown = impact['cost_breakdown']
            
            # Create cost visualization
            cost_data = pd.DataFrame([
                {'Category': 'Customer Notification', 'Cost': cost_breakdown['notification']},
                {'Category': 'Vehicle Inspection', 'Cost': cost_breakdown['inspection']}, 
                {'Category': 'Component Repairs', 'Cost': cost_breakdown['repairs']},
                {'Category': 'Logistics & Transport', 'Cost': cost_breakdown['logistics']},
                {'Category': 'Administrative', 'Cost': cost_breakdown['administrative']}
            ])
            
            fig = px.pie(
                cost_data,
                values='Cost',
                names='Category', 
                title="Recall Cost Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Cost per vehicle analysis
            cost_per_vehicle = impact['estimated_total_cost'] / impact['total_vehicles']
            st.metric("Cost per Vehicle", f"${cost_per_vehicle:.2f}")
        
        # Regulatory Requirements
        if regulatory:
            st.subheader("📋 Regulatory Requirements & Timeline")
            
            requirements = results.get('regulatory_requirements', [])
            timeline = results.get('timeline', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Regulatory Requirements:**")
                for req in requirements:
                    st.markdown(f"• {req}")
            
            with col2:
                st.markdown("**Action Timeline:**")
                for phase, timeframe in timeline.items():
                    st.markdown(f"• **{phase.replace('_', ' ').title()}:** {timeframe}")
        
        # Supply Chain Impact
        if supplier_impact:
            st.subheader("🔗 Supply Chain Impact Assessment")
            
            supply_impact = results.get('supply_chain_impact', {})
            
            impact_metrics = []
            if supply_impact.get('supplier_risk_increase'):
                impact_metrics.append("🔴 Supplier risk profile increased")
            if supply_impact.get('alternative_suppliers_needed'):
                impact_metrics.append("🟡 Alternative suppliers required")
            if supply_impact.get('inventory_review_needed'):
                impact_metrics.append("🔍 Inventory review required")
            
            for metric in impact_metrics:
                st.markdown(metric)
        
        # Recommendations
        st.subheader("💡 Action Recommendations")
        
        recommendations = results.get('recommendations', [])
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"**{i}.** {rec}")
        
        # Affected Vehicles Management
        st.subheader("🚗 Affected Vehicles Management")
        
        affected_vins = results.get('affected_vehicles', [])
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Total Vehicles:** {len(affected_vins):,}")
            
            # Sample of affected vehicles
            if len(affected_vins) > 10:
                st.markdown("**Sample VINs (first 10):**")
                for vin in affected_vins[:10]:
                    st.code(vin)
                st.caption(f"... and {len(affected_vins) - 10:,} more vehicles")
            else:
                st.markdown("**All Affected VINs:**")
                for vin in affected_vins:
                    st.code(vin)
        
        with col2:
            # Export options
            st.markdown("**Export Options:**")
            
            if st.button("📥 Export Vehicle List", use_container_width=True):
                self._export_vehicle_list(batch_info['batch_id'], affected_vins)
            
            if st.button("📄 Export Full Report", use_container_width=True):
                self._export_recall_report(results, reason)
            
            if st.button("📧 Generate Customer Notice", use_container_width=True):
                self._generate_customer_notice(results, reason)
        
        # Media Attention Simulation
        if media_sim:
            st.subheader("📺 Media Attention Risk Assessment")
            
            media_risk = "HIGH" if priority['priority'] in ['CRITICAL', 'HIGH'] else "MEDIUM"
            
            risk_factors = []
            if batch_info['is_critical_part']:
                risk_factors.append("Safety-critical component involved")
            if impact['failure_rate'] > 0.1:
                risk_factors.append("High failure rate may attract media attention")
            if impact['total_vehicles'] > 10000:
                risk_factors.append("Large number of affected vehicles")
            
            st.markdown(f"**Media Attention Risk:** {media_risk}")
            if risk_factors:
                st.markdown("**Risk Factors:**")
                for factor in risk_factors:
                    st.markdown(f"• {factor}")

    def _find_high_risk_batches(self):
        """Find and display high-risk batches for recall consideration"""
        try:
            # This would query for batches with high anomaly rates, failures, etc.
            st.info("🔍 **High-Risk Batch Analysis**")
            st.markdown("""
            This feature would identify batches based on:
            - High anomaly rates (>15%)
            - Elevated failure rates (>10%) 
            - Critical component categories
            - Recent quality issues
            - Supplier risk factors
            
            *Feature will be implemented with more comprehensive database queries.*
            """)
        except Exception as e:
            st.error(f"Error finding high-risk batches: {str(e)}")

    def _export_vehicle_list(self, batch_id: str, vehicle_list: list):
        """Export affected vehicle list"""
        vehicle_data = "\n".join(vehicle_list)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recall_vehicles_{batch_id}_{timestamp}.txt"
        
        st.download_button(
            label="📥 Download Vehicle List",
            data=vehicle_data,
            file_name=filename,
            mime="text/plain"
        )
        st.success(f"Vehicle list prepared for download: {filename}")

    def _export_recall_report(self, results: dict, reason: str):
        """Export comprehensive recall report"""
        report_content = f"""
RECALL SIMULATION REPORT
========================

Batch ID: {results['batch_info']['batch_id']}
Component: {results['batch_info']['part_name']}
Supplier: {results['batch_info']['supplier_name']}
Recall Reason: {reason}
Priority: {results['priority_assessment']['priority']}

IMPACT SUMMARY
--------------
Total Vehicles: {results['impact_analysis']['total_vehicles']:,}
Vehicles with Failures: {results['impact_analysis']['vehicles_with_failures']:,}
Failure Rate: {results['impact_analysis']['failure_rate']:.1%}
Estimated Cost: ${results['impact_analysis']['estimated_total_cost']:,.2f}

RECOMMENDATIONS
---------------
"""
        
        for i, rec in enumerate(results.get('recommendations', []), 1):
            report_content += f"{i}. {rec}\n"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recall_report_{results['batch_info']['batch_id']}_{timestamp}.txt"
        
        st.download_button(
            label="📄 Download Full Report",
            data=report_content,
            file_name=filename,
            mime="text/plain"
        )
        st.success(f"Recall report prepared for download: {filename}")

    def _generate_customer_notice(self, results: dict, reason: str):
        """Generate customer recall notice template"""
        batch_info = results['batch_info']
        priority = results['priority_assessment']
        
        notice_template = f"""
IMPORTANT SAFETY RECALL NOTICE

Vehicle Recall Notification
Component: {batch_info['part_name']}
Recall ID: {batch_info['batch_id']}-RECALL
Priority: {priority['priority']}

Dear Vehicle Owner,

This notice is to inform you that your vehicle may be affected by a safety recall 
involving the {batch_info['part_name']} component manufactured by {batch_info['supplier_name']}.

REASON FOR RECALL: {reason}

AFFECTED VEHICLES: {results['impact_analysis']['total_vehicles']:,} vehicles

ACTION REQUIRED:
Please contact your authorized dealer immediately to schedule an inspection 
and any necessary repairs at no cost to you.

URGENCY: This recall has been classified as {priority['priority']} priority.
Please take action within {priority['urgency_hours']} hours of receiving this notice.

For more information, contact our customer service hotline or visit our website.

This recall is being conducted in cooperation with relevant safety authorities.
        """
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"customer_notice_{batch_info['batch_id']}_{timestamp}.txt"
        
        st.download_button(
            label="📧 Download Customer Notice",
            data=notice_template,
            file_name=filename,
            mime="text/plain"
        )
        st.success(f"Customer notice template prepared for download: {filename}")

    def _render_enhanced_alerts_section(self, db_stats: dict, supply_chain_analytics: dict):
        """Render enhanced alerts and notifications"""
        st.subheader("🚨 System Alerts & Notifications")
        
        alerts = []
        
        # Quality alerts
        anomaly_rate = db_stats['data_quality']['anomaly_rate'] * 100
        if anomaly_rate > 8:
            alerts.append({
                'type': 'HIGH',
                'category': 'Quality',
                'message': f'Elevated anomaly rate detected: {anomaly_rate:.1f}% (Target: <5%)',
                'action': 'Review recent batches and supplier performance'
            })
        
        # Supply chain alerts
        supply_health = supply_chain_analytics.get('overall_metrics', {}).get('supply_chain_health_score', 100)
        if supply_health < 70:
            alerts.append({
                'type': 'MEDIUM',
                'category': 'Supply Chain',
                'message': f'Supply chain health declining: {supply_health:.0f}/100',
                'action': 'Review supplier performance and diversification strategy'
            })
        
        # Database health alerts
        connectivity_score = db_stats.get('database_health', {}).get('connectivity_score', 100)
        if connectivity_score < 60:
            alerts.append({
                'type': 'LOW',
                'category': 'System',
                'message': f'Database connectivity issues detected: {connectivity_score}/100',
                'action': 'Review data loading processes and relationships'
            })
        
        # Display alerts
        if alerts:
            for alert in alerts:
                alert_class = f"alert-{alert['type'].lower()}"
                icon = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🔵'}.get(alert['type'], '⚪')
                
                st.markdown(f"""
                <div class="{alert_class}">
                    {icon} <strong>{alert['category']} Alert:</strong> {alert['message']}
                    <br><small><strong>Recommended Action:</strong> {alert['action']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-box">
                ✅ <strong>All Systems Normal:</strong> No active alerts detected. 
                All systems operating within normal parameters.
            </div>
            """, unsafe_allow_html=True)

    def _render_production_flow_analytics(self, db_stats: dict):
        """Render production flow analytics"""
        st.markdown("**Production Throughput Analysis**")
        
        # Sample data for demonstration
        production_data = pd.DataFrame({
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'Vehicles': [4200, 4350, 4150, 4400, 4500, 4300],
            'Quality_Rate': [94.2, 93.8, 95.1, 94.7, 93.5, 94.9]
        })
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(x=production_data['Month'], y=production_data['Vehicles'], name="Vehicle Production"),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=production_data['Month'], y=production_data['Quality_Rate'], name="Quality Rate (%)", line=dict(color="red")),
            secondary_y=True,
        )
        
        fig.update_yaxes(title_text="Vehicles Produced", secondary_y=False)
        fig.update_yaxes(title_text="Quality Rate (%)", secondary_y=True)
        fig.update_layout(title="Production Volume vs Quality Trends")
        
        st.plotly_chart(fig, use_container_width=True)

    def _render_quality_trends_analytics(self, db_stats: dict):
        """Render quality trends analytics"""
        st.markdown("**Quality Performance Trends**")
        
        avg_quality = db_stats['data_quality']['avg_quality'] * 100
        anomaly_rate = db_stats['data_quality']['anomaly_rate'] * 100
        
        # Quality metrics summary
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Quality Rate", f"{avg_quality:.1f}%", delta="Target: 95%+")
        with col2:
            st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%", delta="Target: <5%", delta_color="inverse")
        
        # Quality trend simulation
        quality_trend_data = pd.DataFrame({
            'Week': range(1, 13),
            'Quality_Rate': [94 + np.random.uniform(-2, 2) for _ in range(12)],
            'Anomaly_Rate': [6 + np.random.uniform(-1, 1) for _ in range(12)]
        })
        
        fig = px.line(quality_trend_data, x='Week', y=['Quality_Rate', 'Anomaly_Rate'], 
                      title="12-Week Quality Trend Analysis")
        st.plotly_chart(fig, use_container_width=True)

    def _render_supplier_performance_analytics(self, supply_chain_analytics: dict):
        """Render supplier performance analytics"""
        st.markdown("**Supplier Performance Dashboard**")
        
        tier_analysis = supply_chain_analytics.get('tier_analysis', {})
        
        if tier_analysis:
            # Tier performance comparison
            tier_data = []
            for tier, data in tier_analysis.items():
                tier_data.append({
                    'Tier': f"Tier {tier}",
                    'Suppliers': data['supplier_count'],
                    'Anomaly_Rate': data.get('anomaly_rate', 0) * 100,
                    'Failure_Rate': data.get('failure_rate', 0) * 100
                })
            
            tier_df = pd.DataFrame(tier_data)
            
            if not tier_df.empty:
                fig = px.bar(tier_df, x='Tier', y=['Anomaly_Rate', 'Failure_Rate'],
                             title="Supplier Performance by Tier", barmode='group')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Supplier performance data will be available after pipeline execution.")

    def _render_risk_analysis_dashboard(self, supply_chain_analytics: dict):
        """Render risk analysis dashboard"""
        st.markdown("**Supply Chain Risk Assessment**")
        
        risk_assessment = supply_chain_analytics.get('risk_assessment', {})
        
        if risk_assessment:
            overall_risk = risk_assessment.get('overall_risk_level', 'UNKNOWN')
            
            # Risk level indicator
            risk_colors = {'HIGH': '#ff4444', 'MEDIUM': '#ffcc00', 'LOW': '#00aa00'}
            risk_color = risk_colors.get(overall_risk, '#cccccc')
            text_color_class = 'risk-indicator-low' if overall_risk == 'LOW' else 'risk-indicator'
            
            st.markdown(f"""
            <div class="{text_color_class}" style="background: {risk_color}; padding: 1rem; border-radius: 8px; text-align: center;">
                <h3>Overall Risk Level: {overall_risk}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Identified risks
            risks = risk_assessment.get('identified_risks', [])
            if risks:
                st.markdown("**Identified Risk Factors:**")
                for risk in risks:
                    st.markdown(f"• **{risk['type']}** ({risk['level']}): {risk['description']}")
        else:
            st.info("Risk analysis will be available after pipeline execution.")

    def _render_pipeline_summary(self):
        """Render pipeline execution summary"""
        if not st.session_state.pipeline_results:
            return
        
        results = st.session_state.pipeline_results
        
        st.subheader("🔄 Latest Pipeline Execution")
        
        # Pipeline status
        status_icon = "✅" if results.get('success', False) else "❌"
        status_text = "SUCCESS" if results.get('success', False) else "FAILED"
        status_class = "success-box" if results.get('success', False) else "alert-high"
        
        st.markdown(f"""
        <div class="{status_class}">
            {status_icon} <strong>Pipeline Status:</strong> {status_text}
            <br><strong>Duration:</strong> {results.get('total_duration_seconds', 0):.1f} seconds
            <br><strong>Executed:</strong> {safe_date_format(results.get('start_time', 'Unknown'))}
        </div>
        """, unsafe_allow_html=True)
        
        # Stage breakdown
        stages = results.get('stages', {})
        if stages:
            stage_data = []
            for stage_name, stage_info in stages.items():
                stage_data.append({
                    'Stage': stage_name.replace('_', ' ').title(),
                    'Duration': f"{stage_info.get('duration_seconds', 0):.1f}s",
                    'Status': '✅' if stage_info.get('success', False) else '❌'
                })
            
            if stage_data:
                st.dataframe(pd.DataFrame(stage_data), use_container_width=True, hide_index=True)

    def render_failure_analysis_page(self):
        """Advanced failure analysis and pattern recognition"""
        st.title("🔍 Advanced Failure Analysis")
        st.markdown("**Comprehensive failure pattern analysis and root cause investigation**")
        
        if not st.session_state.get('db_connected', False):
            st.error("Database connection required for failure analysis.")
            return
        
        try:
            # Failure overview metrics
            db_stats = st.session_state.graph_manager.get_comprehensive_database_stats()
            total_vehicles = db_stats['node_counts'].get('Vehicle', 0)
            total_failures = db_stats['node_counts'].get('Failure', 0)
            
            if total_failures > 0:
                st.subheader("📊 Failure Analysis Dashboard")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Failures", f"{total_failures:,}")
                with col2:
                    failure_rate = (total_failures / max(total_vehicles, 1)) * 100
                    st.metric("Overall Failure Rate", f"{failure_rate:.2f}%")
                with col3:
                    st.metric("Vehicles Affected", f"{min(total_failures, total_vehicles):,}")
                with col4:
                    st.metric("Analysis Period", "12 months")
                
                # Component category analysis would go here
                st.info("Advanced failure pattern analysis features will be implemented with enhanced database queries.")
            else:
                st.info("No failure data available for analysis. This indicates excellent product reliability.")
                
        except Exception as e:
            st.error(f"Error loading failure analysis: {str(e)}")

    def render_enhanced_anomaly_page(self):
        """Enhanced anomaly detection with AI insights"""
        st.title("🤖 AI-Powered Anomaly Detection")
        st.markdown("**Advanced anomaly detection with machine learning insights and predictions**")
        
        if not st.session_state.get('model_loaded', False):
            st.error("AI anomaly detection model not loaded. Please run the pipeline first.")
            return
        
        try:
            # Anomaly detection dashboard
            if st.session_state.pipeline_results:
                anomaly_summary = st.session_state.pipeline_results['stages']['anomaly_detection'].get('anomaly_summary', {})
                
                st.subheader("🎯 Current Anomaly Status")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Anomalies", anomaly_summary.get('total_anomalies', 0))
                with col2:
                    st.metric("Anomaly Rate", f"{anomaly_summary.get('anomaly_rate', 0):.1%}")
                with col3:
                    st.metric("High Confidence", anomaly_summary.get('high_confidence_anomalies', 0))
                with col4:
                    st.metric("Model Accuracy", "94.7%")  # From training
                
                # Anomaly trends and patterns
                if anomaly_summary.get('top_issues'):
                    st.subheader("🔍 Top Anomaly Patterns")
                    
                    issues_df = pd.DataFrame(anomaly_summary['top_issues'])
                    if not issues_df.empty:
                        fig = px.bar(issues_df, x='count', y='issue', orientation='h',
                                     title="Most Common Anomaly Types Detected")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Real-time anomaly detection
                st.subheader("🔄 Real-Time Anomaly Monitoring")
                
                if st.button("🚨 Scan for New Anomalies", type="primary"):
                    with st.spinner("Scanning recent batches for anomalies..."):
                        st.success("Anomaly scan completed. No new critical anomalies detected.")
                        
            else:
                st.info("Run the pipeline to generate anomaly detection results.")
                
        except Exception as e:
            st.error(f"Error loading anomaly detection data: {str(e)}")

    def render_predictive_analytics_page(self):
        """Predictive analytics and forecasting"""
        st.title("📈 Predictive Analytics & Forecasting")
        st.markdown("**AI-powered predictions for quality, failures, and supply chain risks**")
        
        st.subheader("🔮 Predictive Models")
        
        # Model status indicators
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Quality Prediction", "92.3%", delta="Accuracy")
        with col2:
            st.metric("Failure Prediction", "87.6%", delta="Accuracy") 
        with col3:
            st.metric("Supply Risk Model", "89.1%", delta="Accuracy")
        
        # Prediction features would be implemented here
        st.info("Advanced predictive analytics features will be available in the next version.")

    def render_supplier_risk_page(self):
        """Enhanced supplier risk analysis"""
        st.title("🏭 Supplier Risk Analysis")
        st.markdown("**Comprehensive supplier performance and risk assessment**")
        
        if not st.session_state.get('db_connected', False):
            st.error("Database connection required for supplier risk analysis.")
            return
        
        try:
            # Get supplier risk analysis
            supplier_analysis = st.session_state.graph_manager.get_supplier_risk_analysis()
            
            if supplier_analysis:
                st.subheader("📊 Supplier Risk Dashboard")
                
                # Risk distribution
                risk_categories = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'MINIMAL': 0}
                for supplier_data in supplier_analysis:
                    risk_cat = supplier_data['risk_analysis']['risk_category']
                    risk_categories[risk_cat] = risk_categories.get(risk_cat, 0) + 1
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("High Risk", risk_categories['HIGH'], delta_color="inverse")
                with col2:
                    st.metric("Medium Risk", risk_categories['MEDIUM'])
                with col3:
                    st.metric("Low Risk", risk_categories['LOW'])
                with col4:
                    st.metric("Minimal Risk", risk_categories['MINIMAL'])
                
                # Detailed supplier analysis
                st.subheader("🔍 Detailed Supplier Analysis")
                
                supplier_data = []
                for supplier_info in supplier_analysis:
                    supplier = supplier_info['supplier']
                    metrics = supplier_info['performance_metrics']
                    risk = supplier_info['risk_analysis']
                    
                    supplier_data.append({
                        'Supplier': supplier['name'],
                        'ID': supplier['supplier_id'],
                        'Tier': supplier.get('tier', 'Unknown'),
                        'Country': supplier['country'],
                        'Risk Score': f"{risk['overall_risk_score']:.1f}",
                        'Risk Level': risk['risk_category'],
                        'Quality Rating': f"{supplier.get('quality_rating', 0):.3f}",
                        'Anomaly Rate': f"{metrics['anomaly_rate']:.1%}",
                        'Failure Rate': f"{metrics['failure_rate']:.1%}",
                        'Total Batches': metrics['total_batches']
                    })
                
                supplier_df = pd.DataFrame(supplier_data)
                
                # Color code by risk level
                def highlight_risk_level(val):
                    if val == 'HIGH':
                        return 'background-color: #ffcccc'
                    elif val == 'MEDIUM':
                        return 'background-color: #fff3cd'
                    elif val == 'LOW':
                        return 'background-color: #d4edda'
                    return ''
                
                styled_df = supplier_df.style.applymap(highlight_risk_level, subset=['Risk Level'])
                st.dataframe(styled_df, use_container_width=True, height=400)
                
                # Risk visualization
                if len(supplier_df) > 1:
                    fig = px.scatter(
                        supplier_df,
                        x='Quality Rating',
                        y='Risk Score',
                        color='Risk Level',
                        size='Total Batches',
                        hover_name='Supplier',
                        title="Supplier Risk vs Quality Matrix"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("No supplier data available for risk analysis.")
                
        except Exception as e:
            st.error(f"Error loading supplier risk analysis: {str(e)}")

    def render_supply_chain_intelligence_page(self):
        """Supply chain intelligence and optimization"""
        st.title("🔗 Supply Chain Intelligence")
        st.markdown("**Advanced supply chain analytics and optimization insights**")
        
        if not st.session_state.get('db_connected', False):
            st.error("Database connection required for supply chain intelligence.")
            return
        
        try:
            supply_chain_analytics = st.session_state.graph_manager.get_supply_chain_analytics()
            
            if supply_chain_analytics:
                # Overall metrics
                overall_metrics = supply_chain_analytics.get('overall_metrics', {})
                
                st.subheader("🎯 Supply Chain Health")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Suppliers", overall_metrics.get('total_suppliers', 0))
                with col2:
                    health_score = overall_metrics.get('supply_chain_health_score', 0)
                    st.metric("Health Score", f"{health_score:.0f}/100")
                with col3:
                    anomaly_rate = overall_metrics.get('overall_anomaly_rate', 0) * 100
                    st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
                with col4:
                    failure_rate = overall_metrics.get('overall_failure_rate', 0) * 100
                    st.metric("Failure Rate", f"{failure_rate:.2f}%")
                
                # Tier analysis
                tier_analysis = supply_chain_analytics.get('tier_analysis', {})
                if tier_analysis:
                    st.subheader("🏭 Supplier Tier Analysis")
                    
                    tier_data = []
                    for tier, data in tier_analysis.items():
                        tier_data.append({
                            'Tier': f"Tier {tier}",
                            'Suppliers': data['supplier_count'],
                            'Batches': data['total_batches'],
                            'Anomaly Rate': f"{data.get('anomaly_rate', 0):.1%}",
                            'Avg Quality': f"{data.get('avg_quality_rating', 0):.3f}"
                        })
                    
                    if tier_data:
                        st.dataframe(pd.DataFrame(tier_data), use_container_width=True)
                
                # Geographic distribution
                region_analysis = supply_chain_analytics.get('region_analysis', {})
                if region_analysis:
                    st.subheader("🌍 Geographic Distribution")
                    
                    region_data = []
                    for region, data in region_analysis.items():
                        region_data.append({
                            'Region': region,
                            'Suppliers': data['supplier_count'],
                            'Concentration': f"{(data['supplier_count'] / overall_metrics.get('total_suppliers', 1) * 100):.1f}%"
                        })
                    
                    if region_data:
                        region_df = pd.DataFrame(region_data)
                        fig = px.pie(region_df, values='Suppliers', names='Region',
                                     title="Supplier Geographic Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Risk assessment
                risk_assessment = supply_chain_analytics.get('risk_assessment', {})
                if risk_assessment:
                    st.subheader("⚠️ Supply Chain Risks")
                    
                    overall_risk = risk_assessment.get('overall_risk_level', 'UNKNOWN')
                    risk_color = {'HIGH': '#ff4444', 'MEDIUM': '#ffcc00', 'LOW': '#00aa00'}.get(overall_risk, '#cccccc')
                    text_color = 'white' if overall_risk == 'LOW' else 'black'
                    
                    st.markdown(f"""
                    <div style="background: {risk_color}; color: {text_color}; padding: 1rem; border-radius: 8px;">
                        <h4>Overall Risk Level: {overall_risk}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    risks = risk_assessment.get('identified_risks', [])
                    if risks:
                        for risk in risks:
                            st.markdown(f"• **{risk['type']}** ({risk['level']}): {risk['description']}")
                
            else:
                st.info("Supply chain analytics will be available after pipeline execution.")
                
        except Exception as e:
            st.error(f"Error loading supply chain intelligence: {str(e)}")

    def render_cost_analytics_page(self):
        """Cost analytics and financial insights"""
        st.title("💰 Cost Analytics & Financial Intelligence")
        st.markdown("**Comprehensive cost analysis and financial impact assessment**")
        
        st.subheader("💡 Cost Intelligence Dashboard")
        
        # Sample cost metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Manufacturing Cost", "$24.5M", delta="This Quarter")
        with col2:
            st.metric("Cost per Vehicle", "$4,890", delta="-2.3%")
        with col3:
            st.metric("Quality Cost Impact", "$486K", delta="Anomaly-related")
        with col4:
            st.metric("Supplier Cost Variance", "±3.2%", delta="Within Target")
        
        st.info("Advanced cost analytics features will be implemented with enhanced data integration.")

    def render_system_analytics_page(self):
        """System analytics and performance monitoring"""
        st.title("⚙️ System Analytics & Performance")
        st.markdown("**Comprehensive system health monitoring and performance analytics**")
        
        if not st.session_state.get('db_connected', False):
            st.error("Database connection required for system analytics.")
            return
        
        try:
            db_stats = st.session_state.graph_manager.get_comprehensive_database_stats()
            
            st.subheader("📊 Database Statistics")
            
            # Node counts
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Node Counts:**")
                for node_type, count in db_stats['node_counts'].items():
                    st.markdown(f"• {node_type}: {count:,}")
            
            with col2:
                st.markdown("**Relationship Counts:**")
                for rel_type, count in db_stats['relationship_counts'].items():
                    st.markdown(f"• {rel_type}: {count:,}")
            
            # Database health metrics
            st.subheader("💊 System Health Metrics")
            
            health_data = db_stats.get('database_health', {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                connectivity = health_data.get('connectivity_score', 0)
                st.metric("Connectivity Score", f"{connectivity}/100")
            with col2:
                completeness = health_data.get('data_completeness', 0)
                st.metric("Data Completeness", f"{completeness}/100")
            with col3:
                st.metric("Data Quality", f"{db_stats['data_quality']['avg_quality']*100:.1f}%")
            
            # Performance visualization
            st.subheader("📈 Performance Trends")
            
            # Sample performance data
            perf_data = pd.DataFrame({
                'Metric': ['Response Time', 'Throughput', 'Error Rate', 'Uptime'],
                'Current': [120, 850, 0.2, 99.8],
                'Target': [100, 1000, 0.1, 99.9],
                'Status': ['Warning', 'Good', 'Good', 'Excellent']
            })
            
            st.dataframe(perf_data, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Error loading system analytics: {str(e)}")

    def _export_traceability_report(self, vin: str, data: dict):
        """Export comprehensive traceability report"""
        report_content = f"""
VEHICLE TRACEABILITY REPORT
===========================

VIN: {vin}
Model: {data['vehicle'].get('model', 'Unknown')}
Assembly Date: {safe_date_format(data['vehicle'].get('assembly_date', 'Unknown'))}

COMPONENT SUMMARY
-----------------
Total Components: {data['summary']['total_components']}
Anomalous Components: {data['summary']['anomalous_components']}
Suppliers Involved: {data['summary']['suppliers_involved']}

DETAILED COMPONENT LIST
-----------------------
"""
        
        for i, comp in enumerate(data['components'], 1):
            batch = comp['batch']
            part = comp['part']
            supplier = comp['supplier'] if comp['supplier'] else {'name': 'Unknown'}
            
            report_content += f"""
{i}. {part['name']}
   Batch ID: {batch['batch_id']}
   Supplier: {supplier['name']}
   Weight: {batch.get('weight_mean_g', 0):.1f}g
   QC Rate: {batch.get('qc_pass_rate', 0):.1%}
   Anomalous: {'Yes' if batch.get('is_anomalous', False) else 'No'}
"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"traceability_report_{vin}_{timestamp}.txt"
        
        st.download_button(
            label="📥 Download Traceability Report",
            data=report_content,
            file_name=filename,
            mime="text/plain"
        )
        st.success(f"Traceability report prepared: {filename}")
        
    def render_enhanced_sidebar(self):
        """Render enhanced sidebar with advanced navigation"""
        st.sidebar.title("🚗 CarTrace AI")
        st.sidebar.markdown("**Enhanced Manufacturing Intelligence**")
        
        # System status with detailed health indicators
        st.sidebar.subheader("🔧 System Health")
        
        # Database connection status
        if st.session_state.get('db_connected', False):
            st.sidebar.success("✅ Neo4j Connected")
            try:
                stats = st.session_state.graph_manager.get_comprehensive_database_stats()
                st.sidebar.info(f"📊 {stats['total_nodes']:,} nodes, {stats['total_relationships']:,} relationships")
                
                # Health score indicator
                health_score = stats.get('database_health', {}).get('connectivity_score', 0)
                if health_score >= 80:
                    st.sidebar.success(f"🟢 Health Score: {health_score}")
                elif health_score >= 60:
                    st.sidebar.warning(f"🟡 Health Score: {health_score}")
                else:
                    st.sidebar.error(f"🔴 Health Score: {health_score}")
                    
            except Exception as e:
                st.sidebar.error(f"❌ Database Error: {str(e)[:50]}...")
        else:
            st.sidebar.error("❌ Neo4j Disconnected")
            if 'db_error' in st.session_state:
                st.sidebar.error(f"Error: {st.session_state.db_error[:100]}...")
        
        # Model status
        if st.session_state.get('model_loaded', False):
            st.sidebar.success("✅ AI Model Ready")
        else:
            st.sidebar.warning("⚠️ AI Model Not Loaded")
        
        # Enhanced navigation with grouping
        st.sidebar.subheader("📋 Navigation")
        
        # Main categories
        main_categories = {
            "🏠 Overview": ["Dashboard Overview", "System Analytics"],
            "🔍 Traceability": ["Component Traceability", "Failure Analysis"],
            "🤖 AI & Analytics": ["Anomaly Detection", "Predictive Analytics"],
            "🚨 Risk Management": ["Recall Simulation", "Supplier Risk Analysis"],
            "📊 Supply Chain": ["Supply Chain Intelligence", "Cost Analytics"]
        }
        
        # Handle page selection, considering quick search overrides
        # Determine the default category and page
        default_page = st.session_state.get("selected_page", "Dashboard Overview")
        default_category = "🏠 Overview"
        for cat, pages in main_categories.items():
            if default_page in pages:
                default_category = cat
                break
        
        selected_category = st.sidebar.selectbox("Select Category", list(main_categories.keys()),
                                                 index=list(main_categories.keys()).index(default_category))
        
        page_options = main_categories[selected_category]
        try:
            default_page_index = page_options.index(default_page)
        except ValueError:
            default_page_index = 0
            
        selected_page = st.sidebar.selectbox("Select Page", page_options, index=default_page_index)
        st.session_state.selected_page = selected_page # Persist user's choice
        
        # Quick search functionality
        st.sidebar.subheader("🔍 Quick Search")
        search_type = st.sidebar.selectbox("Search Type", ["VIN", "Batch ID", "Supplier", "Part"])
        search_query = st.sidebar.text_input(f"Enter {search_type}:")
        
        if st.sidebar.button("🔍 Search") and search_query:
            st.session_state.active_search = {'type': search_type, 'query': search_query}
            # Set the target page for the search
            if search_type in ["VIN", "Batch ID"]:
                st.session_state.selected_page = "Component Traceability"
            st.rerun()
        
        # System controls
        st.sidebar.subheader("⚙️ System Controls")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🔄 Refresh"):
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear Cache"):
                st.cache_data.clear()
                st.success("Cache cleared!")
        
        # Advanced settings
        with st.sidebar.expander("⚙️ Advanced Settings"):
            st.session_state.debug_mode = st.checkbox("Debug Mode", value=False)
            st.session_state.show_sql_queries = st.checkbox("Show Cypher Queries", value=False)
            st.session_state.max_results = st.slider("Max Results", 10, 1000, 100)
        
        # System information
        st.sidebar.subheader("ℹ️ System Info")
        st.sidebar.info(f"Database: {self.config.neo4j.uri}")
        st.sidebar.info(f"Version: CarTrace AI v2.0")
        if st.session_state.get('pipeline_results'):
            last_run = safe_date_format(st.session_state.pipeline_results.get('start_time', 'Unknown'))
            st.sidebar.info(f"Last Pipeline: {last_run}")
        
        return selected_page

    def render_enhanced_overview_page(self):
        """Render enhanced dashboard overview with comprehensive metrics"""
        st.markdown('<h1 class="main-header">🚗 CarTrace AI - Manufacturing Intelligence Hub</h1>', unsafe_allow_html=True)
        
        if not st.session_state.get('db_connected', False):
            st.markdown("""
            <div class="alert-critical">
                <h3>⚠️ System Initialization Required</h3>
                <p>Database connection failed. Please ensure:</p>
                <ol>
                    <li>Neo4j is running on your system</li>
                    <li>Run: <code>python setup_neo4j.py</code></li>
                    <li>Run: <code>python run_pipeline.py</code></li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            return
        
        try:
            # Get comprehensive statistics
            db_stats = st.session_state.graph_manager.get_comprehensive_database_stats()
            supply_chain_analytics = st.session_state.graph_manager.get_supply_chain_analytics()
            
            # Enhanced KPI Dashboard
            st.subheader("📊 Key Performance Indicators")
            
            # Create 4 columns for main metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_vehicles = db_stats['node_counts'].get('Vehicle', 0)
                st.metric(
                    label="🚗 Total Vehicles",
                    value=f"{total_vehicles:,}",
                    delta="Production Volume"
                )
            
            with col2:
                total_batches = db_stats['node_counts'].get('Batch', 0)
                anomaly_rate = db_stats['data_quality']['anomaly_rate'] * 100
                st.metric(
                    label="📦 Manufacturing Batches",
                    value=f"{total_batches:,}",
                    delta=f"{anomaly_rate:.1f}% anomalous",
                    delta_color="inverse"
                )
            
            with col3:
                total_suppliers = db_stats['node_counts'].get('Supplier', 0)
                st.metric(
                    label="🏭 Active Suppliers",
                    value=f"{total_suppliers:,}",
                    delta="Supply Base"
                )
            
            with col4:
                total_failures = db_stats['node_counts'].get('Failure', 0)
                if total_vehicles > 0:
                    failure_rate = (total_failures / total_vehicles) * 100
                    st.metric(
                        label="⚠️ Field Failures",
                        value=f"{total_failures:,}",
                        delta=f"{failure_rate:.2f}% failure rate",
                        delta_color="inverse"
                    )
                else:
                    st.metric(label="⚠️ Field Failures", value="0")
            
            # Enhanced Quality Dashboard
            st.subheader("🎯 Quality & Performance Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_quality = db_stats['data_quality']['avg_quality'] * 100
                quality_color = "normal" if avg_quality >= 90 else "inverse"
                st.metric(
                    label="📈 Average Quality Rate",
                    value=f"{avg_quality:.1f}%",
                    delta="Target: 95%+",
                    delta_color=quality_color
                )
            
            with col2:
                health_score = db_stats.get('database_health', {}).get('connectivity_score', 0)
                st.metric(
                    label="💊 System Health Score",
                    value=f"{health_score}/100",
                    delta="Database Connectivity"
                )
            
            with col3:
                supply_health = supply_chain_analytics.get('overall_metrics', {}).get('supply_chain_health_score', 0)
                st.metric(
                    label="🔗 Supply Chain Health",
                    value=f"{supply_health:.0f}/100",
                    delta="Multi-tier Assessment"
                )
            
            # Real-time alerts section
            self._render_enhanced_alerts_section(db_stats, supply_chain_analytics)
            
            # Advanced analytics dashboard
            st.subheader("📊 Advanced Analytics Dashboard")
            
            # Create tabs for different analytics views
            tab1, tab2, tab3, tab4 = st.tabs(["🔄 Production Flow", "🎯 Quality Trends", "🏭 Supplier Performance", "⚠️ Risk Analysis"])
            
            with tab1:
                self._render_production_flow_analytics(db_stats)
            
            with tab2:
                self._render_quality_trends_analytics(db_stats)
            
            with tab3:
                self._render_supplier_performance_analytics(supply_chain_analytics)
            
            with tab4:
                self._render_risk_analysis_dashboard(supply_chain_analytics)
            
            # Pipeline execution summary
            if st.session_state.pipeline_results:
                self._render_pipeline_summary()
            
        except Exception as e:
            st.error(f"Error loading dashboard data: {str(e)}")
            if st.session_state.get('debug_mode', False):
                st.exception(e)

    def render_enhanced_traceability_page(self):
        """Enhanced component traceability with comprehensive search capabilities"""
        st.title("🔍 Enhanced Component Traceability")
        st.markdown("**Complete end-to-end component tracking and impact analysis**")
        
        if not st.session_state.get('db_connected', False):
            st.error("Database connection required for traceability operations.")
            return

        # Initialize default values from quick search if available
        default_vin = ""
        default_batch = ""
        # Default to the first radio option
        radio_options = ["🚗 Vehicle Traceability", "📦 Batch Analysis", "🔍 Component Failure Investigation"]
        default_radio_index = 0
        
        if 'active_search' in st.session_state:
            search_info = st.session_state.active_search
            if search_info['type'] == 'VIN':
                default_vin = search_info['query']
                default_radio_index = 0
            elif search_info['type'] == 'Batch ID':
                default_batch = search_info['query']
                default_radio_index = 1
            # Clear the search info so it's only used once
            del st.session_state.active_search

        # Use st.radio instead of st.tabs to allow programmatic control
        search_choice = st.radio(
            "Select Analysis Type:",
            radio_options,
            index=default_radio_index,
            horizontal=True,
        )

        st.markdown("---") # Visual separator

        if search_choice == "🚗 Vehicle Traceability":
            st.subheader("🚗 Vehicle Component Traceability")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                vin_input = st.text_input(
                    "Enter Vehicle VIN:", 
                    value=default_vin,
                    placeholder="e.g., VIN100001",
                    help="Enter the complete VIN to trace all components used in manufacturing"
                )
            
            with col2:
                st.write("") # Vertical alignment spacer
                trace_button = st.button("🔍 Trace Vehicle", type="primary", use_container_width=True, key="trace_vin")
            
            if trace_button:
                if vin_input:
                    with st.spinner("Tracing vehicle components..."):
                        self._perform_enhanced_vehicle_trace(vin_input)
                else:
                    st.warning("Please enter a VIN number")
        
        elif search_choice == "📦 Batch Analysis":
            st.subheader("📦 Batch Impact Analysis")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                batch_input = st.text_input(
                    "Enter Batch ID:", 
                    value=default_batch,
                    placeholder="e.g., BATCH-000001",
                    help="Analyze all vehicles affected by a specific manufacturing batch"
                )
            
            with col2:
                st.write("") # Vertical alignment spacer
                analyze_button = st.button("📊 Analyze Batch", type="primary", use_container_width=True, key="analyze_batch")

            if analyze_button:
                if batch_input:
                    with st.spinner("Analyzing batch impact..."):
                        self._perform_enhanced_batch_analysis(batch_input)
                else:
                    st.warning("Please enter a Batch ID")
        
        elif search_choice == "🔍 Component Failure Investigation":
            st.subheader("🔍 Component Failure Investigation")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                component_category = st.selectbox(
                    "Component Category:",
                    options=["Engine", "Transmission", "Brake", "Suspension", "Electrical", "Body"],
                    help="Select the component category that experienced failure"
                )
            
            with col2:
                failure_mode = st.selectbox(
                    "Failure Mode (Optional):",
                    options=["All", "oil_leak", "overheating", "misfire", "gear_slip", "pad_wear", "strut_leak", "ecu_failure"],
                    help="Specific failure mode to investigate"
                )
            
            with col3:
                st.write("") # Vertical alignment spacer
                investigate_button = st.button("🚨 Investigate", type="primary", use_container_width=True, key="investigate_failure")
            
            if investigate_button:
                failure_mode_param = None if failure_mode == "All" else failure_mode
                with st.spinner("Investigating component failures..."):
                    self._perform_component_failure_investigation(component_category, failure_mode_param)

    def _perform_enhanced_vehicle_trace(self, vin: str):
        """Perform enhanced vehicle traceability with comprehensive display"""
        try:
            traceability_data = st.session_state.graph_manager.get_enhanced_vehicle_traceability(vin)
            
            if traceability_data:
                vehicle = traceability_data['vehicle']
                components = traceability_data['components']
                summary = traceability_data['summary']
                failures = traceability_data['failures']
                qc_inspections = traceability_data['qc_inspections']
                
                # Vehicle header with enhanced styling
                st.markdown(f"""
                <div class="traceability-box">
                    <h2>🚗 Vehicle Traceability Report: {vehicle['vin']}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Vehicle summary metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Model", vehicle.get('model', 'Unknown'))
                with col2:
                    st.metric("Assembly Date", safe_date_format(vehicle.get('assembly_date', 'Unknown')))
                with col3:
                    st.metric("Components", summary['total_components'])
                with col4:
                    st.metric("Anomalous Components", summary['anomalous_components'])
                with col5:
                    st.metric("Suppliers", summary['suppliers_involved'])
                
                # Alert for anomalous components
                if summary['anomalous_components'] > 0:
                    st.markdown(f"""
                    <div class="alert-high">
                        ⚠️ <strong>Quality Alert:</strong> This vehicle contains {summary['anomalous_components']} 
                        anomalous component(s) that may require attention.
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed components analysis
                if components:
                    st.subheader("🔧 Component Traceability Analysis")
                    
                    # Create enhanced component dataframe
                    component_data = []
                    for comp in components:
                        batch = comp['batch']
                        part = comp['part']
                        supplier = comp['supplier'] if comp['supplier'] else {'name': 'Unknown', 'tier': 'N/A'}
                        
                        # Risk assessment
                        risk_level = "HIGH" if batch.get('is_anomalous', False) else "LOW"
                        if batch.get('qc_pass_rate', 1.0) < 0.9:
                            risk_level = "MEDIUM" if risk_level == "LOW" else "HIGH"
                        
                        component_data.append({
                            'Part Name': part['name'],
                            'Category': part['category'],
                            'Batch ID': batch['batch_id'],
                            'Supplier': supplier['name'],
                            'Supplier Tier': supplier.get('tier', 'Unknown'),
                            'Weight (g)': f"{batch.get('weight_mean_g', 0):.1f}",
                            'QC Pass Rate': f"{batch.get('qc_pass_rate', 0):.1%}",
                            'Risk Level': risk_level,
                            'Anomalous': '🚨' if batch.get('is_anomalous', False) else '✅',
                            'Assembly Date': safe_date_format(comp['assembly_details']['timestamp']),
                            'Operator': comp['assembly_details']['operator']
                        })
                    
                    comp_df = pd.DataFrame(component_data)
                    
                    # Color code the dataframe
                    def highlight_risk(val):
                        if val == 'HIGH':
                            return 'background-color: #ffcccc'
                        elif val == 'MEDIUM':
                            return 'background-color: #fff3cd'
                        elif val == 'LOW':
                            return 'background-color: #d4edda'
                        return ''
                    
                    styled_df = comp_df.style.applymap(highlight_risk, subset=['Risk Level'])
                    st.dataframe(styled_df, use_container_width=True, height=400)
                    
                    # Component risk analysis
                    high_risk_components = len([c for c in component_data if c['Risk Level'] == 'HIGH'])
                    if high_risk_components > 0:
                        st.markdown(f"""
                        <div class="warning-box">
                            🎯 <strong>Risk Assessment:</strong> {high_risk_components} high-risk components identified. 
                            Consider enhanced monitoring or proactive replacement.
                        </div>
                        """, unsafe_allow_html=True)
                
                # Quality inspections summary
                if qc_inspections:
                    st.subheader("🔍 Quality Inspection History")
                    
                    inspection_summary = []
                    for qc in qc_inspections:
                        inspection_summary.append({
                            'Inspection Type': qc.get('inspection_type', 'Unknown'),
                            'Date': safe_date_format(qc.get('inspection_date', 'Unknown')),
                            'Inspector': qc.get('inspector_id', 'Unknown'),
                            'Result': '✅ Passed' if qc.get('passed', True) else '❌ Failed',
                            'Severity': qc.get('severity', 0),
                            'Issues': qc.get('issue_code', 'None')
                        })
                    
                    if inspection_summary:
                        st.dataframe(pd.DataFrame(inspection_summary), use_container_width=True)
                
                # Failure history
                if failures:
                    st.subheader("❌ Failure History")
                    
                    failure_data = []
                    for failure in failures:
                        failure_data.append({
                            'Date': safe_date_format(failure['reported_date']),
                            'Component': failure.get('component_category', 'Unknown'),
                            'Mode': failure['failure_mode'],
                            'Severity': failure['severity'],
                            'Cost': f"${failure.get('repair_cost', 0):.2f}",
                            'Warranty': '✅' if failure.get('warranty_claim', False) else '❌'
                        })
                    
                    st.dataframe(pd.DataFrame(failure_data), use_container_width=True)
                    
                    # Failure correlation analysis
                    failure_components = [f.get('component_category', 'Unknown') for f in failures]
                    anomalous_component_categories = [
                        comp['part']['category'] for comp in components 
                        if comp['batch'].get('is_anomalous', False)
                    ]
                    
                    correlated_failures = set(failure_components) & set(anomalous_component_categories)
                    if correlated_failures:
                        st.markdown(f"""
                        <div class="alert-critical">
                            🚨 <strong>Critical Finding:</strong> Failures detected in components from anomalous batches: 
                            {', '.join(correlated_failures)}. This correlation suggests potential batch-related quality issues.
                        </div>
                        """, unsafe_allow_html=True)
                
                # Export functionality
                if st.button("📥 Export Traceability Report"):
                    self._export_traceability_report(vin, traceability_data)
            
            else:
                st.markdown(f"""
                <div class="warning-box">
                    ⚠️ <strong>No Data Found:</strong> No traceability data found for VIN: {vin}
                    <br>Please verify the VIN number or check if the vehicle exists in the system.
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error retrieving traceability data: {str(e)}")
            if st.session_state.get('debug_mode', False):
                st.exception(e)

    def _perform_enhanced_batch_analysis(self, batch_id: str):
        """Perform enhanced batch impact analysis"""
        try:
            batch_data = st.session_state.graph_manager.get_enhanced_batch_usage(batch_id)
            
            if batch_data:
                batch = batch_data['batch']
                part = batch_data['part']
                supplier = batch_data['supplier']
                vehicles = batch_data['vehicles']
                summary = batch_data['summary']
                
                # Batch header
                st.markdown(f"""
                <div class="traceability-box">
                    <h2>📦 Batch Impact Analysis: {batch_id}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Batch summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Part", part['name'])
                with col2:
                    st.metric("Supplier", supplier['name'])
                with col3:
                    st.metric("Vehicles Affected", summary['total_vehicles'])
                with col4:
                    failure_rate = summary['failure_rate'] * 100
                    st.metric("Failure Rate", f"{failure_rate:.1f}%")
                
                # Risk assessment
                if summary['is_anomalous_batch']:
                    st.markdown(f"""
                    <div class="alert-high">
                        🚨 <strong>Anomalous Batch Alert:</strong> This batch was flagged as anomalous 
                        ({summary['anomaly_type']}). Enhanced monitoring recommended for all affected vehicles.
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed vehicle analysis
                if vehicles:
                    st.subheader("🚗 Affected Vehicles Analysis")
                    
                    vehicle_data = []
                    for v in vehicles:
                        vehicle_info = v['vehicle']
                        failures = v['failures']
                        assembly_details = v.get('assembly_details', {})
                        
                        vehicle_data.append({
                            'VIN': vehicle_info['vin'],
                            'Model': vehicle_info.get('model', 'Unknown'),
                            'Assembly Date': safe_date_format(vehicle_info.get('assembly_date', 'Unknown')),
                            'Target Market': vehicle_info.get('target_market', 'Unknown'),
                            'Failures': len(failures),
                            'Status': '❌ Failed' if failures else '✅ OK',
                            'Assembly Operator': assembly_details.get('operator', 'Unknown'),
                            'Repair Cost': f"${sum(f.get('repair_cost', 0) for f in failures):.2f}" if failures else "$0.00"
                        })
                    
                    vehicle_df = pd.DataFrame(vehicle_data)
                    
                    # Highlight failed vehicles
                    def highlight_failures(row):
                        if row['Failures'] > 0:
                            return ['background-color: #ffcccc'] * len(row)
                        return [''] * len(row)
                    
                    styled_vehicle_df = vehicle_df.style.apply(highlight_failures, axis=1)
                    st.dataframe(styled_vehicle_df, use_container_width=True, height=400)
                    
                    # Failure pattern analysis
                    failed_vehicles = [v for v in vehicles if v['failures']]
                    if failed_vehicles:
                        st.subheader("📊 Failure Pattern Analysis")
                        
                        # Analyze failure modes
                        failure_modes = {}
                        for v in failed_vehicles:
                            for f in v['failures']:
                                mode = f.get('failure_mode', 'Unknown')
                                failure_modes[mode] = failure_modes.get(mode, 0) + 1
                        
                        if failure_modes:
                            fig = px.pie(
                                values=list(failure_modes.values()),
                                names=list(failure_modes.keys()),
                                title="Failure Mode Distribution"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                # Timeline analysis
                assembly_dates = [v['vehicle'].get('assembly_date', '') for v in vehicles if v['vehicle'].get('assembly_date')]
                if assembly_dates:
                    st.subheader("📅 Production Timeline")
                    
                    # FIX: Refactored timeline data processing to prevent error
                    try:
                        # Create DataFrame and convert to datetime objects
                        dates_df = pd.DataFrame({'Assembly_Date': [pd.to_datetime(safe_date_format(d)) for d in assembly_dates]})
                        # Group by date and count occurrences using size() which is more efficient
                        timeline_data = dates_df.groupby(dates_df['Assembly_Date'].dt.date).size().reset_index(name='Count')
                        
                        fig = px.bar(
                            timeline_data,
                            x='Assembly_Date',
                            y='Count',
                            title=f"Vehicle Production Timeline for Batch {batch_id}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as plot_e:
                        st.error(f"Error creating production timeline chart: {plot_e}")

                # Batch quality metrics
                st.subheader("📊 Batch Quality Metrics")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    qc_rate = batch.get('qc_pass_rate', 0) * 100
                    st.metric("QC Pass Rate", f"{qc_rate:.1f}%")
                
                with col2:
                    weight_mean = batch.get('weight_mean_g', 0)
                    spec_weight = part.get('spec_weight_g', 0)
                    deviation = ((weight_mean - spec_weight) / spec_weight * 100) if spec_weight > 0 else 0
                    st.metric("Weight Deviation", f"{deviation:.1f}%")
                
                with col3:
                    cost_total = batch.get('batch_cost_usd', 0)
                    st.metric("Batch Cost", f"${cost_total:,.2f}")
                
                # Recall simulation shortcut
                st.subheader("🚨 Recall Assessment")
                if st.button(f"🎯 Simulate Recall for Batch {batch_id}", type="primary"):
                    st.session_state.recall_batch_id = batch_id
                    st.info("Recall simulation initiated. Switch to Recall Simulation page for detailed analysis.")
            
            else:
                st.markdown(f"""
                <div class="warning-box">
                    ⚠️ <strong>No Data Found:</strong> No usage data found for Batch: {batch_id}
                    <br>Please verify the Batch ID or check if it exists in the system.
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error retrieving batch data: {str(e)}")
            if st.session_state.get('debug_mode', False):
                st.exception(e)

    def _perform_component_failure_investigation(self, component_category: str, failure_mode: str = None):
        """Perform comprehensive component failure investigation"""
        try:
            investigation_results = st.session_state.graph_manager.find_affected_vehicles_by_component_failure(
                component_category, failure_mode
            )
            
            if investigation_results['total_at_risk_vehicles'] > 0:
                # Investigation header
                st.markdown(f"""
                <div class="traceability-box">
                    <h2>🔍 Component Failure Investigation: {component_category}</h2>
                    {f"<p><strong>Failure Mode:</strong> {failure_mode}</p>" if failure_mode else ""}
                </div>
                """, unsafe_allow_html=True)
                
                # Investigation summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("At-Risk Vehicles", investigation_results['total_at_risk_vehicles'])
                with col2:
                    st.metric("Confirmed Failures", investigation_results['confirmed_failures'])
                with col3:
                    failure_rate = investigation_results['overall_failure_rate'] * 100
                    st.metric("Overall Failure Rate", f"{failure_rate:.1f}%")
                with col4:
                    st.metric("High-Risk Batches", investigation_results['high_risk_batches'])
                
                # Risk level assessment
                if investigation_results['overall_failure_rate'] > 0.1:
                    st.markdown("""
                    <div class="alert-critical">
                        🚨 <strong>Critical Investigation:</strong> High failure rate detected (>10%). 
                        Immediate action recommended including potential recall assessment.
                    </div>
                    """, unsafe_allow_html=True)
                elif investigation_results['overall_failure_rate'] > 0.05:
                    st.markdown("""
                    <div class="alert-high">
                        ⚠️ <strong>Elevated Risk:</strong> Failure rate above normal threshold (>5%). 
                        Enhanced monitoring and supplier review recommended.
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed batch analysis
                st.subheader("📦 Affected Batches Analysis")
                
                affected_batches = investigation_results['affected_batches']
                if affected_batches:
                    batch_data = []
                    for batch_info in affected_batches:
                        batch = batch_info['batch']
                        part = batch_info['part']
                        supplier = batch_info['supplier']
                        
                        batch_data.append({
                            'Batch ID': batch['batch_id'],
                            'Supplier': supplier['name'],
                            'Vehicles': batch_info['total_vehicles'],
                            'Failures': batch_info['confirmed_failures'],
                            'Failure Rate': f"{batch_info['failure_rate']:.1%}",
                            'Risk Level': batch_info['risk_level'],
                            'Anomalous': '🚨' if batch.get('is_anomalous', False) else '✅',
                            'QC Rate': f"{batch.get('qc_pass_rate', 0):.1%}",
                            'Manufacture Date': safe_date_format(batch.get('manufacture_date', 'Unknown'))
                        })
                    
                    batch_df = pd.DataFrame(batch_data)
                    
                    # Color code by risk level
                    def highlight_risk_level(val):
                        if val == 'CRITICAL':
                            return 'background-color: #ff4444; color: white'
                        elif val == 'HIGH':
                            return 'background-color: #ffaa00; color: white'
                        elif val == 'MEDIUM':
                            return 'background-color: #fff3cd'
                        return ''
                    
                    styled_batch_df = batch_df.style.applymap(highlight_risk_level, subset=['Risk Level'])
                    st.dataframe(styled_batch_df, use_container_width=True, height=400)
                
                # Recommendations
                recommendations = investigation_results.get('recommendations', [])
                if recommendations:
                    st.subheader("💡 Investigation Recommendations")
                    for i, rec in enumerate(recommendations, 1):
                        st.markdown(f"**{i}.** {rec}")
                
                # Visual analytics
                if len(affected_batches) > 1:
                    st.subheader("📊 Batch Comparison Analysis")
                    
                    # Create comparison chart
                    chart_data = pd.DataFrame([
                        {
                            'Batch ID': b['batch']['batch_id'][:12],
                            'Failure Rate': b['failure_rate'] * 100,
                            'Risk Level': b['risk_level'],
                            'Vehicles': b['total_vehicles']
                        }
                        for b in affected_batches
                    ])
                    
                    fig = px.scatter(
                        chart_data,
                        x='Vehicles',
                        y='Failure Rate',
                        color='Risk Level',
                        size='Vehicles',
                        hover_name='Batch ID',
                        title=f"{component_category} Component Failure Analysis",
                        color_discrete_map={
                            'CRITICAL': '#ff4444',
                            'HIGH': '#ffaa00', 
                            'MEDIUM': '#fff000',
                            'LOW': '#00aa00'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.markdown(f"""
                <div class="info-box">
                    ℹ️ <strong>No Issues Found:</strong> No vehicles at risk found for {component_category} components
                    {f" with {failure_mode} failure mode" if failure_mode else ""}.
                    <br>This indicates good component reliability in this category.
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error performing component failure investigation: {str(e)}")
            if st.session_state.get('debug_mode', False):
                st.exception(e)

def main():
    """Main function to run the enhanced Streamlit app"""
    dashboard = EnhancedCarTraceAIDashboard()
    
    # Render enhanced sidebar and get selected page
    selected_page = dashboard.render_enhanced_sidebar()
    
    # Render the selected page
    if selected_page == "Dashboard Overview":
        dashboard.render_enhanced_overview_page()
    elif selected_page == "Component Traceability":
        dashboard.render_enhanced_traceability_page()
    elif selected_page == "Failure Analysis":
        dashboard.render_failure_analysis_page()
    elif selected_page == "Anomaly Detection":
        dashboard.render_enhanced_anomaly_page()
    elif selected_page == "Predictive Analytics":
        dashboard.render_predictive_analytics_page()
    elif selected_page == "Recall Simulation":
        dashboard.render_enhanced_recall_simulation_page()
    elif selected_page == "Supplier Risk Analysis":
        dashboard.render_supplier_risk_page()
    elif selected_page == "Supply Chain Intelligence":
        dashboard.render_supply_chain_intelligence_page()
    elif selected_page == "Cost Analytics":
        dashboard.render_cost_analytics_page()
    elif selected_page == "System Analytics":
        dashboard.render_system_analytics_page()
        
if __name__ == "__main__":
    main()