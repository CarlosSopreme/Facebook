# marketing_analytics.py - COMPREHENSIVE FUNNEL ANALYSIS + ANOMALY DETECTION
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import os
import json
from datetime import datetime
from enum import Enum

# BigQuery imports (for spend data)
try:
    from google.cloud import bigquery
    import pandas as pd
    from google.oauth2 import service_account
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False

# Claude import for AI insights
try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

# BigQuery setup
PROJECT_ID = "gtm-p3gj3zzk-nthlo"
DATASET_ID = "last_14_days_analysis"

if BIGQUERY_AVAILABLE:
    try:
        credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if credentials_json:
            credentials_info = json.loads(credentials_json)
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            bigquery_client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
        else:
            bigquery_client = bigquery.Client(project=PROJECT_ID)
    except Exception as e:
        bigquery_client = None
else:
    bigquery_client = None

# Load API key from environment variable (set in Railway dashboard)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set")

# Initialize Claude client
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if CLAUDE_AVAILABLE else None

# MODELS
class SeverityLevel(str, Enum):
    YELLOW = "yellow"
    ORANGE = "orange" 
    RED = "red"

class FunnelDataPoint(BaseModel):
    date: str
    leads: int
    start_flows: int
    estimates: int
    closings: int
    funded: int
    rpts: int

class CampaignData(BaseModel):
    platform: str
    campaign_name: str
    adset_name: Optional[str] = None
    campaign_id: Optional[str] = None
    utm_campaign: Optional[str] = None
    analysis_date: str
    comparison_date: str
    
    # Core metrics with change percentages
    recent_spend: Optional[float] = None
    baseline_spend: Optional[float] = None
    spend_change_pct: Optional[float] = None
    
    recent_impressions: Optional[float] = None
    baseline_impressions: Optional[float] = None
    impressions_change_pct: Optional[float] = None
    
    recent_clicks: Optional[float] = None
    baseline_clicks: Optional[float] = None
    clicks_change_pct: Optional[float] = None
    
    recent_ctr: Optional[float] = None
    baseline_ctr: Optional[float] = None
    ctr_change_pct: Optional[float] = None
    
    recent_cpc: Optional[float] = None
    baseline_cpc: Optional[float] = None
    cpc_change_pct: Optional[float] = None
    
    recent_cpm: Optional[float] = None
    baseline_cpm: Optional[float] = None
    cpm_change_pct: Optional[float] = None
    
    recent_cost_per_lead: Optional[float] = None
    baseline_cost_per_lead: Optional[float] = None
    cost_per_lead_change_pct: Optional[float] = None
    
    # Additional metrics for comprehensive analysis
    recent_leads: Optional[float] = None
    baseline_leads: Optional[float] = None
    leads_change_pct: Optional[float] = None
    
    recent_start_flows: Optional[float] = None
    baseline_start_flows: Optional[float] = None
    start_flows_change_pct: Optional[float] = None
    
    recent_estimates: Optional[float] = None
    baseline_estimates: Optional[float] = None
    estimates_change_pct: Optional[float] = None
    
    recent_closings: Optional[float] = None
    baseline_closings: Optional[float] = None
    closings_change_pct: Optional[float] = None
    
    recent_funded: Optional[float] = None
    baseline_funded: Optional[float] = None
    funded_change_pct: Optional[float] = None
    
    recent_revenue: Optional[float] = None
    baseline_revenue: Optional[float] = None
    revenue_change_pct: Optional[float] = None
    
    recent_cost_per_estimate: Optional[float] = None
    baseline_cost_per_estimate: Optional[float] = None
    cost_per_estimate_change_pct: Optional[float] = None
    
    recent_cost_per_closing: Optional[float] = None
    baseline_cost_per_closing: Optional[float] = None
    cost_per_closing_change_pct: Optional[float] = None
    
    recent_cost_per_funded: Optional[float] = None
    baseline_cost_per_funded: Optional[float] = None
    cost_per_funded_change_pct: Optional[float] = None
    
    recent_estimate_cvr: Optional[float] = None
    baseline_estimate_cvr: Optional[float] = None
    estimate_cvr_change_pct: Optional[float] = None
    
    recent_closings_cvr: Optional[float] = None
    baseline_closings_cvr: Optional[float] = None
    closings_cvr_change_pct: Optional[float] = None
    
    recent_funded_cvr: Optional[float] = None
    baseline_funded_cvr: Optional[float] = None
    funded_cvr_change_pct: Optional[float] = None
    
    recent_roas: Optional[float] = None
    baseline_roas: Optional[float] = None
    roas_change_pct: Optional[float] = None
    
    # Google-specific metrics
    recent_cpa: Optional[float] = None
    baseline_cpa: Optional[float] = None
    cpa_change_pct: Optional[float] = None
    
    recent_ad_roas: Optional[float] = None
    baseline_ad_roas: Optional[float] = None
    ad_roas_change_pct: Optional[float] = None
    
    recent_ad_conversions: Optional[float] = None
    baseline_ad_conversions: Optional[float] = None
    ad_conversions_change_pct: Optional[float] = None
    
    has_anomaly: Optional[str] = None
    performance_status: Optional[str] = None

class AnomalyAlert(BaseModel):
    severity: SeverityLevel
    platform: str
    campaign_name: str
    adset_name: Optional[str] = None
    metric: str
    current_value: Optional[float] = None
    previous_value: Optional[float] = None
    change_pct: Optional[float] = None
    message: str
    slack_color: str

class CustomThresholds(BaseModel):
    # Meta thresholds (Ad Set level) - NULL means no alerting
    # Based on percentage change of actual values
    meta_spend: Optional[float] = None  # % change that triggers alert
    meta_impressions: Optional[float] = None
    meta_cpm: Optional[float] = None
    meta_clicks: Optional[float] = None
    meta_ctr: Optional[float] = None
    meta_cpc: Optional[float] = None
    meta_leads: Optional[float] = None
    meta_start_flows: Optional[float] = None
    meta_estimates: Optional[float] = None
    meta_closings: Optional[float] = None
    meta_funded: Optional[float] = None
    meta_revenue: Optional[float] = None
    meta_cost_per_closing: Optional[float] = None
    meta_cost_per_lead: Optional[float] = None
    meta_cost_per_estimate: Optional[float] = None
    meta_cost_per_funded: Optional[float] = None
    meta_estimate_cvr: Optional[float] = None
    meta_closings_cvr: Optional[float] = None
    meta_funded_cvr: Optional[float] = None
    meta_roas: Optional[float] = None
    
    # Google thresholds (Campaign level) - NULL means no alerting
    google_spend: Optional[float] = None
    google_impressions: Optional[float] = None
    google_cpm: Optional[float] = None
    google_clicks: Optional[float] = None
    google_ctr: Optional[float] = None
    google_cpc: Optional[float] = None
    google_leads: Optional[float] = None
    google_start_flows: Optional[float] = None
    google_estimates: Optional[float] = None
    google_closings: Optional[float] = None
    google_funded: Optional[float] = None
    google_revenue: Optional[float] = None
    google_cost_per_closing: Optional[float] = None
    google_cost_per_lead: Optional[float] = None
    google_cost_per_estimate: Optional[float] = None
    google_cost_per_funded: Optional[float] = None
    google_estimate_cvr: Optional[float] = None
    google_closings_cvr: Optional[float] = None
    google_funded_cvr: Optional[float] = None
    google_roas: Optional[float] = None
    google_cpa: Optional[float] = None
    google_ad_roas: Optional[float] = None
    google_ad_conversions: Optional[float] = None

class FunnelAnalysisRequest(BaseModel):
    funnel_data: List[FunnelDataPoint]

class FunnelAnalysisResponse(BaseModel):
    status: str
    colorCode: Optional[str] = None
    formatted_metrics: Optional[Dict[str, Any]] = None
    day_over_day_metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ComprehensiveAnalysisRequest(BaseModel):
    funnel_data: Optional[List[FunnelDataPoint]] = None
    dod_campaign_data: Optional[List[CampaignData]] = None
    custom_thresholds: Optional[CustomThresholds] = None

class ComprehensiveAnalysisResponse(BaseModel):
    anomaly_alerts: List[str] = []

# ANOMALY DETECTION CLASS
class AnomalyDetector:
    """Simplified anomaly detection - only alerts when user sets thresholds"""
    
    def detect_anomalies(self, campaign: CampaignData, custom_thresholds: Optional[CustomThresholds] = None) -> List[str]:
        """Detect anomalies based on user-defined thresholds only"""
        alerts = []
        
        # No thresholds = no alerts
        if not custom_thresholds:
            return alerts
        
        platform = campaign.platform.lower()
        prefix = "meta" if platform == "meta" else "google"
        
        # Define metrics with their funnel priority (1 = highest priority/furthest down funnel)
        # Lower numbers = higher priority (closer to revenue)
        metrics_to_check = [
            # Revenue metrics (highest priority)
            (1, "revenue_change_pct", f"{prefix}_revenue", "Revenue", campaign.recent_revenue, campaign.baseline_revenue, "negative"),
            (1, "roas_change_pct", f"{prefix}_roas", "ROAS", campaign.recent_roas, campaign.baseline_roas, "negative"),
            
            # Funded metrics
            (2, "funded_change_pct", f"{prefix}_funded", "Funded", campaign.recent_funded, campaign.baseline_funded, "negative"),
            (2, "cost_per_funded_change_pct", f"{prefix}_cost_per_funded", "Cost per Funded", campaign.recent_cost_per_funded, campaign.baseline_cost_per_funded, "positive"),
            (2, "funded_cvr_change_pct", f"{prefix}_funded_cvr", "Funded CVR", campaign.recent_funded_cvr, campaign.baseline_funded_cvr, "negative"),
            
            # Closings metrics
            (3, "closings_change_pct", f"{prefix}_closings", "Closings", campaign.recent_closings, campaign.baseline_closings, "negative"),
            (3, "cost_per_closing_change_pct", f"{prefix}_cost_per_closing", "Cost per Closing", campaign.recent_cost_per_closing, campaign.baseline_cost_per_closing, "positive"),
            (3, "closings_cvr_change_pct", f"{prefix}_closings_cvr", "Closings CVR", campaign.recent_closings_cvr, campaign.baseline_closings_cvr, "negative"),
            
            # Estimates metrics
            (4, "estimates_change_pct", f"{prefix}_estimates", "Estimates", campaign.recent_estimates, campaign.baseline_estimates, "negative"),
            (4, "cost_per_estimate_change_pct", f"{prefix}_cost_per_estimate", "Cost per Estimate", campaign.recent_cost_per_estimate, campaign.baseline_cost_per_estimate, "positive"),
            (4, "estimate_cvr_change_pct", f"{prefix}_estimate_cvr", "Estimate CVR", campaign.recent_estimate_cvr, campaign.baseline_estimate_cvr, "negative"),
            
            # Start Flows metrics (between leads and estimates)
            (5, "start_flows_change_pct", f"{prefix}_start_flows", "Start Flows", campaign.recent_start_flows, campaign.baseline_start_flows, "negative"),
            
            # Leads metrics
            (6, "leads_change_pct", f"{prefix}_leads", "Leads", campaign.recent_leads, campaign.baseline_leads, "negative"),
            (6, "cost_per_lead_change_pct", f"{prefix}_cost_per_lead", "Cost per Lead", campaign.recent_cost_per_lead, campaign.baseline_cost_per_lead, "positive"),
            
            # Click metrics
            (7, "clicks_change_pct", f"{prefix}_clicks", "Clicks", campaign.recent_clicks, campaign.baseline_clicks, "negative"),
            (7, "ctr_change_pct", f"{prefix}_ctr", "CTR", campaign.recent_ctr, campaign.baseline_ctr, "negative"),
            (7, "cpc_change_pct", f"{prefix}_cpc", "CPC", campaign.recent_cpc, campaign.baseline_cpc, "positive"),
            
            # Impression metrics (lowest priority)
            (8, "impressions_change_pct", f"{prefix}_impressions", "Impressions", campaign.recent_impressions, campaign.baseline_impressions, "negative"),
            (8, "cpm_change_pct", f"{prefix}_cpm", "CPM", campaign.recent_cpm, campaign.baseline_cpm, "positive"),
            
            # Spend (special case)
            (9, "spend_change_pct", f"{prefix}_spend", "Spend", campaign.recent_spend, campaign.baseline_spend, "either"),
        ]
        
        # Add Google-specific metrics
        if platform == "google":
            metrics_to_check.extend([
                (7, "cpa_change_pct", "google_cpa", "CPA", campaign.recent_cpa, campaign.baseline_cpa, "positive"),
                (1, "ad_roas_change_pct", "google_ad_roas", "Ad ROAS", campaign.recent_ad_roas, campaign.baseline_ad_roas, "negative"),
                (6, "ad_conversions_change_pct", "google_ad_conversions", "Ad Conversions", campaign.recent_ad_conversions, campaign.baseline_ad_conversions, "negative"),
            ])
        
        # Check each metric against its threshold and collect potential alerts
        potential_alerts = []
        
        for priority, change_field, threshold_field, metric_name, recent_value, baseline_value, bad_direction in metrics_to_check:
            # Get the change percentage from campaign data
            change_pct = getattr(campaign, change_field, None)
            
            # Get the threshold from custom_thresholds
            threshold = getattr(custom_thresholds, threshold_field, None)
            
            # Skip if no threshold set or no change data
            if threshold is None or change_pct is None:
                continue
            
            # Check if change is in the "bad" direction and exceeds threshold
            should_alert = False
            
            if bad_direction == "negative":
                # Alert only on negative changes (drops)
                should_alert = change_pct <= -threshold
            elif bad_direction == "positive":
                # Alert only on positive changes (increases)
                should_alert = change_pct >= threshold
            elif bad_direction == "either":
                # Alert on any significant change in either direction
                should_alert = abs(change_pct) >= threshold
            
            if should_alert:
                alert = self._create_simple_alert(
                    campaign, metric_name, recent_value, baseline_value, change_pct, threshold
                )
                potential_alerts.append((priority, alert))
        
        # Special case: Check for spending stop (recent_spend = 0 when baseline > 0)
        if (campaign.recent_spend == 0 and 
            campaign.baseline_spend is not None and 
            campaign.baseline_spend > 0):
            
            spend_threshold = getattr(custom_thresholds, f"{prefix}_spend", None)
            if spend_threshold is not None:  # Only alert if threshold is set
                # Build campaign info
                campaign_info = f"{campaign.platform.title()} - {campaign.campaign_name}"
                if campaign.adset_name:
                    campaign_info += f" ({campaign.adset_name})"
                
                spend_stop_message = f"🔴 {campaign_info}: Campaign has stopped spending (was ${campaign.baseline_spend:.2f})"
                potential_alerts.append((0, spend_stop_message))  # Highest priority
        
        # If we have potential alerts, return only the highest priority one (lowest number)
        if potential_alerts:
            # Sort by priority (ascending) and return only the first one
            potential_alerts.sort(key=lambda x: x[0])
            alerts.append(potential_alerts[0][1])  # Take the message from the highest priority tuple
        
        return alerts
    
    def _create_simple_alert(self, campaign: CampaignData, metric: str, current: Optional[float], 
                           previous: Optional[float], change_pct: float, threshold: float) -> str:
        """Create a simple alert message with colored circle"""
        
        # Determine severity and circle color
        if abs(change_pct) >= threshold * 2:  # Very severe
            circle = "🔴"  # Red circle
        elif abs(change_pct) >= threshold * 1.5:  # Moderate
            circle = "🟠"  # Orange circle  
        else:  # Minor
            circle = "🟡"  # Yellow circle
        
        # Format values based on metric type
        if metric in ['CPC', 'CPM', 'Cost per Lead', 'Cost per Estimate', 'Cost per Closing', 'Cost per Funded', 'CPA']:
            current_str = f"${current:.2f}" if current is not None else "N/A"
            previous_str = f"${previous:.2f}" if previous is not None else "N/A"
        elif metric in ['CTR', 'Estimate CVR', 'Closings CVR', 'Funded CVR', 'ROAS', 'Ad ROAS']:
            current_str = f"{current:.2f}%" if current is not None else "N/A"
            previous_str = f"{previous:.2f}%" if previous is not None else "N/A"
        elif metric in ['Revenue']:
            current_str = f"${current:,.0f}" if current is not None else "N/A"
            previous_str = f"${previous:,.0f}" if previous is not None else "N/A"
        else:
            current_str = f"{current:,.0f}" if current is not None else "N/A"
            previous_str = f"{previous:,.0f}" if previous is not None else "N/A"
        
        # Build campaign info
        campaign_info = f"{campaign.platform.title()} - {campaign.campaign_name}"
        if campaign.adset_name:
            campaign_info += f" ({campaign.adset_name})"
        
        # Create final message
        message = f"{circle} {campaign_info}: {metric} changed {change_pct:+.1f}% - {metric}: {current_str} (was {previous_str})"
        
        return message

# CLAUDE ANALYZER CLASS
class ClaudeAnalyzer:
    """Use Claude for advanced anomaly analysis and insights"""
    
    def __init__(self, client):
        self.client = client
    
    def analyze_patterns(self, alerts: List[AnomalyAlert], campaign_data: List[CampaignData], funnel_summary: Optional[Dict] = None) -> str:
        """Get Claude's analysis of anomaly patterns and funnel performance"""
        
        if not self.client:
            return "Claude analysis unavailable - API key not configured"
        
        # Prepare data summary for Claude
        alert_summary = {
            'total_alerts': len(alerts),
            'by_platform': {},
            'by_severity': {},
            'by_metric': {}
        }
        
        for alert in alerts:
            alert_summary['by_platform'][alert.platform] = alert_summary['by_platform'].get(alert.platform, 0) + 1
            alert_summary['by_severity'][alert.severity] = alert_summary['by_severity'].get(alert.severity, 0) + 1
            alert_summary['by_metric'][alert.metric] = alert_summary['by_metric'].get(alert.metric, 0) + 1
        
        # Build prompt
        prompt_parts = [
            "Analyze this marketing performance data and provide a brief executive summary.",
            f"\nCampaign Anomaly Summary:",
            f"- Total alerts: {alert_summary['total_alerts']}",
            f"- By platform: {alert_summary['by_platform']}",
            f"- By severity: {alert_summary['by_severity']}",
            f"- By metric: {alert_summary['by_metric']}"
        ]
        
        if alerts:
            top_alerts = sorted(alerts, key=lambda x: (x.severity == 'red', abs(x.change_pct or 0)), reverse=True)[:3]
            prompt_parts.append(f"\nTop 3 most critical alerts:")
            for alert in top_alerts:
                campaign_info = f"{alert.platform} - {alert.campaign_name}"
                if alert.adset_name:
                    campaign_info += f" ({alert.adset_name})"
                prompt_parts.append(f"- {campaign_info}: {alert.metric} {alert.change_pct:+.1f}% ({alert.severity})")
        
        if funnel_summary:
            prompt_parts.append(f"\nFunnel Performance:")
            prompt_parts.append(f"- Overall status: {funnel_summary.get('colorCode', 'unknown')}")
            if funnel_summary.get('key_metrics'):
                prompt_parts.append(f"- Key changes: {funnel_summary['key_metrics']}")
        
        prompt_parts.append("""
        Provide a 2-3 sentence executive summary focusing on:
        1. Overall health assessment
        2. Most critical issues to address immediately
        3. Any patterns or recommendations
        
        Keep it concise and actionable for a marketing team.
        """)
        
        prompt = "\n".join(prompt_parts)
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return f"Unable to generate analysis: {str(e)}"

# SLACK MESSAGE BUILDER
class SlackMessageBuilder:
    """Build Slack-ready messages for n8n"""
    
    @staticmethod
    def build_comprehensive_message(alerts: List[AnomalyAlert], funnel_summary: Optional[Dict], claude_insights: Optional[str], total_campaigns: int) -> Dict[str, Any]:
        """Build a comprehensive Slack message"""
        
        # Determine overall status
        if alerts:
            red_count = len([a for a in alerts if a.severity == SeverityLevel.RED])
            orange_count = len([a for a in alerts if a.severity == SeverityLevel.ORANGE])
            yellow_count = len([a for a in alerts if a.severity == SeverityLevel.YELLOW])
            
            if red_count > 0:
                emoji = "🚨"
                color = "#F44336"
                status = "Critical Issues Detected"
            elif orange_count > 0:
                emoji = "⚠️"
                color = "#FF9800"
                status = "Moderate Issues Detected"
            else:
                emoji = "📊"
                color = "#FFEB3B"
                status = "Minor Issues Detected"
            
            summary_text = f"{red_count} critical, {orange_count} moderate, {yellow_count} minor"
        else:
            emoji = "✅"
            color = "#4CAF50"
            status = "All Systems Healthy"
            summary_text = "No anomalies detected"
        
        # Build alert details
        alert_fields = []
        if alerts:
            # Group by severity
            red_alerts = [a for a in alerts if a.severity == SeverityLevel.RED]
            orange_alerts = [a for a in alerts if a.severity == SeverityLevel.ORANGE]
            
            # Show critical alerts
            if red_alerts:
                critical_text = []
                for alert in red_alerts[:3]:  # Top 3
                    campaign_info = f"{alert.platform} - {alert.campaign_name}"
                    if alert.adset_name:
                        campaign_info += f" ({alert.adset_name})"
                    critical_text.append(f"• {campaign_info}: {alert.message}")
                
                alert_fields.append({
                    "title": f"🚨 Critical Issues ({len(red_alerts)})",
                    "value": "\n".join(critical_text),
                    "short": False
                })
            
            # Show moderate alerts if not too many
            if orange_alerts and len(orange_alerts) <= 3:
                moderate_text = []
                for alert in orange_alerts:
                    campaign_info = f"{alert.platform} - {alert.campaign_name}"
                    if alert.adset_name:
                        campaign_info += f" ({alert.adset_name})"
                    moderate_text.append(f"• {campaign_info}: {alert.message}")
                
                alert_fields.append({
                    "title": f"⚠️ Moderate Issues ({len(orange_alerts)})",
                    "value": "\n".join(moderate_text),
                    "short": False
                })
        
        # Main message structure
        message = {
            "text": f"{emoji} Marketing Performance Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "attachments": [
                {
                    "color": color,
                    "fields": [
                        {
                            "title": "Campaign Analysis",
                            "value": f"Analyzed {total_campaigns} campaigns\n{summary_text}",
                            "short": True
                        }
                    ] + alert_fields
                }
            ]
        }
        
        # Add funnel summary if available
        if funnel_summary and funnel_summary.get('colorCode'):
            funnel_color = {
                'green': '#4CAF50',
                'yellow': '#FFEB3B', 
                'red': '#F44336'
            }.get(funnel_summary['colorCode'], '#36C5F0')
            
            funnel_fields = []
            if funnel_summary.get('key_metrics'):
                funnel_fields.append({
                    "title": "📈 Funnel Performance",
                    "value": funnel_summary['key_metrics'],
                    "short": False
                })
            
            if funnel_fields:
                message["attachments"].append({
                    "color": funnel_color,
                    "fields": funnel_fields
                })
        
        # Add Claude insights
        if claude_insights:
            message["attachments"].append({
                "color": "#36C5F0",  # Slack blue
                "fields": [
                    {
                        "title": "🤖 AI Analysis",
                        "value": claude_insights,
                        "short": False
                    }
                ]
            })
        
        return message

# Initialize services
detector = AnomalyDetector()
claude_analyzer = ClaudeAnalyzer(claude_client)

# Router
marketing_router = APIRouter(prefix="/marketing", tags=["marketing"])

@marketing_router.get("/")
async def marketing_root():
    return {
        "message": "Marketing Analytics API - Comprehensive Analysis",
        "version": "16.0.0",
        "status": "running",
        "features": ["funnel_analysis", "anomaly_detection", "custom_thresholds", "claude_insights", "slack_integration"]
    }

# UTILITY FUNCTIONS
def get_spend_data():
    """Get total spend from BigQuery"""
    if not bigquery_client:
        return []
    
    try:
        query = """
        WITH meta_daily AS (
          SELECT 
            m.date,
            SUM(SAFE_CAST(m.spend AS FLOAT64)) as meta_spend
          FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data_mapping` mm
          JOIN `gtm-p3gj3zzk-nthlo.last_14_days_analysis.meta_data` m 
            ON mm.adset_name_mapped = m.adset_name
          GROUP BY m.date
        ),
        google_daily AS (
          SELECT 
            date,
            SUM(spend_usd) as google_spend
          FROM `gtm-p3gj3zzk-nthlo.last_14_days_analysis.google_data`
          GROUP BY date
        )
        
        SELECT 
          COALESCE(m.date, g.date) as date,
          COALESCE(m.meta_spend, 0) + COALESCE(g.google_spend, 0) as total_spend
        FROM meta_daily m
        FULL OUTER JOIN google_daily g ON m.date = g.date
        ORDER BY date DESC
        """
        
        df = bigquery_client.query(query).to_dataframe()
        return [{"date": str(row['date']), "total_spend": float(row['total_spend'])} for _, row in df.iterrows()]
    except:
        return []

def calculate_changes(funnel_data: List[FunnelDataPoint]):
    """Calculate both week-over-week and day-over-day changes"""
    
    if len(funnel_data) < 8:
        return {"error": "Need at least 8 days of data for day-over-day comparison"}
    
    # Sort by date (most recent first)
    sorted_data = sorted(funnel_data, key=lambda x: x.date, reverse=True)
    
    # Week-over-week: Last 7 days vs previous 7 days
    if len(sorted_data) >= 14:
        last_7 = sorted_data[:7]
        prev_7 = sorted_data[7:14]
        week_over_week = True
    else:
        week_over_week = False
    
    # Day-over-day: Most recent day vs same day last week (-7 days)
    most_recent_day = sorted_data[0]
    same_day_last_week = sorted_data[7] if len(sorted_data) > 7 else None
    
    def sum_metric(days, metric):
        return sum(getattr(day, metric) for day in days)
    
    def get_metric(day, metric):
        return getattr(day, metric)
    
    # Calculate week-over-week totals
    wow_data = {}
    if week_over_week:
        last_totals = {
            'leads': sum_metric(last_7, 'leads'),
            'start_flows': sum_metric(last_7, 'start_flows'), 
            'estimates': sum_metric(last_7, 'estimates'),
            'closings': sum_metric(last_7, 'closings'),
            'funded': sum_metric(last_7, 'funded'),
            'rpts': sum_metric(last_7, 'rpts')
        }
        
        prev_totals = {
            'leads': sum_metric(prev_7, 'leads'),
            'start_flows': sum_metric(prev_7, 'start_flows'),
            'estimates': sum_metric(prev_7, 'estimates'), 
            'closings': sum_metric(prev_7, 'closings'),
            'funded': sum_metric(prev_7, 'funded'),
            'rpts': sum_metric(prev_7, 'rpts')
        }
        
        wow_data = {
            'last_totals': last_totals,
            'prev_totals': prev_totals
        }
    
    # Calculate day-over-day totals
    dod_data = {}
    if same_day_last_week:
        recent_day_totals = {
            'leads': get_metric(most_recent_day, 'leads'),
            'start_flows': get_metric(most_recent_day, 'start_flows'),
            'estimates': get_metric(most_recent_day, 'estimates'),
            'closings': get_metric(most_recent_day, 'closings'),
            'funded': get_metric(most_recent_day, 'funded'),
            'rpts': get_metric(most_recent_day, 'rpts')
        }
        
        same_day_totals = {
            'leads': get_metric(same_day_last_week, 'leads'),
            'start_flows': get_metric(same_day_last_week, 'start_flows'),
            'estimates': get_metric(same_day_last_week, 'estimates'),
            'closings': get_metric(same_day_last_week, 'closings'),
            'funded': get_metric(same_day_last_week, 'funded'),
            'rpts': get_metric(same_day_last_week, 'rpts')
        }
        
        dod_data = {
            'recent_day_totals': recent_day_totals,
            'same_day_totals': same_day_totals,
            'recent_date': most_recent_day.date,
            'comparison_date': same_day_last_week.date
        }
    
    # Get spend data
    spend_data = get_spend_data()
    spend_dict = {item['date']: item['total_spend'] for item in spend_data}
    
    # Add spend to week-over-week
    if week_over_week:
        last_7_dates = [day.date for day in last_7]
        prev_7_dates = [day.date for day in prev_7]
        
        last_spend = sum(spend_dict.get(date, 0) for date in last_7_dates)
        prev_spend = sum(spend_dict.get(date, 0) for date in prev_7_dates)
        
        wow_data.update({
            'last_spend': last_spend,
            'prev_spend': prev_spend
        })
    
    # Add spend to day-over-day
    if same_day_last_week:
        recent_spend = spend_dict.get(most_recent_day.date, 0)
        comparison_spend = spend_dict.get(same_day_last_week.date, 0)
        
        dod_data.update({
            'recent_spend': recent_spend,
            'comparison_spend': comparison_spend
        })
    
    return {
        'week_over_week': wow_data if week_over_week else {},
        'day_over_day': dod_data if same_day_last_week else {},
        'has_week_data': week_over_week,
        'has_day_data': bool(same_day_last_week)
    }

def format_metric(current, previous, is_currency=False):
    """Format metric with change percentage"""
    if previous == 0:
        change_pct = 100.0 if current > 0 else 0.0
    else:
        change_pct = ((current - previous) / previous) * 100
    
    # Format current value
    if is_currency:
        if current >= 1000:
            current_str = f"${current/1000:,.0f}k"
        else:
            current_str = f"${current:,.0f}"
    else:
        current_str = f"{current:,.0f}"
    
    # Format change
    sign = "+" if change_pct >= 0 else ""
    return f"{current_str} ({sign}{change_pct:.1f}%)"

def determine_color(changes):
    """Determine color based on performance"""
    
    # Use week-over-week data if available, otherwise day-over-day
    if changes.get('has_week_data'):
        last = changes['week_over_week']['last_totals']
        prev = changes['week_over_week']['prev_totals']
    elif changes.get('has_day_data'):
        last = changes['day_over_day']['recent_day_totals']
        prev = changes['day_over_day']['same_day_totals']
    else:
        return "yellow"  # No data for comparison
    
    def pct_change(curr, prev):
        return ((curr - prev) / prev) * 100 if prev > 0 else 0
    
    funded_change = pct_change(last['funded'], prev['funded'])
    revenue_change = pct_change(last['rpts'], prev['rpts'])
    
    if funded_change > 10 and revenue_change > 15:
        return "green"
    elif funded_change < -10 or revenue_change < -15:
        return "red"
    else:
        return "yellow"

# MAIN ENDPOINTS

@marketing_router.post("/funnel-analysis", response_model=FunnelAnalysisResponse)
async def analyze_funnel(request: FunnelAnalysisRequest):
    """Simple funnel analysis - original endpoint"""
    
    try:
        # Calculate changes
        changes = calculate_changes(request.funnel_data)
        
        if 'error' in changes:
            return FunnelAnalysisResponse(
                status="error",
                error=changes['error']
            )
        
        # Format week-over-week metrics (if available)
        formatted_metrics = {}
        if changes.get('has_week_data'):
            wow = changes['week_over_week']
            last = wow['last_totals']
            prev = wow['prev_totals']
            
            formatted_metrics = {
                "spendMetrics": {
                    "channel": "adSpend",
                    "period_type": "weekOverWeekPulse",
                    "totalSpend": format_metric(wow['last_spend'], wow['prev_spend'], True)
                },
                "funnelMetrics": {
                    "channel": "funnelPerformance", 
                    "period_type": "weekOverWeekPulse",
                    "totalLeads": format_metric(last['leads'], prev['leads']),
                    "startFlows": format_metric(last['start_flows'], prev['start_flows']),
                    "estimates": format_metric(last['estimates'], prev['estimates']),
                    "closings": format_metric(last['closings'], prev['closings']),
                    "funded": format_metric(last['funded'], prev['funded']),
                    "revenue": format_metric(last['rpts'], prev['rpts'], True)
                }
            }
        
        # Format day-over-day metrics (if available)
        day_over_day_metrics = {}
        if changes.get('has_day_data'):
            dod = changes['day_over_day']
            recent = dod['recent_day_totals']
            comparison = dod['same_day_totals']
            
            day_over_day_metrics = {
                "spendMetrics": {
                    "channel": "adSpend",
                    "period_type": "dayOverDayPulse",
                    "totalSpend": format_metric(dod['recent_spend'], dod['comparison_spend'], True)
                },
                "funnelMetrics": {
                    "channel": "funnelPerformance",
                    "period_type": "dayOverDayPulse", 
                    "totalLeads": format_metric(recent['leads'], comparison['leads']),
                    "startFlows": format_metric(recent['start_flows'], comparison['start_flows']),
                    "estimates": format_metric(recent['estimates'], comparison['estimates']),
                    "closings": format_metric(recent['closings'], comparison['closings']),
                    "funded": format_metric(recent['funded'], comparison['funded']),
                    "revenue": format_metric(recent['rpts'], comparison['rpts'], True)
                },
                "comparison_info": {
                    "recent_date": dod['recent_date'],
                    "comparison_date": dod['comparison_date']
                }
            }
        
        # Determine color
        color_code = determine_color(changes)
        
        return FunnelAnalysisResponse(
            status="success",
            colorCode=color_code,
            formatted_metrics=formatted_metrics if formatted_metrics else None,
            day_over_day_metrics=day_over_day_metrics if day_over_day_metrics else None
        )
        
    except Exception as e:
        return FunnelAnalysisResponse(
            status="error",
            error=str(e)
        )

@marketing_router.post("/campaign-alerts", response_model=ComprehensiveAnalysisResponse)
async def campaign_alerts_analysis(request: ComprehensiveAnalysisRequest):
    """
    Campaign alerts endpoint that does everything:
    1. Funnel analysis (if funnel_data provided)
    2. Anomaly detection (if dod_campaign_data provided) 
    3. Claude insights
    4. Slack-ready message for n8n
    """
    
    try:
        analysis_timestamp = datetime.now().isoformat()
        
        # 1. FUNNEL ANALYSIS (keeping original structure but not using it)
        funnel_analysis = None
        funnel_summary = None
        
        if request.funnel_data:
            try:
                changes = calculate_changes(request.funnel_data)
                
                if 'error' not in changes:
                    color_code = determine_color(changes)
                    funnel_analysis = {"colorCode": color_code}
                    funnel_summary = {"colorCode": color_code, "key_metrics": f"Funnel status: {color_code}"}
                
            except Exception as e:
                funnel_analysis = {"error": str(e)}
        
        # 2. ANOMALY DETECTION
        all_alerts = []
        total_campaigns = 0
        
        if request.dod_campaign_data:
            total_campaigns = len(request.dod_campaign_data)
            
            # Process each campaign and get alerts (already deduplicated by priority)
            for campaign in request.dod_campaign_data:
                campaign_alerts = detector.detect_anomalies(campaign, request.custom_thresholds)
                all_alerts.extend(campaign_alerts)
        
        return ComprehensiveAnalysisResponse(
            anomaly_alerts=all_alerts
        )
        
    except Exception as e:
        return ComprehensiveAnalysisResponse(
            anomaly_alerts=[f"🔴 Error: {str(e)}"]
        )

@marketing_router.get("/health")
async def health_check():
    """Health check endpoint with service status"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "services": {
            "bigquery": bigquery_client is not None,
            "claude": claude_client is not None,
            "anthropic_key_configured": ANTHROPIC_API_KEY is not None
        }
    }
