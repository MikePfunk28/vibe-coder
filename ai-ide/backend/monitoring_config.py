"""
Configuration for the monitoring and alerting system
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class MonitoringConfig:
    """Configuration class for monitoring system"""
    
    # General settings
    retention_hours: int = 48
    health_check_interval: int = 30
    metrics_collection_interval: int = 60
    
    # Database settings
    database_url: str = "postgresql://localhost:5432/ai_ide_monitoring"
    redis_url: str = "redis://localhost:6379/0"
    
    # Alert settings
    alert_evaluation_interval: int = 60
    max_alerts_per_component: int = 5
    alert_cooldown_minutes: int = 15
    
    # Notification settings
    email_enabled: bool = False
    slack_enabled: bool = False
    webhook_enabled: bool = False
    
    # Email configuration
    smtp_server: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    email_recipients: List[str] = None
    
    # Slack configuration
    slack_webhook_url: str = ""
    slack_channel: str = "#alerts"
    
    # Webhook configuration
    webhook_url: str = ""
    webhook_timeout: int = 30
    
    # Component-specific settings
    component_timeouts: Dict[str, int] = None
    component_retry_counts: Dict[str, int] = None
    
    # Performance thresholds
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    disk_threshold: float = 90.0
    response_time_threshold: float = 2.0
    
    # User experience settings
    satisfaction_threshold: float = 3.5
    success_rate_threshold: float = 0.9
    
    def __post_init__(self):
        """Initialize default values after creation"""
        if self.email_recipients is None:
            self.email_recipients = []
        
        if self.component_timeouts is None:
            self.component_timeouts = {
                'agents': 30,
                'reasoning_engine': 60,
                'web_search': 120,
                'semantic_search': 60,
                'darwin_godel': 300,
                'database': 30,
                'mcp_integration': 60,
                'pocketflow': 45
            }
        
        if self.component_retry_counts is None:
            self.component_retry_counts = {
                'agents': 3,
                'reasoning_engine': 2,
                'web_search': 3,
                'semantic_search': 2,
                'darwin_godel': 1,
                'database': 3,
                'mcp_integration': 2,
                'pocketflow': 3
            }


def load_config_from_env() -> MonitoringConfig:
    """Load configuration from environment variables"""
    
    config = MonitoringConfig()
    
    # General settings
    config.retention_hours = int(os.getenv('MONITORING_RETENTION_HOURS', '48'))
    config.health_check_interval = int(os.getenv('MONITORING_HEALTH_CHECK_INTERVAL', '30'))
    config.metrics_collection_interval = int(os.getenv('MONITORING_METRICS_INTERVAL', '60'))
    
    # Database settings
    config.database_url = os.getenv('MONITORING_DATABASE_URL', config.database_url)
    config.redis_url = os.getenv('MONITORING_REDIS_URL', config.redis_url)
    
    # Alert settings
    config.alert_evaluation_interval = int(os.getenv('MONITORING_ALERT_INTERVAL', '60'))
    config.max_alerts_per_component = int(os.getenv('MONITORING_MAX_ALERTS', '5'))
    config.alert_cooldown_minutes = int(os.getenv('MONITORING_ALERT_COOLDOWN', '15'))
    
    # Email configuration
    config.email_enabled = os.getenv('MONITORING_EMAIL_ENABLED', 'false').lower() == 'true'
    config.smtp_server = os.getenv('MONITORING_SMTP_SERVER', '')
    config.smtp_port = int(os.getenv('MONITORING_SMTP_PORT', '587'))
    config.smtp_username = os.getenv('MONITORING_SMTP_USERNAME', '')
    config.smtp_password = os.getenv('MONITORING_SMTP_PASSWORD', '')
    
    email_recipients = os.getenv('MONITORING_EMAIL_RECIPIENTS', '')
    if email_recipients:
        config.email_recipients = [email.strip() for email in email_recipients.split(',')]
    
    # Slack configuration
    config.slack_enabled = os.getenv('MONITORING_SLACK_ENABLED', 'false').lower() == 'true'
    config.slack_webhook_url = os.getenv('MONITORING_SLACK_WEBHOOK', '')
    config.slack_channel = os.getenv('MONITORING_SLACK_CHANNEL', '#alerts')
    
    # Webhook configuration
    config.webhook_enabled = os.getenv('MONITORING_WEBHOOK_ENABLED', 'false').lower() == 'true'
    config.webhook_url = os.getenv('MONITORING_WEBHOOK_URL', '')
    config.webhook_timeout = int(os.getenv('MONITORING_WEBHOOK_TIMEOUT', '30'))
    
    # Performance thresholds
    config.cpu_threshold = float(os.getenv('MONITORING_CPU_THRESHOLD', '80.0'))
    config.memory_threshold = float(os.getenv('MONITORING_MEMORY_THRESHOLD', '85.0'))
    config.disk_threshold = float(os.getenv('MONITORING_DISK_THRESHOLD', '90.0'))
    config.response_time_threshold = float(os.getenv('MONITORING_RESPONSE_TIME_THRESHOLD', '2.0'))
    
    # User experience settings
    config.satisfaction_threshold = float(os.getenv('MONITORING_SATISFACTION_THRESHOLD', '3.5'))
    config.success_rate_threshold = float(os.getenv('MONITORING_SUCCESS_RATE_THRESHOLD', '0.9'))
    
    return config


def load_config_from_file(config_path: str) -> MonitoringConfig:
    """Load configuration from JSON file"""
    import json
    
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    config = MonitoringConfig()
    
    # Update config with values from file
    for key, value in config_data.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def get_default_alert_rules(config: MonitoringConfig) -> List[Dict[str, Any]]:
    """Get default alert rules based on configuration"""
    
    return [
        # System resource alerts
        {
            'name': 'high_cpu_usage',
            'condition': {
                'type': 'metric_threshold',
                'component': 'system',
                'metric': 'cpu_percent',
                'threshold': config.cpu_threshold,
                'operator': '>'
            },
            'severity': 'high',
            'message': f'System CPU usage is above {config.cpu_threshold}%',
            'component': 'system',
            'cooldown_minutes': config.alert_cooldown_minutes
        },
        {
            'name': 'high_memory_usage',
            'condition': {
                'type': 'metric_threshold',
                'component': 'system',
                'metric': 'memory_percent',
                'threshold': config.memory_threshold,
                'operator': '>'
            },
            'severity': 'high',
            'message': f'System memory usage is above {config.memory_threshold}%',
            'component': 'system',
            'cooldown_minutes': config.alert_cooldown_minutes
        },
        {
            'name': 'high_disk_usage',
            'condition': {
                'type': 'metric_threshold',
                'component': 'system',
                'metric': 'disk_percent',
                'threshold': config.disk_threshold,
                'operator': '>'
            },
            'severity': 'high',
            'message': f'System disk usage is above {config.disk_threshold}%',
            'component': 'system',
            'cooldown_minutes': config.alert_cooldown_minutes
        },
        
        # Component health alerts
        {
            'name': 'database_unhealthy',
            'condition': {
                'type': 'health_status',
                'component': 'database',
                'status': 'unhealthy'
            },
            'severity': 'critical',
            'message': 'Database component is unhealthy',
            'component': 'database',
            'cooldown_minutes': 5
        },
        {
            'name': 'agents_unhealthy',
            'condition': {
                'type': 'health_status',
                'component': 'agents',
                'status': 'unhealthy'
            },
            'severity': 'critical',
            'message': 'Multi-agent system is unhealthy',
            'component': 'agents',
            'cooldown_minutes': 5
        },
        {
            'name': 'reasoning_engine_unhealthy',
            'condition': {
                'type': 'health_status',
                'component': 'reasoning_engine',
                'status': 'unhealthy'
            },
            'severity': 'high',
            'message': 'Reasoning engine is unhealthy',
            'component': 'reasoning_engine',
            'cooldown_minutes': 10
        },
        
        # Performance alerts
        {
            'name': 'slow_agent_response',
            'condition': {
                'type': 'response_time',
                'component': 'agents',
                'threshold': config.response_time_threshold
            },
            'severity': 'medium',
            'message': f'Agent response time is above {config.response_time_threshold} seconds',
            'component': 'agents',
            'cooldown_minutes': config.alert_cooldown_minutes
        },
        {
            'name': 'low_reasoning_accuracy',
            'condition': {
                'type': 'metric_threshold',
                'component': 'reasoning_engine',
                'metric': 'accuracy_score',
                'threshold': 0.7,
                'operator': '<'
            },
            'severity': 'medium',
            'message': 'Reasoning engine accuracy is below 70%',
            'component': 'reasoning_engine',
            'cooldown_minutes': config.alert_cooldown_minutes
        },
        
        # User experience alerts
        {
            'name': 'low_user_satisfaction',
            'condition': {
                'type': 'metric_threshold',
                'component': 'user_experience',
                'metric': 'satisfaction_score',
                'threshold': config.satisfaction_threshold,
                'operator': '<'
            },
            'severity': 'medium',
            'message': f'User satisfaction is below {config.satisfaction_threshold}',
            'component': 'user_experience',
            'cooldown_minutes': config.alert_cooldown_minutes * 2
        },
        {
            'name': 'low_success_rate',
            'condition': {
                'type': 'metric_threshold',
                'component': 'user_experience',
                'metric': 'success_rate',
                'threshold': config.success_rate_threshold,
                'operator': '<'
            },
            'severity': 'medium',
            'message': f'User interaction success rate is below {config.success_rate_threshold * 100}%',
            'component': 'user_experience',
            'cooldown_minutes': config.alert_cooldown_minutes * 2
        },
        
        # Web search specific alerts
        {
            'name': 'web_search_degraded',
            'condition': {
                'type': 'health_status',
                'component': 'web_search',
                'status': 'degraded'
            },
            'severity': 'low',
            'message': 'Web search system is degraded',
            'component': 'web_search',
            'cooldown_minutes': config.alert_cooldown_minutes
        },
        
        # Darwin-Gödel system alerts
        {
            'name': 'self_improvement_failure',
            'condition': {
                'type': 'metric_threshold',
                'component': 'darwin_godel',
                'metric': 'improvement_success_rate',
                'threshold': 0.5,
                'operator': '<'
            },
            'severity': 'medium',
            'message': 'Darwin-Gödel self-improvement success rate is low',
            'component': 'darwin_godel',
            'cooldown_minutes': config.alert_cooldown_minutes * 3
        }
    ]


def get_default_health_check_configs(config: MonitoringConfig) -> Dict[str, Dict[str, Any]]:
    """Get default health check configurations"""
    
    return {
        'agents': {
            'interval': config.component_timeouts.get('agents', 30),
            'timeout': 10,
            'retry_count': config.component_retry_counts.get('agents', 3)
        },
        'reasoning_engine': {
            'interval': config.component_timeouts.get('reasoning_engine', 60),
            'timeout': 15,
            'retry_count': config.component_retry_counts.get('reasoning_engine', 2)
        },
        'web_search': {
            'interval': config.component_timeouts.get('web_search', 120),
            'timeout': 30,
            'retry_count': config.component_retry_counts.get('web_search', 3)
        },
        'semantic_search': {
            'interval': config.component_timeouts.get('semantic_search', 60),
            'timeout': 20,
            'retry_count': config.component_retry_counts.get('semantic_search', 2)
        },
        'darwin_godel': {
            'interval': config.component_timeouts.get('darwin_godel', 300),
            'timeout': 60,
            'retry_count': config.component_retry_counts.get('darwin_godel', 1)
        },
        'database': {
            'interval': config.component_timeouts.get('database', 30),
            'timeout': 5,
            'retry_count': config.component_retry_counts.get('database', 3)
        },
        'mcp_integration': {
            'interval': config.component_timeouts.get('mcp_integration', 60),
            'timeout': 15,
            'retry_count': config.component_retry_counts.get('mcp_integration', 2)
        },
        'pocketflow': {
            'interval': config.component_timeouts.get('pocketflow', 45),
            'timeout': 10,
            'retry_count': config.component_retry_counts.get('pocketflow', 3)
        }
    }


# Example configuration files
EXAMPLE_CONFIG_JSON = """
{
    "retention_hours": 72,
    "health_check_interval": 30,
    "metrics_collection_interval": 60,
    "database_url": "postgresql://localhost:5432/ai_ide_monitoring",
    "redis_url": "redis://localhost:6379/0",
    "email_enabled": true,
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "smtp_username": "alerts@yourcompany.com",
    "smtp_password": "your_app_password",
    "email_recipients": ["admin@yourcompany.com", "devops@yourcompany.com"],
    "slack_enabled": true,
    "slack_webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
    "slack_channel": "#ai-ide-alerts",
    "cpu_threshold": 75.0,
    "memory_threshold": 80.0,
    "disk_threshold": 85.0,
    "response_time_threshold": 1.5,
    "satisfaction_threshold": 4.0,
    "success_rate_threshold": 0.95
}
"""

EXAMPLE_ENV_FILE = """
# Monitoring Configuration
MONITORING_RETENTION_HOURS=72
MONITORING_HEALTH_CHECK_INTERVAL=30
MONITORING_METRICS_INTERVAL=60

# Database
MONITORING_DATABASE_URL=postgresql://localhost:5432/ai_ide_monitoring
MONITORING_REDIS_URL=redis://localhost:6379/0

# Email Notifications
MONITORING_EMAIL_ENABLED=true
MONITORING_SMTP_SERVER=smtp.gmail.com
MONITORING_SMTP_PORT=587
MONITORING_SMTP_USERNAME=alerts@yourcompany.com
MONITORING_SMTP_PASSWORD=your_app_password
MONITORING_EMAIL_RECIPIENTS=admin@yourcompany.com,devops@yourcompany.com

# Slack Notifications
MONITORING_SLACK_ENABLED=true
MONITORING_SLACK_WEBHOOK=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
MONITORING_SLACK_CHANNEL=#ai-ide-alerts

# Performance Thresholds
MONITORING_CPU_THRESHOLD=75.0
MONITORING_MEMORY_THRESHOLD=80.0
MONITORING_DISK_THRESHOLD=85.0
MONITORING_RESPONSE_TIME_THRESHOLD=1.5

# User Experience
MONITORING_SATISFACTION_THRESHOLD=4.0
MONITORING_SUCCESS_RATE_THRESHOLD=0.95
"""


if __name__ == "__main__":
    # Example usage
    print("Loading configuration from environment...")
    config = load_config_from_env()
    print(f"Retention hours: {config.retention_hours}")
    print(f"Email enabled: {config.email_enabled}")
    print(f"CPU threshold: {config.cpu_threshold}%")
    
    print("\nDefault alert rules:")
    alert_rules = get_default_alert_rules(config)
    for rule in alert_rules[:3]:  # Show first 3 rules
        print(f"- {rule['name']}: {rule['message']}")
    
    print(f"\nTotal alert rules: {len(alert_rules)}")