# AI IDE Monitoring and Alerting System

## Overview

The AI IDE Monitoring and Alerting System provides comprehensive monitoring, alerting, and diagnostic capabilities for all components of the AI-powered IDE. It includes health checks, performance monitoring, alerting for critical failures, diagnostic tools, and user experience tracking.

## Features

### 🏥 Health Check System
- **Component Health Monitoring**: Continuous health checks for all system components
- **Health History**: Track health status over time with configurable retention
- **Async Support**: Support for both synchronous and asynchronous health check functions
- **Automatic Recovery**: Built-in retry mechanisms and graceful degradation

### 📊 Performance Monitoring
- **Multi-dimensional Metrics**: Collect and store performance metrics with tags and metadata
- **Real-time Collection**: Continuous metric collection with configurable intervals
- **Statistical Analysis**: Automatic calculation of min, max, average, and trend analysis
- **Time-series Data**: Efficient storage and retrieval of time-series performance data

### 🚨 Intelligent Alerting
- **Rule-based Alerts**: Flexible alert rules based on health status, metrics, and thresholds
- **Multiple Severity Levels**: Low, Medium, High, and Critical alert severities
- **Notification Channels**: Email, Slack, and webhook notification support
- **Alert Deduplication**: Prevent duplicate alerts for the same issue
- **Alert Resolution**: Track alert lifecycle from creation to resolution

### 🔍 Diagnostic Tools
- **Component Diagnostics**: Collect logs, metrics, and traces for troubleshooting
- **System Information**: Automatic collection of system resource information
- **Trace Storage**: Store and retrieve execution traces for debugging
- **Custom Collectors**: Register custom diagnostic data collectors

### 👤 User Experience Monitoring
- **Interaction Tracking**: Monitor user interactions and success rates
- **Satisfaction Scoring**: Track user satisfaction scores and trends
- **Performance Impact**: Measure how system performance affects user experience
- **Experience Metrics**: Calculate comprehensive UX metrics and trends

### 🌐 REST API Endpoints
- **Health Endpoints**: Check individual and overall system health
- **Metrics Endpoints**: Retrieve performance metrics and summaries
- **Alert Endpoints**: View active alerts and alert history
- **Diagnostic Endpoints**: Access diagnostic information for components
- **UX Endpoints**: Monitor user experience metrics

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Web Server                       │
│                  (Health Check Endpoints)                   │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│              MonitoringAlertingSystem                       │
│                   (Main Coordinator)                        │
└─────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼────────┐    ┌────────▼────────┐    ┌────────▼────────┐
│ HealthCheck    │    │  Performance    │    │     Alert       │
│   Manager      │    │    Monitor      │    │    Manager      │
└────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
┌───────▼────────┐    ┌────────▼────────┐    ┌────────▼────────┐
│  Diagnostic    │    │ UserExperience  │    │ Notification    │
│   Manager      │    │    Monitor      │    │   Channels      │
└────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### 1. Installation

```bash
# Install required dependencies
pip install fastapi uvicorn psutil requests

# For email notifications
pip install secure-smtplib

# For advanced features
pip install redis postgresql-adapter
```

### 2. Basic Usage

```python
from monitoring_alerting_system import MonitoringAlertingSystem

# Create monitoring system
monitoring_system = MonitoringAlertingSystem()

# Start monitoring
monitoring_system.start_monitoring()

# Run web server
import uvicorn
uvicorn.run(monitoring_system.app, host="0.0.0.0", port=8000)
```

### 3. Using the Startup Script

```bash
# Start with default configuration
python start_monitoring.py

# Start with custom configuration file
python start_monitoring.py --config monitoring_config.json

# Start on custom host/port
python start_monitoring.py --host localhost --port 9000

# Enable debug logging
python start_monitoring.py --log-level DEBUG
```

## Configuration

### Environment Variables

```bash
# General Settings
MONITORING_RETENTION_HOURS=48
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
```

### Configuration File (JSON)

```json
{
    "retention_hours": 72,
    "health_check_interval": 30,
    "metrics_collection_interval": 60,
    "email_enabled": true,
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "smtp_username": "alerts@yourcompany.com",
    "smtp_password": "your_app_password",
    "email_recipients": ["admin@yourcompany.com"],
    "slack_enabled": true,
    "slack_webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
    "cpu_threshold": 75.0,
    "memory_threshold": 80.0,
    "response_time_threshold": 1.5
}
```

## API Endpoints

### Health Check Endpoints

#### Get Overall System Health
```http
GET /health
```

Response:
```json
{
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00Z",
    "components": {
        "agents": {
            "component": "agents",
            "status": "healthy",
            "response_time": 0.15,
            "message": "Agents are responsive"
        }
    },
    "unhealthy_components": []
}
```

#### Get Component Health
```http
GET /health/{component}
```

#### Get Component Health History
```http
GET /health/{component}/history?limit=10
```

### Metrics Endpoints

#### Get Component Metrics
```http
GET /metrics/{component}?metric=response_time&hours=1
```

#### Get Metrics Summary
```http
GET /metrics/{component}/summary?metric=cpu_usage&minutes=60
```

Response:
```json
{
    "count": 60,
    "min": 45.2,
    "max": 78.9,
    "avg": 62.1,
    "latest": 65.3
}
```

### Alert Endpoints

#### Get Active Alerts
```http
GET /alerts
```

#### Get Alert History
```http
GET /alerts/history?limit=50
```

#### Resolve Alert
```http
POST /alerts/{alert_id}/resolve
```

### Diagnostic Endpoints

#### Get Component Diagnostics
```http
GET /diagnostics/{component}
```

Response:
```json
{
    "component": "agents",
    "timestamp": "2024-01-15T10:30:00Z",
    "logs": ["Agent system initialized", "Code agent active"],
    "metrics": {"active_agents": 4, "total_requests": 150},
    "traces": [{"action": "code_generation", "duration": 1.2}],
    "system_info": {"cpu_percent": 45.2, "memory_percent": 62.1}
}
```

### User Experience Endpoints

#### Get UX Metrics
```http
GET /user-experience?hours=24
```

#### Record User Interaction
```http
POST /user-experience/interaction
Content-Type: application/json

{
    "type": "code_completion",
    "duration": 1.5,
    "success": true,
    "details": {"language": "python"}
}
```

#### Record Satisfaction Score
```http
POST /user-experience/satisfaction
Content-Type: application/json

{
    "score": 4.5,
    "context": "code_generation"
}
```

## Monitored Components

The system monitors the following AI IDE components:

- **agents**: Multi-agent system (CodeAgent, SearchAgent, ReasoningAgent, TestAgent)
- **reasoning_engine**: Chain-of-thought and deep reasoning capabilities
- **web_search**: Web search integration and internet-enabled reasoning
- **semantic_search**: Semantic similarity search and code embeddings
- **darwin_godel**: Self-improving model system
- **reinforcement_learning**: User preference learning system
- **langchain_orchestrator**: LangChain workflow management
- **database**: Database connections and data persistence
- **mcp_integration**: Model Context Protocol integrations
- **pocketflow**: PocketFlow workflow engine

## Default Alert Rules

### System Resource Alerts
- **High CPU Usage**: Triggers when CPU usage > 80%
- **High Memory Usage**: Triggers when memory usage > 85%
- **High Disk Usage**: Triggers when disk usage > 90%

### Component Health Alerts
- **Database Unhealthy**: Critical alert when database is down
- **Agents Unhealthy**: Critical alert when multi-agent system fails
- **Reasoning Engine Unhealthy**: High priority alert for reasoning failures

### Performance Alerts
- **Slow Agent Response**: Medium alert when response time > 2 seconds
- **Low Reasoning Accuracy**: Medium alert when accuracy < 70%

### User Experience Alerts
- **Low User Satisfaction**: Medium alert when satisfaction < 3.5
- **Low Success Rate**: Medium alert when success rate < 90%

## Custom Health Checks

### Registering Health Checks

```python
# Synchronous health check
def check_my_component():
    return {
        'status': ComponentStatus.HEALTHY,
        'message': 'Component is operational',
        'details': {'version': '1.0.0'}
    }

# Asynchronous health check
async def check_my_async_component():
    # Perform async operations
    await some_async_check()
    return {
        'status': ComponentStatus.HEALTHY,
        'message': 'Async component is operational'
    }

# Register health checks
monitoring_system.health_manager.register_health_check(
    'my_component', check_my_component, interval=60
)
monitoring_system.health_manager.register_health_check(
    'my_async_component', check_my_async_component, interval=120
)
```

## Custom Metrics

### Recording Metrics

```python
# Record a simple metric
monitoring_system.performance_monitor.record_metric(
    component='my_service',
    metric_name='request_count',
    value=150,
    unit='count',
    tags={'endpoint': '/api/v1/data'}
)

# Register a metric collector
def collect_custom_metrics():
    # Collect your metrics
    request_count = get_request_count()
    monitoring_system.performance_monitor.record_metric(
        'my_service', 'requests_per_minute', request_count, 'count'
    )

monitoring_system.performance_monitor.register_metric_collector(
    'custom_metrics', collect_custom_metrics, interval=60
)
```

## Custom Alert Rules

### Adding Alert Rules

```python
# Metric threshold alert
monitoring_system.alert_manager.add_alert_rule({
    'name': 'high_error_rate',
    'condition': {
        'type': 'metric_threshold',
        'component': 'my_service',
        'metric': 'error_rate',
        'threshold': 5.0,
        'operator': '>'
    },
    'severity': 'high',
    'message': 'Error rate is above 5%',
    'component': 'my_service'
})

# Health status alert
monitoring_system.alert_manager.add_alert_rule({
    'name': 'service_down',
    'condition': {
        'type': 'health_status',
        'component': 'my_service',
        'status': 'unhealthy'
    },
    'severity': 'critical',
    'message': 'My service is down',
    'component': 'my_service'
})
```

## Custom Notification Channels

### Email Notifications

```python
from monitoring_alerting_system import EmailNotificationChannel

email_channel = EmailNotificationChannel(
    smtp_server='smtp.gmail.com',
    smtp_port=587,
    username='alerts@yourcompany.com',
    password='your_app_password',
    recipients=['admin@yourcompany.com', 'devops@yourcompany.com']
)

monitoring_system.alert_manager.add_notification_channel(email_channel)
```

### Slack Notifications

```python
from monitoring_alerting_system import SlackNotificationChannel

slack_channel = SlackNotificationChannel(
    webhook_url='https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
)

monitoring_system.alert_manager.add_notification_channel(slack_channel)
```

### Custom Notification Channel

```python
def custom_notification_channel(alert):
    # Send notification to your custom system
    payload = {
        'alert_id': alert.id,
        'component': alert.component,
        'severity': alert.severity.value,
        'message': alert.message,
        'timestamp': alert.timestamp.isoformat()
    }
    
    # Send to your notification system
    send_to_custom_system(payload)

monitoring_system.alert_manager.add_notification_channel(custom_notification_channel)
```

## Diagnostic Data Collection

### Custom Diagnostic Collectors

```python
def collect_my_service_diagnostics():
    return {
        'logs': get_recent_logs(),
        'metrics': {
            'active_connections': get_connection_count(),
            'cache_hit_rate': get_cache_hit_rate()
        },
        'traces': get_recent_traces()
    }

monitoring_system.diagnostic_manager.register_diagnostic_collector(
    'my_service', collect_my_service_diagnostics
)
```

## Testing

### Running Tests

```bash
# Run all tests
pytest test_monitoring_alerting_system.py -v

# Run specific test categories
pytest test_monitoring_alerting_system.py::TestHealthCheckManager -v
pytest test_monitoring_alerting_system.py::TestPerformanceMonitor -v
pytest test_monitoring_alerting_system.py::TestAlertManager -v

# Run with coverage
pytest test_monitoring_alerting_system.py --cov=monitoring_alerting_system
```

### Test Categories

- **Unit Tests**: Test individual components and functions
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete monitoring workflows
- **Performance Tests**: Test system performance under load

## Troubleshooting

### Common Issues

#### Health Checks Failing
```bash
# Check component logs
curl http://localhost:8000/diagnostics/agents

# Check system resources
curl http://localhost:8000/metrics/system/summary?metric=cpu_percent
```

#### Alerts Not Firing
```bash
# Check alert rules
curl http://localhost:8000/alerts/history

# Verify metric collection
curl http://localhost:8000/metrics/system?hours=1
```

#### Notifications Not Sending
- Verify SMTP settings for email notifications
- Check Slack webhook URL and permissions
- Review notification channel logs

### Debug Mode

```bash
# Start with debug logging
python start_monitoring.py --log-level DEBUG

# Check specific component health
curl http://localhost:8000/health/agents
```

## Performance Considerations

### Resource Usage
- **Memory**: ~100-200MB for typical workloads
- **CPU**: <5% average CPU usage
- **Storage**: Configurable retention (default 48 hours)

### Scaling
- Use Redis for distributed caching
- PostgreSQL for persistent storage
- Horizontal scaling with load balancers

### Optimization Tips
- Adjust health check intervals based on component criticality
- Use metric aggregation for high-frequency data
- Configure appropriate retention periods
- Monitor the monitoring system itself

## Security

### Best Practices
- Use environment variables for sensitive configuration
- Secure SMTP credentials and API keys
- Implement proper access controls for endpoints
- Use HTTPS in production environments
- Regularly rotate notification credentials

### Network Security
- Restrict access to monitoring endpoints
- Use VPN or private networks when possible
- Implement rate limiting for API endpoints
- Monitor for unusual access patterns

## Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd ai-ide/backend

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest test_monitoring_alerting_system.py -v

# Run linting
flake8 monitoring_alerting_system.py
black monitoring_alerting_system.py
```

### Adding New Features

1. **Health Checks**: Add new component health checks in `_setup_default_health_checks()`
2. **Metrics**: Add new metric collectors in `_setup_default_metrics()`
3. **Alerts**: Add new alert rules in `get_default_alert_rules()`
4. **Endpoints**: Add new API endpoints in `_setup_endpoints()`

## License

This monitoring system is part of the AI IDE project and follows the same licensing terms.

## Support

For issues and questions:
- Check the troubleshooting section above
- Review the test files for usage examples
- Check system logs and diagnostic endpoints
- Create an issue in the project repository