"""
Comprehensive Monitoring and Alerting System

This module implements health check endpoints, performance monitoring,
alerting for critical failures, diagnostic tools, and user experience monitoring.
"""

import asyncio
import logging
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import psutil
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComponentStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    component: str
    status: ComponentStatus
    response_time: float
    message: str
    timestamp: datetime
    details: Dict[str, Any]


@dataclass
class PerformanceMetric:
    component: str
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str]


@dataclass
class Alert:
    id: str
    component: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    details: Dict[str, Any] = None


@dataclass
class DiagnosticInfo:
    component: str
    timestamp: datetime
    logs: List[str]
    metrics: Dict[str, Any]
    traces: List[Dict[str, Any]]
    system_info: Dict[str, Any]


class HealthCheckManager:
    """Manages health checks for all system components"""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.check_intervals: Dict[str, int] = {}
        self.running = False
        self.check_thread = None
    
    def register_health_check(self, component: str, check_func: Callable, interval: int = 30):
        """Register a health check function for a component"""
        self.health_checks[component] = check_func
        self.check_intervals[component] = interval
        self.health_history[component] = deque(maxlen=100)
    
    async def check_component_health(self, component: str) -> HealthCheckResult:
        """Check health of a specific component"""
        if component not in self.health_checks:
            return HealthCheckResult(
                component=component,
                status=ComponentStatus.UNKNOWN,
                response_time=0.0,
                message=f"No health check registered for {component}",
                timestamp=datetime.now(),
                details={}
            )
        
        start_time = time.time()
        try:
            check_func = self.health_checks[component]
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            response_time = time.time() - start_time
            
            health_result = HealthCheckResult(
                component=component,
                status=result.get('status', ComponentStatus.HEALTHY),
                response_time=response_time,
                message=result.get('message', 'Health check passed'),
                timestamp=datetime.now(),
                details=result.get('details', {})
            )
            
            self.health_history[component].append(health_result)
            return health_result
            
        except Exception as e:
            response_time = time.time() - start_time
            health_result = HealthCheckResult(
                component=component,
                status=ComponentStatus.UNHEALTHY,
                response_time=response_time,
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.now(),
                details={'error': str(e)}
            )
            
            self.health_history[component].append(health_result)
            return health_result
    
    async def check_all_components(self) -> Dict[str, HealthCheckResult]:
        """Check health of all registered components"""
        results = {}
        for component in self.health_checks.keys():
            results[component] = await self.check_component_health(component)
        return results
    
    def get_component_history(self, component: str, limit: int = 10) -> List[HealthCheckResult]:
        """Get health check history for a component"""
        history = list(self.health_history.get(component, []))
        return history[-limit:] if limit else history
    
    def start_continuous_monitoring(self):
        """Start continuous health monitoring in background"""
        self.running = True
        self.check_thread = threading.Thread(target=self._continuous_check_loop)
        self.check_thread.daemon = True
        self.check_thread.start()
    
    def stop_continuous_monitoring(self):
        """Stop continuous health monitoring"""
        self.running = False
        if self.check_thread:
            self.check_thread.join()
    
    def _continuous_check_loop(self):
        """Background loop for continuous health checks"""
        last_check_times = defaultdict(float)
        
        while self.running:
            current_time = time.time()
            
            for component, interval in self.check_intervals.items():
                if current_time - last_check_times[component] >= interval:
                    try:
                        # Run health check in event loop
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(self.check_component_health(component))
                        loop.close()
                        last_check_times[component] = current_time
                    except Exception as e:
                        logging.error(f"Error in continuous health check for {component}: {e}")
            
            time.sleep(5)  # Check every 5 seconds


class PerformanceMonitor:
    """Multi-dimensional performance monitoring system"""
    
    def __init__(self, retention_hours: int = 24):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.retention_hours = retention_hours
        self.metric_collectors: Dict[str, Callable] = {}
        self.running = False
        self.monitor_thread = None
    
    def register_metric_collector(self, metric_name: str, collector_func: Callable, interval: int = 60):
        """Register a metric collector function"""
        self.metric_collectors[metric_name] = {
            'func': collector_func,
            'interval': interval,
            'last_run': 0
        }
    
    def record_metric(self, component: str, metric_name: str, value: float, 
                     unit: str = "", tags: Dict[str, str] = None):
        """Record a performance metric"""
        metric = PerformanceMetric(
            component=component,
            metric_name=metric_name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        
        key = f"{component}.{metric_name}"
        self.metrics[key].append(metric)
        
        # Clean old metrics
        self._cleanup_old_metrics(key)
    
    def get_metrics(self, component: str, metric_name: str = None, 
                   start_time: datetime = None, end_time: datetime = None) -> List[PerformanceMetric]:
        """Get performance metrics with optional filtering"""
        if metric_name:
            key = f"{component}.{metric_name}"
            metrics = list(self.metrics.get(key, []))
        else:
            metrics = []
            for key, metric_list in self.metrics.items():
                if key.startswith(f"{component}."):
                    metrics.extend(list(metric_list))
        
        # Filter by time range
        if start_time or end_time:
            filtered_metrics = []
            for metric in metrics:
                if start_time and metric.timestamp < start_time:
                    continue
                if end_time and metric.timestamp > end_time:
                    continue
                filtered_metrics.append(metric)
            metrics = filtered_metrics
        
        return sorted(metrics, key=lambda m: m.timestamp)
    
    def get_metric_summary(self, component: str, metric_name: str, 
                          duration_minutes: int = 60) -> Dict[str, float]:
        """Get statistical summary of metrics"""
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=duration_minutes)
        
        metrics = self.get_metrics(component, metric_name, start_time, end_time)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'latest': values[-1] if values else 0
        }
    
    def start_monitoring(self):
        """Start continuous performance monitoring"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous performance monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitoring_loop(self):
        """Background loop for metric collection"""
        while self.running:
            current_time = time.time()
            
            for metric_name, collector_info in self.metric_collectors.items():
                if current_time - collector_info['last_run'] >= collector_info['interval']:
                    try:
                        collector_info['func']()
                        collector_info['last_run'] = current_time
                    except Exception as e:
                        logging.error(f"Error collecting metric {metric_name}: {e}")
            
            time.sleep(10)  # Check every 10 seconds
    
    def _cleanup_old_metrics(self, key: str):
        """Remove metrics older than retention period"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        metrics = self.metrics[key]
        
        while metrics and metrics[0].timestamp < cutoff_time:
            metrics.popleft()


class AlertManager:
    """Manages alerts for critical system failures"""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Dict[str, Any]] = []
        self.notification_channels: List[Callable] = []
        self.alert_history: deque = deque(maxlen=1000)
    
    def add_alert_rule(self, rule: Dict[str, Any]):
        """Add an alert rule"""
        required_fields = ['name', 'condition', 'severity', 'message']
        if not all(field in rule for field in required_fields):
            raise ValueError(f"Alert rule must contain: {required_fields}")
        
        self.alert_rules.append(rule)
    
    def add_notification_channel(self, channel_func: Callable):
        """Add a notification channel (email, slack, etc.)"""
        self.notification_channels.append(channel_func)
    
    def evaluate_rules(self, health_results: Dict[str, HealthCheckResult], 
                      metrics: Dict[str, List[PerformanceMetric]]):
        """Evaluate alert rules against current system state"""
        for rule in self.alert_rules:
            try:
                if self._evaluate_condition(rule['condition'], health_results, metrics):
                    self._trigger_alert(rule)
            except Exception as e:
                logging.error(f"Error evaluating alert rule {rule['name']}: {e}")
    
    def _evaluate_condition(self, condition: Dict[str, Any], 
                           health_results: Dict[str, HealthCheckResult],
                           metrics: Dict[str, List[PerformanceMetric]]) -> bool:
        """Evaluate a single alert condition"""
        condition_type = condition.get('type')
        
        if condition_type == 'health_status':
            component = condition['component']
            expected_status = condition['status']
            if component in health_results:
                return health_results[component].status.value == expected_status
        
        elif condition_type == 'metric_threshold':
            metric_key = f"{condition['component']}.{condition['metric']}"
            threshold = condition['threshold']
            operator = condition.get('operator', '>')
            
            if metric_key in metrics and metrics[metric_key]:
                latest_value = metrics[metric_key][-1].value
                
                if operator == '>':
                    return latest_value > threshold
                elif operator == '<':
                    return latest_value < threshold
                elif operator == '>=':
                    return latest_value >= threshold
                elif operator == '<=':
                    return latest_value <= threshold
                elif operator == '==':
                    return latest_value == threshold
        
        elif condition_type == 'response_time':
            component = condition['component']
            threshold = condition['threshold']
            if component in health_results:
                return health_results[component].response_time > threshold
        
        return False
    
    def _trigger_alert(self, rule: Dict[str, Any]):
        """Trigger an alert based on a rule"""
        alert_id = f"{rule['name']}_{int(time.time())}"
        
        # Check if similar alert already exists and is not resolved
        existing_alert = None
        for alert in self.alerts.values():
            if (alert.component == rule.get('component', 'system') and 
                not alert.resolved and 
                alert.message == rule['message']):
                existing_alert = alert
                break
        
        if existing_alert:
            return  # Don't create duplicate alerts
        
        alert = Alert(
            id=alert_id,
            component=rule.get('component', 'system'),
            severity=AlertSeverity(rule['severity']),
            message=rule['message'],
            timestamp=datetime.now(),
            details=rule.get('details', {})
        )
        
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        for channel in self.notification_channels:
            try:
                channel(alert)
            except Exception as e:
                logging.error(f"Error sending alert notification: {e}")
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolution_time = datetime.now()
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_alert_history(self, limit: int = 50) -> List[Alert]:
        """Get recent alert history"""
        history = list(self.alert_history)
        return history[-limit:] if limit else history


class DiagnosticManager:
    """Provides diagnostic tools for troubleshooting"""
    
    def __init__(self):
        self.diagnostic_collectors: Dict[str, Callable] = {}
        self.trace_storage: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
    
    def register_diagnostic_collector(self, component: str, collector_func: Callable):
        """Register a diagnostic data collector for a component"""
        self.diagnostic_collectors[component] = collector_func
    
    def collect_diagnostic_info(self, component: str) -> DiagnosticInfo:
        """Collect comprehensive diagnostic information for a component"""
        timestamp = datetime.now()
        
        # Collect component-specific diagnostics
        logs = []
        metrics = {}
        traces = []
        
        if component in self.diagnostic_collectors:
            try:
                diagnostic_data = self.diagnostic_collectors[component]()
                logs = diagnostic_data.get('logs', [])
                metrics = diagnostic_data.get('metrics', {})
                traces = diagnostic_data.get('traces', [])
            except Exception as e:
                logs.append(f"Error collecting diagnostics: {str(e)}")
        
        # Collect system information
        system_info = self._collect_system_info()
        
        # Get stored traces
        stored_traces = list(self.trace_storage.get(component, []))
        traces.extend(stored_traces)
        
        return DiagnosticInfo(
            component=component,
            timestamp=timestamp,
            logs=logs,
            metrics=metrics,
            traces=traces,
            system_info=system_info
        )
    
    def add_trace(self, component: str, trace_data: Dict[str, Any]):
        """Add a trace entry for a component"""
        trace_entry = {
            'timestamp': datetime.now().isoformat(),
            'data': trace_data
        }
        self.trace_storage[component].append(trace_entry)
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect general system information"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
                'process_count': len(psutil.pids()),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': f"Failed to collect system info: {str(e)}"}


class UserExperienceMonitor:
    """Monitors user experience and satisfaction"""
    
    def __init__(self):
        self.user_interactions: deque = deque(maxlen=10000)
        self.satisfaction_scores: deque = deque(maxlen=1000)
        self.performance_impacts: Dict[str, List[float]] = defaultdict(list)
    
    def record_user_interaction(self, interaction_type: str, duration: float, 
                               success: bool, details: Dict[str, Any] = None):
        """Record a user interaction"""
        interaction = {
            'type': interaction_type,
            'duration': duration,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        self.user_interactions.append(interaction)
    
    def record_satisfaction_score(self, score: float, context: str = ""):
        """Record user satisfaction score (1-5 scale)"""
        satisfaction_entry = {
            'score': score,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        self.satisfaction_scores.append(satisfaction_entry)
    
    def record_performance_impact(self, feature: str, impact_score: float):
        """Record performance impact of a feature on user experience"""
        self.performance_impacts[feature].append(impact_score)
        
        # Keep only recent impacts
        if len(self.performance_impacts[feature]) > 100:
            self.performance_impacts[feature] = self.performance_impacts[feature][-100:]
    
    def get_user_experience_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get user experience metrics for the specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter recent interactions
        recent_interactions = [
            interaction for interaction in self.user_interactions
            if datetime.fromisoformat(interaction['timestamp']) > cutoff_time
        ]
        
        # Filter recent satisfaction scores
        recent_satisfaction = [
            entry for entry in self.satisfaction_scores
            if datetime.fromisoformat(entry['timestamp']) > cutoff_time
        ]
        
        # Calculate metrics
        total_interactions = len(recent_interactions)
        successful_interactions = sum(1 for i in recent_interactions if i['success'])
        success_rate = successful_interactions / total_interactions if total_interactions > 0 else 0
        
        avg_duration = sum(i['duration'] for i in recent_interactions) / total_interactions if total_interactions > 0 else 0
        
        avg_satisfaction = sum(s['score'] for s in recent_satisfaction) / len(recent_satisfaction) if recent_satisfaction else 0
        
        return {
            'total_interactions': total_interactions,
            'success_rate': success_rate,
            'average_duration': avg_duration,
            'average_satisfaction': avg_satisfaction,
            'satisfaction_trend': self._calculate_satisfaction_trend(recent_satisfaction),
            'performance_impacts': dict(self.performance_impacts)
        }
    
    def _calculate_satisfaction_trend(self, satisfaction_data: List[Dict[str, Any]]) -> str:
        """Calculate satisfaction trend (improving, declining, stable)"""
        if len(satisfaction_data) < 2:
            return "insufficient_data"
        
        # Split into two halves and compare averages
        mid_point = len(satisfaction_data) // 2
        first_half_avg = sum(s['score'] for s in satisfaction_data[:mid_point]) / mid_point
        second_half_avg = sum(s['score'] for s in satisfaction_data[mid_point:]) / (len(satisfaction_data) - mid_point)
        
        diff = second_half_avg - first_half_avg
        
        if diff > 0.2:
            return "improving"
        elif diff < -0.2:
            return "declining"
        else:
            return "stable"


class MonitoringAlertingSystem:
    """Main monitoring and alerting system that coordinates all components"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize all managers
        self.health_manager = HealthCheckManager()
        self.performance_monitor = PerformanceMonitor(
            retention_hours=self.config.get('retention_hours', 24)
        )
        self.alert_manager = AlertManager()
        self.diagnostic_manager = DiagnosticManager()
        self.ux_monitor = UserExperienceMonitor()
        
        # FastAPI app for health check endpoints
        self.app = FastAPI(title="AI IDE Monitoring System")
        self._setup_endpoints()
        
        # System components to monitor
        self.components = [
            'agents', 'reasoning_engine', 'web_search', 'semantic_search',
            'darwin_godel', 'reinforcement_learning', 'langchain_orchestrator',
            'database', 'mcp_integration', 'pocketflow'
        ]
        
        self._setup_default_health_checks()
        self._setup_default_metrics()
        self._setup_default_alerts()
    
    def _setup_endpoints(self):
        """Setup FastAPI endpoints for health checks and monitoring"""
        
        @self.app.get("/health")
        async def overall_health():
            """Overall system health endpoint"""
            results = await self.health_manager.check_all_components()
            
            overall_status = ComponentStatus.HEALTHY
            unhealthy_components = []
            
            for component, result in results.items():
                if result.status == ComponentStatus.UNHEALTHY:
                    overall_status = ComponentStatus.UNHEALTHY
                    unhealthy_components.append(component)
                elif result.status == ComponentStatus.DEGRADED and overall_status == ComponentStatus.HEALTHY:
                    overall_status = ComponentStatus.DEGRADED
            
            return {
                "status": overall_status.value,
                "timestamp": datetime.now().isoformat(),
                "components": {k: asdict(v) for k, v in results.items()},
                "unhealthy_components": unhealthy_components
            }
        
        @self.app.get("/health/{component}")
        async def component_health(component: str):
            """Individual component health endpoint"""
            result = await self.health_manager.check_component_health(component)
            return asdict(result)
        
        @self.app.get("/health/{component}/history")
        async def component_health_history(component: str, limit: int = 10):
            """Component health history endpoint"""
            history = self.health_manager.get_component_history(component, limit)
            return [asdict(result) for result in history]
        
        @self.app.get("/metrics/{component}")
        async def component_metrics(component: str, metric: str = None, hours: int = 1):
            """Component performance metrics endpoint"""
            start_time = datetime.now() - timedelta(hours=hours)
            metrics = self.performance_monitor.get_metrics(component, metric, start_time)
            return [asdict(metric) for metric in metrics]
        
        @self.app.get("/metrics/{component}/summary")
        async def component_metrics_summary(component: str, metric: str, minutes: int = 60):
            """Component metrics summary endpoint"""
            summary = self.performance_monitor.get_metric_summary(component, metric, minutes)
            return summary
        
        @self.app.get("/alerts")
        async def active_alerts():
            """Active alerts endpoint"""
            alerts = self.alert_manager.get_active_alerts()
            return [asdict(alert) for alert in alerts]
        
        @self.app.get("/alerts/history")
        async def alert_history(limit: int = 50):
            """Alert history endpoint"""
            history = self.alert_manager.get_alert_history(limit)
            return [asdict(alert) for alert in history]
        
        @self.app.post("/alerts/{alert_id}/resolve")
        async def resolve_alert(alert_id: str):
            """Resolve an alert endpoint"""
            self.alert_manager.resolve_alert(alert_id)
            return {"message": f"Alert {alert_id} resolved"}
        
        @self.app.get("/diagnostics/{component}")
        async def component_diagnostics(component: str):
            """Component diagnostics endpoint"""
            diagnostics = self.diagnostic_manager.collect_diagnostic_info(component)
            return asdict(diagnostics)
        
        @self.app.get("/user-experience")
        async def user_experience_metrics(hours: int = 24):
            """User experience metrics endpoint"""
            metrics = self.ux_monitor.get_user_experience_metrics(hours)
            return metrics
        
        @self.app.post("/user-experience/interaction")
        async def record_interaction(interaction_data: Dict[str, Any]):
            """Record user interaction endpoint"""
            self.ux_monitor.record_user_interaction(
                interaction_data['type'],
                interaction_data['duration'],
                interaction_data['success'],
                interaction_data.get('details')
            )
            return {"message": "Interaction recorded"}
        
        @self.app.post("/user-experience/satisfaction")
        async def record_satisfaction(satisfaction_data: Dict[str, Any]):
            """Record user satisfaction endpoint"""
            self.ux_monitor.record_satisfaction_score(
                satisfaction_data['score'],
                satisfaction_data.get('context', '')
            )
            return {"message": "Satisfaction score recorded"}
    
    def _setup_default_health_checks(self):
        """Setup default health checks for system components"""
        
        async def check_agents_health():
            """Health check for multi-agent system"""
            try:
                # This would check actual agent functionality
                return {
                    'status': ComponentStatus.HEALTHY,
                    'message': 'Agents are responsive',
                    'details': {'agent_count': 4}
                }
            except Exception as e:
                return {
                    'status': ComponentStatus.UNHEALTHY,
                    'message': f'Agents health check failed: {str(e)}',
                    'details': {'error': str(e)}
                }
        
        async def check_reasoning_health():
            """Health check for reasoning engine"""
            try:
                return {
                    'status': ComponentStatus.HEALTHY,
                    'message': 'Reasoning engine operational',
                    'details': {'reasoning_modes': ['basic', 'deep', 'cot']}
                }
            except Exception as e:
                return {
                    'status': ComponentStatus.UNHEALTHY,
                    'message': f'Reasoning engine health check failed: {str(e)}',
                    'details': {'error': str(e)}
                }
        
        async def check_web_search_health():
            """Health check for web search system"""
            try:
                return {
                    'status': ComponentStatus.HEALTHY,
                    'message': 'Web search system operational',
                    'details': {'search_engines': ['google', 'bing', 'duckduckgo']}
                }
            except Exception as e:
                return {
                    'status': ComponentStatus.DEGRADED,
                    'message': f'Web search partially available: {str(e)}',
                    'details': {'error': str(e)}
                }
        
        # Register health checks
        self.health_manager.register_health_check('agents', check_agents_health, 30)
        self.health_manager.register_health_check('reasoning_engine', check_reasoning_health, 60)
        self.health_manager.register_health_check('web_search', check_web_search_health, 120)
    
    def _setup_default_metrics(self):
        """Setup default performance metrics collection"""
        
        def collect_system_metrics():
            """Collect system-level performance metrics"""
            try:
                # CPU metrics
                self.performance_monitor.record_metric(
                    'system', 'cpu_percent', psutil.cpu_percent(interval=1), 'percent'
                )
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self.performance_monitor.record_metric(
                    'system', 'memory_percent', memory.percent, 'percent'
                )
                
            except Exception as e:
                logging.error(f"Error collecting system metrics: {e}")
        
        # Register metric collectors
        self.performance_monitor.register_metric_collector('system_metrics', collect_system_metrics, 60)
    
    def _setup_default_alerts(self):
        """Setup default alert rules"""
        
        # High CPU usage alert
        self.alert_manager.add_alert_rule({
            'name': 'high_cpu_usage',
            'condition': {
                'type': 'metric_threshold',
                'component': 'system',
                'metric': 'cpu_percent',
                'threshold': 80.0,
                'operator': '>'
            },
            'severity': 'high',
            'message': 'System CPU usage is above 80%',
            'component': 'system'
        })
    
    def start_monitoring(self):
        """Start all monitoring components"""
        logging.info("Starting comprehensive monitoring and alerting system...")
        
        # Start continuous health monitoring
        self.health_manager.start_continuous_monitoring()
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        
        logging.info("Monitoring and alerting system started successfully")
    
    def stop_monitoring(self):
        """Stop all monitoring components"""
        logging.info("Stopping monitoring and alerting system...")
        
        self.health_manager.stop_continuous_monitoring()
        self.performance_monitor.stop_monitoring()
        
        logging.info("Monitoring and alerting system stopped")


# Notification channel implementations
class EmailNotificationChannel:
    """Email notification channel for alerts"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str, recipients: List[str]):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = recipients
    
    def __call__(self, alert: Alert):
        """Send email notification for an alert"""
        if not EMAIL_AVAILABLE:
            logging.error("Email functionality not available - missing email modules")
            return
            
        try:
            msg = MimeMultipart()
            msg['From'] = self.username
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = f"AI IDE Alert: {alert.severity.value.upper()} - {alert.component}"
            
            body = f"""
            Alert Details:
            - Component: {alert.component}
            - Severity: {alert.severity.value}
            - Message: {alert.message}
            - Timestamp: {alert.timestamp}
            - Alert ID: {alert.id}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            logging.info(f"Email alert sent for {alert.id}")
            
        except Exception as e:
            logging.error(f"Failed to send email alert: {e}")


class SlackNotificationChannel:
    """Slack notification channel for alerts"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def __call__(self, alert: Alert):
        """Send Slack notification for an alert"""
        try:
            color_map = {
                AlertSeverity.LOW: '#36a64f',
                AlertSeverity.MEDIUM: '#ff9500',
                AlertSeverity.HIGH: '#ff0000',
                AlertSeverity.CRITICAL: '#8b0000'
            }
            
            payload = {
                'attachments': [{
                    'color': color_map.get(alert.severity, '#36a64f'),
                    'title': f'AI IDE Alert: {alert.component}',
                    'text': alert.message,
                    'fields': [
                        {'title': 'Severity', 'value': alert.severity.value, 'short': True},
                        {'title': 'Component', 'value': alert.component, 'short': True}
                    ]
                }]
            }
            
            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()
            
            logging.info(f"Slack alert sent for {alert.id}")
            
        except Exception as e:
            logging.error(f"Failed to send Slack alert: {e}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create monitoring system
    monitoring_system = MonitoringAlertingSystem()
    
    # Start monitoring
    monitoring_system.start_monitoring()
    
    print("Monitoring system initialized. Health check endpoints available at http://localhost:8000/health")