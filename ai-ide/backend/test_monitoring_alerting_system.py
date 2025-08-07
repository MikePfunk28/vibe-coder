"""
Comprehensive tests for the monitoring and alerting system
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from monitoring_alerting_system import (
    MonitoringAlertingSystem,
    HealthCheckManager,
    PerformanceMonitor,
    AlertManager,
    DiagnosticManager,
    UserExperienceMonitor,
    ComponentStatus,
    AlertSeverity,
    HealthCheckResult,
    PerformanceMetric,
    Alert,
    DiagnosticInfo,
    EmailNotificationChannel,
    SlackNotificationChannel
)


class TestHealthCheckManager:
    """Test health check management functionality"""
    
    @pytest.fixture
    def health_manager(self):
        return HealthCheckManager()
    
    def test_register_health_check(self, health_manager):
        """Test registering health check functions"""
        def dummy_check():
            return {'status': ComponentStatus.HEALTHY, 'message': 'OK'}
        
        health_manager.register_health_check('test_component', dummy_check, 60)
        
        assert 'test_component' in health_manager.health_checks
        assert health_manager.check_intervals['test_component'] == 60
    
    @pytest.mark.asyncio
    async def test_check_component_health_success(self, health_manager):
        """Test successful component health check"""
        def healthy_check():
            return {
                'status': ComponentStatus.HEALTHY,
                'message': 'Component is healthy',
                'details': {'version': '1.0.0'}
            }
        
        health_manager.register_health_check('test_component', healthy_check)
        result = await health_manager.check_component_health('test_component')
        
        assert result.component == 'test_component'
        assert result.status == ComponentStatus.HEALTHY
        assert result.message == 'Component is healthy'
        assert result.details == {'version': '1.0.0'}
        assert result.response_time > 0
    
    @pytest.mark.asyncio
    async def test_check_component_health_failure(self, health_manager):
        """Test failed component health check"""
        def failing_check():
            raise Exception("Component is down")
        
        health_manager.register_health_check('test_component', failing_check)
        result = await health_manager.check_component_health('test_component')
        
        assert result.component == 'test_component'
        assert result.status == ComponentStatus.UNHEALTHY
        assert 'Component is down' in result.message
        assert 'error' in result.details
    
    @pytest.mark.asyncio
    async def test_check_component_health_async(self, health_manager):
        """Test async health check function"""
        async def async_check():
            await asyncio.sleep(0.1)
            return {'status': ComponentStatus.HEALTHY, 'message': 'Async OK'}
        
        health_manager.register_health_check('async_component', async_check)
        result = await health_manager.check_component_health('async_component')
        
        assert result.status == ComponentStatus.HEALTHY
        assert result.message == 'Async OK'
        assert result.response_time >= 0.1
    
    @pytest.mark.asyncio
    async def test_check_unknown_component(self, health_manager):
        """Test checking health of unregistered component"""
        result = await health_manager.check_component_health('unknown_component')
        
        assert result.component == 'unknown_component'
        assert result.status == ComponentStatus.UNKNOWN
        assert 'No health check registered' in result.message
    
    @pytest.mark.asyncio
    async def test_check_all_components(self, health_manager):
        """Test checking all registered components"""
        def check1():
            return {'status': ComponentStatus.HEALTHY, 'message': 'OK1'}
        
        def check2():
            return {'status': ComponentStatus.DEGRADED, 'message': 'OK2'}
        
        health_manager.register_health_check('component1', check1)
        health_manager.register_health_check('component2', check2)
        
        results = await health_manager.check_all_components()
        
        assert len(results) == 2
        assert 'component1' in results
        assert 'component2' in results
        assert results['component1'].status == ComponentStatus.HEALTHY
        assert results['component2'].status == ComponentStatus.DEGRADED
    
    def test_get_component_history(self, health_manager):
        """Test getting component health history"""
        # Add some history manually
        result1 = HealthCheckResult(
            component='test_component',
            status=ComponentStatus.HEALTHY,
            response_time=0.1,
            message='OK',
            timestamp=datetime.now(),
            details={}
        )
        result2 = HealthCheckResult(
            component='test_component',
            status=ComponentStatus.DEGRADED,
            response_time=0.2,
            message='Slow',
            timestamp=datetime.now(),
            details={}
        )
        
        health_manager.health_history['test_component'].append(result1)
        health_manager.health_history['test_component'].append(result2)
        
        history = health_manager.get_component_history('test_component', 1)
        assert len(history) == 1
        assert history[0].status == ComponentStatus.DEGRADED
        
        full_history = health_manager.get_component_history('test_component')
        assert len(full_history) == 2


class TestPerformanceMonitor:
    """Test performance monitoring functionality"""
    
    @pytest.fixture
    def performance_monitor(self):
        return PerformanceMonitor(retention_hours=1)
    
    def test_record_metric(self, performance_monitor):
        """Test recording performance metrics"""
        performance_monitor.record_metric(
            'test_component', 'response_time', 0.5, 'seconds', {'env': 'test'}
        )
        
        metrics = performance_monitor.get_metrics('test_component', 'response_time')
        assert len(metrics) == 1
        assert metrics[0].value == 0.5
        assert metrics[0].unit == 'seconds'
        assert metrics[0].tags == {'env': 'test'}
    
    def test_get_metrics_with_time_filter(self, performance_monitor):
        """Test getting metrics with time filtering"""
        now = datetime.now()
        old_time = now - timedelta(hours=2)
        
        # Record old metric
        old_metric = PerformanceMetric(
            component='test_component',
            metric_name='cpu_usage',
            value=50.0,
            unit='percent',
            timestamp=old_time,
            tags={}
        )
        performance_monitor.metrics['test_component.cpu_usage'].append(old_metric)
        
        # Record new metric
        performance_monitor.record_metric('test_component', 'cpu_usage', 75.0, 'percent')
        
        # Get metrics from last hour
        start_time = now - timedelta(hours=1)
        recent_metrics = performance_monitor.get_metrics(
            'test_component', 'cpu_usage', start_time
        )
        
        assert len(recent_metrics) == 1
        assert recent_metrics[0].value == 75.0
    
    def test_get_metric_summary(self, performance_monitor):
        """Test getting metric summary statistics"""
        # Record multiple metrics
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for value in values:
            performance_monitor.record_metric('test_component', 'latency', value, 'ms')
        
        summary = performance_monitor.get_metric_summary('test_component', 'latency', 60)
        
        assert summary['count'] == 5
        assert summary['min'] == 10.0
        assert summary['max'] == 50.0
        assert summary['avg'] == 30.0
        assert summary['latest'] == 50.0
    
    def test_register_metric_collector(self, performance_monitor):
        """Test registering metric collector functions"""
        def dummy_collector():
            pass
        
        performance_monitor.register_metric_collector('test_metric', dummy_collector, 30)
        
        assert 'test_metric' in performance_monitor.metric_collectors
        assert performance_monitor.metric_collectors['test_metric']['interval'] == 30


class TestAlertManager:
    """Test alert management functionality"""
    
    @pytest.fixture
    def alert_manager(self):
        return AlertManager()
    
    def test_add_alert_rule(self, alert_manager):
        """Test adding alert rules"""
        rule = {
            'name': 'high_cpu',
            'condition': {'type': 'metric_threshold'},
            'severity': 'high',
            'message': 'CPU usage is high'
        }
        
        alert_manager.add_alert_rule(rule)
        assert len(alert_manager.alert_rules) == 1
        assert alert_manager.alert_rules[0]['name'] == 'high_cpu'
    
    def test_add_invalid_alert_rule(self, alert_manager):
        """Test adding invalid alert rule"""
        invalid_rule = {
            'name': 'incomplete_rule'
            # Missing required fields
        }
        
        with pytest.raises(ValueError):
            alert_manager.add_alert_rule(invalid_rule)
    
    def test_evaluate_metric_threshold_rule(self, alert_manager):
        """Test evaluating metric threshold alert rule"""
        rule = {
            'name': 'high_cpu',
            'condition': {
                'type': 'metric_threshold',
                'component': 'system',
                'metric': 'cpu_usage',
                'threshold': 80.0,
                'operator': '>'
            },
            'severity': 'high',
            'message': 'CPU usage is high'
        }
        
        alert_manager.add_alert_rule(rule)
        
        # Create mock metrics that should trigger alert
        high_cpu_metric = PerformanceMetric(
            component='system',
            metric_name='cpu_usage',
            value=85.0,
            unit='percent',
            timestamp=datetime.now(),
            tags={}
        )
        
        metrics = {'system.cpu_usage': [high_cpu_metric]}
        health_results = {}
        
        # Mock notification channel
        notification_mock = Mock()
        alert_manager.add_notification_channel(notification_mock)
        
        alert_manager.evaluate_rules(health_results, metrics)
        
        # Check that alert was triggered
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].severity == AlertSeverity.HIGH
        assert 'CPU usage is high' in active_alerts[0].message
        
        # Check that notification was sent
        notification_mock.assert_called_once()
    
    def test_evaluate_health_status_rule(self, alert_manager):
        """Test evaluating health status alert rule"""
        rule = {
            'name': 'database_down',
            'condition': {
                'type': 'health_status',
                'component': 'database',
                'status': 'unhealthy'
            },
            'severity': 'critical',
            'message': 'Database is down'
        }
        
        alert_manager.add_alert_rule(rule)
        
        # Create unhealthy health result
        unhealthy_result = HealthCheckResult(
            component='database',
            status=ComponentStatus.UNHEALTHY,
            response_time=0.0,
            message='Connection failed',
            timestamp=datetime.now(),
            details={}
        )
        
        health_results = {'database': unhealthy_result}
        metrics = {}
        
        notification_mock = Mock()
        alert_manager.add_notification_channel(notification_mock)
        
        alert_manager.evaluate_rules(health_results, metrics)
        
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].severity == AlertSeverity.CRITICAL
    
    def test_resolve_alert(self, alert_manager):
        """Test resolving alerts"""
        # Create an alert manually
        alert = Alert(
            id='test_alert_1',
            component='test',
            severity=AlertSeverity.HIGH,
            message='Test alert',
            timestamp=datetime.now()
        )
        alert_manager.alerts['test_alert_1'] = alert
        
        # Resolve the alert
        alert_manager.resolve_alert('test_alert_1')
        
        assert alert_manager.alerts['test_alert_1'].resolved is True
        assert alert_manager.alerts['test_alert_1'].resolution_time is not None
    
    def test_duplicate_alert_prevention(self, alert_manager):
        """Test that duplicate alerts are not created"""
        rule = {
            'name': 'test_rule',
            'condition': {
                'type': 'metric_threshold',
                'component': 'system',
                'metric': 'cpu_usage',
                'threshold': 80.0,
                'operator': '>'
            },
            'severity': 'high',
            'message': 'Test alert',
            'component': 'system'
        }
        
        alert_manager.add_alert_rule(rule)
        
        high_cpu_metric = PerformanceMetric(
            component='system',
            metric_name='cpu_usage',
            value=85.0,
            unit='percent',
            timestamp=datetime.now(),
            tags={}
        )
        
        metrics = {'system.cpu_usage': [high_cpu_metric]}
        health_results = {}
        
        # Evaluate rules twice
        alert_manager.evaluate_rules(health_results, metrics)
        alert_manager.evaluate_rules(health_results, metrics)
        
        # Should only have one alert
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) == 1


class TestDiagnosticManager:
    """Test diagnostic management functionality"""
    
    @pytest.fixture
    def diagnostic_manager(self):
        return DiagnosticManager()
    
    def test_register_diagnostic_collector(self, diagnostic_manager):
        """Test registering diagnostic collectors"""
        def dummy_collector():
            return {'logs': [], 'metrics': {}, 'traces': []}
        
        diagnostic_manager.register_diagnostic_collector('test_component', dummy_collector)
        
        assert 'test_component' in diagnostic_manager.diagnostic_collectors
    
    def test_collect_diagnostic_info(self, diagnostic_manager):
        """Test collecting diagnostic information"""
        def test_collector():
            return {
                'logs': ['Log entry 1', 'Log entry 2'],
                'metrics': {'requests': 100, 'errors': 5},
                'traces': [{'action': 'test', 'duration': 0.5}]
            }
        
        diagnostic_manager.register_diagnostic_collector('test_component', test_collector)
        
        with patch('monitoring_alerting_system.psutil') as mock_psutil:
            mock_psutil.cpu_percent.return_value = 50.0
            mock_psutil.virtual_memory.return_value.percent = 60.0
            mock_psutil.disk_usage.return_value.percent = 70.0
            mock_psutil.pids.return_value = [1, 2, 3]
            
            diagnostics = diagnostic_manager.collect_diagnostic_info('test_component')
        
        assert diagnostics.component == 'test_component'
        assert len(diagnostics.logs) == 2
        assert diagnostics.metrics['requests'] == 100
        assert len(diagnostics.traces) >= 1  # At least the one from collector
        assert 'cpu_percent' in diagnostics.system_info
    
    def test_add_trace(self, diagnostic_manager):
        """Test adding trace entries"""
        trace_data = {'action': 'test_action', 'result': 'success'}
        diagnostic_manager.add_trace('test_component', trace_data)
        
        traces = list(diagnostic_manager.trace_storage['test_component'])
        assert len(traces) == 1
        assert traces[0]['data'] == trace_data


class TestUserExperienceMonitor:
    """Test user experience monitoring functionality"""
    
    @pytest.fixture
    def ux_monitor(self):
        return UserExperienceMonitor()
    
    def test_record_user_interaction(self, ux_monitor):
        """Test recording user interactions"""
        ux_monitor.record_user_interaction(
            'code_completion', 1.5, True, {'language': 'python'}
        )
        
        interactions = list(ux_monitor.user_interactions)
        assert len(interactions) == 1
        assert interactions[0]['type'] == 'code_completion'
        assert interactions[0]['duration'] == 1.5
        assert interactions[0]['success'] is True
    
    def test_record_satisfaction_score(self, ux_monitor):
        """Test recording satisfaction scores"""
        ux_monitor.record_satisfaction_score(4.5, 'code_generation')
        
        scores = list(ux_monitor.satisfaction_scores)
        assert len(scores) == 1
        assert scores[0]['score'] == 4.5
        assert scores[0]['context'] == 'code_generation'
    
    def test_record_performance_impact(self, ux_monitor):
        """Test recording performance impacts"""
        ux_monitor.record_performance_impact('semantic_search', 0.8)
        ux_monitor.record_performance_impact('semantic_search', 0.9)
        
        impacts = ux_monitor.performance_impacts['semantic_search']
        assert len(impacts) == 2
        assert impacts == [0.8, 0.9]
    
    def test_get_user_experience_metrics(self, ux_monitor):
        """Test getting user experience metrics"""
        # Record some interactions
        ux_monitor.record_user_interaction('action1', 1.0, True)
        ux_monitor.record_user_interaction('action2', 2.0, False)
        ux_monitor.record_user_interaction('action3', 1.5, True)
        
        # Record satisfaction scores
        ux_monitor.record_satisfaction_score(4.0)
        ux_monitor.record_satisfaction_score(5.0)
        
        metrics = ux_monitor.get_user_experience_metrics(24)
        
        assert metrics['total_interactions'] == 3
        assert metrics['success_rate'] == 2/3  # 2 successful out of 3
        assert metrics['average_duration'] == 1.5  # (1.0 + 2.0 + 1.5) / 3
        assert metrics['average_satisfaction'] == 4.5  # (4.0 + 5.0) / 2


class TestNotificationChannels:
    """Test notification channel implementations"""
    
    def test_email_notification_channel(self):
        """Test email notification channel"""
        with patch('monitoring_alerting_system.smtplib.SMTP') as mock_smtp:
            mock_server = Mock()
            mock_smtp.return_value = mock_server
            
            email_channel = EmailNotificationChannel(
                'smtp.test.com', 587, 'test@test.com', 'password', ['admin@test.com']
            )
            
            alert = Alert(
                id='test_alert',
                component='test',
                severity=AlertSeverity.HIGH,
                message='Test alert message',
                timestamp=datetime.now()
            )
            
            email_channel(alert)
            
            mock_smtp.assert_called_once_with('smtp.test.com', 587)
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once_with('test@test.com', 'password')
            mock_server.send_message.assert_called_once()
            mock_server.quit.assert_called_once()
    
    def test_slack_notification_channel(self):
        """Test Slack notification channel"""
        with patch('monitoring_alerting_system.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            slack_channel = SlackNotificationChannel('https://hooks.slack.com/test')
            
            alert = Alert(
                id='test_alert',
                component='test',
                severity=AlertSeverity.CRITICAL,
                message='Critical alert message',
                timestamp=datetime.now()
            )
            
            slack_channel(alert)
            
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == 'https://hooks.slack.com/test'
            assert 'attachments' in call_args[1]['json']


class TestMonitoringAlertingSystem:
    """Test the main monitoring and alerting system"""
    
    @pytest.fixture
    def monitoring_system(self):
        config = {'retention_hours': 24}
        return MonitoringAlertingSystem(config)
    
    def test_initialization(self, monitoring_system):
        """Test system initialization"""
        assert monitoring_system.health_manager is not None
        assert monitoring_system.performance_monitor is not None
        assert monitoring_system.alert_manager is not None
        assert monitoring_system.diagnostic_manager is not None
        assert monitoring_system.ux_monitor is not None
        assert monitoring_system.app is not None
    
    def test_get_system_overview(self, monitoring_system):
        """Test getting system overview"""
        overview = monitoring_system.get_system_overview()
        
        assert 'timestamp' in overview
        assert 'health_status' in overview
        assert 'active_alerts' in overview
        assert 'monitored_components' in overview
        assert 'performance_summary' in overview
        assert 'user_experience' in overview
    
    @patch('monitoring_alerting_system.logging')
    def test_start_stop_monitoring(self, mock_logging, monitoring_system):
        """Test starting and stopping monitoring"""
        with patch.object(monitoring_system.health_manager, 'start_continuous_monitoring') as mock_health_start, \
             patch.object(monitoring_system.performance_monitor, 'start_monitoring') as mock_perf_start, \
             patch.object(monitoring_system.health_manager, 'stop_continuous_monitoring') as mock_health_stop, \
             patch.object(monitoring_system.performance_monitor, 'stop_monitoring') as mock_perf_stop:
            
            monitoring_system.start_monitoring()
            mock_health_start.assert_called_once()
            mock_perf_start.assert_called_once()
            
            monitoring_system.stop_monitoring()
            mock_health_stop.assert_called_once()
            mock_perf_stop.assert_called_once()


# Integration tests
class TestIntegration:
    """Integration tests for the monitoring system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_monitoring_flow(self):
        """Test complete monitoring flow from health check to alert"""
        monitoring_system = MonitoringAlertingSystem()
        
        # Register a failing health check
        def failing_check():
            raise Exception("Component failure")
        
        monitoring_system.health_manager.register_health_check('test_component', failing_check)
        
        # Add alert rule for unhealthy component
        monitoring_system.alert_manager.add_alert_rule({
            'name': 'component_failure',
            'condition': {
                'type': 'health_status',
                'component': 'test_component',
                'status': 'unhealthy'
            },
            'severity': 'critical',
            'message': 'Test component has failed',
            'component': 'test_component'
        })
        
        # Mock notification channel
        notification_mock = Mock()
        monitoring_system.alert_manager.add_notification_channel(notification_mock)
        
        # Run health check
        health_results = await monitoring_system.health_manager.check_all_components()
        
        # Evaluate alert rules
        monitoring_system.alert_manager.evaluate_rules(health_results, {})
        
        # Verify alert was created and notification sent
        active_alerts = monitoring_system.alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].component == 'test_component'
        assert active_alerts[0].severity == AlertSeverity.CRITICAL
        
        notification_mock.assert_called_once()
    
    def test_performance_monitoring_with_alerts(self):
        """Test performance monitoring triggering alerts"""
        monitoring_system = MonitoringAlertingSystem()
        
        # Add performance alert rule
        monitoring_system.alert_manager.add_alert_rule({
            'name': 'high_response_time',
            'condition': {
                'type': 'metric_threshold',
                'component': 'api',
                'metric': 'response_time',
                'threshold': 1.0,
                'operator': '>'
            },
            'severity': 'medium',
            'message': 'API response time is high'
        })
        
        # Record high response time metric
        monitoring_system.performance_monitor.record_metric(
            'api', 'response_time', 2.5, 'seconds'
        )
        
        # Get metrics for alert evaluation
        metrics = {
            'api.response_time': monitoring_system.performance_monitor.get_metrics('api', 'response_time')
        }
        
        # Mock notification
        notification_mock = Mock()
        monitoring_system.alert_manager.add_notification_channel(notification_mock)
        
        # Evaluate rules
        monitoring_system.alert_manager.evaluate_rules({}, metrics)
        
        # Verify alert was triggered
        active_alerts = monitoring_system.alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert 'response time is high' in active_alerts[0].message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])