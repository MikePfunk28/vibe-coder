"""
Integration tests for the monitoring and alerting system
"""

import asyncio
import time
import pytest
from monitoring_alerting_system import MonitoringAlertingSystem, ComponentStatus, AlertSeverity


class TestMonitoringIntegration:
    """Integration tests for the complete monitoring system"""
    
    @pytest.fixture
    def monitoring_system(self):
        """Create a monitoring system for testing"""
        config = {
            'retention_hours': 1,
            'health_check_interval': 5,
            'metrics_collection_interval': 10
        }
        return MonitoringAlertingSystem(config)
    
    @pytest.mark.asyncio
    async def test_health_check_to_alert_flow(self, monitoring_system):
        """Test complete flow from health check failure to alert generation"""
        
        # Register a failing health check
        def failing_health_check():
            raise Exception("Component is down")
        
        monitoring_system.health_manager.register_health_check(
            'test_component', failing_health_check, 30
        )
        
        # Add alert rule for unhealthy component
        monitoring_system.alert_manager.add_alert_rule({
            'name': 'test_component_failure',
            'condition': {
                'type': 'health_status',
                'component': 'test_component',
                'status': 'unhealthy'
            },
            'severity': 'critical',
            'message': 'Test component has failed',
            'component': 'test_component'
        })
        
        # Mock notification channel to capture alerts
        captured_alerts = []
        def mock_notification(alert):
            captured_alerts.append(alert)
        
        monitoring_system.alert_manager.add_notification_channel(mock_notification)
        
        # Run health check
        health_results = await monitoring_system.health_manager.check_all_components()
        
        # Verify health check failed
        assert 'test_component' in health_results
        assert health_results['test_component'].status == ComponentStatus.UNHEALTHY
        
        # Evaluate alert rules
        monitoring_system.alert_manager.evaluate_rules(health_results, {})
        
        # Verify alert was created and notification sent
        active_alerts = monitoring_system.alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].component == 'test_component'
        assert active_alerts[0].severity == AlertSeverity.CRITICAL
        
        # Verify notification was sent
        assert len(captured_alerts) == 1
        assert captured_alerts[0].component == 'test_component'
    
    def test_performance_monitoring_with_alerts(self, monitoring_system):
        """Test performance monitoring triggering alerts"""
        
        # Add performance alert rule
        monitoring_system.alert_manager.add_alert_rule({
            'name': 'high_response_time',
            'condition': {
                'type': 'metric_threshold',
                'component': 'test_service',
                'metric': 'response_time',
                'threshold': 1.0,
                'operator': '>'
            },
            'severity': 'medium',
            'message': 'Service response time is high'
        })
        
        # Record high response time metric
        monitoring_system.performance_monitor.record_metric(
            'test_service', 'response_time', 2.5, 'seconds'
        )
        
        # Get metrics for alert evaluation
        metrics = {
            'test_service.response_time': monitoring_system.performance_monitor.get_metrics(
                'test_service', 'response_time'
            )
        }
        
        # Mock notification
        captured_alerts = []
        def mock_notification(alert):
            captured_alerts.append(alert)
        
        monitoring_system.alert_manager.add_notification_channel(mock_notification)
        
        # Evaluate rules
        monitoring_system.alert_manager.evaluate_rules({}, metrics)
        
        # Verify alert was triggered
        active_alerts = monitoring_system.alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert 'response time is high' in active_alerts[0].message
        assert len(captured_alerts) == 1
    
    def test_user_experience_monitoring(self, monitoring_system):
        """Test user experience monitoring functionality"""
        
        # Record some user interactions
        monitoring_system.ux_monitor.record_user_interaction(
            'code_completion', 1.2, True, {'language': 'python'}
        )
        monitoring_system.ux_monitor.record_user_interaction(
            'code_generation', 3.5, False, {'error': 'timeout'}
        )
        monitoring_system.ux_monitor.record_user_interaction(
            'semantic_search', 0.8, True, {'results': 10}
        )
        
        # Record satisfaction scores
        monitoring_system.ux_monitor.record_satisfaction_score(4.5, 'code_completion')
        monitoring_system.ux_monitor.record_satisfaction_score(2.0, 'code_generation')
        
        # Get UX metrics
        ux_metrics = monitoring_system.ux_monitor.get_user_experience_metrics(24)
        
        # Verify metrics
        assert ux_metrics['total_interactions'] == 3
        assert ux_metrics['success_rate'] == 2/3  # 2 successful out of 3
        assert ux_metrics['average_satisfaction'] == 3.25  # (4.5 + 2.0) / 2
        assert ux_metrics['satisfaction_trend'] in ['improving', 'declining', 'stable', 'insufficient_data']
    
    def test_diagnostic_collection(self, monitoring_system):
        """Test diagnostic information collection"""
        
        # Register a diagnostic collector
        def test_diagnostic_collector():
            return {
                'logs': ['Service started', 'Processing request', 'Request completed'],
                'metrics': {
                    'requests_processed': 150,
                    'errors_encountered': 3,
                    'average_response_time': 1.2
                },
                'traces': [
                    {'operation': 'process_request', 'duration': 1.1, 'success': True},
                    {'operation': 'database_query', 'duration': 0.3, 'success': True}
                ]
            }
        
        monitoring_system.diagnostic_manager.register_diagnostic_collector(
            'test_service', test_diagnostic_collector
        )
        
        # Add some traces
        monitoring_system.diagnostic_manager.add_trace(
            'test_service', 
            {'action': 'user_request', 'result': 'success', 'duration': 0.9}
        )
        
        # Collect diagnostics
        diagnostics = monitoring_system.diagnostic_manager.collect_diagnostic_info('test_service')
        
        # Verify diagnostic information
        assert diagnostics.component == 'test_service'
        assert len(diagnostics.logs) == 3
        assert diagnostics.metrics['requests_processed'] == 150
        assert len(diagnostics.traces) >= 3  # 2 from collector + 1 added trace
        assert 'cpu_percent' in diagnostics.system_info or 'error' in diagnostics.system_info
    
    @pytest.mark.asyncio
    async def test_system_overview(self, monitoring_system):
        """Test getting system overview"""
        
        # Register some health checks
        def healthy_check():
            return {'status': ComponentStatus.HEALTHY, 'message': 'OK'}
        
        def degraded_check():
            return {'status': ComponentStatus.DEGRADED, 'message': 'Slow'}
        
        monitoring_system.health_manager.register_health_check('service1', healthy_check)
        monitoring_system.health_manager.register_health_check('service2', degraded_check)
        
        # Record some metrics
        monitoring_system.performance_monitor.record_metric('service1', 'cpu_usage', 45.0, 'percent')
        monitoring_system.performance_monitor.record_metric('service2', 'memory_usage', 70.0, 'percent')
        
        # Record user interactions
        monitoring_system.ux_monitor.record_user_interaction('test_action', 1.0, True)
        monitoring_system.ux_monitor.record_satisfaction_score(4.2)
        
        # Get system overview
        overview = monitoring_system.get_system_overview()
        
        # Verify overview contains expected information
        assert 'timestamp' in overview
        assert 'health_status' in overview
        assert 'active_alerts' in overview
        assert 'monitored_components' in overview
        assert 'performance_summary' in overview
        assert 'user_experience' in overview
        
        # Verify user experience data
        assert overview['user_experience']['satisfaction_score'] == 4.2
        assert overview['user_experience']['success_rate'] == 1.0
    
    def test_alert_resolution(self, monitoring_system):
        """Test alert resolution workflow"""
        
        # Create an alert manually for testing
        from monitoring_alerting_system import Alert
        
        alert = Alert(
            id='test_alert_123',
            component='test_service',
            severity=AlertSeverity.HIGH,
            message='Test alert for resolution',
            timestamp=monitoring_system.alert_manager.alert_history[0].timestamp if monitoring_system.alert_manager.alert_history else None
        )
        
        # Add alert to manager
        monitoring_system.alert_manager.alerts['test_alert_123'] = alert
        
        # Verify alert is active
        active_alerts = monitoring_system.alert_manager.get_active_alerts()
        assert any(a.id == 'test_alert_123' for a in active_alerts)
        
        # Resolve the alert
        monitoring_system.alert_manager.resolve_alert('test_alert_123')
        
        # Verify alert is resolved
        resolved_alert = monitoring_system.alert_manager.alerts['test_alert_123']
        assert resolved_alert.resolved is True
        assert resolved_alert.resolution_time is not None
        
        # Verify alert is no longer in active alerts
        active_alerts = monitoring_system.alert_manager.get_active_alerts()
        assert not any(a.id == 'test_alert_123' for a in active_alerts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])