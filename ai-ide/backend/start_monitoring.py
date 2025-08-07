#!/usr/bin/env python3
"""
Startup script for the AI IDE monitoring and alerting system
"""

import asyncio
import logging
import signal
import sys
import uvicorn
from pathlib import Path
from monitoring_alerting_system import MonitoringAlertingSystem
from monitoring_config import load_config_from_env, load_config_from_file, get_default_alert_rules
from monitoring_alerting_system import EmailNotificationChannel, SlackNotificationChannel


class MonitoringSystemRunner:
    """Runner class for the monitoring system"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.monitoring_system = None
        self.running = False
        
        # Setup logging
        self._setup_logging()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('monitoring_system.log')
            ]
        )
        
        # Reduce noise from some libraries
        logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
        logging.getLogger('uvicorn.error').setLevel(logging.INFO)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logging.info(f"Received signal {signum}, shutting down...")
        self.running = False
        if self.monitoring_system:
            self.monitoring_system.stop_monitoring()
        sys.exit(0)
    
    def load_configuration(self):
        """Load monitoring system configuration"""
        try:
            if self.config_path and Path(self.config_path).exists():
                logging.info(f"Loading configuration from file: {self.config_path}")
                config = load_config_from_file(self.config_path)
            else:
                logging.info("Loading configuration from environment variables")
                config = load_config_from_env()
            
            logging.info(f"Configuration loaded:")
            logging.info(f"  - Retention hours: {config.retention_hours}")
            logging.info(f"  - Health check interval: {config.health_check_interval}s")
            logging.info(f"  - Metrics collection interval: {config.metrics_collection_interval}s")
            logging.info(f"  - Email notifications: {config.email_enabled}")
            logging.info(f"  - Slack notifications: {config.slack_enabled}")
            
            return config
            
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            sys.exit(1)
    
    def setup_monitoring_system(self, config):
        """Setup the monitoring system with configuration"""
        try:
            # Create monitoring system
            system_config = {
                'retention_hours': config.retention_hours,
                'health_check_interval': config.health_check_interval,
                'metrics_collection_interval': config.metrics_collection_interval
            }
            
            self.monitoring_system = MonitoringAlertingSystem(system_config)
            
            # Setup notification channels
            self._setup_notification_channels(config)
            
            # Setup default alert rules
            self._setup_alert_rules(config)
            
            logging.info("Monitoring system setup completed")
            
        except Exception as e:
            logging.error(f"Failed to setup monitoring system: {e}")
            sys.exit(1)
    
    def _setup_notification_channels(self, config):
        """Setup notification channels based on configuration"""
        channels_added = 0
        
        # Email notifications
        if config.email_enabled and config.smtp_server and config.email_recipients:
            try:
                email_channel = EmailNotificationChannel(
                    smtp_server=config.smtp_server,
                    smtp_port=config.smtp_port,
                    username=config.smtp_username,
                    password=config.smtp_password,
                    recipients=config.email_recipients
                )
                self.monitoring_system.alert_manager.add_notification_channel(email_channel)
                channels_added += 1
                logging.info(f"Email notifications enabled for {len(config.email_recipients)} recipients")
            except Exception as e:
                logging.error(f"Failed to setup email notifications: {e}")
        
        # Slack notifications
        if config.slack_enabled and config.slack_webhook_url:
            try:
                slack_channel = SlackNotificationChannel(config.slack_webhook_url)
                self.monitoring_system.alert_manager.add_notification_channel(slack_channel)
                channels_added += 1
                logging.info(f"Slack notifications enabled for channel {config.slack_channel}")
            except Exception as e:
                logging.error(f"Failed to setup Slack notifications: {e}")
        
        # Webhook notifications
        if config.webhook_enabled and config.webhook_url:
            try:
                def webhook_channel(alert):
                    import requests
                    payload = {
                        'alert_id': alert.id,
                        'component': alert.component,
                        'severity': alert.severity.value,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat(),
                        'details': alert.details
                    }
                    requests.post(config.webhook_url, json=payload, timeout=config.webhook_timeout)
                
                self.monitoring_system.alert_manager.add_notification_channel(webhook_channel)
                channels_added += 1
                logging.info(f"Webhook notifications enabled: {config.webhook_url}")
            except Exception as e:
                logging.error(f"Failed to setup webhook notifications: {e}")
        
        if channels_added == 0:
            logging.warning("No notification channels configured - alerts will only be logged")
        else:
            logging.info(f"Configured {channels_added} notification channel(s)")
    
    def _setup_alert_rules(self, config):
        """Setup default alert rules"""
        try:
            alert_rules = get_default_alert_rules(config)
            
            for rule in alert_rules:
                self.monitoring_system.alert_manager.add_alert_rule(rule)
            
            logging.info(f"Added {len(alert_rules)} default alert rules")
            
        except Exception as e:
            logging.error(f"Failed to setup alert rules: {e}")
    
    def start_monitoring(self):
        """Start the monitoring system"""
        try:
            logging.info("Starting monitoring system...")
            self.monitoring_system.start_monitoring()
            self.running = True
            logging.info("Monitoring system started successfully")
            
        except Exception as e:
            logging.error(f"Failed to start monitoring system: {e}")
            sys.exit(1)
    
    def start_web_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Start the web server for health check endpoints"""
        try:
            logging.info(f"Starting web server on {host}:{port}")
            
            # Configure uvicorn
            config = uvicorn.Config(
                app=self.monitoring_system.app,
                host=host,
                port=port,
                log_level="info",
                access_log=True
            )
            
            server = uvicorn.Server(config)
            
            # Run server
            asyncio.run(server.serve())
            
        except Exception as e:
            logging.error(f"Failed to start web server: {e}")
            sys.exit(1)
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the complete monitoring system"""
        logging.info("AI IDE Monitoring and Alerting System starting...")
        
        # Load configuration
        config = self.load_configuration()
        
        # Setup monitoring system
        self.setup_monitoring_system(config)
        
        # Start monitoring
        self.start_monitoring()
        
        # Print system information
        self._print_system_info(host, port)
        
        # Start web server (this will block)
        self.start_web_server(host, port)
    
    def _print_system_info(self, host: str, port: int):
        """Print system information and available endpoints"""
        print("\n" + "="*60)
        print("AI IDE MONITORING AND ALERTING SYSTEM")
        print("="*60)
        print(f"Web server: http://{host}:{port}")
        print("\nAvailable endpoints:")
        print(f"  - Health check (all):     http://{host}:{port}/health")
        print(f"  - Component health:       http://{host}:{port}/health/{{component}}")
        print(f"  - Component metrics:      http://{host}:{port}/metrics/{{component}}")
        print(f"  - Active alerts:          http://{host}:{port}/alerts")
        print(f"  - Component diagnostics:  http://{host}:{port}/diagnostics/{{component}}")
        print(f"  - User experience:        http://{host}:{port}/user-experience")
        print("\nMonitored components:")
        for component in self.monitoring_system.components:
            print(f"  - {component}")
        print("\nPress Ctrl+C to stop the system")
        print("="*60 + "\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI IDE Monitoring and Alerting System")
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file (JSON format)",
        default=None
    )
    parser.add_argument(
        "--host",
        help="Host to bind web server to",
        default="0.0.0.0"
    )
    parser.add_argument(
        "--port", "-p",
        help="Port to bind web server to",
        type=int,
        default=8000
    )
    parser.add_argument(
        "--log-level",
        help="Logging level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO'
    )
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create and run monitoring system
    runner = MonitoringSystemRunner(args.config)
    
    try:
        runner.run(args.host, args.port)
    except KeyboardInterrupt:
        logging.info("Monitoring system stopped by user")
    except Exception as e:
        logging.error(f"Monitoring system failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()