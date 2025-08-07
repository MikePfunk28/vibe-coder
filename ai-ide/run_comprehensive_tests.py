#!/usr/bin/env python3
"""
Comprehensive Test Execution Script for AI IDE
Main entry point for running all tests with proper configuration
"""

import sys
import os
import argparse
import asyncio
import subprocess
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.test_runner import ComprehensiveTestRunner
from backend.test_config import get_test_config

def setup_environment():
    """Setup test environment"""
    config = get_test_config()
    
    # Set environment variables
    env_vars = config.get_environment_variables()
    for key, value in env_vars.items():
        os.environ[key] = value
        
    # Create test data directory
    config.create_test_data_directory()
    config.generate_sample_test_data()
    
    print(f"Test environment configured for: {os.getenv('AI_IDE_TEST_ENV', 'development')}")
    return config

def install_dependencies():
    """Install test dependencies"""
    print("Installing backend dependencies...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-r", 
        "backend/requirements.txt"
    ], check=True)
    
    print("Installing extension dependencies...")
    subprocess.run([
        "npm", "install"
    ], cwd="extensions/ai-assistant", check=True)

async def run_tests(test_types: list = None, verbose: bool = False):
    """Run comprehensive tests"""
    runner = ComprehensiveTestRunner()
    
    try:
        if not test_types or "all" in test_types:
            print("Running all test suites...")
            results = await runner.run_all_tests()
        else:
            print(f"Running selected test suites: {', '.join(test_types)}")
            results = {}
            
            if "backend" in test_types:
                results["backend"] = await runner.run_backend_tests()
                
            if "extension" in test_types:
                results["extension"] = runner.run_extension_tests()
                
            if "performance" in test_types:
                results["performance"] = runner.run_performance_tests()
                
            if "security" in test_types:
                results["security"] = runner.run_security_tests()
                
            runner.results = results
        
        # Generate and save report
        timestamp = int(time.time())
        report_file = f"comprehensive_test_report_{timestamp}.json"
        runner.save_report(report_file)
        
        # Print summary
        runner.print_summary()
        
        # Determine exit code
        report = runner.generate_comprehensive_report()
        success_rate = report["summary"]["success_rate"]
        
        if success_rate >= 80:
            print(f"\n✅ Tests passed with {success_rate:.1f}% success rate")
            return 0
        else:
            print(f"\n❌ Tests failed with {success_rate:.1f}% success rate")
            return 1
            
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return 1
        
    finally:
        runner.cleanup()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run comprehensive AI IDE tests")
    
    parser.add_argument(
        "--types", 
        nargs="+", 
        choices=["all", "backend", "extension", "performance", "security"],
        default=["all"],
        help="Test types to run"
    )
    
    parser.add_argument(
        "--env",
        choices=["development", "ci", "production"],
        default="development",
        help="Test environment"
    )
    
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install dependencies before running tests"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only setup environment, don't run tests"
    )
    
    args = parser.parse_args()
    
    # Set test environment
    os.environ["AI_IDE_TEST_ENV"] = args.env
    
    try:
        # Setup environment
        config = setup_environment()
        
        # Install dependencies if requested
        if args.install_deps:
            install_dependencies()
            
        # Validate configuration
        issues = config.validate_configuration()
        if issues:
            print("❌ Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return 1
            
        if args.setup_only:
            print("✅ Environment setup complete")
            return 0
            
        # Run tests
        import time
        exit_code = asyncio.run(run_tests(args.types, args.verbose))
        return exit_code
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 130
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())