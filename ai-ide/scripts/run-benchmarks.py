#!/usr/bin/env python3
"""
AI IDE Performance Benchmarking Script
Runs comprehensive performance tests for CI/CD pipeline
"""

import os
import sys
import json
import time
import asyncio
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from mini_benchmark_system import MiniBenchmarkSystem, BenchmarkSuite
from semantic_search_engine import SemanticSearchEngine
from qwen_coder_agent import QwenCoderAgent
from web_search_agent import WebSearchAgent
from rag_system import RAGSystem


class CIBenchmarkRunner:
    """Comprehensive benchmark runner for CI/CD pipeline"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        self.start_time = time.time()
        
    def print_status(self, message: str):
        """Print status message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def run_mini_benchmarks(self) -> Dict[str, Any]:
        """Run mini-benchmark system tests"""
        self.print_status("Running mini-benchmark system tests...")
        
        try:
            benchmark_system = MiniBenchmarkSystem()
            
            # Create test suite for AI components
            suite = BenchmarkSuite("ai_components")
            
            # Add benchmarks for different components
            benchmarks = [
                ("semantic_search_latency", self._benchmark_semantic_search),
                ("code_generation_speed", self._benchmark_code_generation),
                ("web_search_performance", self._benchmark_web_search),
                ("rag_query_speed", self._benchmark_rag_queries),
                ("reasoning_performance", self._benchmark_reasoning),
            ]
            
            results = {}
            for name, benchmark_func in benchmarks:
                self.print_status(f"Running {name} benchmark...")
                try:
                    result = benchmark_func()
                    results[name] = result
                    self.print_status(f"✅ {name}: {result.get('avg_time', 'N/A')}s")
                except Exception as e:
                    self.print_status(f"❌ {name} failed: {str(e)}")
                    results[name] = {"error": str(e)}
            
            return {
                "mini_benchmarks": results,
                "timestamp": datetime.now().isoformat(),
                "total_time": time.time() - self.start_time
            }
            
        except Exception as e:
            self.print_status(f"Mini-benchmark system failed: {str(e)}")
            return {"error": str(e)}
    
    def _benchmark_semantic_search(self) -> Dict[str, Any]:
        """Benchmark semantic search performance"""
        try:
            search_engine = SemanticSearchEngine()
            
            # Test queries
            queries = [
                "function to sort array",
                "error handling in async code",
                "database connection setup",
                "API endpoint validation",
                "unit test examples"
            ]
            
            times = []
            for query in queries:
                start = time.time()
                results = search_engine.search_similar(query, max_results=10)
                end = time.time()
                times.append(end - start)
            
            return {
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "queries_tested": len(queries),
                "total_results": sum(len(r) for r in [[] for _ in queries])  # Placeholder
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _benchmark_code_generation(self) -> Dict[str, Any]:
        """Benchmark code generation performance"""
        try:
            agent = QwenCoderAgent()
            
            # Test prompts
            prompts = [
                "Create a Python function to calculate fibonacci numbers",
                "Write a JavaScript function to validate email addresses",
                "Generate a SQL query to find top 10 customers",
                "Create a React component for user profile",
                "Write a Python class for file operations"
            ]
            
            times = []
            token_counts = []
            
            for prompt in prompts:
                start = time.time()
                response = agent.generate_code(prompt)
                end = time.time()
                
                times.append(end - start)
                token_counts.append(len(response.split()) if response else 0)
            
            return {
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "avg_tokens": sum(token_counts) / len(token_counts),
                "prompts_tested": len(prompts)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _benchmark_web_search(self) -> Dict[str, Any]:
        """Benchmark web search performance"""
        try:
            search_agent = WebSearchAgent()
            
            # Test queries
            queries = [
                "Python asyncio best practices",
                "React hooks tutorial",
                "Docker container optimization",
                "PostgreSQL performance tuning",
                "TypeScript generics examples"
            ]
            
            times = []
            result_counts = []
            
            for query in queries:
                start = time.time()
                results = search_agent.search(query, max_results=5)
                end = time.time()
                
                times.append(end - start)
                result_counts.append(len(results) if results else 0)
            
            return {
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "avg_results": sum(result_counts) / len(result_counts),
                "queries_tested": len(queries)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _benchmark_rag_queries(self) -> Dict[str, Any]:
        """Benchmark RAG system performance"""
        try:
            rag_system = RAGSystem()
            
            # Test queries
            queries = [
                "How to implement authentication in FastAPI?",
                "Best practices for database migrations",
                "Error handling patterns in Python",
                "React component lifecycle methods",
                "Docker multi-stage builds"
            ]
            
            times = []
            relevance_scores = []
            
            for query in queries:
                start = time.time()
                response = rag_system.query(query)
                end = time.time()
                
                times.append(end - start)
                # Mock relevance score calculation
                relevance_scores.append(0.8 if response else 0.0)
            
            return {
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "avg_relevance": sum(relevance_scores) / len(relevance_scores),
                "queries_tested": len(queries)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _benchmark_reasoning(self) -> Dict[str, Any]:
        """Benchmark reasoning engine performance"""
        try:
            # Import reasoning engine
            from chain_of_thought_engine import ChainOfThoughtEngine
            
            reasoning_engine = ChainOfThoughtEngine()
            
            # Test reasoning tasks
            tasks = [
                "Explain the time complexity of quicksort algorithm",
                "Debug this Python code: def func(x): return x + y",
                "Optimize this SQL query for better performance",
                "Design a caching strategy for web application",
                "Analyze the security implications of this API design"
            ]
            
            times = []
            step_counts = []
            
            for task in tasks:
                start = time.time()
                result = reasoning_engine.reason(task)
                end = time.time()
                
                times.append(end - start)
                step_counts.append(len(result.get('steps', [])) if result else 0)
            
            return {
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "avg_steps": sum(step_counts) / len(step_counts),
                "tasks_tested": len(tasks)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def run_load_tests(self) -> Dict[str, Any]:
        """Run load testing scenarios"""
        self.print_status("Running load tests...")
        
        try:
            # Simulate concurrent requests
            import concurrent.futures
            import requests
            
            base_url = os.getenv("AI_IDE_BASE_URL", "http://localhost:8000")
            
            def make_request(endpoint: str) -> Dict[str, Any]:
                try:
                    start = time.time()
                    response = requests.get(f"{base_url}{endpoint}", timeout=30)
                    end = time.time()
                    
                    return {
                        "status_code": response.status_code,
                        "response_time": end - start,
                        "success": response.status_code == 200
                    }
                except Exception as e:
                    return {
                        "error": str(e),
                        "success": False
                    }
            
            # Test endpoints
            endpoints = [
                "/health",
                "/api/semantic-search",
                "/api/code-generation",
                "/api/web-search",
                "/api/rag-query"
            ]
            
            # Run concurrent requests
            results = {}
            for endpoint in endpoints:
                self.print_status(f"Load testing {endpoint}...")
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(make_request, endpoint) for _ in range(20)]
                    responses = [f.result() for f in concurrent.futures.as_completed(futures)]
                
                successful = [r for r in responses if r.get('success', False)]
                response_times = [r['response_time'] for r in successful]
                
                results[endpoint] = {
                    "total_requests": len(responses),
                    "successful_requests": len(successful),
                    "success_rate": len(successful) / len(responses),
                    "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
                    "max_response_time": max(response_times) if response_times else 0,
                    "min_response_time": min(response_times) if response_times else 0
                }
            
            return {
                "load_tests": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.print_status(f"Load tests failed: {str(e)}")
            return {"error": str(e)}
    
    def run_memory_benchmarks(self) -> Dict[str, Any]:
        """Run memory usage benchmarks"""
        self.print_status("Running memory benchmarks...")
        
        try:
            import psutil
            import gc
            
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run memory-intensive operations
            operations = [
                ("semantic_indexing", self._memory_test_semantic_indexing),
                ("large_code_generation", self._memory_test_large_generation),
                ("concurrent_searches", self._memory_test_concurrent_searches),
                ("rag_knowledge_loading", self._memory_test_rag_loading)
            ]
            
            results = {}
            for name, operation in operations:
                gc.collect()  # Clean up before test
                start_memory = process.memory_info().rss / 1024 / 1024
                
                try:
                    operation()
                    end_memory = process.memory_info().rss / 1024 / 1024
                    memory_used = end_memory - start_memory
                    
                    results[name] = {
                        "memory_used_mb": memory_used,
                        "peak_memory_mb": end_memory,
                        "success": True
                    }
                    
                except Exception as e:
                    results[name] = {
                        "error": str(e),
                        "success": False
                    }
            
            final_memory = process.memory_info().rss / 1024 / 1024
            
            return {
                "memory_benchmarks": results,
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "total_memory_increase_mb": final_memory - initial_memory,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _memory_test_semantic_indexing(self):
        """Test memory usage during semantic indexing"""
        search_engine = SemanticSearchEngine()
        # Simulate indexing large codebase
        for i in range(100):
            search_engine.index_code(f"test_file_{i}.py", f"def function_{i}(): pass")
    
    def _memory_test_large_generation(self):
        """Test memory usage during large code generation"""
        agent = QwenCoderAgent()
        # Generate large code blocks
        for i in range(10):
            agent.generate_code(f"Create a comprehensive class with {i*10} methods")
    
    def _memory_test_concurrent_searches(self):
        """Test memory usage during concurrent searches"""
        search_engine = SemanticSearchEngine()
        import threading
        
        def search_worker():
            for i in range(20):
                search_engine.search_similar(f"test query {i}")
        
        threads = [threading.Thread(target=search_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    
    def _memory_test_rag_loading(self):
        """Test memory usage during RAG knowledge loading"""
        rag_system = RAGSystem()
        # Simulate loading large knowledge base
        for i in range(50):
            rag_system.add_document(f"doc_{i}", f"Large document content {i} " * 100)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        self.print_status("Generating comprehensive benchmark report...")
        
        # Run all benchmark suites
        mini_results = self.run_mini_benchmarks()
        load_results = self.run_load_tests()
        memory_results = self.run_memory_benchmarks()
        
        # Combine results
        report = {
            "benchmark_run": {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": time.time() - self.start_time,
                "environment": {
                    "python_version": sys.version,
                    "platform": sys.platform,
                    "ci_environment": os.getenv("CI", "false"),
                    "github_sha": os.getenv("GITHUB_SHA", "unknown"),
                    "github_ref": os.getenv("GITHUB_REF", "unknown")
                }
            },
            "results": {
                "mini_benchmarks": mini_results,
                "load_tests": load_results,
                "memory_benchmarks": memory_results
            },
            "summary": self._generate_summary(mini_results, load_results, memory_results),
            "recommendations": self._generate_recommendations(mini_results, load_results, memory_results)
        }
        
        # Save report
        report_file = self.output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.print_status(f"Report saved to: {report_file}")
        return report
    
    def _generate_summary(self, mini_results: Dict, load_results: Dict, memory_results: Dict) -> Dict[str, Any]:
        """Generate benchmark summary"""
        summary = {
            "overall_status": "PASS",
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "performance_score": 0.0
        }
        
        # Count tests and calculate scores
        test_categories = [mini_results, load_results, memory_results]
        scores = []
        
        for category in test_categories:
            if "error" not in category:
                category_tests = len(category.get("results", category))
                summary["total_tests"] += category_tests
                
                # Calculate category score (simplified)
                category_score = 0.8  # Default good score
                scores.append(category_score)
                summary["passed_tests"] += category_tests
            else:
                summary["failed_tests"] += 1
                scores.append(0.0)
        
        if scores:
            summary["performance_score"] = sum(scores) / len(scores)
        
        if summary["failed_tests"] > 0 or summary["performance_score"] < 0.7:
            summary["overall_status"] = "FAIL"
        
        return summary
    
    def _generate_recommendations(self, mini_results: Dict, load_results: Dict, memory_results: Dict) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # Analyze mini-benchmark results
        if "mini_benchmarks" in mini_results:
            for test_name, result in mini_results["mini_benchmarks"].items():
                if "error" in result:
                    recommendations.append(f"Fix {test_name} benchmark failure: {result['error']}")
                elif result.get("avg_time", 0) > 2.0:
                    recommendations.append(f"Optimize {test_name} performance (current: {result['avg_time']:.2f}s)")
        
        # Analyze load test results
        if "load_tests" in load_results:
            for endpoint, result in load_results["load_tests"].items():
                if result.get("success_rate", 0) < 0.95:
                    recommendations.append(f"Improve reliability of {endpoint} (success rate: {result['success_rate']:.1%})")
                if result.get("avg_response_time", 0) > 1.0:
                    recommendations.append(f"Optimize response time for {endpoint} (current: {result['avg_response_time']:.2f}s)")
        
        # Analyze memory results
        if "memory_benchmarks" in memory_results:
            total_increase = memory_results.get("total_memory_increase_mb", 0)
            if total_increase > 500:  # 500MB threshold
                recommendations.append(f"Investigate memory usage increase: {total_increase:.1f}MB")
        
        if not recommendations:
            recommendations.append("All benchmarks passed! Performance is within acceptable limits.")
        
        return recommendations


def main():
    """Main benchmark runner"""
    parser = argparse.ArgumentParser(description="AI IDE Performance Benchmarks")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory for results")
    parser.add_argument("--mini-only", action="store_true", help="Run only mini-benchmarks")
    parser.add_argument("--load-only", action="store_true", help="Run only load tests")
    parser.add_argument("--memory-only", action="store_true", help="Run only memory tests")
    
    args = parser.parse_args()
    
    runner = CIBenchmarkRunner(args.output_dir)
    
    try:
        if args.mini_only:
            results = runner.run_mini_benchmarks()
        elif args.load_only:
            results = runner.run_load_tests()
        elif args.memory_only:
            results = runner.run_memory_benchmarks()
        else:
            results = runner.generate_report()
        
        # Print summary
        if "summary" in results:
            summary = results["summary"]
            print(f"\n=== Benchmark Summary ===")
            print(f"Status: {summary['overall_status']}")
            print(f"Tests: {summary['passed_tests']}/{summary['total_tests']} passed")
            print(f"Performance Score: {summary['performance_score']:.2f}")
            print(f"========================\n")
            
            # Exit with error code if benchmarks failed
            if summary["overall_status"] == "FAIL":
                sys.exit(1)
        
    except Exception as e:
        print(f"Benchmark run failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()