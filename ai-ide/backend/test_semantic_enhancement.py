#!/usr/bin/env python3
"""
Test script for enhanced semantic awareness in AI IDE
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.pocketflow_integration import create_ai_ide_flow
from backend.semantic_engine import get_semantic_index, get_performance_tracker

async def test_semantic_indexing():
    """Test semantic indexing functionality"""
    print("Testing semantic indexing...")
    
    # Create a test workspace
    test_workspace = Path.cwd()
    semantic_index = get_semantic_index(str(test_workspace))
    
    # Index the current workspace
    start_time = time.time()
    semantic_index.index_workspace(['.py'])
    index_time = time.time() - start_time
    
    print(f"Indexed {len(semantic_index.contexts)} files in {index_time:.2f} seconds")
    
    # Show some indexed content
    for file_path, context in list(semantic_index.contexts.items())[:3]:
        features = context.semantic_features
        print(f"\nFile: {file_path}")
        print(f"  Language: {context.language}")
        print(f"  Functions: {len(features.get('functions', []))}")
        print(f"  Classes: {len(features.get('classes', []))}")
        print(f"  Complexity: {features.get('complexity', {}).get('cyclomatic_complexity', 0)}")
    
    return len(semantic_index.contexts) > 0

async def test_enhanced_semantic_search():
    """Test enhanced semantic search"""
    print("\nTesting enhanced semantic search...")
    
    flow = create_ai_ide_flow()
    
    test_queries = [
        "function that handles file operations",
        "class for managing data",
        "error handling code",
        "async programming patterns",
        "test functions"
    ]
    
    results = []
    
    for query in test_queries:
        print(f"\nSearching for: '{query}'")
        
        task = {
            "id": f"test_search_{query.replace(' ', '_')}",
            "type": "semantic_search",
            "input": {
                "query": query,
                "options": {"maxResults": 5}
            },
            "context": {
                "working_dir": str(Path.cwd())
            }
        }
        
        start_time = time.time()
        result = flow.execute_task(task)
        search_time = time.time() - start_time
        
        if result['success']:
            history = result.get('history', [])
            for entry in history:
                if entry.get('tool') == 'semantic_search':
                    search_result = entry.get('result', {})
                    matches = search_result.get('matches', [])
                    
                    print(f"  Found {len(matches)} matches in {search_time:.3f}s")
                    for match in matches[:3]:  # Show top 3
                        print(f"    - {match.get('file', 'unknown')}:{match.get('line', 0)} "
                              f"(score: {match.get('semantic_score', 0):.2f}) "
                              f"- {match.get('content', '')[:60]}...")
                    
                    results.append({
                        'query': query,
                        'matches': len(matches),
                        'time': search_time
                    })
                    break
        else:
            print(f"  Search failed: {result.get('error', 'Unknown error')}")
    
    return len(results) > 0

async def test_dynamic_flow_generation():
    """Test dynamic flow generation"""
    print("\nTesting dynamic flow generation...")
    
    flow = create_ai_ide_flow()
    
    test_cases = [
        {
            "query": "Create a simple function to add two numbers",
            "expected_complexity": "low",
            "type": "code_generation"
        },
        {
            "query": "Design a complex microservices architecture with multiple databases and caching layers",
            "expected_complexity": "high",
            "type": "code_generation"
        },
        {
            "query": "Find all authentication related functions in the codebase",
            "expected_complexity": "medium",
            "type": "semantic_search"
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\nTesting: '{test_case['query'][:50]}...'")
        
        # Test complexity assessment
        complexity = flow._assess_task_complexity(test_case['query'])
        print(f"  Assessed complexity: {complexity} (expected: {test_case['expected_complexity']})")
        
        # Test flow generation
        flow_sequence = flow.generate_dynamic_flow(test_case['type'], complexity)
        print(f"  Generated flow: {' -> '.join(flow_sequence)}")
        
        # Execute the task
        task = {
            "id": f"test_dynamic_{len(results)}",
            "type": test_case['type'],
            "input": {"prompt": test_case['query']},
            "context": {"working_dir": str(Path.cwd())}
        }
        
        start_time = time.time()
        result = flow.execute_task(task)
        execution_time = time.time() - start_time
        
        print(f"  Execution time: {execution_time:.3f}s")
        print(f"  Success: {result['success']}")
        print(f"  Nodes executed: {result.get('metrics', {}).get('nodes_executed', [])}")
        
        results.append({
            'complexity_match': complexity == test_case['expected_complexity'],
            'execution_success': result['success'],
            'execution_time': execution_time
        })
    
    return all(r['execution_success'] for r in results)

async def test_performance_tracking():
    """Test performance tracking"""
    print("\nTesting performance tracking...")
    
    tracker = get_performance_tracker()
    
    # Simulate some operations
    for i in range(5):
        tracker.record_search_time(0.1 + i * 0.05)
        tracker.record_index_time(0.5 + i * 0.1)
        if i % 2 == 0:
            tracker.record_cache_hit()
        else:
            tracker.record_cache_miss()
    
    stats = tracker.get_stats()
    
    print(f"Performance Statistics:")
    print(f"  Average search time: {stats['avg_search_time']:.3f}s")
    print(f"  Average index time: {stats['avg_index_time']:.3f}s")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"  Total searches: {stats['total_searches']}")
    print(f"  Total indexes: {stats['total_indexes']}")
    
    return stats['total_searches'] > 0

async def main():
    """Run all semantic enhancement tests"""
    print("AI IDE Semantic Enhancement Test Suite")
    print("=" * 50)
    
    tests = [
        ("Semantic Indexing", test_semantic_indexing),
        ("Enhanced Semantic Search", test_enhanced_semantic_search),
        ("Dynamic Flow Generation", test_dynamic_flow_generation),
        ("Performance Tracking", test_performance_tracking)
    ]
    
    results = []
    total_start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            start_time = time.time()
            success = await test_func()
            test_time = time.time() - start_time
            
            results.append((test_name, success, test_time))
            print(f"\n{test_name}: {'PASS' if success else 'FAIL'} ({test_time:.2f}s)")
            
        except Exception as e:
            print(f"\n{test_name}: FAIL - Exception: {e}")
            results.append((test_name, False, 0))
    
    total_time = time.time() - total_start_time
    
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, success, test_time in results:
        status = "PASS" if success else "FAIL"
        print(f"  {test_name:<30} {status:>6} ({test_time:.2f}s)")
        if success:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    print(f"Total execution time: {total_time:.2f}s")
    
    # Show final performance stats
    tracker = get_performance_tracker()
    final_stats = tracker.get_stats()
    if final_stats['total_searches'] > 0:
        print(f"\nFinal Performance Metrics:")
        print(f"  Average search time: {final_stats['avg_search_time']:.3f}s")
        print(f"  Cache hit rate: {final_stats['cache_hit_rate']:.2%}")

if __name__ == "__main__":
    asyncio.run(main())