"""
PocketFlow Integration for AI IDE
Enhanced version of the existing PocketFlow implementation with AI IDE specific features
"""

import sys
import os
import json
import yaml
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Add the parent directory to the path to import existing utilities
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Import semantic engine
from semantic_engine import get_semantic_index, get_performance_tracker

# Import existing PocketFlow and utilities
try:
    from pocketflow import Node, Flow, BatchNode
    from utils.call_llm import call_llm
    from utils.read_file import read_file
    from utils.delete_file import delete_file
    from utils.replace_file import replace_file
    from utils.search_ops import grep_search
    from utils.dir_ops import list_dir
    from utils.insert_file import insert_file
    from utils.remove_file import remove_file
except ImportError as e:
    logging.error(f"Failed to import existing utilities: {e}")
    # Create mock implementations for development
    class Node:
        def prep(self, shared): pass
        def exec(self, inputs): pass
        def post(self, shared, prep_res, exec_res): pass
    
    class BatchNode(Node):
        pass
    
    class Flow:
        def __init__(self, shared): pass
        def run(self): pass
    
    def call_llm(prompt, **kwargs): return "Mock LLM response"
    def read_file(path): return ("Mock file content", True)
    def grep_search(**kwargs): return (True, [])
    def delete_file(path): return (True, "File deleted")
    def replace_file(path, content): return (True, "File replaced")
    def list_dir(path): return (True, "Directory listing")
    def insert_file(path, content): return (True, "Content inserted")
    def remove_file(path): return (True, "File removed")

logger = logging.getLogger('pocketflow_integration')

class CrossLanguageErrorHandler:
    """Handles errors and communication between TypeScript and Python"""
    
    def __init__(self):
        self.error_counts = {}
        self.last_errors = []
        self.max_error_history = 50
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with context for debugging"""
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Track error frequency
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
        
        # Store error details
        error_details = {
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "message": error_msg,
            "context": context,
            "count": self.error_counts[error_type]
        }
        
        self.last_errors.append(error_details)
        if len(self.last_errors) > self.max_error_history:
            self.last_errors.pop(0)
        
        # Log with appropriate level
        if self.error_counts[error_type] > 5:
            logger.critical(f"Frequent error ({self.error_counts[error_type]}x): {error_type} in {context}: {error_msg}")
        else:
            logger.error(f"{error_type} in {context}: {error_msg}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors for debugging"""
        return {
            "error_counts": self.error_counts,
            "recent_errors": self.last_errors[-10:],  # Last 10 errors
            "total_errors": len(self.last_errors)
        }
    
    def should_retry(self, error: Exception) -> bool:
        """Determine if an operation should be retried"""
        error_type = type(error).__name__
        
        # Don't retry certain types of errors
        non_retryable = [
            'ValueError', 'TypeError', 'AttributeError',
            'FileNotFoundError', 'PermissionError'
        ]
        
        if error_type in non_retryable:
            return False
        
        # Don't retry if we've seen this error too many times
        if self.error_counts.get(error_type, 0) > 3:
            return False
        
        return True

# Global error handler instance
error_handler = CrossLanguageErrorHandler()

class AIIDESharedMemory:
    """Enhanced shared memory for AI IDE with semantic awareness"""
    
    def __init__(self, working_dir: str = None):
        self.data = {
            # Core PocketFlow compatibility
            "user_query": "",
            "working_dir": working_dir or os.getcwd(),
            "history": [],
            "edit_operations": [],
            "response": "",
            
            # AI IDE enhancements
            "semantic_context": {},
            "active_agents": [],
            "reasoning_trace": [],
            "context_windows": [],
            "performance_metrics": {},
            "user_preferences": {},
            "model_improvements": []
        }
    
    def get(self, key: str, default=None):
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any):
        self.data[key] = value
    
    def update(self, updates: Dict[str, Any]):
        self.data.update(updates)
    
    def to_dict(self) -> Dict[str, Any]:
        return self.data.copy()

class SemanticRouter:
    """Routes tasks based on semantic understanding"""
    
    def __init__(self):
        self.routing_rules = {
            'code_generation': ['write', 'create', 'implement', 'generate', 'build'],
            'code_analysis': ['analyze', 'review', 'check', 'examine', 'inspect'],
            'semantic_search': ['find', 'search', 'locate', 'discover', 'lookup'],
            'reasoning': ['explain', 'reason', 'think', 'solve', 'understand'],
            'refactoring': ['refactor', 'improve', 'optimize', 'restructure', 'clean']
        }
    
    def route_task(self, query: str) -> str:
        """Determine the best task type based on query content"""
        query_lower = query.lower()
        
        scores = {}
        for task_type, keywords in self.routing_rules.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                scores[task_type] = score
        
        if scores:
            return max(scores, key=scores.get)
        
        return 'code_generation'  # Default fallback

class EnhancedMainDecisionAgent(Node):
    """Enhanced version of MainDecisionAgent with semantic routing"""
    
    def __init__(self):
        self.semantic_router = SemanticRouter()
    
    def prep(self, shared: AIIDESharedMemory) -> Tuple[str, List[Dict[str, Any]], str]:
        user_query = shared.get("user_query", "")
        history = shared.get("history", [])
        
        # Determine task type using semantic routing
        task_type = self.semantic_router.route_task(user_query)
        
        return user_query, history, task_type
    
    def exec(self, inputs: Tuple[str, List[Dict[str, Any]], str]) -> Dict[str, Any]:
        user_query, history, task_type = inputs
        
        logger.info(f"Processing query: {user_query[:100]}... (Type: {task_type})")
        
        # Enhanced context for AI IDE
        context = {
            "task_type": task_type,
            "history_length": len(history),
            "timestamp": datetime.now().isoformat()
        }
        
        # Create enhanced prompt with semantic context
        prompt = self._create_enhanced_prompt(user_query, history, task_type, context)
        
        # Use the existing call_llm with intelligent model selection
        response = call_llm(prompt, model_name="reasoning-plus")
        
        # Parse response (simplified for now)
        decision = self._parse_llm_response(response, task_type)
        
        return decision
    
    def post(self, shared: AIIDESharedMemory, prep_res: Any, exec_res: Dict[str, Any]) -> str:
        # Add to history with enhanced metadata
        history_entry = {
            "tool": exec_res["tool"],
            "reason": exec_res["reason"],
            "params": exec_res.get("params", {}),
            "result": None,
            "timestamp": datetime.now().isoformat(),
            "task_type": prep_res[2],  # task_type from prep
            "semantic_context": shared.get("semantic_context", {})
        }
        
        history = shared.get("history", [])
        history.append(history_entry)
        shared.set("history", history)
        
        return exec_res["tool"]
    
    def _create_enhanced_prompt(self, query: str, history: List, task_type: str, context: Dict) -> str:
        """Create an enhanced prompt with semantic context"""
        
        # Format history summary for context
        history_str = self._format_history_summary(history)
        
        base_prompt = f"""You are an advanced AI coding assistant with semantic understanding capabilities.

Task Type: {task_type}
User Query: {query}

Context:
- Working in AI IDE environment
- Task automatically classified as: {task_type}
- History entries: {len(history)}

Previous Actions:
{history_str}

Available tools:
1. read_file - Read content from a file
   - Parameters: target_file (path)
   
2. edit_file - Make changes to a file with AI assistance
   - Parameters: target_file (path), instructions, code_edit
   
3. delete_file - Remove a file
   - Parameters: target_file (path)
   
4. grep_search - Search for patterns in files
   - Parameters: query, case_sensitive (optional), include_pattern (optional), exclude_pattern (optional)
   
5. list_dir - List contents of a directory
   - Parameters: relative_workspace_path
   
6. semantic_search - Search code using semantic similarity
   - Parameters: query, max_results (optional)
   
7. reasoning_task - Perform complex reasoning
   - Parameters: problem, mode (basic/chain-of-thought/deep)
   
8. finish - End the process and provide final response
   - No parameters required

DECISION RULES:
- If user asks for CODE/FUNCTIONS/IMPLEMENTATIONS: Use edit_file to CREATE new code
- If user asks to FIND/SEARCH existing code: Use semantic_search or grep_search
- If user asks to READ/VIEW existing files: Use read_file
- If user asks to see directory contents: Use list_dir
- If user asks to delete files: Use delete_file
- If user asks complex questions requiring reasoning: Use reasoning_task

Respond with a YAML object containing:
```yaml
tool: selected_tool_name
reason: |
  detailed explanation of why you chose this tool and what you intend to do
params:
  param1: value1
  param2: value2
```

If you believe no more actions are needed, use "finish" as the tool."""
        
        return base_prompt
    
    def _format_history_summary(self, history: List[Dict[str, Any]]) -> str:
        """Format history summary for context"""
        if not history:
            return "No previous actions."

        history_str = "\n"
        for i, action in enumerate(history):
            history_str += f"Action {i+1}:\n"
            history_str += f"- Tool: {action['tool']}\n"
            history_str += f"- Reason: {action['reason']}\n"
            
            result = action.get("result")
            if result:
                success = result.get("success", False)
                history_str += f"- Result: {'Success' if success else 'Failed'}\n"
            
            history_str += "\n"
        
        return history_str
    
    def _parse_llm_response(self, response: str, task_type: str) -> Dict[str, Any]:
        """Parse LLM response with YAML and JSON fallback logic"""
        
        # Try to extract YAML first
        yaml_content = ""
        if "```yaml" in response:
            yaml_blocks = response.split("```yaml")
            if len(yaml_blocks) > 1:
                yaml_content = yaml_blocks[1].split("```")[0].strip()
        elif "```yml" in response:
            yaml_blocks = response.split("```yml")
            if len(yaml_blocks) > 1:
                yaml_content = yaml_blocks[1].split("```")[0].strip()
        elif "```" in response:
            yaml_blocks = response.split("```")
            if len(yaml_blocks) > 1:
                yaml_content = yaml_blocks[1].strip()
        
        if yaml_content:
            try:
                decision = yaml.safe_load(yaml_content)
                if isinstance(decision, dict) and "tool" in decision:
                    # Ensure params exist
                    if "params" not in decision:
                        decision["params"] = {}
                    return decision
            except yaml.YAMLError as e:
                logger.warning(f"YAML parsing failed: {e}")
        
        # Try JSON fallback
        try:
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
                decision = json.loads(json_str)
                if "params" not in decision:
                    decision["params"] = {}
                return decision
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback based on task type
        fallback_decisions = {
            'code_generation': {
                "tool": "edit_file",
                "reason": "Generating code based on user request",
                "params": {
                    "target_file": "generated_code.py", 
                    "instructions": "Generate code as requested",
                    "code_edit": "# Generated code will be placed here"
                }
            },
            'semantic_search': {
                "tool": "semantic_search", 
                "reason": "Searching for relevant code",
                "params": {"query": "search query", "max_results": 10}
            },
            'reasoning': {
                "tool": "reasoning_task",
                "reason": "Performing complex reasoning",
                "params": {"problem": "reasoning problem", "mode": "basic"}
            },
            'code_analysis': {
                "tool": "semantic_search",
                "reason": "Analyzing code structure",
                "params": {"query": "code analysis", "max_results": 5}
            }
        }
        
        return fallback_decisions.get(task_type, fallback_decisions['code_generation'])

class SemanticSearchNode(Node):
    """Enhanced semantic search with context awareness"""
    
    def __init__(self):
        self.semantic_index = None
        self.performance_tracker = get_performance_tracker()
    
    def prep(self, shared: AIIDESharedMemory) -> Dict[str, Any]:
        history = shared.get("history", [])
        last_action = history[-1] if history else {}
        
        params = last_action.get("params", {})
        working_dir = shared.get("working_dir", "")
        
        # Initialize semantic index if not already done
        if self.semantic_index is None:
            self.semantic_index = get_semantic_index(working_dir)
        
        return {
            "query": params.get("query", ""),
            "working_dir": working_dir,
            "semantic_context": shared.get("semantic_context", {}),
            "context_windows": shared.get("context_windows", []),
            "max_results": params.get("max_results", 10)
        }
    
    def exec(self, params: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
        query = params["query"]
        working_dir = params["working_dir"]
        max_results = params.get("max_results", 10)
        
        logger.info(f"Performing enhanced semantic search for: {query}")
        
        start_time = datetime.now()
        
        try:
            # Use enhanced semantic search
            if self.semantic_index:
                search_results = self.semantic_index.search_semantic(query, max_results)
                
                # Convert to expected format
                enhanced_matches = []
                for result in search_results:
                    context = result['context']
                    for match in result['matches']:
                        enhanced_matches.append({
                            'file': result['file'],
                            'line': match['line'],
                            'content': match['content'],
                            'semantic_score': result['score'],
                            'match_type': match['type'],
                            'language': context.language,
                            'similarity': min(result['score'] / 10.0, 1.0)  # Normalize
                        })
                
                # Record performance
                duration = (datetime.now() - start_time).total_seconds()
                self.performance_tracker.record_search_time(duration)
                
                if enhanced_matches:
                    return True, enhanced_matches
            
            # Fallback to grep search with semantic enhancement
            success, matches = grep_search(
                query=query,
                working_dir=working_dir,
                include_pattern="*.py,*.js,*.ts,*.java,*.cpp,*.c,*.h"
            )
            
            if success:
                # Enhance matches with semantic scoring
                enhanced_matches = []
                for match in matches:
                    enhanced_match = match.copy()
                    enhanced_match['semantic_score'] = self._calculate_semantic_score(
                        query, match.get('content', '')
                    )
                    enhanced_match['match_type'] = 'grep_enhanced'
                    enhanced_matches.append(enhanced_match)
                
                # Sort by semantic score
                enhanced_matches.sort(key=lambda x: x.get('semantic_score', 0), reverse=True)
                return True, enhanced_matches[:max_results]
            
            return False, []
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return False, []
    
    def post(self, shared: AIIDESharedMemory, prep_res: Dict, exec_res: Tuple[bool, List]) -> str:
        success, matches = exec_res
        
        # Update history
        history = shared.get("history", [])
        if history:
            history[-1]["result"] = {
                "success": success,
                "matches": matches,
                "semantic_enhanced": True
            }
            shared.set("history", history)
        
        return "decide_next"
    
    def _calculate_semantic_score(self, query: str, content: str) -> float:
        """Calculate semantic similarity score (simplified implementation)"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        intersection = query_words.intersection(content_words)
        return len(intersection) / len(query_words)

class ReasoningNode(Node):
    """Enhanced reasoning node with trace generation"""
    
    def prep(self, shared: AIIDESharedMemory) -> Dict[str, Any]:
        history = shared.get("history", [])
        last_action = history[-1] if history else {}
        
        params = last_action.get("params", {})
        
        return {
            "problem": params.get("problem", ""),
            "mode": params.get("mode", "basic"),
            "context": shared.get("semantic_context", {})
        }
    
    def exec(self, params: Dict[str, Any]) -> Dict[str, Any]:
        problem = params["problem"]
        mode = params["mode"]
        
        logger.info(f"Performing {mode} reasoning for: {problem[:100]}...")
        
        # Create reasoning prompt based on mode
        if mode == "chain-of-thought":
            prompt = f"""Think step by step to solve this problem:

Problem: {problem}

Please provide:
1. Your reasoning process step by step
2. The final solution
3. Confidence level (0-1)

Format your response as:
Reasoning:
1. [Step 1]
2. [Step 2]
...

Solution: [Your solution]
Confidence: [0.0-1.0]"""
        
        elif mode == "deep":
            prompt = f"""Perform deep analysis of this problem:

Problem: {problem}

Consider:
- Multiple approaches
- Edge cases
- Potential issues
- Alternative solutions

Provide detailed reasoning and the best solution."""
        
        else:  # basic
            prompt = f"Solve this problem: {problem}"
        
        # Use reasoning model for complex thinking
        response = call_llm(prompt, model_name="reasoning-plus")
        
        # Parse reasoning trace
        reasoning_steps = self._extract_reasoning_steps(response)
        solution = self._extract_solution(response)
        confidence = self._extract_confidence(response)
        
        return {
            "solution": solution,
            "reasoning": reasoning_steps,
            "confidence": confidence,
            "mode": mode,
            "full_response": response
        }
    
    def post(self, shared: AIIDESharedMemory, prep_res: Dict, exec_res: Dict) -> str:
        # Add reasoning trace to shared memory
        reasoning_trace = {
            "id": f"trace_{datetime.now().timestamp()}",
            "problem": prep_res["problem"],
            "mode": prep_res["mode"],
            "steps": exec_res["reasoning"],
            "solution": exec_res["solution"],
            "confidence": exec_res["confidence"],
            "timestamp": datetime.now().isoformat()
        }
        
        traces = shared.get("reasoning_trace", [])
        traces.append(reasoning_trace)
        shared.set("reasoning_trace", traces)
        
        # Update history
        history = shared.get("history", [])
        if history:
            history[-1]["result"] = exec_res
            shared.set("history", history)
        
        return "decide_next"
    
    def _extract_reasoning_steps(self, response: str) -> List[str]:
        """Extract reasoning steps from response"""
        steps = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '•')):
                steps.append(line)
        
        return steps if steps else ["Reasoning step extracted from response"]
    
    def _extract_solution(self, response: str) -> str:
        """Extract solution from response"""
        if "Solution:" in response:
            parts = response.split("Solution:")
            if len(parts) > 1:
                return parts[1].split("Confidence:")[0].strip()
        
        # Fallback: return last paragraph
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        return paragraphs[-1] if paragraphs else response[:200]
    
    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from response"""
        if "Confidence:" in response:
            try:
                conf_part = response.split("Confidence:")[1].strip()
                conf_str = conf_part.split()[0]
                return float(conf_str)
            except (IndexError, ValueError):
                pass
        
        return 0.8  # Default confidence

class ReadFileActionNode(Node):
    """Ported ReadFileAction from original PocketFlow"""
    
    def prep(self, shared: AIIDESharedMemory) -> str:
        history = shared.get("history", [])
        if not history:
            raise ValueError("No history found")

        last_action = history[-1]
        file_path = last_action["params"].get("target_file")

        if not file_path:
            raise ValueError("Missing target_file parameter")

        working_dir = shared.get("working_dir", "")
        full_path = os.path.join(working_dir, file_path) if working_dir else file_path

        reason = last_action.get("reason", "No reason provided")
        logger.info(f"ReadFileAction: {reason}")

        return full_path

    def exec(self, file_path: str) -> Tuple[str, bool]:
        return read_file(file_path)

    def post(self, shared: AIIDESharedMemory, prep_res: str, exec_res: Tuple[str, bool]) -> str:
        content, success = exec_res

        history = shared.get("history", [])
        if history:
            history[-1]["result"] = {
                "success": success,
                "content": content
            }
            shared.set("history", history)

        return "decide_next"

class GrepSearchActionNode(Node):
    """Ported GrepSearchAction from original PocketFlow"""
    
    def prep(self, shared: AIIDESharedMemory) -> Dict[str, Any]:
        history = shared.get("history", [])
        if not history:
            raise ValueError("No history found")

        last_action = history[-1]
        params = last_action["params"]

        if "query" not in params:
            raise ValueError("Missing query parameter")

        reason = last_action.get("reason", "No reason provided")
        logger.info(f"GrepSearchAction: {reason}")

        working_dir = shared.get("working_dir", "")

        return {
            "query": params["query"],
            "case_sensitive": params.get("case_sensitive", False),
            "include_pattern": params.get("include_pattern"),
            "exclude_pattern": params.get("exclude_pattern"),
            "working_dir": working_dir
        }

    def exec(self, params: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
        working_dir = params.pop("working_dir", "")

        return grep_search(
            query=params["query"],
            case_sensitive=params.get("case_sensitive", False),
            include_pattern=params.get("include_pattern"),
            exclude_pattern=params.get("exclude_pattern"),
            working_dir=working_dir
        )

    def post(self, shared: AIIDESharedMemory, prep_res: Dict[str, Any], exec_res: Tuple[bool, List[Dict[str, Any]]]) -> str:
        success, matches = exec_res  # Fixed order to match grep_search return

        history = shared.get("history", [])
        if history:
            history[-1]["result"] = {
                "success": success,
                "matches": matches
            }
            shared.set("history", history)

        return "decide_next"

class ListDirActionNode(Node):
    """Ported ListDirAction from original PocketFlow"""
    
    def prep(self, shared: AIIDESharedMemory) -> str:
        history = shared.get("history", [])
        if not history:
            raise ValueError("No history found")

        last_action = history[-1]
        path = last_action["params"].get("relative_workspace_path", ".")

        reason = last_action.get("reason", "No reason provided")
        logger.info(f"ListDirAction: {reason}")

        working_dir = shared.get("working_dir", "")
        full_path = os.path.join(working_dir, path) if working_dir else path

        return full_path

    def exec(self, path: str) -> Tuple[bool, str]:
        success, tree_str = list_dir(path)
        return success, tree_str

    def post(self, shared: AIIDESharedMemory, prep_res: str, exec_res: Tuple[bool, str]) -> str:
        success, tree_str = exec_res

        history = shared.get("history", [])
        if history:
            history[-1]["result"] = {
                "success": success,
                "tree_visualization": tree_str
            }
            shared.set("history", history)

        return "decide_next"

class DeleteFileActionNode(Node):
    """Ported DeleteFileAction from original PocketFlow"""
    
    def prep(self, shared: AIIDESharedMemory) -> str:
        history = shared.get("history", [])
        if not history:
            raise ValueError("No history found")

        last_action = history[-1]
        file_path = last_action["params"].get("target_file")

        if not file_path:
            raise ValueError("Missing target_file parameter")

        reason = last_action.get("reason", "No reason provided")
        logger.info(f"DeleteFileAction: {reason}")

        working_dir = shared.get("working_dir", "")
        full_path = os.path.join(working_dir, file_path) if working_dir else file_path

        return full_path

    def exec(self, file_path: str) -> Tuple[bool, str]:
        return delete_file(file_path)

    def post(self, shared: AIIDESharedMemory, prep_res: str, exec_res: Tuple[bool, str]) -> str:
        success, message = exec_res

        history = shared.get("history", [])
        if history:
            history[-1]["result"] = {
                "success": success,
                "message": message
            }
            shared.set("history", history)

        return "decide_next"

class EditFileActionNode(Node):
    """Enhanced edit file action with semantic awareness"""
    
    def prep(self, shared: AIIDESharedMemory) -> Dict[str, Any]:
        history = shared.get("history", [])
        if not history:
            raise ValueError("No history found")

        last_action = history[-1]
        params = last_action["params"]
        
        file_path = params.get("target_file")
        instructions = params.get("instructions")
        code_edit = params.get("code_edit")

        if not file_path:
            raise ValueError("Missing target_file parameter")
        if not instructions:
            raise ValueError("Missing instructions parameter")
        if not code_edit:
            raise ValueError("Missing code_edit parameter")

        working_dir = shared.get("working_dir", "")
        full_path = os.path.join(working_dir, file_path) if working_dir else file_path

        # Read current file content
        current_content, file_exists = read_file(full_path)
        if not file_exists:
            current_content = ""

        return {
            "file_path": full_path,
            "current_content": current_content,
            "instructions": instructions,
            "code_edit": code_edit,
            "file_exists": file_exists
        }

    def exec(self, params: Dict[str, Any]) -> Dict[str, Any]:
        file_path = params["file_path"]
        current_content = params["current_content"]
        instructions = params["instructions"]
        code_edit = params["code_edit"]
        file_exists = params["file_exists"]

        logger.info(f"EditFileAction: Processing {file_path}")

        # Use LLM to analyze and plan the edit
        edit_prompt = f"""
As a code editing assistant, convert this edit instruction into the final file content.

CURRENT FILE CONTENT:
{current_content}

EDIT INSTRUCTIONS:
{instructions}

CODE EDIT PATTERN:
{code_edit}

Generate the complete updated file content. If the file doesn't exist, create new content.
Respond with only the final file content, no explanations.
"""

        try:
            # Use reasoning model for code editing
            updated_content = call_llm(edit_prompt, model="reasoning")
            
            # Clean up the response to extract just the code
            if "```" in updated_content:
                # Extract code from markdown blocks
                parts = updated_content.split("```")
                for i, part in enumerate(parts):
                    if i % 2 == 1:  # Odd indices are code blocks
                        # Skip language identifier line
                        lines = part.strip().split('\n')
                        if lines and not lines[0].strip().startswith(('def ', 'class ', 'import ', 'from ')):
                            lines = lines[1:]  # Remove language identifier
                        updated_content = '\n'.join(lines)
                        break
            
            # Write the updated content
            success, message = replace_file(file_path, updated_content)
            
            return {
                "success": success,
                "message": message,
                "updated_content": updated_content,
                "operations": 1,
                "reasoning": f"Applied edit: {instructions}"
            }
            
        except Exception as e:
            logger.error(f"Edit file failed: {e}")
            return {
                "success": False,
                "message": f"Edit failed: {e}",
                "operations": 0,
                "reasoning": f"Failed to apply edit: {e}"
            }

    def post(self, shared: AIIDESharedMemory, prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        history = shared.get("history", [])
        if history:
            history[-1]["result"] = exec_res
            shared.set("history", history)

        return "decide_next"

class AIIDEFlow(Flow):
    """Enhanced Flow for AI IDE with semantic routing and agent coordination"""
    
    def __init__(self, working_dir: str = None):
        self.shared = AIIDESharedMemory(working_dir)
        self.semantic_index = get_semantic_index(working_dir)
        self.performance_tracker = get_performance_tracker()
        
        self.nodes = {
            "main_decision": EnhancedMainDecisionAgent(),
            "semantic_search": SemanticSearchNode(),
            "reasoning_task": ReasoningNode(),
            "read_file": ReadFileActionNode(),
            "grep_search": GrepSearchActionNode(),
            "list_dir": ListDirActionNode(),
            "delete_file": DeleteFileActionNode(),
            "edit_file": EditFileActionNode(),
        }
        
        # Performance tracking
        self.execution_metrics = {
            "start_time": None,
            "end_time": None,
            "nodes_executed": [],
            "total_llm_calls": 0,
            "semantic_operations": 0
        }
        
        # Dynamic flow generation with ported nodes
        self.flow_patterns = {
            'code_generation': ['main_decision', 'edit_file', 'done'],
            'semantic_search': ['main_decision', 'semantic_search', 'done'],
            'code_analysis': ['main_decision', 'semantic_search', 'reasoning_task', 'done'],
            'refactoring': ['main_decision', 'read_file', 'reasoning_task', 'edit_file', 'done'],
            'file_operations': ['main_decision', 'read_file', 'done'],
            'directory_listing': ['main_decision', 'list_dir', 'done'],
            'file_search': ['main_decision', 'grep_search', 'done']
        }
    
    def generate_dynamic_flow(self, task_type: str, complexity: str = "medium") -> List[str]:
        """Generate dynamic flow based on task type and complexity"""
        base_flow = self.flow_patterns.get(task_type, ['main_decision', 'done'])
        
        # Enhance flow based on complexity
        if complexity == "high":
            # Add additional reasoning and validation steps
            enhanced_flow = []
            for node in base_flow:
                enhanced_flow.append(node)
                if node == 'reasoning_task':
                    enhanced_flow.append('validation_step')  # Would need to implement
                elif node == 'semantic_search':
                    enhanced_flow.append('context_enrichment')  # Would need to implement
            return enhanced_flow
        elif complexity == "low":
            # Simplify flow
            return [node for node in base_flow if node not in ['reasoning_task']]
        
        return base_flow
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with enhanced tracking and error handling"""
        self.execution_metrics["start_time"] = datetime.now()
        task_id = task.get("id", "unknown")
        
        try:
            logger.info(f"Starting task execution: {task_id}")
            
            # Set up shared memory with enhanced context
            user_query = task.get("input", {}).get("prompt", "")
            if not user_query:
                user_query = task.get("input", {}).get("query", "")
            
            self.shared.set("user_query", user_query)
            self.shared.set("task_id", task_id)
            self.shared.update(task.get("context", {}))
            
            # Determine task complexity for dynamic flow generation
            task_type = task.get("type", "code_generation")
            complexity = self._assess_task_complexity(user_query)
            
            # Generate dynamic flow
            flow_sequence = self.generate_dynamic_flow(task_type, complexity)
            logger.info(f"Generated flow sequence for {task_type}: {flow_sequence}")
            
            # Execute dynamic flow with error handling
            current_node_idx = 0
            max_iterations = len(flow_sequence) * 3  # Allow more flexibility
            iteration = 0
            last_error = None
            
            while current_node_idx < len(flow_sequence) and iteration < max_iterations:
                current_node = flow_sequence[current_node_idx]
                
                if current_node == "done":
                    logger.info("Flow completed successfully")
                    break
                
                if current_node in self.nodes:
                    try:
                        logger.info(f"Executing node: {current_node} (iteration {iteration})")
                        
                        node = self.nodes[current_node]
                        self.execution_metrics["nodes_executed"].append({
                            "node": current_node,
                            "iteration": iteration,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Execute node with error handling
                        prep_result = node.prep(self.shared)
                        exec_result = node.exec(prep_result)
                        next_node = node.post(self.shared, prep_result, exec_result)
                        
                        # Track operations
                        if current_node == "semantic_search":
                            self.execution_metrics["semantic_operations"] += 1
                        elif current_node in ["edit_file", "read_file", "delete_file"]:
                            self.execution_metrics["file_operations"] = self.execution_metrics.get("file_operations", 0) + 1
                        
                        # Handle node routing
                        if next_node == "decide_next" or next_node is None:
                            current_node_idx += 1
                        elif next_node == "finish":
                            logger.info("Node requested finish")
                            break
                        elif next_node in self.nodes:
                            # Node requested specific next node
                            try:
                                current_node_idx = flow_sequence.index(next_node)
                            except ValueError:
                                # Next node not in sequence, add it
                                flow_sequence.insert(current_node_idx + 1, next_node)
                                current_node_idx += 1
                        else:
                            current_node_idx += 1
                        
                        # Reset error tracking on successful execution
                        last_error = None
                        
                    except Exception as node_error:
                        error_handler.log_error(node_error, f"Node: {current_node}")
                        last_error = node_error
                        
                        # Try to recover or skip node
                        if error_handler.should_retry(node_error) and iteration < max_iterations - 1:
                            logger.warning(f"Retrying node {current_node} after error: {node_error}")
                            # Don't increment node index, retry same node
                        else:
                            logger.error(f"Skipping node {current_node} after error: {node_error}")
                            current_node_idx += 1
                    
                    iteration += 1
                else:
                    logger.warning(f"Unknown node: {current_node}")
                    current_node_idx += 1
                    iteration += 1
            
            self.execution_metrics["end_time"] = datetime.now()
            self.execution_metrics["total_iterations"] = iteration
            
            # Extract final result from history or shared memory
            final_result = self._extract_final_result(task_type)
            
            # Return enhanced result
            result = {
                "success": True,
                "result": final_result,
                "history": self.shared.get("history", []),
                "reasoning_trace": self.shared.get("reasoning_trace", []),
                "metrics": self.execution_metrics,
                "flow_sequence": flow_sequence,
                "semantic_context": self.shared.get("semantic_context", {}),
                "performance_stats": self.performance_tracker.get_stats(),
                "task_id": task_id
            }
            
            logger.info(f"Task {task_id} completed successfully")
            return result
            
        except Exception as e:
            error_handler.log_error(e, f"Task execution: {task_id}")
            self.execution_metrics["end_time"] = datetime.now()
            
            error_result = {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "metrics": self.execution_metrics,
                "task_id": task_id,
                "error_summary": error_handler.get_error_summary()
            }
            
            logger.error(f"Task {task_id} failed: {e}")
            return error_result
    
    def _extract_final_result(self, task_type: str) -> Any:
        """Extract the final result based on task type"""
        history = self.shared.get("history", [])
        
        if not history:
            return "No operations performed"
        
        # Get the last successful operation result
        for entry in reversed(history):
            result = entry.get("result", {})
            if result.get("success"):
                if task_type == "code_generation" and entry.get("tool") == "edit_file":
                    return {
                        "code": result.get("updated_content", ""),
                        "operations": result.get("operations", 0),
                        "reasoning": result.get("reasoning", "")
                    }
                elif task_type == "semantic_search" and entry.get("tool") == "semantic_search":
                    return {
                        "matches": result.get("matches", []),
                        "total": len(result.get("matches", [])),
                        "semantic_enhanced": True
                    }
                elif task_type == "reasoning" and entry.get("tool") == "reasoning_task":
                    return result
                elif entry.get("tool") == "read_file":
                    return {
                        "content": result.get("content", ""),
                        "file_read": True
                    }
                elif entry.get("tool") == "list_dir":
                    return {
                        "tree_visualization": result.get("tree_visualization", ""),
                        "directory_listed": True
                    }
        
        # Fallback to generic result
        return {
            "message": "Task completed",
            "operations": len(history),
            "last_action": history[-1].get("tool", "unknown") if history else "none"
        }
    
    def _assess_task_complexity(self, query: str) -> str:
        """Assess task complexity based on query content"""
        query_lower = query.lower()
        
        # High complexity indicators
        high_complexity_keywords = [
            'architecture', 'design pattern', 'refactor', 'optimize',
            'complex', 'algorithm', 'performance', 'scalability',
            'multiple', 'integrate', 'system'
        ]
        
        # Low complexity indicators
        low_complexity_keywords = [
            'simple', 'basic', 'quick', 'small', 'fix', 'typo',
            'comment', 'rename', 'format'
        ]
        
        high_score = sum(1 for keyword in high_complexity_keywords if keyword in query_lower)
        low_score = sum(1 for keyword in low_complexity_keywords if keyword in query_lower)
        
        if high_score > low_score and high_score >= 2:
            return "high"
        elif low_score > high_score and low_score >= 1:
            return "low"
        else:
            return "medium"

# Factory function for creating flows
def create_ai_ide_flow(working_dir: str = None) -> AIIDEFlow:
    """Create a new AI IDE flow instance"""
    return AIIDEFlow(working_dir)

# Test function
def test_pocketflow_integration():
    """Test the PocketFlow integration"""
    flow = create_ai_ide_flow()
    
    test_task = {
        "id": "test_001",
        "type": "code_generation",
        "input": {
            "prompt": "Create a function to calculate fibonacci numbers"
        },
        "context": {
            "working_dir": os.getcwd()
        }
    }
    
    result = flow.execute_task(test_task)
    print(f"Test result: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    test_pocketflow_integration()