import os
import yaml  # Add YAML support
import re
import logging
from datetime import datetime
from typing: List, Dict, Any, Tuple

# Import utility functions
from utils.call_llm import call_llm
from utils.read_file import read_file
from utils.delete_file import delete_file
from utils.replace_file import replace_file
from utils.search_ops import grep_search
from utils.dir_ops import list_dir

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('coding_agent.log')
    ]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger('coding_agent')


def format_history_summary(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "No previous actions."

    history_str = "\n"

    for i, action in enumerate(history):
        # Header for all entries - removed timestamp
        history_str += f"Action {i+1}:\n"
        history_str += f"- Tool: {action['tool']}\n"
        history_str += f"- Reason: {action['reason']}\n"

        # Add parameters
        params = action.get("params", {})
        if params:
            history_str += f"- Parameters:\n"
            for k, v in params.items():
                history_str += f"  - {k}: {v}\n"

        # Add detailed result information
        result = action.get("result")
        if result:
            if isinstance(result, dict):
                success = result.get("success", False)
                history_str += f"- Result: {'Success' if success else 'Failed'}\n"

                # Add tool-specific details
                if action['tool'] == 'read_file' and success:
                    content = result.get("content", "")
                    # Show full content without truncating
                    history_str += f"- Content: {content}\n"
                elif action['tool'] == 'grep_search' and success:
                    matches = result.get("matches", [])
                    history_str += f"- Matches: {len(matches)}\n"
                    # Show all matches without limiting to first 3
                    for j, match in enumerate(matches):
                        history_str += f"  {j+1}. {match.get('file')}:{match.get('line')}: {match.get('content')}\n"
                elif action['tool'] == 'edit_file' and success:
                    operations = result.get("operations", 0)
                    history_str += f"- Operations: {operations}\n"

                    # Include the reasoning if available
                    reasoning = result.get("reasoning", "")
                    if reasoning:
                        history_str += f"- Reasoning: {reasoning}\n"
                elif action['tool'] == 'list_dir' and success:
                    # Get the tree visualization string
                    tree_visualization = result.get("tree_visualization", "")
                    history_str += "- Directory structure:\n"

                    # Properly handle and format the tree visualization
                    if tree_visualization and isinstance(tree_visualization, str):
                        # First, ensure we handle any special line ending characters properly
                        clean_tree = tree_visualization.replace(
                            '\r\n', '\n').strip()

                        if clean_tree:
                            # Add each line with proper indentation
                            for line in clean_tree.split('\n'):
                                # Ensure the line is properly indented
                                if line.strip():  # Only include non-empty lines
                                    history_str += f"  {line}\n"
                        else:
                            history_str += "  (No tree structure data)\n"
                    else:
                        history_str += "  (Empty or inaccessible directory)\n"
                        logger.debug(
                            f"Tree visualization missing or invalid: {tree_visualization}")
            else:
                history_str += f"- Result: {result}\n"

        # Add separator between actions
        history_str += "\n" if i < len(history) - 1 else ""

    return history_str

#############################################
# Main Decision Agent Node
#############################################


class MainDecisionAgent(Node):
    def prep(self, shared: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
        # Get user query and history
        user_query = shared.get("user_query", "")
        history = shared.get("history", [])

        return user_query, history

    def exec(self, inputs: Tuple[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        user_query, history = inputs
        logger.info(f"MainDecisionAgent: Analyzing user query: {user_query}")

        # Format history using the utility function with 'basic' detail level
        history_str = format_history_summary(history)

        # Create prompt for the LLM using YAML instead of JSON
        prompt = f"""You are a coding assistant that helps modify and navigate code. Given the following request,
decide which tool to use from the available options.

User request: {user_query}

Here are the actions you performed:
{history_str}

IMPORTANT DECISION RULES:
- If the user asks for CODE, FUNCTIONS, or IMPLEMENTATIONS (like "write binary search", "create a function", "implement sorting"): Use edit_file to CREATE new code
- If the user asks to FIND or SEARCH existing code (like "find all functions", "search for imports"): Use grep_search
- If the user asks to READ/view existing files: Use read_file
- If the user asks to see directory contents: Use list_dir
- If the user asks to delete files: Use delete_file

EXAMPLES:
- "write binary search code" 
 edit_file (CREATE new code)
- "implement sorting function" 
 edit_file (CREATE new code)
- "find all logger calls" 
 grep_search (SEARCH existing code)
- "show me the main.py file" 
 read_file (READ existing file)
- "what files are in utils/" 
 list_dir (LIST directory)

Available tools:
1. read_file: Read content from a file
   - Parameters: target_file (path)
   - Example:
     tool: read_file
     reason: I need to read the main.py file to understand its structure
     params:
       target_file: main.py

2. edit_file: Make changes to a file
   - Parameters: target_file (path), instructions, code_edit
   - Code_edit_instructions:
       - The code changes with context, following these rules:
       - Use "// ... existing code ..." to represent unchanged code between edits
       - Include sufficient context around the changes to resolve ambiguity
       - Minimize repeating unchanged code
       - Never omit code without using the "// ... existing code ..." marker
       - No need to specify line numbers - the context helps locate the changes
   - Example:
     tool: edit_file
     reason: I need to add error handling to the file reading function
     params:
       target_file: utils/read_file.py
       instructions: Add try-except block around the file reading operation
       code_edit: |
            // ... existing file reading code ...
            function newEdit() {{
                // new code here
            }}
            // ... existing file reading code ...

3. delete_file: Remove a file
   - Parameters: target_file (path)
   - Example:
     tool: delete_file
     reason: The temporary file is no longer needed
     params:
       target_file: temp.txt

4. grep_search: Search for patterns in files
   - Parameters: query, case_sensitive (optional), include_pattern (optional), exclude_pattern (optional)
   - Example:
     tool: grep_search
     reason: I need to find all occurrences of 'logger' in Python files
     params:
       query: logger
       include_pattern: "*.py"
       case_sensitive: false

5. list_dir: List contents of a directory
   - Parameters: relative_workspace_path
   - Example:
     tool: list_dir
     reason: I need to see all files in the utils directory
     params:
       relative_workspace_path: utils
   - Result: Returns a tree visualization of the directory structure

6. finish: End the process and provide final response
   - No parameters required
   - Example:
     tool: finish
     reason: I have completed the requested task of finding all logger instances
     params: {}

Respond with a YAML object containing:
```yaml
tool: one of: read_file, edit_file, delete_file, grep_search, list_dir, finish
reason: |
  detailed explanation of why you chose this tool and what you intend to do
params:
  # parameters specific to the chosen tool
```

For CODE GENERATION requests (like "write binary search code"):
- Use edit_file to create the code
- If no specific file is mentioned, create it in an appropriate file (like main.py, utils.py, etc.)

Example for code generation:
```yaml
tool: edit_file
reason: |
  The user is asking for binary search code implementation. This is a code generation request,
  so I need to create new code content using edit_file.
params:
  target_file: main.py
  instructions: Add a binary search function implementation
  code_edit: |
    def binary_search(arr, target):
        left, right = 0, len(arr) - 1

        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        return -1
```

Respond with a YAML object containing:
```yaml
tool: one of: read_file, edit_file, delete_file, grep_search, list_dir, finish
reason: |
  detailed explanation of why you chose this tool and what you intend to do
params:
  # parameters specific to the chosen tool
```

If you believe no more actions are needed, use "finish" as the tool and explain why in the reason.
"""
        # Call LLM to decide action - use reasoning-plus for complex decision making
        response = call_llm(prompt, model="reasoning-plus")

        # DEBUG: Log the full response
        print(f"DEBUG: Full LLM Response in MainDecisionAgent:\n{response}")
        print(f"DEBUG: Response length in MainDecisionAgent: {len(response)}")

        
        
        
        
        
        yaml_content = ""
        # Use regex to find the first YAML block
        match = re.search(r"```(?:yaml|yml)?\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            yaml_content = match.group(1).strip()
            print(f"DEBUG: Extracted YAML block using regex in MainDecisionAgent:\n{yaml_content}")
            try:
                test_parse = yaml.safe_load(yaml_content)
                if not (isinstance(test_parse, dict) and "tool" in test_parse and "reason" in test_parse):
                    yaml_content = "" # Reset if not valid or missing keys
            except yaml.YAMLError:
                yaml_content = "" # Reset if parsing fails

        if not yaml_content: # Fallback if no valid YAML block found or keys missing
            print(f"DEBUG: No valid YAML block found in MainDecisionAgent, attempting to parse entire response as YAML.\nResponse: {response.strip()}")
            try:
                test_parse = yaml.safe_load(response.strip())
                if isinstance(test_parse, dict) and "tool" in test_parse and "reason" in test_parse:
                    yaml_content = response.strip()
            except yaml.YAMLError as e:
                print(f"DEBUG: Fallback YAML parsing failed in MainDecisionAgent: {e}")
                pass

        # DEBUG: Log the extracted YAML
        print(f"DEBUG: Extracted YAML:\n{yaml_content}")

        # Clean up the YAML content to remove any trailing text
        if yaml_content:
            # No aggressive cleaning, assume LLM provides clean YAML within code blocks
            pass

        # DEBUG: Log the cleaned YAML
        print(f"DEBUG: Cleaned YAML:\n{yaml_content}")

        if yaml_content:
            try:
                decision = yaml.safe_load(yaml_content)
            except yaml.YAMLError as e:
                logger.error(f"YAML parsing error: {e}")
                logger.error(f"YAML content: {yaml_content}")

                # Try to fix common YAML issues
                try:
                    # Convert JavaScript-style object notation to proper YAML
                    fixed_content = yaml_content


                    # Fix params: { key: "value" } format to proper YAML
                    # Match the pattern and convert to proper YAML indentation
                    def fix_params_block(match):
                        params_content = match.group(1).strip()
                        lines = []

                        # Split by commas and process each key-value pair
                        for pair in params_content.split(','):
                            pair = pair.strip()
                            if ':' in pair:
                                key, value = pair.split(':', 1)
                                key = key.strip()
                                value = value.strip()
                                # Remove quotes if present
                                if (value.startswith('"') and value.endswith('"')) or (value.startswith("'') and value.endswith("'")):
                                    value = value[1:-1]
                                lines.append(f"  {key}: \"{value}\"")

                        return "params:\n" + "\n".join(lines)

                    # Replace the problematic params block
                    fixed_content = re.sub(
                        r'params:\s*{\s*([^}]*)\s*}', fix_params_block, fixed_content, flags=re.MULTILINE | re.DOTALL)

                    print(f"DEBUG: Attempting to fix YAML:\n{fixed_content}")
                    decision = yaml.safe_load(fixed_content)
                except Exception as e2:
                    logger.error(f"Failed to fix YAML: {e2}")
                    raise ValueError(f"Invalid YAML response: {e}")

            # Validate the required fields
            assert "tool" in decision, "Tool name is missing"
            assert "reason" in decision, "Reason is missing"

            # For tools other than "finish", params must be present
            if decision["tool"] != "finish":
                if "params" not in decision:
                    # Add debug info and raise a more helpful error
                    print(f"DEBUG: Decision object: {decision}")
                    print(f"DEBUG: Tool: {decision.get('tool')}")
                    print(f"DEBUG: Available keys: {list(decision.keys())}")
                    raise ValueError(
                        f"Parameters are missing for tool '{decision['tool']}'. The LLM response must include a 'params' section.")

                # Special validation for edit_file tool
                if decision["tool"] == "edit_file":
                    params = decision["params"]
                    assert "target_file" in params, "target_file parameter is missing for edit_file"
                    assert "instructions" in params, "instructions parameter is missing for edit_file"
                    assert "code_edit" in params, "code_edit parameter is missing for edit_file"

                    # Ensure code_edit is not empty
                    if not params["code_edit"] or not params["code_edit"].strip():
                        raise ValueError(
                            "code_edit parameter cannot be empty for edit_file")
            else:
                decision["params"] = {}

            return decision
        else:
            raise ValueError("No YAML object found in response")

    def post(self, shared: Dict[str, Any], prep_res: Any, exec_res: Dict[str, Any]) -> str:
        logger.info(f"MainDecisionAgent: Selected tool: {exec_res['tool']}")

        # Initialize history if not present
        if "history" not in shared:
            shared["history"] = []

        # Add this action to history
        shared["history"].append({
            "tool": exec_res["tool"],
            "reason": exec_res["reason"],
            "params": exec_res.get("params", {}),
            "result": None,  # Will be filled in by action nodes
            "timestamp": datetime.now().isoformat()
        })

        # Return the action to take
        return exec_res["tool"]

#############################################
# Read File Action Node
#############################################


class ReadFileAction(Node):
    def prep(self, shared: Dict[str, Any]) -> str:
        # Get parameters from the last history entry
        history = shared.get("history", [])
        if not history:
            raise ValueError("No history found")

        last_action = history[-1]
        file_path = last_action["params"].get("target_file")

        if not file_path:
            raise ValueError("Missing target_file parameter")

        # Ensure path is relative to working directory
        working_dir = shared.get("working_dir", "")
        full_path = os.path.join(
            working_dir, file_path) if working_dir else file_path

        # Use the reason for logging instead of explanation
        reason = last_action.get("reason", "No reason provided")
        logger.info(f"ReadFileAction: {reason}")

        return full_path

    def exec(self, file_path: str) -> Tuple[str, bool]:
        # Call read_file utility which returns a tuple of (content, success)
        return read_file(file_path)

    def post(self, shared: Dict[str, Any], prep_res: str, exec_res: Tuple[str, bool]) -> str:
        # Unpack the tuple returned by read_file()
        content, success = exec_res

        # Update the result in the last history entry
        history = shared.get("history", [])
        if history:
            history[-1]["result"] = {
                "success": success,
                "content": content
            }

#############################################
# Grep Search Action Node
#############################################


class GrepSearchAction(Node):
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        # Get parameters from the last history entry
        history = shared.get("history", [])
        if not history:
            raise ValueError("No history found")

        last_action = history[-1]
        params = last_action["params"]

        if "query" not in params:
            raise ValueError("Missing query parameter")

        # Use the reason for logging instead of explanation
        reason = last_action.get("reason", "No reason provided")
        logger.info(f"GrepSearchAction: {reason}")

        # Ensure paths are relative to working directory
        working_dir = shared.get("working_dir", "")

        return {
            "query": params["query"],
            "case_sensitive": params.get("case_sensitive", False),
            "include_pattern": params.get("include_pattern"),
            "exclude_pattern": params.get("exclude_pattern"),
            "working_dir": working_dir
        }

    def exec(self, params: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
        # Use current directory if not specified
        working_dir = params.pop("working_dir", "")

        # Call grep_search utility which returns (success, matches)
        return grep_search(
            query=params["query"],
            case_sensitive=params.get("case_sensitive", False),
            include_pattern=params.get("include_pattern"),
            exclude_pattern=params.get("exclude_pattern"),
            working_dir=working_dir
        )

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Tuple[bool, List[Dict[str, Any]]]) -> str:
        matches, success = exec_res

        # Update the result in the last history entry
        history = shared.get("history", [])
        if history:
            history[-1]["result"] = {
                "success": success,
                "matches": matches
            }

#############################################
# List Directory Action Node
#############################################


class ListDirAction(Node):
    def prep(self, shared: Dict[str, Any]) -> str:
        # Get parameters from the last history entry
        history = shared.get("history", [])
        if not history:
            raise ValueError("No history found")

        last_action = history[-1]
        path = last_action["params"].get("relative_workspace_path", ".")

        # Use the reason for logging instead of explanation
        reason = last_action.get("reason", "No reason provided")
        logger.info(f"ListDirAction: {reason}")

        # Ensure path is relative to working directory
        working_dir = shared.get("working_dir", "")
        full_path = os.path.join(working_dir, path) if working_dir else path

        return full_path

    def exec(self, path: str) -> Tuple[bool, str]:
        # Call list_dir utility which now returns (success, tree_str)
        success, tree_str = list_dir(path)

        return success, tree_str

    def post(self, shared: Dict[str, Any], prep_res: str, exec_res: Tuple[bool, str]) -> str:
        success, tree_str = exec_res

        # Update the result in the last history entry with the new structure
        history = shared.get("history", [])
        if history:
            history[-1]["result"] = {
                "success": success,
                "tree_visualization": tree_str
            }

#############################################
# Delete File Action Node
#############################################


class DeleteFileAction(Node):
    def prep(self, shared: Dict[str, Any]) -> str:
        # Get parameters from the last history entry
        history = shared.get("history", [])
        if not history:
            raise ValueError("No history found")

        last_action = history[-1]
        file_path = last_action["params"].get("target_file")

        if not file_path:
            raise ValueError("Missing target_file parameter")

        # Use the reason for logging instead of explanation
        reason = last_action.get("reason", "No reason provided")
        logger.info(f"DeleteFileAction: {reason}")

        # Ensure path is relative to working directory
        working_dir = shared.get("working_dir", "")
        full_path = os.path.join(
            working_dir, file_path) if working_dir else file_path

        return full_path

    def exec(self, file_path: str) -> Tuple[bool, str]:
        # Call delete_file utility which returns (success, message)
        return delete_file(file_path)

    def post(self, shared: Dict[str, Any], prep_res: str, exec_res: Tuple[bool, str]) -> str:
        success, message = exec_res

        # Update the result in the last history entry
        history = shared.get("history", [])
        if history:
            history[-1]["result"] = {
                "success": success,
                "message": message
            }

#############################################
# Read Target File Node (Edit Agent)
#############################################


class ReadTargetFileNode(Node):
    def prep(self, shared: Dict[str, Any]) -> str:
        # Get parameters from the last history entry
        history = shared.get("history", [])
        if not history:
            raise ValueError("No history found")

        last_action = history[-1]
        file_path = last_action["params"].get("target_file")

        if not file_path:
            raise ValueError("Missing target_file parameter")

        # Ensure path is relative to working directory
        working_dir = shared.get("working_dir", "")
        full_path = os.path.join(
            working_dir, file_path) if working_dir else file_path

        # Use the reason for logging instead of explanation
        reason = last_action.get("reason", "No reason provided")
        logger.info(f"ReadFileAction: {reason}")

        return full_path

    def exec(self, file_path: str) -> Tuple[str, bool]:
        # Call read_file utility which returns (content, success)
        return read_file(file_path)

    def post(self, shared: Dict[str, Any], prep_res: str, exec_res: Tuple[str, bool]) -> str:
        content, success = exec_res
        logger.info("ReadTargetFileNode: File read completed for editing")

        # Store file content in the history entry
        history = shared.get("history", [])
        if history:
            history[-1]["file_content"] = content

#############################################
# Analyze and Plan Changes Node
#############################################


class AnalyzeAndPlanNode(Node):
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        # Get history
        history = shared.get("history", [])
        if not history:
            raise ValueError("No history found")

        last_action = history[-1]
        file_content = last_action.get("file_content")
        instructions = last_action["params"].get("instructions")
        code_edit = last_action["params"].get("code_edit")

        if not file_content:
            raise ValueError("File content not found")
        if not instructions:
            raise ValueError("Missing instructions parameter")
        if not code_edit:
            raise ValueError("Missing code_edit parameter")

        return {
            "file_content": file_content,
            "instructions": instructions,
            "code_edit": code_edit
        }

    def exec(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        file_content = params["file_content"]
        instructions = params["instructions"]
        code_edit = params["code_edit"]

        # File content as lines
        file_lines = file_content.split('\n')
        total_lines = len(file_lines)

        # Generate a prompt for the LLM to analyze the edit using YAML instead of JSON
        prompt = f"""
Given the following file content, edit instructions, and code edit pattern,
generate a YAML object containing your reasoning and a list of specific edit operations.

FILE CONTENT ({total_lines} lines):
{file_content}

EDIT INSTRUCTIONS:
{instructions}

CODE EDIT PATTERN (markers like "// ... existing code ..." indicate unchanged code):
{code_edit}

Return a YAML object with your reasoning and an array of edit operations.
Each operation must include `start_line` (1-indexed, inclusive), `end_line` (1-indexed, inclusive),
and `replacement` (the new code).
If content should be appended, set `start_line` and `end_line` to `{total_lines + 1}`.

```yaml
reasoning: |
  Explain your interpretation of the edit pattern and how you determined the edit locations.
  Be precise with line numbers.

operations:
  - start_line: 10
    end_line: 15
    replacement: |
      def process_file(filename):
          # New implementation with better error handling
          try:
              with open(filename, 'r') as f:
                  return f.read()
          except FileNotFoundError:
              return None

  - start_line: 25
    end_line: 25
    replacement: |
      logger.info("File processing completed")
```
"""

        # Call LLM to analyze - use reasoning model for code analysis
        response = call_llm(prompt, model="reasoning")

        # DEBUG: Log the raw LLM response for AnalyzeAndPlanNode
        print(f"DEBUG: Raw LLM Response in AnalyzeAndPlanNode: {response}")

        yaml_content = ""
        # Use regex to find all YAML blocks
        yaml_blocks = re.findall(r"```(?:yaml|yml)?\s*(.*?)\s*```", response, re.DOTALL)
        
        for block in yaml_blocks:
            try:
                test_parse = yaml.safe_load(block.strip())
                if isinstance(test_parse, dict) and "reasoning" in test_parse and "operations" in test_parse:
                    yaml_content = block.strip()
                    break
            except yaml.YAMLError:
                continue
        
        if not yaml_content: # Fallback if no code blocks found
            try:
                test_parse = yaml.safe_load(response.strip())
                if isinstance(test_parse, dict) and "reasoning" in test_parse and "operations" in test_parse:
                    yaml_content = response.strip()
            except yaml.YAMLError:
                pass

        if yaml_content:
            decision = yaml.safe_load(yaml_content)

            # Validate the required fields
            assert "reasoning" in decision, "Reasoning is missing"
            assert "operations" in decision, "Operations are missing"

            # Ensure operations is a list
            if not isinstance(decision["operations"], list):
                raise ValueError("Operations are not a list")

            # Validate operations
            for op in decision["operations"]:
                assert "start_line" in op, "start_line is missing"
                assert "end_line" in op, "end_line is missing"
                assert "replacement" in op, "replacement is missing"

                # Fix negative line numbers (common LLM error)
                if op["start_line"] < 1:
                    op["start_line"] = 1
                if op["end_line"] < 1:
                    op["end_line"] = 1

                # Ensure we don't exceed file bounds
                if op["start_line"] > total_lines + 1:
                    op["start_line"] = total_lines + 1
                if op["end_line"] > total_lines + 1:
                    op["end_line"] = total_lines + 1

                # Validate line ranges
                assert 1 <= op["start_line"] <= total_lines +
                    1, f"start_line out of range: {op['start_line']}"
                assert 1 <= op["end_line"] <= total_lines +
                    1, f"end_line out of range: {op['end_line']}"
                assert op["start_line"] <= op[
                    "end_line"], f"start_line > end_line: {op['start_line']} > {op['end_line']}"

            return decision
        else:
            raise ValueError("No YAML object found in response")

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Dict[str, Any]) -> str:
        # Store reasoning and edit operations in shared
        shared["edit_reasoning"] = exec_res.get("reasoning", "")
        shared["edit_operations"] = exec_res.get("operations", [])


#############################################
# Apply Changes Batch Node
#############################################
class ApplyChangesNode(BatchNode):
    def prep(self, shared: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Get edit operations
        edit_operations = shared.get("edit_operations", [])
        if not edit_operations:
            logger.warning("No edit operations found")
            return []

        # Sort edit operations in descending order by start_line
        # This ensures that line numbers remain valid as we edit from bottom to top
        sorted_ops = sorted(
            edit_operations, key=lambda op: op["start_line"], reverse=True)

        # Get target file from history
        history = shared.get("history", [])
        if not history:
            raise ValueError("No history found")

        last_action = history[-1]
        target_file = last_action["params"].get("target_file")

        if not target_file:
            raise ValueError("Missing target_file parameter")

        # Ensure path is relative to working directory
        working_dir = shared.get("working_dir", "")
        full_path = os.path.join(
            working_dir, target_file) if working_dir else target_file

        # Attach file path to each operation
        for op in sorted_ops:
            op["target_file"] = full_path

        return sorted_ops

    def exec(self, op: Dict[str, Any]) -> Tuple[bool, str]:
        # Call replace_file utility which returns (success, message)
        return replace_file(
            target_file=op["target_file"],
            start_line=op["start_line"],
            end_line=op["end_line"],
            content=op["replacement"]
        )

    def post(self, shared: Dict[str, Any], prep_res: List[Dict[str, Any]], exec_res_list: List[Tuple[bool, str]]) -> str:
        # Check if all operations were successful
        all_successful = all(success for success, _ in exec_res_list)

        # Format results for history
        result_details = [
            {"success": success, "message": message}
            for success, message in exec_res_list
        ]

        # Update edit result in history
        history = shared.get("history", [])
        if history:
            history[-1]["result"] = {
                "success": all_successful,
                "operations": len(exec_res_list),
                "details": result_details,
                "reasoning": shared.get("edit_reasoning", "")
            }

        # Clear edit operations and reasoning after processing
        shared.pop("edit_operations", None)
        shared.pop("edit_reasoning", None)


#############################################
# Format Response Node
#############################################
class FormatResponseNode(Node):
    def prep(self, shared: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Get history
        history = shared.get("history", [])

        return history

    def exec(self, history: List[Dict[str, Any]]) -> str:
        # If no history, return a generic message
        if not history:
            return "No actions were performed."

        # Generate a summary of actions for the LLM using the utility function
        actions_summary = format_history_summary(history)

        # Prompt for the LLM to generate the final response
        prompt = f"""
You are a coding assistant. You have just performed a series of actions based on the
user's request. Summarize what you did in a clear, helpful response.

Here are the actions you performed:
{actions_summary}

Generate a comprehensive yet concise response that explains:
1. What actions were taken
2. What was found or modified
3. Any next steps the user might want to take

IMPORTANT:
- Focus on the outcomes and results, not the specific tools used
- Write as if you are directly speaking to the user
- When providing code examples or structured information, use YAML format enclosed in triple backticks
"""

        # Call LLM to generate response
        response = call_llm(prompt)

        return response

    def post(self, shared: Dict[str, Any], prep_res: List[Dict[str, Any]], exec_res: str) -> str:
        logger.info(
            f"###### Final Response Generated ######\n{exec_res}\n###### End of Response ######")

        # Store response in shared
        shared["response"] = exec_res

        return "done"

#############################################
# Edit Agent Flow
#############################################


def create_edit_agent() -> Flow:
    # Create nodes
    read_target = ReadTargetFileNode()
    analyze_plan = AnalyzeAndPlanNode()
    apply_changes = ApplyChangesNode()

    # Connect nodes using default action (no named actions)
    read_target >> analyze_plan
    analyze_plan >> apply_changes

    # Create flow
    return Flow(start=read_target)

#############################################
# Main Flow
#############################################


def create_main_flow() -> Flow:
    # Create nodes
    main_agent = MainDecisionAgent()
    read_action = ReadFileAction()
    grep_action = GrepSearchAction()
    list_dir_action = ListDirAction()
    delete_action = DeleteFileAction()
    edit_agent = create_edit_agent()
    format_response = FormatResponseNode()

    # Connect main agent to action nodes
    main_agent - "read_file" >> read_action
    main_agent - "grep_search" >> grep_action
    main_agent - "list_dir" >> list_dir_action
    main_agent - "delete_file" >> delete_action
    main_agent - "edit_file" >> edit_agent
    main_agent - "finish" >> format_response

    # Connect action nodes back to main agent using default action
    read_action >> main_agent
    grep_action >> main_agent
    list_dir_action >> main_agent
    delete_action >> main_agent
    edit_agent >> main_agent

    # Create flow
    return Flow(start=main_agent)


# Create the main flow
coding_agent_flow = create_main_flow()