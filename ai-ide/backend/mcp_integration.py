"""
Model Context Protocol (MCP) Integration System for AI IDE
Provides comprehensive MCP server discovery, management, and tool execution
"""

import asyncio
import json
import logging
import subprocess
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import tempfile
import os

logger = logging.getLogger('mcp_integration')

class MCPServerStatus(Enum):
    """Status of MCP servers"""
    UNKNOWN = "unknown"
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    DISABLED = "disabled"

class MCPToolType(Enum):
    """Types of MCP tools"""
    FUNCTION = "function"
    RESOURCE = "resource"
    PROMPT = "prompt"
    COMPLETION = "completion"

@dataclass
class MCPTool:
    """Represents an MCP tool"""
    name: str
    description: str
    tool_type: MCPToolType
    server_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    schema: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MCPServer:
    """Represents an MCP server configuration"""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    disabled: bool = False
    auto_approve: List[str] = field(default_factory=list)
    working_dir: Optional[str] = None
    timeout: int = 30
    retry_count: int = 3
    
    # Runtime state
    status: MCPServerStatus = MCPServerStatus.UNKNOWN
    process: Optional[subprocess.Popen] = None
    last_error: Optional[str] = None
    start_time: Optional[datetime] = None
    tools: List[MCPTool] = field(default_factory=list)

@dataclass
class MCPExecutionResult:
    """Result of MCP tool execution"""
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    server_name: str = ""
    tool_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class MCPServerManager:
    """Manages MCP server lifecycle and communication"""
    
    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        self.tools: Dict[str, MCPTool] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.config_paths = [
            Path.cwd() / ".kiro" / "settings" / "mcp.json",  # Workspace config
            Path.home() / ".kiro" / "settings" / "mcp.json"  # User config
        ]
        
        # Security settings
        self.sandbox_enabled = True
        self.max_execution_time = 60.0
        self.allowed_commands = set()
        
        # Performance tracking
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
    
    async def initialize(self):
        """Initialize the MCP server manager"""
        logger.info("Initializing MCP Server Manager...")
        
        try:
            # Load configuration
            await self.load_configuration()
            
            # Discover and start servers
            await self.discover_servers()
            await self.start_all_servers()
            
            # Load tools from running servers
            await self.discover_tools()
            
            logger.info(f"MCP Manager initialized with {len(self.servers)} servers and {len(self.tools)} tools")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP Manager: {e}")
            raise
    
    async def load_configuration(self):
        """Load MCP configuration from files"""
        config = {"mcpServers": {}}
        
        # Load configurations in order (user config first, then workspace overrides)
        for config_path in reversed(self.config_paths):
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        file_config = json.load(f)
                        if "mcpServers" in file_config:
                            config["mcpServers"].update(file_config["mcpServers"])
                    logger.info(f"Loaded MCP config from {config_path}")
                except Exception as e:
                    logger.warning(f"Failed to load config from {config_path}: {e}")
        
        # Convert config to server objects
        for server_name, server_config in config["mcpServers"].items():
            server = MCPServer(
                name=server_name,
                command=server_config.get("command", ""),
                args=server_config.get("args", []),
                env=server_config.get("env", {}),
                disabled=server_config.get("disabled", False),
                auto_approve=server_config.get("autoApprove", []),
                working_dir=server_config.get("workingDir"),
                timeout=server_config.get("timeout", 30)
            )
            self.servers[server_name] = server
            
            # Initialize performance metrics
            self.performance_metrics[server_name] = {
                "success_rate": 0.9,
                "avg_response_time": 1.0,
                "total_requests": 0,
                "error_count": 0
            }
    
    async def discover_servers(self):
        """Discover available MCP servers"""
        logger.info("Discovering MCP servers...")
        
        # Add default servers if none configured
        if not self.servers:
            await self._add_default_servers()
        
        # Validate server configurations
        for server_name, server in self.servers.items():
            if not server.command:
                logger.warning(f"Server {server_name} has no command specified")
                server.status = MCPServerStatus.ERROR
                server.last_error = "No command specified"
            elif server.disabled:
                server.status = MCPServerStatus.DISABLED
                logger.info(f"Server {server_name} is disabled")
    
    async def _add_default_servers(self):
        """Add default MCP servers for common integrations"""
        default_servers = {
            "filesystem": {
                "command": "uvx",
                "args": ["mcp-server-filesystem", "--", "/tmp"],
                "description": "File system operations"
            },
            "git": {
                "command": "uvx", 
                "args": ["mcp-server-git", "--", "."],
                "description": "Git repository operations"
            },
            "web-search": {
                "command": "uvx",
                "args": ["mcp-server-brave-search"],
                "env": {"BRAVE_API_KEY": ""},
                "description": "Web search capabilities"
            }
        }
        
        for name, config in default_servers.items():
            if name not in self.servers:
                server = MCPServer(
                    name=name,
                    command=config["command"],
                    args=config["args"],
                    env=config.get("env", {}),
                    disabled=True  # Disabled by default until configured
                )
                self.servers[name] = server
                logger.info(f"Added default server: {name}")
    
    async def start_all_servers(self):
        """Start all enabled MCP servers"""
        logger.info("Starting MCP servers...")
        
        start_tasks = []
        for server_name, server in self.servers.items():
            if not server.disabled and server.status != MCPServerStatus.ERROR:
                start_tasks.append(self.start_server(server_name))
        
        if start_tasks:
            results = await asyncio.gather(*start_tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    server_name = list(self.servers.keys())[i]
                    logger.error(f"Failed to start server {server_name}: {result}")
    
    async def start_server(self, server_name: str) -> bool:
        """Start a specific MCP server"""
        if server_name not in self.servers:
            logger.error(f"Unknown server: {server_name}")
            return False
        
        server = self.servers[server_name]
        
        if server.disabled:
            logger.info(f"Server {server_name} is disabled")
            return False
        
        if server.status == MCPServerStatus.RUNNING:
            logger.info(f"Server {server_name} is already running")
            return True
        
        logger.info(f"Starting MCP server: {server_name}")
        server.status = MCPServerStatus.STARTING
        
        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(server.env)
            
            # Start the process
            cmd = [server.command] + server.args
            server.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=server.working_dir,
                text=True
            )
            
            server.start_time = datetime.now()
            
            # Wait for server to be ready (simplified check)
            await asyncio.sleep(2)
            
            if server.process.poll() is None:
                server.status = MCPServerStatus.RUNNING
                logger.info(f"Server {server_name} started successfully")
                return True
            else:
                # Process exited
                stdout, stderr = server.process.communicate()
                server.status = MCPServerStatus.ERROR
                server.last_error = f"Process exited: {stderr}"
                logger.error(f"Server {server_name} failed to start: {stderr}")
                return False
                
        except Exception as e:
            server.status = MCPServerStatus.ERROR
            server.last_error = str(e)
            logger.error(f"Failed to start server {server_name}: {e}")
            return False
    
    async def stop_server(self, server_name: str) -> bool:
        """Stop a specific MCP server"""
        if server_name not in self.servers:
            return False
        
        server = self.servers[server_name]
        
        if server.process and server.process.poll() is None:
            try:
                server.process.terminate()
                await asyncio.sleep(1)
                
                if server.process.poll() is None:
                    server.process.kill()
                
                server.status = MCPServerStatus.STOPPED
                server.process = None
                logger.info(f"Server {server_name} stopped")
                return True
                
            except Exception as e:
                logger.error(f"Failed to stop server {server_name}: {e}")
                return False
        
        return True
    
    async def discover_tools(self):
        """Discover tools from running MCP servers"""
        logger.info("Discovering MCP tools...")
        
        for server_name, server in self.servers.items():
            if server.status == MCPServerStatus.RUNNING:
                try:
                    tools = await self._get_server_tools(server)
                    server.tools = tools
                    
                    # Add to global tools registry
                    for tool in tools:
                        tool_key = f"{server_name}.{tool.name}"
                        self.tools[tool_key] = tool
                    
                    logger.info(f"Discovered {len(tools)} tools from server {server_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to discover tools from server {server_name}: {e}")
    
    async def _get_server_tools(self, server: MCPServer) -> List[MCPTool]:
        """Get tools from a specific server (simplified implementation)"""
        # In a real implementation, this would communicate with the MCP server
        # using the MCP protocol to get available tools
        
        # For now, return mock tools based on server type
        mock_tools = []
        
        if server.name == "filesystem":
            mock_tools = [
                MCPTool(
                    name="read_file",
                    description="Read contents of a file",
                    tool_type=MCPToolType.FUNCTION,
                    server_name=server.name,
                    parameters={"file_path": {"type": "string", "description": "Path to file"}}
                ),
                MCPTool(
                    name="write_file",
                    description="Write contents to a file",
                    tool_type=MCPToolType.FUNCTION,
                    server_name=server.name,
                    parameters={
                        "file_path": {"type": "string", "description": "Path to file"},
                        "content": {"type": "string", "description": "File content"}
                    }
                ),
                MCPTool(
                    name="list_directory",
                    description="List directory contents",
                    tool_type=MCPToolType.FUNCTION,
                    server_name=server.name,
                    parameters={"directory_path": {"type": "string", "description": "Directory path"}}
                )
            ]
        
        elif server.name == "git":
            mock_tools = [
                MCPTool(
                    name="git_status",
                    description="Get git repository status",
                    tool_type=MCPToolType.FUNCTION,
                    server_name=server.name
                ),
                MCPTool(
                    name="git_commit",
                    description="Create a git commit",
                    tool_type=MCPToolType.FUNCTION,
                    server_name=server.name,
                    parameters={
                        "message": {"type": "string", "description": "Commit message"},
                        "files": {"type": "array", "description": "Files to commit"}
                    }
                ),
                MCPTool(
                    name="git_diff",
                    description="Show git diff",
                    tool_type=MCPToolType.FUNCTION,
                    server_name=server.name,
                    parameters={"file_path": {"type": "string", "description": "File to diff"}}
                )
            ]
        
        elif server.name == "web-search":
            mock_tools = [
                MCPTool(
                    name="web_search",
                    description="Search the web",
                    tool_type=MCPToolType.FUNCTION,
                    server_name=server.name,
                    parameters={
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {"type": "integer", "description": "Maximum results"}
                    }
                )
            ]
        
        return mock_tools
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any], 
                          timeout: Optional[float] = None) -> MCPExecutionResult:
        """Execute an MCP tool"""
        start_time = time.time()
        
        if tool_name not in self.tools:
            return MCPExecutionResult(
                success=False,
                error=f"Tool not found: {tool_name}",
                tool_name=tool_name
            )
        
        tool = self.tools[tool_name]
        server = self.servers[tool.server_name]
        
        if server.status != MCPServerStatus.RUNNING:
            return MCPExecutionResult(
                success=False,
                error=f"Server {tool.server_name} is not running",
                server_name=tool.server_name,
                tool_name=tool_name
            )
        
        logger.info(f"Executing MCP tool: {tool_name} on server {tool.server_name}")
        
        try:
            # Validate parameters
            validation_result = self._validate_parameters(tool, parameters)
            if not validation_result["valid"]:
                return MCPExecutionResult(
                    success=False,
                    error=f"Parameter validation failed: {validation_result['error']}",
                    server_name=tool.server_name,
                    tool_name=tool_name
                )
            
            # Check security constraints
            if not self._check_security_constraints(tool, parameters):
                return MCPExecutionResult(
                    success=False,
                    error="Security constraints not met",
                    server_name=tool.server_name,
                    tool_name=tool_name
                )
            
            # Execute the tool (simplified implementation)
            result = await self._execute_tool_on_server(server, tool, parameters, timeout)
            
            execution_time = time.time() - start_time
            
            # Update performance metrics
            self._update_performance_metrics(tool.server_name, True, execution_time)
            
            # Record execution
            self._record_execution(tool_name, parameters, result, execution_time)
            
            return MCPExecutionResult(
                success=True,
                result=result,
                execution_time=execution_time,
                server_name=tool.server_name,
                tool_name=tool_name
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Update performance metrics
            self._update_performance_metrics(tool.server_name, False, execution_time)
            
            logger.error(f"Tool execution failed: {e}")
            
            return MCPExecutionResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                server_name=tool.server_name,
                tool_name=tool_name
            )
    
    def _validate_parameters(self, tool: MCPTool, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool parameters"""
        # Simplified parameter validation
        required_params = set(tool.parameters.keys())
        provided_params = set(parameters.keys())
        
        missing_params = required_params - provided_params
        if missing_params:
            return {
                "valid": False,
                "error": f"Missing required parameters: {missing_params}"
            }
        
        return {"valid": True}
    
    def _check_security_constraints(self, tool: MCPTool, parameters: Dict[str, Any]) -> bool:
        """Check security constraints for tool execution"""
        if not self.sandbox_enabled:
            return True
        
        # Check if tool is in auto-approve list
        server = self.servers[tool.server_name]
        if tool.name in server.auto_approve:
            return True
        
        # Additional security checks could be added here
        # For now, allow all tools from known servers
        return tool.server_name in self.servers
    
    async def _execute_tool_on_server(self, server: MCPServer, tool: MCPTool, 
                                    parameters: Dict[str, Any], timeout: Optional[float]) -> Any:
        """Execute tool on MCP server (simplified implementation)"""
        
        # In a real implementation, this would send MCP protocol messages
        # to the server and handle the response
        
        # For now, return mock results based on tool type
        if tool.name == "read_file":
            file_path = parameters.get("file_path", "")
            return f"Mock content of file: {file_path}"
        
        elif tool.name == "write_file":
            file_path = parameters.get("file_path", "")
            content = parameters.get("content", "")
            return f"Mock: Wrote {len(content)} characters to {file_path}"
        
        elif tool.name == "list_directory":
            directory_path = parameters.get("directory_path", "")
            return [f"file1.txt", f"file2.py", f"subdirectory/"]
        
        elif tool.name == "git_status":
            return {
                "branch": "main",
                "modified_files": ["file1.py", "file2.js"],
                "untracked_files": ["new_file.txt"]
            }
        
        elif tool.name == "web_search":
            query = parameters.get("query", "")
            max_results = parameters.get("max_results", 5)
            return [
                {"title": f"Result 1 for {query}", "url": "https://example.com/1"},
                {"title": f"Result 2 for {query}", "url": "https://example.com/2"}
            ]
        
        else:
            return f"Mock result for tool: {tool.name}"
    
    def _update_performance_metrics(self, server_name: str, success: bool, execution_time: float):
        """Update performance metrics for a server"""
        if server_name not in self.performance_metrics:
            return
        
        metrics = self.performance_metrics[server_name]
        
        # Update success rate
        total_requests = metrics["total_requests"]
        current_success_rate = metrics["success_rate"]
        new_success_rate = (current_success_rate * total_requests + (1.0 if success else 0.0)) / (total_requests + 1)
        metrics["success_rate"] = new_success_rate
        
        # Update average response time
        current_avg_time = metrics["avg_response_time"]
        new_avg_time = (current_avg_time * total_requests + execution_time) / (total_requests + 1)
        metrics["avg_response_time"] = new_avg_time
        
        # Update counters
        metrics["total_requests"] = total_requests + 1
        if not success:
            metrics["error_count"] = metrics.get("error_count", 0) + 1
    
    def _record_execution(self, tool_name: str, parameters: Dict[str, Any], 
                         result: Any, execution_time: float):
        """Record tool execution for analysis"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "parameters": parameters,
            "result_type": type(result).__name__,
            "execution_time": execution_time,
            "success": True
        }
        
        self.execution_history.append(record)
        
        # Keep only recent history
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
    
    def get_server_status(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific server"""
        if server_name not in self.servers:
            return None
        
        server = self.servers[server_name]
        metrics = self.performance_metrics.get(server_name, {})
        
        return {
            "name": server.name,
            "status": server.status.value,
            "disabled": server.disabled,
            "tools_count": len(server.tools),
            "last_error": server.last_error,
            "start_time": server.start_time.isoformat() if server.start_time else None,
            "performance_metrics": metrics
        }
    
    def list_servers(self) -> List[Dict[str, Any]]:
        """List all servers with their status"""
        return [
            {
                "name": server.name,
                "status": server.status.value,
                "disabled": server.disabled,
                "tools_count": len(server.tools),
                "command": server.command
            }
            for server in self.servers.values()
        ]
    
    def list_tools(self, server_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available tools, optionally filtered by server"""
        tools = []
        
        for tool_key, tool in self.tools.items():
            if server_name is None or tool.server_name == server_name:
                tools.append({
                    "name": tool.name,
                    "full_name": tool_key,
                    "description": tool.description,
                    "type": tool.tool_type.value,
                    "server": tool.server_name,
                    "parameters": tool.parameters
                })
        
        return tools
    
    async def reload_configuration(self):
        """Reload MCP configuration and restart servers"""
        logger.info("Reloading MCP configuration...")
        
        # Stop all servers
        stop_tasks = []
        for server_name in self.servers.keys():
            stop_tasks.append(self.stop_server(server_name))
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        # Clear current state
        self.servers.clear()
        self.tools.clear()
        
        # Reload configuration
        await self.load_configuration()
        await self.discover_servers()
        await self.start_all_servers()
        await self.discover_tools()
        
        logger.info("MCP configuration reloaded")
    
    async def shutdown(self):
        """Shutdown all MCP servers"""
        logger.info("Shutting down MCP servers...")
        
        stop_tasks = []
        for server_name in self.servers.keys():
            stop_tasks.append(self.stop_server(server_name))
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        logger.info("All MCP servers stopped")

# Global MCP server manager instance
_mcp_manager: Optional[MCPServerManager] = None

async def get_mcp_manager() -> MCPServerManager:
    """Get the global MCP server manager instance"""
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MCPServerManager()
        await _mcp_manager.initialize()
    return _mcp_manager