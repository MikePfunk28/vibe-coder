"""
Custom MCP Server Development Framework for AI IDE
Provides tools and utilities for creating custom MCP servers
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import inspect
import sys

logger = logging.getLogger('mcp_server_framework')

class MCPMessageType(Enum):
    """MCP message types"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"

class MCPMethod(Enum):
    """Standard MCP methods"""
    INITIALIZE = "initialize"
    LIST_TOOLS = "tools/list"
    CALL_TOOL = "tools/call"
    LIST_RESOURCES = "resources/list"
    READ_RESOURCE = "resources/read"
    LIST_PROMPTS = "prompts/list"
    GET_PROMPT = "prompts/get"
    COMPLETE = "completion/complete"

@dataclass
class MCPMessage:
    """Represents an MCP protocol message"""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

@dataclass
class MCPToolDefinition:
    """Definition of an MCP tool"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MCPResourceDefinition:
    """Definition of an MCP resource"""
    uri: str
    name: str
    description: str
    mime_type: str
    handler: Callable
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MCPPromptDefinition:
    """Definition of an MCP prompt"""
    name: str
    description: str
    arguments: List[Dict[str, Any]]
    handler: Callable
    metadata: Dict[str, Any] = field(default_factory=dict)

class MCPServerFramework:
    """Framework for building custom MCP servers"""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.tools: Dict[str, MCPToolDefinition] = {}
        self.resources: Dict[str, MCPResourceDefinition] = {}
        self.prompts: Dict[str, MCPPromptDefinition] = {}
        self.capabilities = {
            "tools": {},
            "resources": {},
            "prompts": {},
            "completion": {}
        }
        
        # Message handling
        self.message_handlers: Dict[str, Callable] = {}
        self.middleware: List[Callable] = []
        
        # Server state
        self.initialized = False
        self.client_info: Optional[Dict[str, Any]] = None
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default MCP message handlers"""
        self.message_handlers.update({
            MCPMethod.INITIALIZE.value: self._handle_initialize,
            MCPMethod.LIST_TOOLS.value: self._handle_list_tools,
            MCPMethod.CALL_TOOL.value: self._handle_call_tool,
            MCPMethod.LIST_RESOURCES.value: self._handle_list_resources,
            MCPMethod.READ_RESOURCE.value: self._handle_read_resource,
            MCPMethod.LIST_PROMPTS.value: self._handle_list_prompts,
            MCPMethod.GET_PROMPT.value: self._handle_get_prompt,
            MCPMethod.COMPLETE.value: self._handle_complete
        })
    
    def tool(self, name: str, description: str, input_schema: Optional[Dict[str, Any]] = None):
        """Decorator for registering MCP tools"""
        def decorator(func: Callable):
            # Generate input schema from function signature if not provided
            if input_schema is None:
                schema = self._generate_input_schema(func)
            else:
                schema = input_schema
            
            tool_def = MCPToolDefinition(
                name=name,
                description=description,
                input_schema=schema,
                handler=func
            )
            
            self.tools[name] = tool_def
            logger.info(f"Registered tool: {name}")
            
            return func
        
        return decorator
    
    def resource(self, uri: str, name: str, description: str, mime_type: str = "text/plain"):
        """Decorator for registering MCP resources"""
        def decorator(func: Callable):
            resource_def = MCPResourceDefinition(
                uri=uri,
                name=name,
                description=description,
                mime_type=mime_type,
                handler=func
            )
            
            self.resources[uri] = resource_def
            logger.info(f"Registered resource: {uri}")
            
            return func
        
        return decorator
    
    def prompt(self, name: str, description: str, arguments: Optional[List[Dict[str, Any]]] = None):
        """Decorator for registering MCP prompts"""
        def decorator(func: Callable):
            # Generate arguments from function signature if not provided
            if arguments is None:
                args = self._generate_prompt_arguments(func)
            else:
                args = arguments
            
            prompt_def = MCPPromptDefinition(
                name=name,
                description=description,
                arguments=args,
                handler=func
            )
            
            self.prompts[name] = prompt_def
            logger.info(f"Registered prompt: {name}")
            
            return func
        
        return decorator
    
    def middleware(self, func: Callable):
        """Decorator for registering middleware"""
        self.middleware.append(func)
        logger.info(f"Registered middleware: {func.__name__}")
        return func
    
    def _generate_input_schema(self, func: Callable) -> Dict[str, Any]:
        """Generate JSON schema from function signature"""
        sig = inspect.signature(func)
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            param_schema = {"type": "string"}  # Default type
            
            # Try to infer type from annotation
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_schema["type"] = "integer"
                elif param.annotation == float:
                    param_schema["type"] = "number"
                elif param.annotation == bool:
                    param_schema["type"] = "boolean"
                elif param.annotation == list:
                    param_schema["type"] = "array"
                elif param.annotation == dict:
                    param_schema["type"] = "object"
            
            # Check if parameter is required
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
            else:
                param_schema["default"] = param.default
            
            properties[param_name] = param_schema
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    def _generate_prompt_arguments(self, func: Callable) -> List[Dict[str, Any]]:
        """Generate prompt arguments from function signature"""
        sig = inspect.signature(func)
        arguments = []
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            arg_def = {
                "name": param_name,
                "description": f"Parameter {param_name}",
                "required": param.default == inspect.Parameter.empty
            }
            
            arguments.append(arg_def)
        
        return arguments
    
    async def _handle_initialize(self, message: MCPMessage) -> MCPMessage:
        """Handle initialize request"""
        params = message.params or {}
        
        self.client_info = {
            "name": params.get("clientInfo", {}).get("name", "Unknown"),
            "version": params.get("clientInfo", {}).get("version", "Unknown")
        }
        
        self.initialized = True
        
        return MCPMessage(
            id=message.id,
            result={
                "protocolVersion": "2024-11-05",
                "capabilities": self.capabilities,
                "serverInfo": {
                    "name": self.name,
                    "version": self.version
                }
            }
        )
    
    async def _handle_list_tools(self, message: MCPMessage) -> MCPMessage:
        """Handle list tools request"""
        tools = []
        
        for tool_name, tool_def in self.tools.items():
            tools.append({
                "name": tool_name,
                "description": tool_def.description,
                "inputSchema": tool_def.input_schema
            })
        
        return MCPMessage(
            id=message.id,
            result={"tools": tools}
        )
    
    async def _handle_call_tool(self, message: MCPMessage) -> MCPMessage:
        """Handle call tool request"""
        params = message.params or {}
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name not in self.tools:
            return MCPMessage(
                id=message.id,
                error={
                    "code": -32601,
                    "message": f"Tool not found: {tool_name}"
                }
            )
        
        tool_def = self.tools[tool_name]
        
        try:
            # Call the tool handler
            if inspect.iscoroutinefunction(tool_def.handler):
                result = await tool_def.handler(**arguments)
            else:
                result = tool_def.handler(**arguments)
            
            return MCPMessage(
                id=message.id,
                result={
                    "content": [
                        {
                            "type": "text",
                            "text": str(result)
                        }
                    ]
                }
            )
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return MCPMessage(
                id=message.id,
                error={
                    "code": -32603,
                    "message": f"Tool execution failed: {str(e)}"
                }
            )
    
    async def _handle_list_resources(self, message: MCPMessage) -> MCPMessage:
        """Handle list resources request"""
        resources = []
        
        for uri, resource_def in self.resources.items():
            resources.append({
                "uri": uri,
                "name": resource_def.name,
                "description": resource_def.description,
                "mimeType": resource_def.mime_type
            })
        
        return MCPMessage(
            id=message.id,
            result={"resources": resources}
        )
    
    async def _handle_read_resource(self, message: MCPMessage) -> MCPMessage:
        """Handle read resource request"""
        params = message.params or {}
        uri = params.get("uri")
        
        if uri not in self.resources:
            return MCPMessage(
                id=message.id,
                error={
                    "code": -32601,
                    "message": f"Resource not found: {uri}"
                }
            )
        
        resource_def = self.resources[uri]
        
        try:
            # Call the resource handler
            if inspect.iscoroutinefunction(resource_def.handler):
                content = await resource_def.handler(uri)
            else:
                content = resource_def.handler(uri)
            
            return MCPMessage(
                id=message.id,
                result={
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": resource_def.mime_type,
                            "text": str(content)
                        }
                    ]
                }
            )
            
        except Exception as e:
            logger.error(f"Resource read failed: {e}")
            return MCPMessage(
                id=message.id,
                error={
                    "code": -32603,
                    "message": f"Resource read failed: {str(e)}"
                }
            )
    
    async def _handle_list_prompts(self, message: MCPMessage) -> MCPMessage:
        """Handle list prompts request"""
        prompts = []
        
        for prompt_name, prompt_def in self.prompts.items():
            prompts.append({
                "name": prompt_name,
                "description": prompt_def.description,
                "arguments": prompt_def.arguments
            })
        
        return MCPMessage(
            id=message.id,
            result={"prompts": prompts}
        )
    
    async def _handle_get_prompt(self, message: MCPMessage) -> MCPMessage:
        """Handle get prompt request"""
        params = message.params or {}
        prompt_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if prompt_name not in self.prompts:
            return MCPMessage(
                id=message.id,
                error={
                    "code": -32601,
                    "message": f"Prompt not found: {prompt_name}"
                }
            )
        
        prompt_def = self.prompts[prompt_name]
        
        try:
            # Call the prompt handler
            if inspect.iscoroutinefunction(prompt_def.handler):
                result = await prompt_def.handler(**arguments)
            else:
                result = prompt_def.handler(**arguments)
            
            return MCPMessage(
                id=message.id,
                result={
                    "description": prompt_def.description,
                    "messages": [
                        {
                            "role": "user",
                            "content": {
                                "type": "text",
                                "text": str(result)
                            }
                        }
                    ]
                }
            )
            
        except Exception as e:
            logger.error(f"Prompt generation failed: {e}")
            return MCPMessage(
                id=message.id,
                error={
                    "code": -32603,
                    "message": f"Prompt generation failed: {str(e)}"
                }
            )
    
    async def _handle_complete(self, message: MCPMessage) -> MCPMessage:
        """Handle completion request"""
        # Default implementation - can be overridden
        return MCPMessage(
            id=message.id,
            result={
                "completion": {
                    "values": [],
                    "total": 0,
                    "hasMore": False
                }
            }
        )
    
    async def process_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an incoming MCP message"""
        try:
            # Parse message
            message = MCPMessage(
                jsonrpc=message_data.get("jsonrpc", "2.0"),
                id=message_data.get("id"),
                method=message_data.get("method"),
                params=message_data.get("params"),
                result=message_data.get("result"),
                error=message_data.get("error")
            )
            
            # Apply middleware
            for middleware_func in self.middleware:
                if inspect.iscoroutinefunction(middleware_func):
                    message = await middleware_func(message)
                else:
                    message = middleware_func(message)
            
            # Handle the message
            if message.method in self.message_handlers:
                handler = self.message_handlers[message.method]
                response = await handler(message)
            else:
                response = MCPMessage(
                    id=message.id,
                    error={
                        "code": -32601,
                        "message": f"Method not found: {message.method}"
                    }
                )
            
            # Convert response to dict
            response_dict = {
                "jsonrpc": response.jsonrpc
            }
            
            if response.id is not None:
                response_dict["id"] = response.id
            
            if response.result is not None:
                response_dict["result"] = response.result
            
            if response.error is not None:
                response_dict["error"] = response.error
            
            return response_dict
            
        except Exception as e:
            logger.error(f"Message processing failed: {e}")
            return {
                "jsonrpc": "2.0",
                "id": message_data.get("id"),
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
    
    async def run_stdio(self):
        """Run the server using stdio transport"""
        logger.info(f"Starting MCP server: {self.name} v{self.version}")
        
        try:
            while True:
                # Read message from stdin
                line = sys.stdin.readline()
                if not line:
                    break
                
                try:
                    message_data = json.loads(line.strip())
                    response = await self.process_message(message_data)
                    
                    # Write response to stdout
                    print(json.dumps(response), flush=True)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": "Parse error"
                        }
                    }
                    print(json.dumps(error_response), flush=True)
                
        except KeyboardInterrupt:
            logger.info("Server shutting down...")
        except Exception as e:
            logger.error(f"Server error: {e}")
    
    def add_capability(self, capability_type: str, capability_data: Dict[str, Any]):
        """Add a capability to the server"""
        self.capabilities[capability_type] = capability_data
    
    def register_handler(self, method: str, handler: Callable):
        """Register a custom message handler"""
        self.message_handlers[method] = handler
        logger.info(f"Registered handler for method: {method}")

class MCPServerBuilder:
    """Builder class for creating MCP servers with common patterns"""
    
    @staticmethod
    def create_file_server(name: str, base_path: str) -> MCPServerFramework:
        """Create a file system MCP server"""
        server = MCPServerFramework(name, "1.0.0")
        base_path = Path(base_path)
        
        @server.tool("read_file", "Read contents of a file")
        def read_file(file_path: str) -> str:
            """Read file contents"""
            full_path = base_path / file_path
            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            return full_path.read_text()
        
        @server.tool("write_file", "Write contents to a file")
        def write_file(file_path: str, content: str) -> str:
            """Write file contents"""
            full_path = base_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            return f"Written {len(content)} characters to {file_path}"
        
        @server.tool("list_files", "List files in a directory")
        def list_files(directory_path: str = ".") -> List[str]:
            """List files in directory"""
            full_path = base_path / directory_path
            if not full_path.exists():
                raise FileNotFoundError(f"Directory not found: {directory_path}")
            
            return [item.name for item in full_path.iterdir()]
        
        return server
    
    @staticmethod
    def create_api_server(name: str, base_url: str) -> MCPServerFramework:
        """Create an API client MCP server"""
        server = MCPServerFramework(name, "1.0.0")
        
        @server.tool("get_request", "Make a GET request to the API")
        async def get_request(endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
            """Make GET request"""
            import aiohttp
            
            url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    return {
                        "status": response.status,
                        "data": await response.json()
                    }
        
        @server.tool("post_request", "Make a POST request to the API")
        async def post_request(endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
            """Make POST request"""
            import aiohttp
            
            url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    return {
                        "status": response.status,
                        "data": await response.json()
                    }
        
        return server
    
    @staticmethod
    def create_database_server(name: str, connection_string: str) -> MCPServerFramework:
        """Create a database MCP server"""
        server = MCPServerFramework(name, "1.0.0")
        
        @server.tool("execute_query", "Execute a SQL query")
        async def execute_query(query: str, parameters: List[Any] = None) -> Dict[str, Any]:
            """Execute SQL query"""
            # This would need actual database connection implementation
            return {
                "query": query,
                "parameters": parameters or [],
                "result": "Mock database result"
            }
        
        @server.tool("list_tables", "List database tables")
        async def list_tables() -> List[str]:
            """List database tables"""
            # Mock implementation
            return ["users", "products", "orders"]
        
        return server

# Example usage and templates
def create_example_server():
    """Create an example MCP server"""
    server = MCPServerFramework("example-server", "1.0.0")
    
    @server.tool("hello", "Say hello to someone")
    def hello(name: str = "World") -> str:
        """Say hello"""
        return f"Hello, {name}!"
    
    @server.tool("calculate", "Perform basic calculations")
    def calculate(operation: str, a: float, b: float) -> float:
        """Perform calculation"""
        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Cannot divide by zero")
            return a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    @server.resource("config://settings", "Server Settings", "Server configuration settings")
    def get_settings(uri: str) -> Dict[str, Any]:
        """Get server settings"""
        return {
            "name": server.name,
            "version": server.version,
            "tools_count": len(server.tools),
            "resources_count": len(server.resources)
        }
    
    @server.prompt("code_review", "Generate a code review prompt")
    def code_review_prompt(code: str, language: str = "python") -> str:
        """Generate code review prompt"""
        return f"""Please review the following {language} code:

```{language}
{code}
```

Please provide feedback on:
1. Code quality and style
2. Potential bugs or issues
3. Performance considerations
4. Suggestions for improvement
"""
    
    @server.middleware
    async def logging_middleware(message: MCPMessage) -> MCPMessage:
        """Log all messages"""
        if message.method:
            logger.info(f"Processing method: {message.method}")
        return message
    
    return server

if __name__ == "__main__":
    # Example of running a server
    server = create_example_server()
    asyncio.run(server.run_stdio())