"""
Unified Tool Interface for AI IDE
Provides a unified interface for external integrations (GitHub, Jira, Slack, etc.)
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

from mcp_integration import MCPServerManager, MCPExecutionResult, get_mcp_manager

logger = logging.getLogger('unified_tool_interface')

class IntegrationType(Enum):
    """Types of external integrations"""
    VERSION_CONTROL = "version_control"  # Git, GitHub, GitLab
    PROJECT_MANAGEMENT = "project_management"  # Jira, Trello, Asana
    COMMUNICATION = "communication"  # Slack, Teams, Discord
    CLOUD_SERVICES = "cloud_services"  # AWS, Azure, GCP
    DATABASES = "databases"  # PostgreSQL, MongoDB, Redis
    MONITORING = "monitoring"  # Datadog, New Relic, Prometheus
    DOCUMENTATION = "documentation"  # Confluence, Notion, GitBook
    FILE_STORAGE = "file_storage"  # Dropbox, Google Drive, S3
    API_SERVICES = "api_services"  # REST APIs, GraphQL
    DEVELOPMENT_TOOLS = "development_tools"  # Docker, Kubernetes, CI/CD

class ToolCategory(Enum):
    """Categories of tools for organization"""
    READ = "read"  # Read-only operations
    WRITE = "write"  # Write operations
    EXECUTE = "execute"  # Execute operations
    SEARCH = "search"  # Search operations
    ANALYZE = "analyze"  # Analysis operations
    MONITOR = "monitor"  # Monitoring operations

@dataclass
class UnifiedTool:
    """Represents a unified tool interface"""
    id: str
    name: str
    description: str
    integration_type: IntegrationType
    category: ToolCategory
    mcp_server: str
    mcp_tool: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_permissions: List[str] = field(default_factory=list)
    rate_limit: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolExecutionContext:
    """Context for tool execution"""
    user_id: Optional[str] = None
    workspace_id: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    rate_limit_state: Dict[str, Any] = field(default_factory=dict)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UnifiedExecutionResult:
    """Result of unified tool execution"""
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    tool_id: str = ""
    integration_type: Optional[IntegrationType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class UnifiedToolInterface:
    """Unified interface for external tool integrations"""
    
    def __init__(self):
        self.mcp_manager: Optional[MCPServerManager] = None
        self.tools: Dict[str, UnifiedTool] = {}
        self.tool_categories: Dict[IntegrationType, List[str]] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Rate limiting
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        
        # Security and permissions
        self.permission_validator: Optional[Callable] = None
        self.security_policies: Dict[str, Any] = {}
        
        # Cross-tool workflows
        self.workflows: Dict[str, List[str]] = {}
    
    async def initialize(self):
        """Initialize the unified tool interface"""
        logger.info("Initializing Unified Tool Interface...")
        
        try:
            # Initialize MCP manager
            self.mcp_manager = await get_mcp_manager()
            
            # Register default integrations
            await self.register_default_integrations()
            
            # Set up security policies
            self._setup_security_policies()
            
            logger.info(f"Unified Tool Interface initialized with {len(self.tools)} tools")
            
        except Exception as e:
            logger.error(f"Failed to initialize Unified Tool Interface: {e}")
            raise
    
    async def register_default_integrations(self):
        """Register default external integrations"""
        
        # GitHub Integration
        await self._register_github_tools()
        
        # Jira Integration
        await self._register_jira_tools()
        
        # Slack Integration
        await self._register_slack_tools()
        
        # AWS Integration
        await self._register_aws_tools()
        
        # Database Integration
        await self._register_database_tools()
        
        # File System Integration
        await self._register_filesystem_tools()
        
        # Development Tools Integration
        await self._register_development_tools()
    
    async def _register_github_tools(self):
        """Register GitHub integration tools"""
        github_tools = [
            UnifiedTool(
                id="github.list_repositories",
                name="List Repositories",
                description="List GitHub repositories for a user or organization",
                integration_type=IntegrationType.VERSION_CONTROL,
                category=ToolCategory.READ,
                mcp_server="github",
                mcp_tool="list_repositories",
                parameters={
                    "owner": {"type": "string", "description": "Repository owner"},
                    "type": {"type": "string", "description": "Repository type (all, owner, member)"}
                },
                required_permissions=["github:read"]
            ),
            UnifiedTool(
                id="github.create_issue",
                name="Create Issue",
                description="Create a new GitHub issue",
                integration_type=IntegrationType.VERSION_CONTROL,
                category=ToolCategory.WRITE,
                mcp_server="github",
                mcp_tool="create_issue",
                parameters={
                    "owner": {"type": "string", "description": "Repository owner"},
                    "repo": {"type": "string", "description": "Repository name"},
                    "title": {"type": "string", "description": "Issue title"},
                    "body": {"type": "string", "description": "Issue body"},
                    "labels": {"type": "array", "description": "Issue labels"}
                },
                required_permissions=["github:write"],
                rate_limit={"requests_per_hour": 100}
            ),
            UnifiedTool(
                id="github.create_pull_request",
                name="Create Pull Request",
                description="Create a new GitHub pull request",
                integration_type=IntegrationType.VERSION_CONTROL,
                category=ToolCategory.WRITE,
                mcp_server="github",
                mcp_tool="create_pull_request",
                parameters={
                    "owner": {"type": "string", "description": "Repository owner"},
                    "repo": {"type": "string", "description": "Repository name"},
                    "title": {"type": "string", "description": "PR title"},
                    "body": {"type": "string", "description": "PR body"},
                    "head": {"type": "string", "description": "Head branch"},
                    "base": {"type": "string", "description": "Base branch"}
                },
                required_permissions=["github:write"]
            ),
            UnifiedTool(
                id="github.search_code",
                name="Search Code",
                description="Search code across GitHub repositories",
                integration_type=IntegrationType.VERSION_CONTROL,
                category=ToolCategory.SEARCH,
                mcp_server="github",
                mcp_tool="search_code",
                parameters={
                    "query": {"type": "string", "description": "Search query"},
                    "language": {"type": "string", "description": "Programming language"},
                    "repo": {"type": "string", "description": "Repository to search in"}
                },
                required_permissions=["github:read"]
            )
        ]
        
        for tool in github_tools:
            await self.register_tool(tool)
    
    async def _register_jira_tools(self):
        """Register Jira integration tools"""
        jira_tools = [
            UnifiedTool(
                id="jira.create_issue",
                name="Create Jira Issue",
                description="Create a new Jira issue",
                integration_type=IntegrationType.PROJECT_MANAGEMENT,
                category=ToolCategory.WRITE,
                mcp_server="jira",
                mcp_tool="create_issue",
                parameters={
                    "project": {"type": "string", "description": "Project key"},
                    "issue_type": {"type": "string", "description": "Issue type"},
                    "summary": {"type": "string", "description": "Issue summary"},
                    "description": {"type": "string", "description": "Issue description"},
                    "priority": {"type": "string", "description": "Issue priority"}
                },
                required_permissions=["jira:write"]
            ),
            UnifiedTool(
                id="jira.search_issues",
                name="Search Jira Issues",
                description="Search for Jira issues using JQL",
                integration_type=IntegrationType.PROJECT_MANAGEMENT,
                category=ToolCategory.SEARCH,
                mcp_server="jira",
                mcp_tool="search_issues",
                parameters={
                    "jql": {"type": "string", "description": "JQL query"},
                    "max_results": {"type": "integer", "description": "Maximum results"}
                },
                required_permissions=["jira:read"]
            ),
            UnifiedTool(
                id="jira.update_issue",
                name="Update Jira Issue",
                description="Update an existing Jira issue",
                integration_type=IntegrationType.PROJECT_MANAGEMENT,
                category=ToolCategory.WRITE,
                mcp_server="jira",
                mcp_tool="update_issue",
                parameters={
                    "issue_key": {"type": "string", "description": "Issue key"},
                    "fields": {"type": "object", "description": "Fields to update"}
                },
                required_permissions=["jira:write"]
            )
        ]
        
        for tool in jira_tools:
            await self.register_tool(tool)
    
    async def _register_slack_tools(self):
        """Register Slack integration tools"""
        slack_tools = [
            UnifiedTool(
                id="slack.send_message",
                name="Send Slack Message",
                description="Send a message to a Slack channel",
                integration_type=IntegrationType.COMMUNICATION,
                category=ToolCategory.WRITE,
                mcp_server="slack",
                mcp_tool="send_message",
                parameters={
                    "channel": {"type": "string", "description": "Channel ID or name"},
                    "text": {"type": "string", "description": "Message text"},
                    "attachments": {"type": "array", "description": "Message attachments"}
                },
                required_permissions=["slack:write"],
                rate_limit={"requests_per_minute": 60}
            ),
            UnifiedTool(
                id="slack.list_channels",
                name="List Slack Channels",
                description="List Slack channels",
                integration_type=IntegrationType.COMMUNICATION,
                category=ToolCategory.READ,
                mcp_server="slack",
                mcp_tool="list_channels",
                parameters={
                    "types": {"type": "string", "description": "Channel types to include"}
                },
                required_permissions=["slack:read"]
            ),
            UnifiedTool(
                id="slack.search_messages",
                name="Search Slack Messages",
                description="Search for messages in Slack",
                integration_type=IntegrationType.COMMUNICATION,
                category=ToolCategory.SEARCH,
                mcp_server="slack",
                mcp_tool="search_messages",
                parameters={
                    "query": {"type": "string", "description": "Search query"},
                    "channel": {"type": "string", "description": "Channel to search in"},
                    "count": {"type": "integer", "description": "Number of results"}
                },
                required_permissions=["slack:read"]
            )
        ]
        
        for tool in slack_tools:
            await self.register_tool(tool)
    
    async def _register_aws_tools(self):
        """Register AWS integration tools"""
        aws_tools = [
            UnifiedTool(
                id="aws.s3.list_buckets",
                name="List S3 Buckets",
                description="List AWS S3 buckets",
                integration_type=IntegrationType.CLOUD_SERVICES,
                category=ToolCategory.READ,
                mcp_server="aws",
                mcp_tool="s3_list_buckets",
                required_permissions=["aws:s3:read"]
            ),
            UnifiedTool(
                id="aws.s3.upload_file",
                name="Upload File to S3",
                description="Upload a file to AWS S3",
                integration_type=IntegrationType.CLOUD_SERVICES,
                category=ToolCategory.WRITE,
                mcp_server="aws",
                mcp_tool="s3_upload_file",
                parameters={
                    "bucket": {"type": "string", "description": "S3 bucket name"},
                    "key": {"type": "string", "description": "Object key"},
                    "file_path": {"type": "string", "description": "Local file path"}
                },
                required_permissions=["aws:s3:write"]
            ),
            UnifiedTool(
                id="aws.ec2.list_instances",
                name="List EC2 Instances",
                description="List AWS EC2 instances",
                integration_type=IntegrationType.CLOUD_SERVICES,
                category=ToolCategory.READ,
                mcp_server="aws",
                mcp_tool="ec2_list_instances",
                parameters={
                    "region": {"type": "string", "description": "AWS region"},
                    "filters": {"type": "object", "description": "Instance filters"}
                },
                required_permissions=["aws:ec2:read"]
            )
        ]
        
        for tool in aws_tools:
            await self.register_tool(tool)
    
    async def _register_database_tools(self):
        """Register database integration tools"""
        db_tools = [
            UnifiedTool(
                id="postgres.execute_query",
                name="Execute PostgreSQL Query",
                description="Execute a query on PostgreSQL database",
                integration_type=IntegrationType.DATABASES,
                category=ToolCategory.EXECUTE,
                mcp_server="postgres",
                mcp_tool="execute_query",
                parameters={
                    "query": {"type": "string", "description": "SQL query"},
                    "parameters": {"type": "array", "description": "Query parameters"}
                },
                required_permissions=["postgres:execute"]
            ),
            UnifiedTool(
                id="redis.get_key",
                name="Get Redis Key",
                description="Get value from Redis key",
                integration_type=IntegrationType.DATABASES,
                category=ToolCategory.READ,
                mcp_server="redis",
                mcp_tool="get_key",
                parameters={
                    "key": {"type": "string", "description": "Redis key"}
                },
                required_permissions=["redis:read"]
            ),
            UnifiedTool(
                id="redis.set_key",
                name="Set Redis Key",
                description="Set value for Redis key",
                integration_type=IntegrationType.DATABASES,
                category=ToolCategory.WRITE,
                mcp_server="redis",
                mcp_tool="set_key",
                parameters={
                    "key": {"type": "string", "description": "Redis key"},
                    "value": {"type": "string", "description": "Value to set"},
                    "ttl": {"type": "integer", "description": "Time to live in seconds"}
                },
                required_permissions=["redis:write"]
            )
        ]
        
        for tool in db_tools:
            await self.register_tool(tool)
    
    async def _register_filesystem_tools(self):
        """Register file system integration tools"""
        fs_tools = [
            UnifiedTool(
                id="fs.read_file",
                name="Read File",
                description="Read contents of a file",
                integration_type=IntegrationType.FILE_STORAGE,
                category=ToolCategory.READ,
                mcp_server="filesystem",
                mcp_tool="read_file",
                parameters={
                    "file_path": {"type": "string", "description": "Path to file"}
                },
                required_permissions=["fs:read"]
            ),
            UnifiedTool(
                id="fs.write_file",
                name="Write File",
                description="Write contents to a file",
                integration_type=IntegrationType.FILE_STORAGE,
                category=ToolCategory.WRITE,
                mcp_server="filesystem",
                mcp_tool="write_file",
                parameters={
                    "file_path": {"type": "string", "description": "Path to file"},
                    "content": {"type": "string", "description": "File content"}
                },
                required_permissions=["fs:write"]
            ),
            UnifiedTool(
                id="fs.list_directory",
                name="List Directory",
                description="List contents of a directory",
                integration_type=IntegrationType.FILE_STORAGE,
                category=ToolCategory.READ,
                mcp_server="filesystem",
                mcp_tool="list_directory",
                parameters={
                    "directory_path": {"type": "string", "description": "Directory path"}
                },
                required_permissions=["fs:read"]
            )
        ]
        
        for tool in fs_tools:
            await self.register_tool(tool)
    
    async def _register_development_tools(self):
        """Register development tools integration"""
        dev_tools = [
            UnifiedTool(
                id="docker.list_containers",
                name="List Docker Containers",
                description="List Docker containers",
                integration_type=IntegrationType.DEVELOPMENT_TOOLS,
                category=ToolCategory.READ,
                mcp_server="docker",
                mcp_tool="list_containers",
                parameters={
                    "all": {"type": "boolean", "description": "Include stopped containers"}
                },
                required_permissions=["docker:read"]
            ),
            UnifiedTool(
                id="docker.start_container",
                name="Start Docker Container",
                description="Start a Docker container",
                integration_type=IntegrationType.DEVELOPMENT_TOOLS,
                category=ToolCategory.EXECUTE,
                mcp_server="docker",
                mcp_tool="start_container",
                parameters={
                    "container_id": {"type": "string", "description": "Container ID or name"}
                },
                required_permissions=["docker:execute"]
            ),
            UnifiedTool(
                id="git.status",
                name="Git Status",
                description="Get git repository status",
                integration_type=IntegrationType.VERSION_CONTROL,
                category=ToolCategory.READ,
                mcp_server="git",
                mcp_tool="git_status",
                required_permissions=["git:read"]
            ),
            UnifiedTool(
                id="git.commit",
                name="Git Commit",
                description="Create a git commit",
                integration_type=IntegrationType.VERSION_CONTROL,
                category=ToolCategory.WRITE,
                mcp_server="git",
                mcp_tool="git_commit",
                parameters={
                    "message": {"type": "string", "description": "Commit message"},
                    "files": {"type": "array", "description": "Files to commit"}
                },
                required_permissions=["git:write"]
            )
        ]
        
        for tool in dev_tools:
            await self.register_tool(tool)
    
    async def register_tool(self, tool: UnifiedTool):
        """Register a unified tool"""
        self.tools[tool.id] = tool
        
        # Organize by integration type
        if tool.integration_type not in self.tool_categories:
            self.tool_categories[tool.integration_type] = []
        self.tool_categories[tool.integration_type].append(tool.id)
        
        logger.info(f"Registered unified tool: {tool.id} ({tool.integration_type.value})")
    
    def _setup_security_policies(self):
        """Set up security policies for tool execution"""
        self.security_policies = {
            "require_permissions": True,
            "sandbox_execution": True,
            "rate_limiting": True,
            "audit_logging": True,
            "dangerous_operations": [
                "delete", "remove", "drop", "truncate", "destroy"
            ],
            "restricted_integrations": [
                IntegrationType.CLOUD_SERVICES,
                IntegrationType.DATABASES
            ]
        }
    
    async def execute_tool(self, tool_id: str, parameters: Dict[str, Any], 
                          context: Optional[ToolExecutionContext] = None) -> UnifiedExecutionResult:
        """Execute a unified tool"""
        start_time = datetime.now()
        
        if tool_id not in self.tools:
            return UnifiedExecutionResult(
                success=False,
                error=f"Tool not found: {tool_id}",
                tool_id=tool_id
            )
        
        tool = self.tools[tool_id]
        context = context or ToolExecutionContext()
        
        logger.info(f"Executing unified tool: {tool_id}")
        
        try:
            # Security validation
            security_result = await self._validate_security(tool, parameters, context)
            if not security_result["valid"]:
                return UnifiedExecutionResult(
                    success=False,
                    error=f"Security validation failed: {security_result['error']}",
                    tool_id=tool_id,
                    integration_type=tool.integration_type
                )
            
            # Rate limiting check
            rate_limit_result = await self._check_rate_limit(tool, context)
            if not rate_limit_result["allowed"]:
                return UnifiedExecutionResult(
                    success=False,
                    error=f"Rate limit exceeded: {rate_limit_result['error']}",
                    tool_id=tool_id,
                    integration_type=tool.integration_type
                )
            
            # Execute via MCP
            mcp_result = await self.mcp_manager.execute_tool(
                f"{tool.mcp_server}.{tool.mcp_tool}",
                parameters
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Record execution
            self._record_execution(tool_id, parameters, mcp_result, execution_time, context)
            
            return UnifiedExecutionResult(
                success=mcp_result.success,
                result=mcp_result.result,
                error=mcp_result.error,
                execution_time=execution_time,
                tool_id=tool_id,
                integration_type=tool.integration_type,
                metadata={
                    "mcp_server": tool.mcp_server,
                    "mcp_tool": tool.mcp_tool,
                    "category": tool.category.value
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.error(f"Unified tool execution failed: {e}")
            
            return UnifiedExecutionResult(
                success=False,
                error=str(e),
                execution_time=execution_time,
                tool_id=tool_id,
                integration_type=tool.integration_type
            )
    
    async def _validate_security(self, tool: UnifiedTool, parameters: Dict[str, Any], 
                                context: ToolExecutionContext) -> Dict[str, Any]:
        """Validate security constraints for tool execution"""
        
        # Check permissions
        if self.security_policies.get("require_permissions", True):
            if not all(perm in context.permissions for perm in tool.required_permissions):
                missing_perms = set(tool.required_permissions) - set(context.permissions)
                return {
                    "valid": False,
                    "error": f"Missing required permissions: {missing_perms}"
                }
        
        # Check for dangerous operations
        dangerous_ops = self.security_policies.get("dangerous_operations", [])
        for param_value in parameters.values():
            if isinstance(param_value, str):
                if any(dangerous_op in param_value.lower() for dangerous_op in dangerous_ops):
                    return {
                        "valid": False,
                        "error": "Dangerous operation detected"
                    }
        
        # Check restricted integrations
        restricted = self.security_policies.get("restricted_integrations", [])
        if tool.integration_type in restricted:
            if not context.user_id or "admin" not in context.permissions:
                return {
                    "valid": False,
                    "error": f"Restricted integration: {tool.integration_type.value}"
                }
        
        return {"valid": True}
    
    async def _check_rate_limit(self, tool: UnifiedTool, context: ToolExecutionContext) -> Dict[str, Any]:
        """Check rate limiting for tool execution"""
        
        if not self.security_policies.get("rate_limiting", True):
            return {"allowed": True}
        
        if not tool.rate_limit:
            return {"allowed": True}
        
        # Simple rate limiting implementation
        rate_limit_key = f"{tool.id}:{context.user_id or 'anonymous'}"
        current_time = datetime.now()
        
        if rate_limit_key not in self.rate_limits:
            self.rate_limits[rate_limit_key] = {
                "requests": [],
                "last_reset": current_time
            }
        
        rate_data = self.rate_limits[rate_limit_key]
        
        # Clean old requests based on rate limit window
        if "requests_per_minute" in tool.rate_limit:
            window_start = current_time.timestamp() - 60
            rate_data["requests"] = [
                req_time for req_time in rate_data["requests"] 
                if req_time > window_start
            ]
            
            if len(rate_data["requests"]) >= tool.rate_limit["requests_per_minute"]:
                return {
                    "allowed": False,
                    "error": "Rate limit exceeded (per minute)"
                }
        
        elif "requests_per_hour" in tool.rate_limit:
            window_start = current_time.timestamp() - 3600
            rate_data["requests"] = [
                req_time for req_time in rate_data["requests"] 
                if req_time > window_start
            ]
            
            if len(rate_data["requests"]) >= tool.rate_limit["requests_per_hour"]:
                return {
                    "allowed": False,
                    "error": "Rate limit exceeded (per hour)"
                }
        
        # Record this request
        rate_data["requests"].append(current_time.timestamp())
        
        return {"allowed": True}
    
    def _record_execution(self, tool_id: str, parameters: Dict[str, Any], 
                         mcp_result: MCPExecutionResult, execution_time: float,
                         context: ToolExecutionContext):
        """Record tool execution for audit and analysis"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "tool_id": tool_id,
            "user_id": context.user_id,
            "workspace_id": context.workspace_id,
            "parameters": parameters,
            "success": mcp_result.success,
            "execution_time": execution_time,
            "error": mcp_result.error,
            "integration_type": self.tools[tool_id].integration_type.value
        }
        
        self.execution_history.append(record)
        
        # Keep only recent history
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
    
    async def execute_workflow(self, workflow_name: str, parameters: Dict[str, Any],
                             context: Optional[ToolExecutionContext] = None) -> List[UnifiedExecutionResult]:
        """Execute a cross-tool workflow"""
        if workflow_name not in self.workflows:
            return [UnifiedExecutionResult(
                success=False,
                error=f"Workflow not found: {workflow_name}"
            )]
        
        tool_ids = self.workflows[workflow_name]
        results = []
        
        logger.info(f"Executing workflow: {workflow_name} with {len(tool_ids)} tools")
        
        # Execute tools in sequence
        workflow_context = parameters.copy()
        
        for tool_id in tool_ids:
            # Use results from previous tools as context
            tool_params = self._prepare_workflow_parameters(tool_id, workflow_context, results)
            
            result = await self.execute_tool(tool_id, tool_params, context)
            results.append(result)
            
            # Add result to workflow context for next tool
            if result.success:
                workflow_context[f"{tool_id}_result"] = result.result
            else:
                # Stop workflow on failure
                logger.warning(f"Workflow {workflow_name} stopped due to tool failure: {tool_id}")
                break
        
        return results
    
    def _prepare_workflow_parameters(self, tool_id: str, workflow_context: Dict[str, Any],
                                   previous_results: List[UnifiedExecutionResult]) -> Dict[str, Any]:
        """Prepare parameters for a tool in a workflow"""
        tool = self.tools[tool_id]
        tool_params = {}
        
        # Map workflow context to tool parameters
        for param_name in tool.parameters.keys():
            if param_name in workflow_context:
                tool_params[param_name] = workflow_context[param_name]
        
        # Use results from previous tools if needed
        for i, result in enumerate(previous_results):
            if result.success and result.result:
                # Simple mapping - could be more sophisticated
                if isinstance(result.result, dict):
                    for key, value in result.result.items():
                        if key in tool.parameters:
                            tool_params[key] = value
        
        return tool_params
    
    def register_workflow(self, workflow_name: str, tool_ids: List[str]):
        """Register a cross-tool workflow"""
        # Validate all tools exist
        for tool_id in tool_ids:
            if tool_id not in self.tools:
                raise ValueError(f"Tool not found in workflow: {tool_id}")
        
        self.workflows[workflow_name] = tool_ids
        logger.info(f"Registered workflow: {workflow_name} with {len(tool_ids)} tools")
    
    def list_tools(self, integration_type: Optional[IntegrationType] = None,
                  category: Optional[ToolCategory] = None) -> List[Dict[str, Any]]:
        """List available tools with optional filtering"""
        tools = []
        
        for tool_id, tool in self.tools.items():
            if integration_type and tool.integration_type != integration_type:
                continue
            if category and tool.category != category:
                continue
            
            tools.append({
                "id": tool.id,
                "name": tool.name,
                "description": tool.description,
                "integration_type": tool.integration_type.value,
                "category": tool.category.value,
                "parameters": tool.parameters,
                "required_permissions": tool.required_permissions
            })
        
        return tools
    
    def list_integrations(self) -> Dict[str, List[str]]:
        """List available integrations and their tools"""
        return {
            integration_type.value: tool_ids
            for integration_type, tool_ids in self.tool_categories.items()
        }
    
    def list_workflows(self) -> Dict[str, List[str]]:
        """List available workflows"""
        return self.workflows.copy()
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        return self.execution_history[-limit:]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all integrations"""
        health_status = {
            "overall_status": "healthy",
            "integrations": {},
            "total_tools": len(self.tools),
            "active_workflows": len(self.workflows)
        }
        
        # Check MCP server status
        if self.mcp_manager:
            servers = self.mcp_manager.list_servers()
            for server in servers:
                health_status["integrations"][server["name"]] = {
                    "status": server["status"],
                    "tools_count": server["tools_count"]
                }
                
                if server["status"] != "running":
                    health_status["overall_status"] = "degraded"
        
        return health_status

# Global unified tool interface instance
_unified_tool_interface: Optional[UnifiedToolInterface] = None

async def get_unified_tool_interface() -> UnifiedToolInterface:
    """Get the global unified tool interface instance"""
    global _unified_tool_interface
    if _unified_tool_interface is None:
        _unified_tool_interface = UnifiedToolInterface()
        await _unified_tool_interface.initialize()
    return _unified_tool_interface