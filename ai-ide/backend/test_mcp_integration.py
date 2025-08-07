"""
Tests for MCP Integration System
"""

import pytest
import asyncio
import tempfile
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from mcp_integration import (
    MCPServerManager, MCPServer, MCPTool, MCPServerStatus, MCPToolType,
    MCPExecutionResult, get_mcp_manager
)
from unified_tool_interface import (
    UnifiedToolInterface, UnifiedTool, IntegrationType, ToolCategory,
    ToolExecutionContext, UnifiedExecutionResult, get_unified_tool_interface
)
from tool_security_sandbox import (
    SecurityValidator, ToolSandbox, SecurityPolicy, SecurityLevel,
    SandboxConfig, SandboxType, ValidationResult, get_security_validator, get_tool_sandbox
)

class TestMCPServerManager:
    """Test cases for MCPServerManager"""
    
    @pytest.fixture
    def mcp_manager(self):
        """Create an MCP manager for testing"""
        return MCPServerManager()
    
    @pytest.fixture
    def sample_config(self):
        """Sample MCP configuration"""
        return {
            "mcpServers": {
                "test_server": {
                    "command": "uvx",
                    "args": ["test-mcp-server"],
                    "env": {"TEST_VAR": "test_value"},
                    "disabled": False,
                    "autoApprove": ["safe_tool"]
                }
            }
        }
    
    def test_mcp_manager_initialization(self, mcp_manager):
        """Test MCPServerManager initialization"""
        assert mcp_manager.servers == {}
        assert mcp_manager.tools == {}
        assert mcp_manager.execution_history == []
        assert mcp_manager.sandbox_enabled is True
        assert mcp_manager.max_execution_time == 60.0
    
    @pytest.mark.asyncio
    async def test_load_configuration(self, mcp_manager, sample_config):
        """Test loading MCP configuration"""
        # Mock configuration file
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open_with_content(json.dumps(sample_config))):
                await mcp_manager.load_configuration()
        
        assert "test_server" in mcp_manager.servers
        server = mcp_manager.servers["test_server"]
        assert server.name == "test_server"
        assert server.command == "uvx"
        assert server.args == ["test-mcp-server"]
        assert server.env == {"TEST_VAR": "test_value"}
        assert server.disabled is False
        assert "safe_tool" in server.auto_approve
    
    @pytest.mark.asyncio
    async def test_discover_servers(self, mcp_manager):
        """Test server discovery"""
        # Add a test server
        mcp_manager.servers["test_server"] = MCPServer(
            name="test_server",
            command="test_command",
            args=["arg1", "arg2"]
        )
        
        await mcp_manager.discover_servers()
        
        # Server should be validated
        server = mcp_manager.servers["test_server"]
        assert server.status != MCPServerStatus.ERROR
    
    @pytest.mark.asyncio
    async def test_start_server_success(self, mcp_manager):
        """Test successful server startup"""
        # Create a test server
        server = MCPServer(
            name="test_server",
            command="echo",
            args=["test"]
        )
        mcp_manager.servers["test_server"] = server
        
        # Mock subprocess
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process is running
        
        with patch('subprocess.Popen', return_value=mock_process):
            result = await mcp_manager.start_server("test_server")
        
        assert result is True
        assert server.status == MCPServerStatus.RUNNING
        assert server.process == mock_process
    
    @pytest.mark.asyncio
    async def test_start_server_failure(self, mcp_manager):
        """Test server startup failure"""
        # Create a test server
        server = MCPServer(
            name="test_server",
            command="nonexistent_command",
            args=[]
        )
        mcp_manager.servers["test_server"] = server
        
        # Mock subprocess to raise exception
        with patch('subprocess.Popen', side_effect=FileNotFoundError("Command not found")):
            result = await mcp_manager.start_server("test_server")
        
        assert result is False
        assert server.status == MCPServerStatus.ERROR
        assert "Command not found" in server.last_error
    
    @pytest.mark.asyncio
    async def test_discover_tools(self, mcp_manager):
        """Test tool discovery from servers"""
        # Create a running server
        server = MCPServer(
            name="filesystem",
            command="test_command",
            status=MCPServerStatus.RUNNING
        )
        mcp_manager.servers["filesystem"] = server
        
        await mcp_manager.discover_tools()
        
        # Check that tools were discovered
        assert len(server.tools) > 0
        assert "filesystem.read_file" in mcp_manager.tools
        
        # Verify tool properties
        read_file_tool = mcp_manager.tools["filesystem.read_file"]
        assert read_file_tool.name == "read_file"
        assert read_file_tool.server_name == "filesystem"
        assert read_file_tool.tool_type == MCPToolType.FUNCTION
    
    @pytest.mark.asyncio
    async def test_execute_tool_success(self, mcp_manager):
        """Test successful tool execution"""
        # Set up a tool
        tool = MCPTool(
            name="read_file",
            description="Read a file",
            tool_type=MCPToolType.FUNCTION,
            server_name="filesystem",
            parameters={"file_path": {"type": "string"}}
        )
        mcp_manager.tools["filesystem.read_file"] = tool
        
        # Set up a running server
        server = MCPServer(
            name="filesystem",
            command="test_command",
            status=MCPServerStatus.RUNNING
        )
        mcp_manager.servers["filesystem"] = server
        
        # Execute tool
        result = await mcp_manager.execute_tool(
            "filesystem.read_file",
            {"file_path": "/test/file.txt"}
        )
        
        assert result.success is True
        assert result.server_name == "filesystem"
        assert result.tool_name == "filesystem.read_file"
        assert "Mock content of file" in result.result
    
    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self, mcp_manager):
        """Test tool execution with non-existent tool"""
        result = await mcp_manager.execute_tool(
            "nonexistent.tool",
            {}
        )
        
        assert result.success is False
        assert "Tool not found" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_tool_server_not_running(self, mcp_manager):
        """Test tool execution with stopped server"""
        # Set up a tool with stopped server
        tool = MCPTool(
            name="test_tool",
            description="Test tool",
            tool_type=MCPToolType.FUNCTION,
            server_name="stopped_server"
        )
        mcp_manager.tools["stopped_server.test_tool"] = tool
        
        server = MCPServer(
            name="stopped_server",
            command="test_command",
            status=MCPServerStatus.STOPPED
        )
        mcp_manager.servers["stopped_server"] = server
        
        result = await mcp_manager.execute_tool(
            "stopped_server.test_tool",
            {}
        )
        
        assert result.success is False
        assert "is not running" in result.error
    
    def test_get_server_status(self, mcp_manager):
        """Test getting server status"""
        server = MCPServer(
            name="test_server",
            command="test_command",
            status=MCPServerStatus.RUNNING,
            disabled=False
        )
        server.tools = [Mock(), Mock()]  # 2 mock tools
        mcp_manager.servers["test_server"] = server
        
        status = mcp_manager.get_server_status("test_server")
        
        assert status is not None
        assert status["name"] == "test_server"
        assert status["status"] == "running"
        assert status["disabled"] is False
        assert status["tools_count"] == 2
    
    def test_list_servers(self, mcp_manager):
        """Test listing servers"""
        server1 = MCPServer(name="server1", command="cmd1", status=MCPServerStatus.RUNNING)
        server2 = MCPServer(name="server2", command="cmd2", status=MCPServerStatus.STOPPED, disabled=True)
        
        mcp_manager.servers["server1"] = server1
        mcp_manager.servers["server2"] = server2
        
        servers = mcp_manager.list_servers()
        
        assert len(servers) == 2
        assert any(s["name"] == "server1" and s["status"] == "running" for s in servers)
        assert any(s["name"] == "server2" and s["disabled"] is True for s in servers)
    
    def test_list_tools(self, mcp_manager):
        """Test listing tools"""
        tool1 = MCPTool(name="tool1", description="Tool 1", tool_type=MCPToolType.FUNCTION, server_name="server1")
        tool2 = MCPTool(name="tool2", description="Tool 2", tool_type=MCPToolType.RESOURCE, server_name="server2")
        
        mcp_manager.tools["server1.tool1"] = tool1
        mcp_manager.tools["server2.tool2"] = tool2
        
        # List all tools
        all_tools = mcp_manager.list_tools()
        assert len(all_tools) == 2
        
        # List tools for specific server
        server1_tools = mcp_manager.list_tools("server1")
        assert len(server1_tools) == 1
        assert server1_tools[0]["name"] == "tool1"

class TestUnifiedToolInterface:
    """Test cases for UnifiedToolInterface"""
    
    @pytest.fixture
    async def unified_interface(self):
        """Create a unified tool interface for testing"""
        interface = UnifiedToolInterface()
        # Mock MCP manager initialization
        with patch('unified_tool_interface.get_mcp_manager', new_callable=AsyncMock):
            await interface.initialize()
        return interface
    
    @pytest.mark.asyncio
    async def test_unified_interface_initialization(self, unified_interface):
        """Test UnifiedToolInterface initialization"""
        assert len(unified_interface.tools) > 0
        assert len(unified_interface.tool_categories) > 0
        
        # Check that different integration types are registered
        assert IntegrationType.VERSION_CONTROL in unified_interface.tool_categories
        assert IntegrationType.PROJECT_MANAGEMENT in unified_interface.tool_categories
        assert IntegrationType.COMMUNICATION in unified_interface.tool_categories
    
    @pytest.mark.asyncio
    async def test_register_tool(self, unified_interface):
        """Test tool registration"""
        tool = UnifiedTool(
            id="test.custom_tool",
            name="Custom Tool",
            description="A custom test tool",
            integration_type=IntegrationType.DEVELOPMENT_TOOLS,
            category=ToolCategory.EXECUTE,
            mcp_server="test_server",
            mcp_tool="custom_tool"
        )
        
        await unified_interface.register_tool(tool)
        
        assert "test.custom_tool" in unified_interface.tools
        assert "test.custom_tool" in unified_interface.tool_categories[IntegrationType.DEVELOPMENT_TOOLS]
    
    @pytest.mark.asyncio
    async def test_execute_tool_success(self, unified_interface):
        """Test successful tool execution"""
        # Mock MCP manager
        mock_mcp_result = MCPExecutionResult(
            success=True,
            result="Mock execution result",
            execution_time=1.5,
            server_name="test_server",
            tool_name="test_tool"
        )
        
        unified_interface.mcp_manager = Mock()
        unified_interface.mcp_manager.execute_tool = AsyncMock(return_value=mock_mcp_result)
        
        # Execute a GitHub tool (should exist from default registration)
        context = ToolExecutionContext(
            user_id="test_user",
            permissions=["github:read"]
        )
        
        result = await unified_interface.execute_tool(
            "github.list_repositories",
            {"owner": "test_owner"},
            context
        )
        
        assert result.success is True
        assert result.result == "Mock execution result"
        assert result.integration_type == IntegrationType.VERSION_CONTROL
    
    @pytest.mark.asyncio
    async def test_execute_tool_security_failure(self, unified_interface):
        """Test tool execution with security validation failure"""
        # Try to execute a tool without required permissions
        context = ToolExecutionContext(
            user_id="test_user",
            permissions=[]  # No permissions
        )
        
        result = await unified_interface.execute_tool(
            "github.create_issue",
            {"owner": "test_owner", "repo": "test_repo", "title": "Test Issue"},
            context
        )
        
        assert result.success is False
        assert "Security validation failed" in result.error
    
    def test_list_tools_filtering(self, unified_interface):
        """Test tool listing with filtering"""
        # List all tools
        all_tools = unified_interface.list_tools()
        assert len(all_tools) > 0
        
        # List only version control tools
        vc_tools = unified_interface.list_tools(integration_type=IntegrationType.VERSION_CONTROL)
        assert all(tool["integration_type"] == "version_control" for tool in vc_tools)
        
        # List only read operations
        read_tools = unified_interface.list_tools(category=ToolCategory.READ)
        assert all(tool["category"] == "read" for tool in read_tools)
    
    def test_list_integrations(self, unified_interface):
        """Test listing integrations"""
        integrations = unified_interface.list_integrations()
        
        assert "version_control" in integrations
        assert "project_management" in integrations
        assert "communication" in integrations
        
        # Check that each integration has tools
        for integration_type, tool_ids in integrations.items():
            assert len(tool_ids) > 0

class TestSecurityValidator:
    """Test cases for SecurityValidator"""
    
    @pytest.fixture
    def security_validator(self):
        """Create a security validator for testing"""
        return SecurityValidator()
    
    def test_security_validator_initialization(self, security_validator):
        """Test SecurityValidator initialization"""
        assert len(security_validator.policies) > 0
        assert len(security_validator.threat_patterns) > 0
        
        # Check default policies exist
        assert "low_security" in security_validator.policies
        assert "medium_security" in security_validator.policies
        assert "high_security" in security_validator.policies
        assert "critical_security" in security_validator.policies
    
    def test_validate_safe_request(self, security_validator):
        """Test validation of safe request"""
        result = security_validator.validate_execution_request(
            "github.list_repositories",
            {"owner": "test_owner", "type": "all"},
            "medium_security"
        )
        
        assert result["result"] == ValidationResult.ALLOWED
        assert len(result["violations"]) == 0
    
    def test_validate_dangerous_request(self, security_validator):
        """Test validation of dangerous request"""
        result = security_validator.validate_execution_request(
            "system.execute_command",
            {"command": "rm -rf /"},
            "medium_security"
        )
        
        assert result["result"] == ValidationResult.BLOCKED
        assert len(result["violations"]) > 0
        
        # Check that threat patterns were detected
        violation_types = [v["type"] for v in result["violations"]]
        assert "threat_pattern" in violation_types
    
    def test_validate_command_injection(self, security_validator):
        """Test detection of command injection attempts"""
        result = security_validator.validate_execution_request(
            "file.read",
            {"file_path": "/etc/passwd; cat /etc/shadow"},
            "high_security"
        )
        
        assert result["result"] == ValidationResult.BLOCKED
        
        # Check for command injection detection
        violations = result["violations"]
        assert any(v["name"] == "command_injection" for v in violations)
    
    def test_validate_path_traversal(self, security_validator):
        """Test detection of path traversal attempts"""
        result = security_validator.validate_execution_request(
            "file.read",
            {"file_path": "../../../etc/passwd"},
            "medium_security"
        )
        
        assert result["result"] in [ValidationResult.BLOCKED, ValidationResult.REQUIRES_APPROVAL]
        
        # Check for path traversal detection
        violations = result["violations"]
        assert any(v["name"] == "path_traversal" for v in violations)
    
    def test_policy_enforcement(self, security_validator):
        """Test security policy enforcement"""
        # High security policy should require approval for most operations
        result = security_validator.validate_execution_request(
            "safe.operation",
            {"param": "safe_value"},
            "high_security"
        )
        
        # High security policy requires approval by default
        assert result["result"] == ValidationResult.REQUIRES_APPROVAL
        
        # Critical security policy should be even more restrictive
        result = security_validator.validate_execution_request(
            "safe.operation",
            {"param": "safe_value"},
            "critical_security"
        )
        
        assert result["result"] == ValidationResult.REQUIRES_APPROVAL
    
    def test_get_validation_statistics(self, security_validator):
        """Test validation statistics"""
        # Perform some validations
        security_validator.validate_execution_request("tool1", {}, "low_security")
        security_validator.validate_execution_request("tool2", {"cmd": "rm -rf /"}, "high_security")
        security_validator.validate_execution_request("tool3", {}, "medium_security")
        
        stats = security_validator.get_validation_statistics()
        
        assert stats["total_validations"] == 3
        assert "allowed" in stats
        assert "blocked" in stats
        assert "requires_approval" in stats
        assert "allowed_percentage" in stats

class TestToolSandbox:
    """Test cases for ToolSandbox"""
    
    @pytest.fixture
    def tool_sandbox(self):
        """Create a tool sandbox for testing"""
        return ToolSandbox()
    
    @pytest.mark.asyncio
    async def test_create_sandbox(self, tool_sandbox):
        """Test sandbox creation"""
        sandbox_id = await tool_sandbox.create_sandbox()
        
        assert sandbox_id.startswith("sandbox_")
        assert sandbox_id in tool_sandbox.active_sandboxes
        
        sandbox_info = tool_sandbox.active_sandboxes[sandbox_id]
        assert sandbox_info["active"] is True
        assert os.path.exists(sandbox_info["directory"])
    
    @pytest.mark.asyncio
    async def test_execute_in_sandbox(self, tool_sandbox):
        """Test command execution in sandbox"""
        sandbox_id = await tool_sandbox.create_sandbox()
        
        # Execute a simple command
        result = await tool_sandbox.execute_in_sandbox(
            sandbox_id,
            ["echo", "Hello, Sandbox!"]
        )
        
        assert result.success is True
        assert result.return_code == 0
        assert "Hello, Sandbox!" in result.stdout
        assert result.sandbox_id == sandbox_id
    
    @pytest.mark.asyncio
    async def test_execute_with_timeout(self, tool_sandbox):
        """Test command execution with timeout"""
        config = SandboxConfig(
            sandbox_type=SandboxType.PROCESS,
            working_directory="/tmp",
            resource_limits={"timeout_seconds": 1}
        )
        
        sandbox_id = await tool_sandbox.create_sandbox(config)
        
        # Execute a command that should timeout
        result = await tool_sandbox.execute_in_sandbox(
            sandbox_id,
            ["sleep", "5"]
        )
        
        assert result.success is False
        assert "timed out" in result.stderr
    
    @pytest.mark.asyncio
    async def test_destroy_sandbox(self, tool_sandbox):
        """Test sandbox destruction"""
        sandbox_id = await tool_sandbox.create_sandbox()
        sandbox_dir = tool_sandbox.active_sandboxes[sandbox_id]["directory"]
        
        # Verify sandbox exists
        assert os.path.exists(sandbox_dir)
        assert sandbox_id in tool_sandbox.active_sandboxes
        
        # Destroy sandbox
        result = await tool_sandbox.destroy_sandbox(sandbox_id)
        
        assert result is True
        assert not os.path.exists(sandbox_dir)
        assert sandbox_id not in tool_sandbox.active_sandboxes
    
    def test_list_active_sandboxes(self, tool_sandbox):
        """Test listing active sandboxes"""
        # Initially no sandboxes
        sandboxes = tool_sandbox.list_active_sandboxes()
        assert len(sandboxes) == 0
    
    @pytest.mark.asyncio
    async def test_cleanup_all_sandboxes(self, tool_sandbox):
        """Test cleaning up all sandboxes"""
        # Create multiple sandboxes
        sandbox1 = await tool_sandbox.create_sandbox()
        sandbox2 = await tool_sandbox.create_sandbox()
        
        assert len(tool_sandbox.active_sandboxes) == 2
        
        # Cleanup all
        await tool_sandbox.cleanup_all_sandboxes()
        
        assert len(tool_sandbox.active_sandboxes) == 0

# Helper functions
def mock_open_with_content(content):
    """Create a mock open function that returns specific content"""
    from unittest.mock import mock_open
    return mock_open(read_data=content)

if __name__ == "__main__":
    pytest.main([__file__])