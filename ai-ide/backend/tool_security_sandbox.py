"""
Tool Execution Sandboxing and Security Validation for AI IDE
Provides secure execution environment for external tools
"""

import asyncio
import logging
import tempfile
import shutil
import os
import subprocess
import time
import resource
import signal
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import hashlib

logger = logging.getLogger('tool_security_sandbox')

class SecurityLevel(Enum):
    """Security levels for tool execution"""
    LOW = "low"          # Basic validation only
    MEDIUM = "medium"    # Sandboxed execution with resource limits
    HIGH = "high"        # Strict sandboxing with network isolation
    CRITICAL = "critical" # Maximum security with audit logging

class SandboxType(Enum):
    """Types of sandboxing available"""
    PROCESS = "process"      # Process-level isolation
    CONTAINER = "container"  # Container-based isolation
    VM = "vm"               # Virtual machine isolation
    CHROOT = "chroot"       # Chroot jail isolation

class ValidationResult(Enum):
    """Results of security validation"""
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    REQUIRES_APPROVAL = "requires_approval"
    QUARANTINED = "quarantined"

@dataclass
class SecurityPolicy:
    """Security policy for tool execution"""
    name: str
    description: str
    security_level: SecurityLevel
    allowed_operations: List[str] = field(default_factory=list)
    blocked_operations: List[str] = field(default_factory=list)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    network_access: bool = False
    file_system_access: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 30
    requires_approval: bool = False

@dataclass
class SandboxConfig:
    """Configuration for sandbox execution"""
    sandbox_type: SandboxType
    working_directory: str
    allowed_paths: List[str] = field(default_factory=list)
    blocked_paths: List[str] = field(default_factory=list)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    network_isolation: bool = True
    environment_variables: Dict[str, str] = field(default_factory=dict)
    cleanup_on_exit: bool = True

@dataclass
class ExecutionResult:
    """Result of sandboxed execution"""
    success: bool
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""
    execution_time: float = 0.0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    security_violations: List[str] = field(default_factory=list)
    sandbox_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class SecurityValidator:
    """Validates tool execution requests against security policies"""
    
    def __init__(self):
        self.policies: Dict[str, SecurityPolicy] = {}
        self.validation_history: List[Dict[str, Any]] = []
        self.threat_patterns: List[Dict[str, Any]] = []
        
        # Initialize default policies
        self._initialize_default_policies()
        self._load_threat_patterns()
    
    def _initialize_default_policies(self):
        """Initialize default security policies"""
        
        # Low security policy
        self.policies["low_security"] = SecurityPolicy(
            name="low_security",
            description="Basic security validation",
            security_level=SecurityLevel.LOW,
            allowed_operations=["read", "list", "search", "analyze"],
            blocked_operations=["delete", "format", "shutdown", "reboot"],
            resource_limits={"memory_mb": 512, "cpu_percent": 50},
            network_access=True,
            timeout_seconds=60
        )
        
        # Medium security policy
        self.policies["medium_security"] = SecurityPolicy(
            name="medium_security",
            description="Sandboxed execution with resource limits",
            security_level=SecurityLevel.MEDIUM,
            allowed_operations=["read", "write", "list", "search", "analyze", "execute"],
            blocked_operations=["delete", "format", "shutdown", "reboot", "install", "uninstall"],
            resource_limits={"memory_mb": 256, "cpu_percent": 25, "disk_mb": 100},
            network_access=False,
            file_system_access=["/tmp", "/var/tmp"],
            timeout_seconds=30
        )
        
        # High security policy
        self.policies["high_security"] = SecurityPolicy(
            name="high_security",
            description="Strict sandboxing with network isolation",
            security_level=SecurityLevel.HIGH,
            allowed_operations=["read", "list", "search"],
            blocked_operations=["write", "delete", "execute", "install", "network"],
            resource_limits={"memory_mb": 128, "cpu_percent": 10, "disk_mb": 50},
            network_access=False,
            file_system_access=["/tmp/sandbox"],
            timeout_seconds=15,
            requires_approval=True
        )
        
        # Critical security policy
        self.policies["critical_security"] = SecurityPolicy(
            name="critical_security",
            description="Maximum security with audit logging",
            security_level=SecurityLevel.CRITICAL,
            allowed_operations=["read"],
            blocked_operations=["write", "delete", "execute", "network", "system"],
            resource_limits={"memory_mb": 64, "cpu_percent": 5, "disk_mb": 10},
            network_access=False,
            file_system_access=[],
            timeout_seconds=10,
            requires_approval=True
        )
    
    def _load_threat_patterns(self):
        """Load known threat patterns for detection"""
        self.threat_patterns = [
            {
                "name": "command_injection",
                "pattern": r"[;&|`$(){}[\]<>]",
                "description": "Potential command injection characters",
                "severity": "high"
            },
            {
                "name": "path_traversal",
                "pattern": r"\.\./",
                "description": "Path traversal attempt",
                "severity": "high"
            },
            {
                "name": "system_commands",
                "pattern": r"\b(rm|del|format|shutdown|reboot|kill|killall)\b",
                "description": "Dangerous system commands",
                "severity": "critical"
            },
            {
                "name": "network_access",
                "pattern": r"\b(curl|wget|nc|netcat|telnet|ssh)\b",
                "description": "Network access commands",
                "severity": "medium"
            },
            {
                "name": "file_operations",
                "pattern": r"\b(chmod|chown|mount|umount)\b",
                "description": "System file operations",
                "severity": "medium"
            },
            {
                "name": "privilege_escalation",
                "pattern": r"\b(sudo|su|passwd|useradd|usermod)\b",
                "description": "Privilege escalation attempts",
                "severity": "critical"
            }
        ]
    
    def validate_execution_request(self, tool_name: str, parameters: Dict[str, Any],
                                 policy_name: str = "medium_security") -> Dict[str, Any]:
        """Validate a tool execution request against security policy"""
        
        if policy_name not in self.policies:
            return {
                "result": ValidationResult.BLOCKED,
                "reason": f"Unknown security policy: {policy_name}"
            }
        
        policy = self.policies[policy_name]
        validation_result = {
            "result": ValidationResult.ALLOWED,
            "reason": "",
            "violations": [],
            "recommendations": []
        }
        
        # Check for threat patterns
        threat_violations = self._check_threat_patterns(parameters)
        if threat_violations:
            validation_result["violations"].extend(threat_violations)
            
            # Determine action based on severity
            critical_threats = [v for v in threat_violations if v["severity"] == "critical"]
            if critical_threats:
                validation_result["result"] = ValidationResult.BLOCKED
                validation_result["reason"] = "Critical security threats detected"
            elif policy.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                validation_result["result"] = ValidationResult.BLOCKED
                validation_result["reason"] = "Security threats detected in high security mode"
            else:
                validation_result["result"] = ValidationResult.REQUIRES_APPROVAL
                validation_result["reason"] = "Security threats detected, approval required"
        
        # Check operation permissions
        operation_violations = self._check_operation_permissions(tool_name, parameters, policy)
        if operation_violations:
            validation_result["violations"].extend(operation_violations)
            if validation_result["result"] == ValidationResult.ALLOWED:
                validation_result["result"] = ValidationResult.REQUIRES_APPROVAL
                validation_result["reason"] = "Operation requires approval"
        
        # Check resource requirements
        resource_violations = self._check_resource_requirements(parameters, policy)
        if resource_violations:
            validation_result["violations"].extend(resource_violations)
            validation_result["recommendations"].extend([
                "Consider reducing resource requirements",
                "Use a lower security policy if appropriate"
            ])
        
        # Apply policy-specific rules
        if policy.requires_approval and validation_result["result"] == ValidationResult.ALLOWED:
            validation_result["result"] = ValidationResult.REQUIRES_APPROVAL
            validation_result["reason"] = "Policy requires approval for all operations"
        
        # Record validation
        self._record_validation(tool_name, parameters, policy_name, validation_result)
        
        return validation_result
    
    def _check_threat_patterns(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check parameters against known threat patterns"""
        violations = []
        
        # Convert all parameter values to strings for pattern matching
        param_strings = []
        for key, value in parameters.items():
            if isinstance(value, (str, int, float)):
                param_strings.append(str(value))
            elif isinstance(value, (list, dict)):
                param_strings.append(json.dumps(value))
        
        combined_params = " ".join(param_strings)
        
        # Check each threat pattern
        import re
        for pattern_info in self.threat_patterns:
            pattern = pattern_info["pattern"]
            if re.search(pattern, combined_params, re.IGNORECASE):
                violations.append({
                    "type": "threat_pattern",
                    "name": pattern_info["name"],
                    "description": pattern_info["description"],
                    "severity": pattern_info["severity"],
                    "matched_pattern": pattern
                })
        
        return violations
    
    def _check_operation_permissions(self, tool_name: str, parameters: Dict[str, Any],
                                   policy: SecurityPolicy) -> List[Dict[str, Any]]:
        """Check if operations are allowed by the security policy"""
        violations = []
        
        # Extract operation type from tool name or parameters
        operation_type = self._extract_operation_type(tool_name, parameters)
        
        # Check against blocked operations
        if operation_type in policy.blocked_operations:
            violations.append({
                "type": "blocked_operation",
                "operation": operation_type,
                "description": f"Operation '{operation_type}' is blocked by policy",
                "severity": "high"
            })
        
        # Check against allowed operations (if whitelist is defined)
        if policy.allowed_operations and operation_type not in policy.allowed_operations:
            violations.append({
                "type": "unauthorized_operation",
                "operation": operation_type,
                "description": f"Operation '{operation_type}' is not in allowed list",
                "severity": "medium"
            })
        
        return violations
    
    def _extract_operation_type(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Extract operation type from tool name and parameters"""
        tool_lower = tool_name.lower()
        
        # Common operation patterns
        if any(op in tool_lower for op in ["read", "get", "fetch", "retrieve"]):
            return "read"
        elif any(op in tool_lower for op in ["write", "create", "post", "put", "save"]):
            return "write"
        elif any(op in tool_lower for op in ["delete", "remove", "drop"]):
            return "delete"
        elif any(op in tool_lower for op in ["list", "search", "find", "query"]):
            return "search"
        elif any(op in tool_lower for op in ["execute", "run", "start", "launch"]):
            return "execute"
        elif any(op in tool_lower for op in ["analyze", "process", "compute"]):
            return "analyze"
        else:
            return "unknown"
    
    def _check_resource_requirements(self, parameters: Dict[str, Any],
                                   policy: SecurityPolicy) -> List[Dict[str, Any]]:
        """Check if resource requirements exceed policy limits"""
        violations = []
        
        # Estimate resource requirements based on parameters
        estimated_memory = self._estimate_memory_usage(parameters)
        estimated_cpu = self._estimate_cpu_usage(parameters)
        estimated_disk = self._estimate_disk_usage(parameters)
        
        # Check against policy limits
        if "memory_mb" in policy.resource_limits:
            if estimated_memory > policy.resource_limits["memory_mb"]:
                violations.append({
                    "type": "resource_limit",
                    "resource": "memory",
                    "estimated": estimated_memory,
                    "limit": policy.resource_limits["memory_mb"],
                    "severity": "medium"
                })
        
        if "cpu_percent" in policy.resource_limits:
            if estimated_cpu > policy.resource_limits["cpu_percent"]:
                violations.append({
                    "type": "resource_limit",
                    "resource": "cpu",
                    "estimated": estimated_cpu,
                    "limit": policy.resource_limits["cpu_percent"],
                    "severity": "medium"
                })
        
        if "disk_mb" in policy.resource_limits:
            if estimated_disk > policy.resource_limits["disk_mb"]:
                violations.append({
                    "type": "resource_limit",
                    "resource": "disk",
                    "estimated": estimated_disk,
                    "limit": policy.resource_limits["disk_mb"],
                    "severity": "medium"
                })
        
        return violations
    
    def _estimate_memory_usage(self, parameters: Dict[str, Any]) -> int:
        """Estimate memory usage in MB based on parameters"""
        # Simple heuristic based on parameter size
        param_size = len(json.dumps(parameters))
        
        # Base memory requirement
        base_memory = 32  # MB
        
        # Additional memory based on parameter complexity
        if param_size > 10000:
            return base_memory + 128
        elif param_size > 1000:
            return base_memory + 64
        else:
            return base_memory
    
    def _estimate_cpu_usage(self, parameters: Dict[str, Any]) -> int:
        """Estimate CPU usage percentage based on parameters"""
        # Simple heuristic
        param_count = len(parameters)
        
        if param_count > 10:
            return 30
        elif param_count > 5:
            return 20
        else:
            return 10
    
    def _estimate_disk_usage(self, parameters: Dict[str, Any]) -> int:
        """Estimate disk usage in MB based on parameters"""
        # Simple heuristic
        param_size = len(json.dumps(parameters))
        
        # Assume temporary files might be created
        return max(10, param_size // 1000)  # MB
    
    def _record_validation(self, tool_name: str, parameters: Dict[str, Any],
                          policy_name: str, validation_result: Dict[str, Any]):
        """Record validation result for analysis"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "policy_name": policy_name,
            "result": validation_result["result"].value,
            "violations_count": len(validation_result["violations"]),
            "parameter_hash": hashlib.md5(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
        }
        
        self.validation_history.append(record)
        
        # Keep only recent history
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        if not self.validation_history:
            return {"total_validations": 0}
        
        total = len(self.validation_history)
        allowed = len([r for r in self.validation_history if r["result"] == "allowed"])
        blocked = len([r for r in self.validation_history if r["result"] == "blocked"])
        requires_approval = len([r for r in self.validation_history if r["result"] == "requires_approval"])
        
        return {
            "total_validations": total,
            "allowed": allowed,
            "blocked": blocked,
            "requires_approval": requires_approval,
            "allowed_percentage": (allowed / total) * 100 if total > 0 else 0,
            "blocked_percentage": (blocked / total) * 100 if total > 0 else 0
        }

class ToolSandbox:
    """Provides sandboxed execution environment for tools"""
    
    def __init__(self):
        self.active_sandboxes: Dict[str, Dict[str, Any]] = {}
        self.sandbox_counter = 0
        self.cleanup_tasks: List[asyncio.Task] = []
        
        # Default sandbox configuration
        self.default_config = SandboxConfig(
            sandbox_type=SandboxType.PROCESS,
            working_directory="/tmp",
            resource_limits={
                "memory_mb": 256,
                "cpu_percent": 25,
                "timeout_seconds": 30
            },
            network_isolation=True,
            cleanup_on_exit=True
        )
    
    async def create_sandbox(self, config: Optional[SandboxConfig] = None) -> str:
        """Create a new sandbox environment"""
        config = config or self.default_config
        
        self.sandbox_counter += 1
        sandbox_id = f"sandbox_{self.sandbox_counter}_{int(time.time())}"
        
        # Create temporary directory for sandbox
        sandbox_dir = tempfile.mkdtemp(prefix=f"ai_ide_sandbox_{sandbox_id}_")
        
        sandbox_info = {
            "id": sandbox_id,
            "config": config,
            "directory": sandbox_dir,
            "created_at": datetime.now(),
            "active": True,
            "processes": [],
            "resource_usage": {}
        }
        
        self.active_sandboxes[sandbox_id] = sandbox_info
        
        # Schedule cleanup if configured
        if config.cleanup_on_exit:
            cleanup_task = asyncio.create_task(
                self._schedule_cleanup(sandbox_id, config.resource_limits.get("timeout_seconds", 30))
            )
            self.cleanup_tasks.append(cleanup_task)
        
        logger.info(f"Created sandbox: {sandbox_id} in {sandbox_dir}")
        
        return sandbox_id
    
    async def execute_in_sandbox(self, sandbox_id: str, command: List[str],
                                input_data: Optional[str] = None,
                                environment: Optional[Dict[str, str]] = None) -> ExecutionResult:
        """Execute a command in the specified sandbox"""
        
        if sandbox_id not in self.active_sandboxes:
            return ExecutionResult(
                success=False,
                return_code=-1,
                stderr="Sandbox not found",
                sandbox_id=sandbox_id
            )
        
        sandbox_info = self.active_sandboxes[sandbox_id]
        config = sandbox_info["config"]
        
        start_time = time.time()
        
        try:
            # Prepare environment
            env = os.environ.copy()
            if environment:
                env.update(environment)
            if config.environment_variables:
                env.update(config.environment_variables)
            
            # Set resource limits
            def set_limits():
                if "memory_mb" in config.resource_limits:
                    memory_bytes = config.resource_limits["memory_mb"] * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
                
                if "cpu_percent" in config.resource_limits:
                    # CPU limiting would require more complex implementation
                    pass
            
            # Execute command
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=sandbox_info["directory"],
                env=env,
                preexec_fn=set_limits,
                text=True
            )
            
            sandbox_info["processes"].append(process)
            
            # Set timeout
            timeout = config.resource_limits.get("timeout_seconds", 30)
            
            try:
                stdout, stderr = process.communicate(input=input_data, timeout=timeout)
                return_code = process.returncode
                
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                return_code = -1
                stderr += "\nExecution timed out"
            
            execution_time = time.time() - start_time
            
            # Collect resource usage (simplified)
            resource_usage = {
                "execution_time": execution_time,
                "return_code": return_code
            }
            
            # Check for security violations
            security_violations = self._check_execution_violations(stdout, stderr, config)
            
            return ExecutionResult(
                success=return_code == 0,
                return_code=return_code,
                stdout=stdout,
                stderr=stderr,
                execution_time=execution_time,
                resource_usage=resource_usage,
                security_violations=security_violations,
                sandbox_id=sandbox_id
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=False,
                return_code=-1,
                stderr=str(e),
                execution_time=execution_time,
                sandbox_id=sandbox_id
            )
        
        finally:
            # Remove process from tracking
            if 'process' in locals():
                try:
                    sandbox_info["processes"].remove(process)
                except ValueError:
                    pass
    
    def _check_execution_violations(self, stdout: str, stderr: str,
                                  config: SandboxConfig) -> List[str]:
        """Check execution output for security violations"""
        violations = []
        
        combined_output = stdout + stderr
        
        # Check for network access attempts
        if not config.network_isolation:
            network_indicators = ["connection", "socket", "network", "dns", "http"]
            if any(indicator in combined_output.lower() for indicator in network_indicators):
                violations.append("Potential network access detected")
        
        # Check for file system access violations
        if config.blocked_paths:
            for blocked_path in config.blocked_paths:
                if blocked_path in combined_output:
                    violations.append(f"Access to blocked path detected: {blocked_path}")
        
        # Check for privilege escalation attempts
        privilege_indicators = ["sudo", "su", "root", "admin"]
        if any(indicator in combined_output.lower() for indicator in privilege_indicators):
            violations.append("Potential privilege escalation detected")
        
        return violations
    
    async def _schedule_cleanup(self, sandbox_id: str, delay_seconds: int):
        """Schedule sandbox cleanup after delay"""
        await asyncio.sleep(delay_seconds)
        await self.destroy_sandbox(sandbox_id)
    
    async def destroy_sandbox(self, sandbox_id: str) -> bool:
        """Destroy a sandbox and clean up resources"""
        if sandbox_id not in self.active_sandboxes:
            return False
        
        sandbox_info = self.active_sandboxes[sandbox_id]
        
        try:
            # Kill any remaining processes
            for process in sandbox_info["processes"]:
                if process.poll() is None:
                    process.terminate()
                    await asyncio.sleep(1)
                    if process.poll() is None:
                        process.kill()
            
            # Remove sandbox directory
            if os.path.exists(sandbox_info["directory"]):
                shutil.rmtree(sandbox_info["directory"])
            
            # Mark as inactive
            sandbox_info["active"] = False
            
            # Remove from active sandboxes
            del self.active_sandboxes[sandbox_id]
            
            logger.info(f"Destroyed sandbox: {sandbox_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to destroy sandbox {sandbox_id}: {e}")
            return False
    
    def list_active_sandboxes(self) -> List[Dict[str, Any]]:
        """List all active sandboxes"""
        return [
            {
                "id": sandbox_id,
                "created_at": info["created_at"].isoformat(),
                "directory": info["directory"],
                "active_processes": len(info["processes"]),
                "sandbox_type": info["config"].sandbox_type.value
            }
            for sandbox_id, info in self.active_sandboxes.items()
            if info["active"]
        ]
    
    async def cleanup_all_sandboxes(self):
        """Clean up all active sandboxes"""
        sandbox_ids = list(self.active_sandboxes.keys())
        
        cleanup_tasks = [
            self.destroy_sandbox(sandbox_id)
            for sandbox_id in sandbox_ids
        ]
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Cancel cleanup tasks
        for task in self.cleanup_tasks:
            if not task.done():
                task.cancel()
        
        self.cleanup_tasks.clear()
        
        logger.info("All sandboxes cleaned up")

# Global instances
_security_validator: Optional[SecurityValidator] = None
_tool_sandbox: Optional[ToolSandbox] = None

def get_security_validator() -> SecurityValidator:
    """Get the global security validator instance"""
    global _security_validator
    if _security_validator is None:
        _security_validator = SecurityValidator()
    return _security_validator

def get_tool_sandbox() -> ToolSandbox:
    """Get the global tool sandbox instance"""
    global _tool_sandbox
    if _tool_sandbox is None:
        _tool_sandbox = ToolSandbox()
    return _tool_sandbox