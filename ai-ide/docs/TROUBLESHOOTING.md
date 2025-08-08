# AI IDE Troubleshooting Guide

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Installation Issues](#installation-issues)
3. [Backend Connection Problems](#backend-connection-problems)
4. [Agent System Issues](#agent-system-issues)
5. [Performance Problems](#performance-problems)
6. [AI Model Issues](#ai-model-issues)
7. [Database Problems](#database-problems)
8. [Extension Issues](#extension-issues)
9. [Network and Connectivity](#network-and-connectivity)
10. [Advanced Debugging](#advanced-debugging)

## Quick Diagnostics

### Health Check Commands

```bash
# Check overall system health
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health/detailed

# Check specific services
curl http://localhost:8001/health  # Web search service
curl http://localhost:8002/health  # RAG service
curl http://localhost:8003/health  # MCP service
```

### System Status

```bash
# Check running processes
ps aux | grep -E "(python|node)" | grep -E "(main.py|ai-ide)"

# Check port usage
netstat -tulpn | grep -E "(8000|8001|8002|8003)"

# Check system resources
free -h  # Memory usage
df -h    # Disk usage
top      # CPU usage
```

### VSCode Extension Status

1. Open Command Palette (`Ctrl+Shift+P`)
2. Run "AI Assistant: Show System Status"
3. Check extension logs: "Developer: Show Logs" → "Extension Host"

## Installation Issues

### Python Version Problems

**Issue**: Wrong Python version or multiple Python installations

**Symptoms**:
- Import errors during installation
- "Python 3.11+ required" messages
- Module not found errors

**Solutions**:
```bash
# Check Python version
python --version
python3 --version
python3.11 --version

# Use specific Python version
python3.11 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Verify virtual environment
which python
python --version
```

### Node.js and npm Issues

**Issue**: Node.js version compatibility or npm permission errors

**Symptoms**:
- "Node.js 18+ required" messages
- npm permission denied errors
- Extension compilation failures

**Solutions**:
```bash
# Check Node.js version
node --version
npm --version

# Fix npm permissions (Linux/macOS)
sudo chown -R $(whoami) ~/.npm
sudo chown -R $(whoami) /usr/local/lib/node_modules

# Use Node Version Manager
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
nvm use 18
```

### Package Installation Failures

**Issue**: Failed to install Python or npm packages

**Symptoms**:
- Timeout errors during installation
- Network connection errors
- Dependency conflicts

**Solutions**:
```bash
# Clear package caches
pip cache purge
npm cache clean --force

# Use alternative package sources
pip install -r requirements.txt -i https://pypi.org/simple/
npm install --registry https://registry.npmjs.org/

# Install with verbose output
pip install -r requirements.txt -v
npm install --verbose

# Fix dependency conflicts
pip install --upgrade pip setuptools wheel
npm audit fix
```

## Backend Connection Problems

### Service Not Starting

**Issue**: Backend service fails to start

**Symptoms**:
- "Connection refused" errors
- No response from health check
- Process exits immediately

**Diagnostic Steps**:
```bash
# Check if port is already in use
lsof -i :8000  # Linux/macOS
netstat -ano | findstr :8000  # Windows

# Start backend with verbose logging
cd backend
source venv/bin/activate
python main.py --debug --log-level DEBUG

# Check for configuration errors
python -c "
import yaml
with open('config/development.yaml') as f:
    config = yaml.safe_load(f)
    print('Configuration loaded successfully')
"
```

**Common Solutions**:
```bash
# Kill processes using the port
sudo kill -9 $(lsof -t -i:8000)  # Linux/macOS
# Windows: Use Task Manager or:
taskkill /F /PID <PID>

# Reset configuration
cp config/development.yaml.template config/development.yaml

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### Database Connection Errors

**Issue**: Cannot connect to PostgreSQL database

**Symptoms**:
- "Connection to database failed" errors
- "Role does not exist" errors
- "Database does not exist" errors

**Solutions**:
```bash
# Check PostgreSQL status
sudo systemctl status postgresql  # Linux
brew services list | grep postgresql  # macOS
# Windows: Check Services in Task Manager

# Start PostgreSQL
sudo systemctl start postgresql  # Linux
brew services start postgresql  # macOS

# Create database and user
sudo -u postgres psql
CREATE USER aiide WITH PASSWORD 'aiide_dev';
CREATE DATABASE ai_ide OWNER aiide;
GRANT ALL PRIVILEGES ON DATABASE ai_ide TO aiide;
\q

# Test connection
psql -h localhost -U aiide -d ai_ide -c "SELECT 1;"
```

### Redis Connection Issues

**Issue**: Cannot connect to Redis cache

**Symptoms**:
- "Redis connection failed" errors
- Slow performance
- Cache-related errors

**Solutions**:
```bash
# Check Redis status
redis-cli ping

# Start Redis
sudo systemctl start redis  # Linux
brew services start redis  # macOS
# Windows: Start Redis service

# Test Redis connection
redis-cli
127.0.0.1:6379> ping
PONG
127.0.0.1:6379> exit

# Clear Redis cache if needed
redis-cli FLUSHALL
```

## Agent System Issues

### Agents Not Responding

**Issue**: AI agents don't respond to requests

**Symptoms**:
- Timeout errors in chat
- "Agent unavailable" messages
- No response from agent commands

**Diagnostic Steps**:
```bash
# Check agent status
curl http://localhost:8000/api/agents/status

# Check agent logs
tail -f backend/logs/agents.log

# Test individual agents
curl -X POST http://localhost:8000/api/agents/code/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test", "context": {}}'
```

**Solutions**:
```bash
# Restart agents
curl -X POST http://localhost:8000/api/agents/restart

# Reset agent configuration
cp backend/config/agents.yaml.template backend/config/agents.yaml

# Check agent dependencies
cd backend
python -c "
from multi_agent_system import MultiAgentSystem
system = MultiAgentSystem()
print('Agent system initialized successfully')
"
```

### Agent Performance Issues

**Issue**: Agents respond slowly or consume too much memory

**Symptoms**:
- Response times > 30 seconds
- High memory usage
- System becomes unresponsive

**Solutions**:
```bash
# Monitor agent performance
curl http://localhost:8000/api/agents/metrics

# Adjust agent configuration
# Edit backend/config/agents.yaml:
agents:
  max_concurrent: 2  # Reduce from default 5
  timeout: 30        # Reduce timeout
  memory_limit: "1GB"  # Set memory limit

# Restart with limited resources
cd backend
python main.py --max-workers 2 --memory-limit 2G
```

## Performance Problems

### Slow Response Times

**Issue**: AI IDE responds slowly to requests

**Symptoms**:
- Long delays in chat responses
- Slow semantic search
- Timeouts in web search

**Diagnostic Steps**:
```bash
# Check system resources
htop  # Linux/macOS
# Windows: Task Manager

# Monitor API response times
curl -w "@curl-format.txt" http://localhost:8000/health

# Create curl-format.txt:
echo "     time_namelookup:  %{time_namelookup}\n
        time_connect:  %{time_connect}\n
     time_appconnect:  %{time_appconnect}\n
    time_pretransfer:  %{time_pretransfer}\n
       time_redirect:  %{time_redirect}\n
  time_starttransfer:  %{time_starttransfer}\n
                     ----------\n
          time_total:  %{time_total}\n" > curl-format.txt
```

**Solutions**:
```bash
# Optimize configuration
# Edit ~/.ai-ide/config.yaml:
performance:
  cache_enabled: true
  max_context_length: 4096  # Reduce from 8192
  concurrent_requests: 3    # Reduce from 10

# Clear caches
redis-cli FLUSHALL
rm -rf backend/cache/*

# Restart services
docker-compose restart  # If using Docker
# or
cd backend && python main.py
```

### High Memory Usage

**Issue**: AI IDE consumes excessive memory

**Symptoms**:
- System becomes slow
- Out of memory errors
- Frequent garbage collection

**Solutions**:
```bash
# Monitor memory usage
python -c "
import psutil
process = psutil.Process()
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"

# Optimize memory settings
# Edit backend/config/development.yaml:
memory:
  max_heap_size: "2GB"
  gc_threshold: 0.8
  cache_size: 1000  # Reduce cache size

# Use memory profiling
pip install memory-profiler
python -m memory_profiler main.py
```

### Disk Space Issues

**Issue**: Running out of disk space

**Symptoms**:
- "No space left on device" errors
- Database write failures
- Log file errors

**Solutions**:
```bash
# Check disk usage
df -h
du -sh backend/logs/
du -sh backend/cache/
du -sh ~/.ai-ide/

# Clean up logs
find backend/logs/ -name "*.log" -mtime +7 -delete
find backend/logs/ -name "*.log.*" -delete

# Clean up cache
rm -rf backend/cache/*
redis-cli FLUSHALL

# Configure log rotation
# Edit backend/config/logging.yaml:
handlers:
  file:
    filename: logs/ai-ide.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
```

## AI Model Issues

### Model Loading Failures

**Issue**: AI models fail to load or initialize

**Symptoms**:
- "Model not found" errors
- Initialization timeouts
- CUDA/GPU errors

**Solutions**:
```bash
# Check model availability
python -c "
from transformers import AutoTokenizer, AutoModel
try:
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    print('Model loading successful')
except Exception as e:
    print(f'Model loading failed: {e}')
"

# Download models manually
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print('Model downloaded successfully')
"

# Check GPU availability
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
"
```

### LM Studio Connection Issues

**Issue**: Cannot connect to LM Studio

**Symptoms**:
- "LM Studio unavailable" errors
- Model generation timeouts
- Connection refused errors

**Solutions**:
```bash
# Check LM Studio status
curl http://localhost:1234/v1/models

# Test LM Studio API
curl -X POST http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-coder",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }'

# Configure LM Studio in AI IDE
# Edit backend/config/development.yaml:
ai_models:
  lm_studio:
    url: "http://localhost:1234"
    timeout: 60
    max_retries: 3
```

## Database Problems

### Migration Failures

**Issue**: Database migrations fail to run

**Symptoms**:
- "Migration failed" errors
- Schema version mismatches
- Table creation errors

**Solutions**:
```bash
# Check migration status
cd backend
python -c "
from database.migrations import get_migration_status
status = get_migration_status()
print(f'Current version: {status}')
"

# Run migrations manually
python database/migrations.py --force

# Reset database (WARNING: This will delete all data)
python -c "
from database.connection import DatabaseManager
db = DatabaseManager()
db.reset_database()
print('Database reset complete')
"

# Check database schema
psql -h localhost -U aiide -d ai_ide -c "\dt"
```

### Performance Issues

**Issue**: Database queries are slow

**Symptoms**:
- Long response times
- Query timeouts
- High CPU usage

**Solutions**:
```bash
# Check database performance
psql -h localhost -U aiide -d ai_ide -c "
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;"

# Analyze slow queries
psql -h localhost -U aiide -d ai_ide -c "
EXPLAIN ANALYZE SELECT * FROM code_embeddings 
WHERE similarity(embedding, '[1,2,3]') > 0.7;"

# Create missing indexes
psql -h localhost -U aiide -d ai_ide -c "
CREATE INDEX CONCURRENTLY idx_code_embeddings_similarity 
ON code_embeddings USING ivfflat (embedding vector_cosine_ops);"

# Update table statistics
psql -h localhost -U aiide -d ai_ide -c "ANALYZE;"
```

## Extension Issues

### Extension Not Loading

**Issue**: VSCode extension fails to load

**Symptoms**:
- AI Assistant panel not visible
- Commands not available
- Extension not in extensions list

**Solutions**:
```bash
# Check if extension is installed
code --list-extensions | grep ai-ide

# Reinstall extension
code --uninstall-extension ai-ide.ai-assistant
code --install-extension dist/ai-ide-extension-*.vsix

# Check extension logs
# In VSCode: Ctrl+Shift+P → "Developer: Show Logs" → "Extension Host"

# Reset VSCode settings
# Backup and remove: ~/.vscode/settings.json
# Restart VSCode
```

### Extension Crashes

**Issue**: Extension crashes or becomes unresponsive

**Symptoms**:
- Extension host crashes
- Unresponsive UI
- Error notifications

**Solutions**:
```bash
# Reload extension host
# In VSCode: Ctrl+Shift+P → "Developer: Reload Window"

# Check for conflicting extensions
# Disable other AI/coding extensions temporarily

# Clear extension cache
rm -rf ~/.vscode/extensions/ai-ide.ai-assistant-*/out/
cd extensions/ai-assistant
npm run compile

# Check extension memory usage
# In VSCode: Ctrl+Shift+P → "Developer: Show Running Extensions"
```

## Network and Connectivity

### Firewall Issues

**Issue**: Firewall blocking connections

**Symptoms**:
- Connection timeouts
- "Connection refused" errors
- Services unreachable

**Solutions**:
```bash
# Check firewall status
sudo ufw status  # Linux
# Windows: Check Windows Firewall settings

# Allow AI IDE ports
sudo ufw allow 8000:8004/tcp  # Linux

# Test connectivity
telnet localhost 8000
nc -zv localhost 8000  # Linux/macOS
```

### Proxy Configuration

**Issue**: Corporate proxy blocking requests

**Symptoms**:
- Web search failures
- Model download errors
- API connection issues

**Solutions**:
```bash
# Configure proxy for Python
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
export NO_PROXY=localhost,127.0.0.1

# Configure proxy for npm
npm config set proxy http://proxy.company.com:8080
npm config set https-proxy http://proxy.company.com:8080

# Configure proxy in AI IDE
# Edit backend/config/development.yaml:
network:
  proxy:
    http: "http://proxy.company.com:8080"
    https: "http://proxy.company.com:8080"
    no_proxy: "localhost,127.0.0.1"
```

## Advanced Debugging

### Enable Debug Mode

```bash
# Backend debug mode
cd backend
python main.py --debug --log-level DEBUG

# Extension debug mode
# In VSCode settings.json:
{
  "ai-ide.debug": true,
  "ai-ide.logLevel": "DEBUG"
}
```

### Collect Debug Information

```bash
# Generate system report
cd backend
python -c "
import sys, platform, pkg_resources, psutil
import json
from datetime import datetime

report = {
    'timestamp': datetime.now().isoformat(),
    'system': {
        'platform': platform.platform(),
        'python_version': sys.version,
        'architecture': platform.architecture(),
    },
    'resources': {
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'cpu_count': psutil.cpu_count(),
        'disk_usage': psutil.disk_usage('/').percent
    },
    'packages': {pkg.key: pkg.version for pkg in pkg_resources.working_set}
}

with open('debug_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print('Debug report saved to debug_report.json')
"
```

### Log Analysis

```bash
# Analyze error patterns
grep -i error backend/logs/ai-ide.log | tail -20

# Check for memory leaks
grep -i "memory" backend/logs/ai-ide.log

# Monitor real-time logs
tail -f backend/logs/ai-ide.log | grep -E "(ERROR|WARNING|CRITICAL)"

# Analyze performance logs
grep -i "slow" backend/logs/ai-ide.log
grep -E "took [0-9]+\.[0-9]+s" backend/logs/ai-ide.log
```

### Performance Profiling

```bash
# Profile Python code
pip install py-spy
py-spy top --pid $(pgrep -f "python main.py")

# Profile memory usage
pip install memory-profiler
python -m memory_profiler main.py

# Profile database queries
# Enable query logging in PostgreSQL:
# Edit postgresql.conf:
log_statement = 'all'
log_min_duration_statement = 1000  # Log queries > 1s
```

## Getting Help

### Before Reporting Issues

1. **Check this troubleshooting guide**
2. **Search existing issues**: https://github.com/ai-ide/ai-ide/issues
3. **Collect debug information** (see above)
4. **Try minimal reproduction** steps

### Reporting Issues

Include the following information:

1. **System Information**:
   - Operating system and version
   - Python version
   - Node.js version
   - VSCode/VSCodium version

2. **AI IDE Information**:
   - AI IDE version
   - Installation method
   - Configuration files (sanitized)

3. **Error Information**:
   - Complete error messages
   - Log files (relevant portions)
   - Steps to reproduce

4. **Debug Report**:
   - System resource usage
   - Network connectivity
   - Service status

### Support Channels

- **GitHub Issues**: https://github.com/ai-ide/ai-ide/issues
- **Discussions**: https://github.com/ai-ide/ai-ide/discussions
- **Discord**: https://discord.gg/ai-ide
- **Email**: support@ai-ide.dev

### Emergency Recovery

If AI IDE becomes completely unusable:

```bash
# Stop all services
pkill -f "python.*ai-ide"
pkill -f "node.*ai-assistant"

# Reset configuration
mv ~/.ai-ide ~/.ai-ide.backup
mv .ai-ide .ai-ide.backup

# Reinstall from scratch
rm -rf ai-ide/
git clone https://github.com/ai-ide/ai-ide.git
cd ai-ide
./scripts/install.sh
```