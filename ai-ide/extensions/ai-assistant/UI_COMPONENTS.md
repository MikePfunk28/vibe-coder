# AI Assistant UI Components

This document describes the advanced UI components implemented for the AI Assistant VSCodium extension.

## Overview

The AI Assistant extension provides two main UI components:

1. **AI Chat Interface** - Advanced multi-agent conversation with reasoning traces
2. **Search & Dashboard** - Unified search and performance monitoring system

Both components are accessible through the AI Assistant activity bar panel.

## AI Chat Interface

### Features

#### Multi-Agent Support
- **General Assistant**: General purpose coding assistance
- **Code Specialist**: Focused on code generation and refactoring
- **Search Expert**: Semantic search and knowledge retrieval
- **Reasoning Engine**: Complex problem solving and analysis
- **Test Engineer**: Test generation and validation

#### Reasoning Modes
- **Basic**: Quick responses with minimal reasoning
- **Chain of Thought**: Step-by-step reasoning process
- **Deep Analysis**: Comprehensive analysis with detailed reasoning
- **Interleaved**: Advanced interleaved reasoning with context windows
- **ReAct**: Reasoning and Acting with tool usage

#### Advanced Features
- **Streaming Responses**: Real-time response generation with typing indicators
- **Reasoning Trace Visualization**: Step-by-step reasoning process display
- **Code Snippet Integration**: Insertable code blocks with syntax highlighting
- **Web Search Integration**: Real-time web search results in conversations
- **Context-Aware Suggestions**: Responses based on current editor context
- **Confidence Indicators**: Visual confidence scores for AI responses
- **Chat Export**: Export conversations to markdown format
- **Agent Selection**: Dynamic agent switching during conversations

### Usage

1. Open the AI Assistant panel from the activity bar
2. Select your preferred agent and reasoning mode
3. Type your question or request
4. View the streaming response with reasoning traces
5. Insert code snippets directly into your editor
6. Export conversations for documentation

### Keyboard Shortcuts

- `Ctrl+Shift+A` (Cmd+Shift+A on Mac): Open AI Chat
- `Enter`: Send message
- `Shift+Enter`: New line in message input

## Search & Dashboard

### Features

#### Unified Search Interface
- **Semantic Search**: Find code patterns using semantic similarity
- **Web Search**: Real-time web search for documentation and examples
- **RAG Search**: Knowledge base search with retrieval-augmented generation
- **Multi-Type Search**: Combine multiple search types in one query
- **Search History**: Track and revisit previous searches
- **Result Export**: Export search results to markdown

#### Performance Dashboard
- **System Metrics**: Response time, accuracy, user satisfaction
- **Agent Performance**: Individual agent success rates and usage
- **Reasoning Quality**: Confidence scores and reasoning effectiveness
- **Trend Analysis**: Performance trends over time
- **Real-time Updates**: Live metric updates during usage

#### Reasoning Trace Explorer
- **Trace Visualization**: Detailed reasoning step analysis
- **Session Tracking**: Group traces by conversation sessions
- **Performance Analysis**: Reasoning duration and confidence metrics
- **Debug Tools**: Identify reasoning bottlenecks and issues
- **Export Capabilities**: Export traces for analysis

#### Knowledge Base Management
- **Content Browser**: Browse and search knowledge base entries
- **CRUD Operations**: Add, update, and delete knowledge entries
- **Tagging System**: Organize content with tags
- **Relevance Scoring**: Automatic relevance calculation
- **Source Tracking**: Track content sources and updates

#### MCP Tool Management
- **Tool Discovery**: Automatic MCP server discovery
- **Configuration Panel**: Configure tool settings and parameters
- **Enable/Disable**: Toggle tools on/off as needed
- **Testing Interface**: Test tool functionality
- **Usage Analytics**: Track tool usage and success rates
- **Auto-Approval Settings**: Configure automatic tool approvals

### Usage

#### Search Tab
1. Enter your search query
2. Select search types (semantic, web, RAG)
3. Click Search or press Enter
4. Click results to open files or web pages
5. Export results for documentation

#### Dashboard Tab
1. View real-time performance metrics
2. Click Refresh to update metrics
3. Export metrics for analysis
4. Monitor system health and performance

#### Reasoning Tab
1. View reasoning traces from conversations
2. Click traces to see detailed analysis
3. Export traces for debugging
4. Clear old traces to free memory

#### Knowledge Tab
1. Search existing knowledge base entries
2. Add new entries with title, content, and tags
3. Browse entries by type and relevance
4. Update or delete entries as needed

#### MCP Tools Tab
1. View configured MCP tools
2. Enable/disable tools as needed
3. Test tool functionality
4. Configure tool settings
5. Monitor usage statistics

### Keyboard Shortcuts

- `Ctrl+Shift+D` (Cmd+Shift+D on Mac): Open Search & Dashboard
- `Ctrl+Shift+F` (Cmd+Shift+F on Mac): Perform Unified Search
- `Ctrl+Shift+S` (Cmd+Shift+S on Mac): Semantic Search

## Integration with Backend

Both UI components integrate seamlessly with the PocketFlow backend system:

### Chat Interface Integration
- Uses PocketFlow reasoning engine for multi-step analysis
- Integrates with semantic search for context-aware responses
- Connects to web search agents for real-time information
- Leverages multi-agent system for specialized responses

### Dashboard Integration
- Retrieves performance metrics from backend systems
- Displays reasoning traces from PocketFlow execution
- Manages MCP tool configurations and status
- Synchronizes knowledge base with backend storage

## Customization

### Themes
The UI components automatically adapt to VSCode themes:
- Dark themes: Optimized for dark backgrounds
- Light themes: Adjusted for light backgrounds
- High contrast: Enhanced visibility for accessibility

### Configuration
Settings can be configured through VSCode settings:
- Default agent selection
- Default reasoning mode
- Search result limits
- Metric refresh intervals
- Knowledge base sources

## Troubleshooting

### Common Issues

1. **Chat not responding**: Check backend connection status
2. **Search results empty**: Verify search types are selected
3. **Metrics not updating**: Click refresh or check backend status
4. **MCP tools not working**: Test individual tools and check configuration

### Debug Information

Enable debug logging by setting the log level in VSCode settings:
```json
{
  "ai-assistant.logLevel": "debug"
}
```

### Support

For issues and feature requests:
1. Check the extension output panel for error messages
2. Review the backend logs for connection issues
3. Export relevant data (chat, search results, metrics) for analysis
4. Report issues with detailed reproduction steps

## Future Enhancements

Planned improvements include:
- Voice input/output support
- Collaborative chat sessions
- Advanced visualization tools
- Custom agent creation
- Plugin system for extensions
- Mobile companion app
- Integration with external IDEs