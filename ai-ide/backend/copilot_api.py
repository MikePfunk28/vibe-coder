#!/usr/bin/env python3
"""
GitHub Copilot API Server for AI IDE Backend
Provides Copilot-compatible REST API endpoints
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from copilot_integration import (
    get_copilot_integration, CopilotDocument, CopilotPosition, CopilotContext
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('copilot-api')

# Create FastAPI app
app = FastAPI(
    title="GitHub Copilot API for AI IDE",
    description="Copilot-compatible API endpoints for Mike-AI-IDE",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class DocumentModel(BaseModel):
    uri: str
    languageId: str
    version: int
    text: str

class PositionModel(BaseModel):
    line: int
    character: int

class ContextModel(BaseModel):
    triggerKind: int
    triggerCharacter: Optional[str] = None

class CompletionRequest(BaseModel):
    document: DocumentModel
    position: PositionModel
    context: ContextModel

class CompletionResponse(BaseModel):
    text: str
    range: Dict[str, int]
    displayText: str
    uuid: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    response: str

class StatusResponse(BaseModel):
    status: str
    user: Optional[Dict[str, Any]] = None

class TelemetryRequest(BaseModel):
    uuid: str

# Global Copilot integration instance
copilot_integration = None

@app.on_event("startup")
async def startup_event():
    """Initialize Copilot integration on startup"""
    global copilot_integration
    try:
        # Try to import and use the main AI backend
        try:
            from main import AIIDEBackend
            ai_backend = AIIDEBackend()
            await ai_backend.initialize()
            copilot_integration = get_copilot_integration(ai_backend)
        except ImportError:
            # Fallback to standalone Copilot integration
            copilot_integration = get_copilot_integration()
        
        # Auto sign-in for local development
        copilot_integration.sign_in({'username': 'local-dev', 'email': 'dev@local.ai'})
        
        logger.info("Copilot API server started successfully")
    except Exception as e:
        logger.error(f"Failed to start Copilot API: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global copilot_integration
    if copilot_integration:
        copilot_integration.sign_out()
    logger.info("Copilot API server shutdown")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "copilot-api",
        "copilot_status": copilot_integration.get_status() if copilot_integration else None
    }

@app.get("/api/copilot/status", response_model=StatusResponse)
async def get_copilot_status():
    """Get Copilot authentication status"""
    if not copilot_integration:
        raise HTTPException(status_code=503, detail="Copilot service not available")
    
    status_info = copilot_integration.get_status()
    return StatusResponse(
        status=status_info['status'],
        user=status_info.get('user')
    )

@app.post("/api/copilot/signin")
async def sign_in_copilot(user_info: Optional[Dict[str, Any]] = None):
    """Sign in to Copilot"""
    if not copilot_integration:
        raise HTTPException(status_code=503, detail="Copilot service not available")
    
    success = copilot_integration.sign_in(user_info)
    if success:
        return {"message": "Signed in successfully", "status": "SignedIn"}
    else:
        raise HTTPException(status_code=401, detail="Sign-in failed")

@app.post("/api/copilot/signout")
async def sign_out_copilot():
    """Sign out from Copilot"""
    if not copilot_integration:
        raise HTTPException(status_code=503, detail="Copilot service not available")
    
    copilot_integration.sign_out()
    return {"message": "Signed out successfully", "status": "SignedOut"}

@app.post("/api/copilot/completions", response_model=Dict[str, List[CompletionResponse]])
async def get_completions(request: CompletionRequest):
    """Get code completions from Copilot"""
    if not copilot_integration:
        raise HTTPException(status_code=503, detail="Copilot service not available")
    
    try:
        # Convert request to internal format
        document = CopilotDocument(
            uri=request.document.uri,
            language_id=request.document.languageId,
            version=request.document.version,
            text=request.document.text
        )
        
        position = CopilotPosition(
            line=request.position.line,
            character=request.position.character
        )
        
        context = CopilotContext(
            trigger_kind=request.context.triggerKind,
            trigger_character=request.context.triggerCharacter
        )
        
        # Get completions
        completions = await copilot_integration.get_completions(document, position, context)
        
        # Convert to response format
        completion_responses = [
            CompletionResponse(
                text=comp.text,
                range=comp.range,
                displayText=comp.display_text,
                uuid=comp.uuid
            )
            for comp in completions
        ]
        
        return {"completions": completion_responses}
        
    except Exception as e:
        logger.error(f"Failed to get completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/copilot/chat", response_model=ChatResponse)
async def copilot_chat(request: ChatRequest):
    """Chat with Copilot"""
    if not copilot_integration:
        raise HTTPException(status_code=503, detail="Copilot service not available")
    
    try:
        # Convert messages to internal format
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        
        # Get chat response
        response = await copilot_integration.get_chat_response(messages)
        
        return ChatResponse(response=response)
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/copilot/accept")
async def accept_completion(request: TelemetryRequest):
    """Record completion acceptance"""
    if not copilot_integration:
        raise HTTPException(status_code=503, detail="Copilot service not available")
    
    copilot_integration.accept_completion(request.uuid)
    return {"message": "Completion acceptance recorded"}

@app.post("/api/copilot/reject")
async def reject_completion(request: TelemetryRequest):
    """Record completion rejection"""
    if not copilot_integration:
        raise HTTPException(status_code=503, detail="Copilot service not available")
    
    copilot_integration.reject_completion(request.uuid)
    return {"message": "Completion rejection recorded"}

@app.get("/api/copilot/telemetry")
async def get_telemetry():
    """Get telemetry data"""
    if not copilot_integration:
        raise HTTPException(status_code=503, detail="Copilot service not available")
    
    return {"telemetry": copilot_integration.get_telemetry_data()}

@app.delete("/api/copilot/telemetry")
async def clear_telemetry():
    """Clear telemetry data"""
    if not copilot_integration:
        raise HTTPException(status_code=503, detail="Copilot service not available")
    
    copilot_integration.clear_telemetry_data()
    return {"message": "Telemetry data cleared"}

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "status_code": 500
    }

def run_copilot_api_server(host: str = "localhost", port: int = 8001):
    """Run the Copilot API server"""
    logger.info(f"Starting Copilot API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GitHub Copilot API Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    
    args = parser.parse_args()
    run_copilot_api_server(args.host, args.port)