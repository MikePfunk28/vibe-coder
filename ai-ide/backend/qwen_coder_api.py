"""
Qwen Coder 3 API Endpoints
REST API endpoints for code completion and generation
"""

import json
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from qwen_coder_agent import (
    QwenCoderAgent, CodeRequest, CodeContext, CodeTaskType,
    get_qwen_coder_agent, complete_code, generate_code, 
    refactor_code, debug_code
)

logger = logging.getLogger('qwen_coder_api')

# Pydantic models for API requests/responses
class CodeContextModel(BaseModel):
    language: str
    file_path: Optional[str] = None
    selected_text: Optional[str] = None
    cursor_position: Optional[int] = None
    surrounding_code: Optional[str] = None
    project_context: Optional[Dict[str, Any]] = None
    imports: Optional[list] = None
    functions: Optional[list] = None
    classes: Optional[list] = None

class CodeCompletionRequest(BaseModel):
    code: str = Field(..., description="Code to complete")
    context: CodeContextModel
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    stream: bool = Field(default=False)

class CodeGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Description of code to generate")
    context: CodeContextModel
    max_tokens: int = Field(default=2048, ge=1, le=4096)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    include_explanation: bool = Field(default=False)
    stream: bool = Field(default=False)

class CodeRefactoringRequest(BaseModel):
    code: str = Field(..., description="Code to refactor")
    refactoring_request: str = Field(..., description="What kind of refactoring to perform")
    context: CodeContextModel
    max_tokens: int = Field(default=2048, ge=1, le=4096)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)

class CodeDebuggingRequest(BaseModel):
    code: str = Field(..., description="Code to debug")
    issue_description: str = Field(..., description="Description of the issue")
    context: CodeContextModel
    max_tokens: int = Field(default=2048, ge=1, le=4096)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)

class CodeDocumentationRequest(BaseModel):
    code: str = Field(..., description="Code to document")
    documentation_request: str = Field(default="Generate comprehensive documentation")
    context: CodeContextModel
    max_tokens: int = Field(default=2048, ge=1, le=4096)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)

class CodeExplanationRequest(BaseModel):
    code: str = Field(..., description="Code to explain")
    explanation_request: str = Field(default="Explain this code in detail")
    context: CodeContextModel
    max_tokens: int = Field(default=2048, ge=1, le=4096)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)

class CodeOptimizationRequest(BaseModel):
    code: str = Field(..., description="Code to optimize")
    optimization_request: str = Field(default="Optimize this code for better performance")
    context: CodeContextModel
    max_tokens: int = Field(default=2048, ge=1, le=4096)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)

class CodeResponseModel(BaseModel):
    code: str
    language: str
    confidence: float
    explanation: Optional[str] = None
    suggestions: Optional[list] = None
    metadata: Optional[Dict[str, Any]] = None
    execution_time: float
    model_info: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
    timestamp: str

# Create FastAPI app
app = FastAPI(
    title="Qwen Coder 3 API",
    description="Advanced code generation API using Qwen Coder 3",
    version="1.0.0"
)

def convert_context_model_to_context(context_model: CodeContextModel) -> CodeContext:
    """Convert Pydantic model to CodeContext dataclass"""
    return CodeContext(
        language=context_model.language,
        file_path=context_model.file_path,
        selected_text=context_model.selected_text,
        cursor_position=context_model.cursor_position,
        surrounding_code=context_model.surrounding_code,
        project_context=context_model.project_context,
        imports=context_model.imports,
        functions=context_model.functions,
        classes=context_model.classes
    )

def convert_response_to_model(response) -> CodeResponseModel:
    """Convert CodeResponse to Pydantic model"""
    return CodeResponseModel(
        code=response.code,
        language=response.language,
        confidence=response.confidence,
        explanation=response.explanation,
        suggestions=response.suggestions,
        metadata=response.metadata,
        execution_time=response.execution_time,
        model_info=response.model_info
    )

@app.on_event("startup")
async def startup_event():
    """Initialize the Qwen Coder agent on startup"""
    try:
        await get_qwen_coder_agent()
        logger.info("Qwen Coder API started successfully")
    except Exception as e:
        logger.error(f"Failed to start Qwen Coder API: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        agent = await get_qwen_coder_agent()
        await agent.close()
        logger.info("Qwen Coder API shutdown successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        agent = await get_qwen_coder_agent()
        stats = agent.get_performance_stats()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "performance": stats
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/complete", response_model=CodeResponseModel)
async def complete_code_endpoint(request: CodeCompletionRequest):
    """Complete code using Qwen Coder 3"""
    try:
        context_dict = {
            'file_path': request.context.file_path,
            'selected_text': request.context.selected_text,
            'surrounding_code': request.context.surrounding_code
        }
        
        response = await complete_code(
            code=request.code,
            language=request.context.language,
            context=context_dict,
            stream=False  # Non-streaming for regular endpoint
        )
        
        return convert_response_to_model(response)
        
    except Exception as e:
        logger.error(f"Code completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/complete/stream")
async def complete_code_stream_endpoint(request: CodeCompletionRequest):
    """Complete code with streaming response"""
    try:
        agent = await get_qwen_coder_agent()
        
        code_context = convert_context_model_to_context(request.context)
        
        code_request = CodeRequest(
            prompt=request.code,
            task_type=CodeTaskType.COMPLETION,
            context=code_context,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=True
        )
        
        async def generate_stream():
            try:
                async for chunk in agent.generate_code_stream(code_request):
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        logger.error(f"Streaming code completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate", response_model=CodeResponseModel)
async def generate_code_endpoint(request: CodeGenerationRequest):
    """Generate code using Qwen Coder 3"""
    try:
        context_dict = {
            'file_path': request.context.file_path,
            'selected_text': request.context.selected_text,
            'project_context': request.context.project_context
        }
        
        response = await generate_code(
            prompt=request.prompt,
            language=request.context.language,
            context=context_dict,
            include_explanation=request.include_explanation,
            stream=False
        )
        
        return convert_response_to_model(response)
        
    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/stream")
async def generate_code_stream_endpoint(request: CodeGenerationRequest):
    """Generate code with streaming response"""
    try:
        agent = await get_qwen_coder_agent()
        
        code_context = convert_context_model_to_context(request.context)
        
        code_request = CodeRequest(
            prompt=request.prompt,
            task_type=CodeTaskType.GENERATION,
            context=code_context,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            include_explanation=request.include_explanation,
            stream=True
        )
        
        async def generate_stream():
            try:
                async for chunk in agent.generate_code_stream(code_request):
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        logger.error(f"Streaming code generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refactor", response_model=CodeResponseModel)
async def refactor_code_endpoint(request: CodeRefactoringRequest):
    """Refactor code using Qwen Coder 3"""
    try:
        context_dict = {
            'file_path': request.context.file_path
        }
        
        response = await refactor_code(
            code=request.code,
            language=request.context.language,
            refactoring_request=request.refactoring_request,
            context=context_dict
        )
        
        return convert_response_to_model(response)
        
    except Exception as e:
        logger.error(f"Code refactoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/debug", response_model=CodeResponseModel)
async def debug_code_endpoint(request: CodeDebuggingRequest):
    """Debug code using Qwen Coder 3"""
    try:
        context_dict = {
            'file_path': request.context.file_path,
            'surrounding_code': request.context.surrounding_code
        }
        
        response = await debug_code(
            code=request.code,
            language=request.context.language,
            issue_description=request.issue_description,
            context=context_dict
        )
        
        return convert_response_to_model(response)
        
    except Exception as e:
        logger.error(f"Code debugging failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/document", response_model=CodeResponseModel)
async def document_code_endpoint(request: CodeDocumentationRequest):
    """Generate documentation for code using Qwen Coder 3"""
    try:
        agent = await get_qwen_coder_agent()
        
        code_context = convert_context_model_to_context(request.context)
        code_context.selected_text = request.code
        
        code_request = CodeRequest(
            prompt=request.documentation_request,
            task_type=CodeTaskType.DOCUMENTATION,
            context=code_context,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            include_explanation=True
        )
        
        response = await agent.generate_code(code_request)
        return convert_response_to_model(response)
        
    except Exception as e:
        logger.error(f"Code documentation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain", response_model=CodeResponseModel)
async def explain_code_endpoint(request: CodeExplanationRequest):
    """Explain code using Qwen Coder 3"""
    try:
        agent = await get_qwen_coder_agent()
        
        code_context = convert_context_model_to_context(request.context)
        code_context.selected_text = request.code
        
        code_request = CodeRequest(
            prompt=request.explanation_request,
            task_type=CodeTaskType.EXPLANATION,
            context=code_context,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            include_explanation=True
        )
        
        response = await agent.generate_code(code_request)
        return convert_response_to_model(response)
        
    except Exception as e:
        logger.error(f"Code explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize", response_model=CodeResponseModel)
async def optimize_code_endpoint(request: CodeOptimizationRequest):
    """Optimize code using Qwen Coder 3"""
    try:
        agent = await get_qwen_coder_agent()
        
        code_context = convert_context_model_to_context(request.context)
        code_context.selected_text = request.code
        
        code_request = CodeRequest(
            prompt=request.optimization_request,
            task_type=CodeTaskType.OPTIMIZATION,
            context=code_context,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            include_explanation=True
        )
        
        response = await agent.generate_code(code_request)
        return convert_response_to_model(response)
        
    except Exception as e:
        logger.error(f"Code optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_performance_stats():
    """Get performance statistics"""
    try:
        agent = await get_qwen_coder_agent()
        stats = agent.get_performance_stats()
        
        # Add LM Studio manager stats if available
        if agent.lm_studio_manager:
            lm_stats = agent.lm_studio_manager.get_performance_stats()
            stats['lm_studio'] = lm_stats
        
        return {
            "timestamp": datetime.now().isoformat(),
            "qwen_coder": stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_available_models():
    """Get information about available models"""
    try:
        agent = await get_qwen_coder_agent()
        
        if agent.lm_studio_manager:
            models_info = agent.lm_studio_manager.get_model_info()
            return {
                "timestamp": datetime.now().isoformat(),
                "models": models_info
            }
        else:
            return {
                "timestamp": datetime.now().isoformat(),
                "models": {},
                "error": "LM Studio manager not available"
            }
            
    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "details": str(exc),
        "timestamp": datetime.now().isoformat()
    }

def run_api_server(host: str = "localhost", port: int = 8001):
    """Run the API server"""
    uvicorn.run(
        "qwen_coder_api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    run_api_server()