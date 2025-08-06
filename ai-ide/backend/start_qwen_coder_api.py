#!/usr/bin/env python3
"""
Startup script for Qwen Coder 3 API server
Runs the FastAPI server for code generation endpoints
"""

import sys
import logging
import argparse
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from qwen_coder_api import run_api_server

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('qwen_coder_startup')

def main():
    """Main startup function"""
    parser = argparse.ArgumentParser(description='Start Qwen Coder 3 API server')
    parser.add_argument('--host', default='localhost', help='Host to bind to (default: localhost)')
    parser.add_argument('--port', type=int, default=8001, help='Port to bind to (default: 8001)')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    
    args = parser.parse_args()
    
    logger.info(f"Starting Qwen Coder 3 API server on {args.host}:{args.port}")
    
    try:
        if args.reload:
            import uvicorn
            uvicorn.run(
                "qwen_coder_api:app",
                host=args.host,
                port=args.port,
                reload=True,
                log_level="info"
            )
        else:
            run_api_server(host=args.host, port=args.port)
            
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()