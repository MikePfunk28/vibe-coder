"""
ReAct Framework Default Tools

Default tool implementations for the ReAct framework.
"""

import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DefaultToolImplementations:
    """Default tool implementations for the ReAct framework."""

    def __init__(self, llm_client, context_manager):
        self.llm_client = llm_client
        self.context_manager = context_manager

    async def execute_search_tool(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute semantic search tool."""
        query = input_data.get("query", "")
        max_results = input_data.get("max_results", 5)

        try:
            # Use context manager for search
            context = self.context_manager.get_relevant_context(query, max_tokens=1024)

            results = []
            for ctx in context[:max_results]:
                results.append({
                    "content": ctx.content[:200] + "..." if len(ctx.content) > 200 else ctx.content,
                    "source": getattr(ctx, 'source', 'unknown'),
                    "relevance": getattr(ctx, 'relevance', 0.5)
                })

            return {
                "results": results,
                "query": query,
                "total_found": len(results)
            }

        except Exception as e:
            return {"error": str(e), "results": []}

    async def execute_analysis_tool(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code analysis tool."""
        code = input_data.get("code", "")
        analysis_type = input_data.get("analysis_type", "general")

        if not code:
            return {"error": "No code provided for analysis"}

        try:
            prompt = f"""
            Analyze the following code for {analysis_type}:
            
            {code}
            
            Provide analysis including:
            1. Code quality assessment
            2. Potential issues
            3. Suggestions for improvement
            
            Analysis:"""

            analysis = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=400,
                temperature=0.2
            )

            return {
                "analysis": analysis.strip(),
                "code_length": len(code),
                "analysis_type": analysis_type
            }

        except Exception as e:
            return {"error": str(e)}

    async def execute_generation_tool(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code generation tool."""
        description = input_data.get("description", "")
        language = input_data.get("language", "python")

        if not description:
            return {"error": "No description provided for code generation"}

        try:
            prompt = f"""
            Generate {language} code for:
            {description}
            
            Provide clean, well-documented code:
            """

            code = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3
            )

            return {
                "generated_code": code.strip(),
                "language": language,
                "description": description
            }

        except Exception as e:
            return {"error": str(e)}

    async def execute_reasoning_tool(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute deep reasoning tool."""
        problem = input_data.get("problem", "")
        context = input_data.get("context", {})

        if not problem:
            return {"error": "No problem provided for reasoning"}

        try:
            prompt = f"""
            Think deeply about this problem:
            {problem}
            
            Context: {json.dumps(context, indent=2)}
            
            Provide step-by-step reasoning:
            1. Problem understanding
            2. Key considerations
            3. Potential approaches
            4. Recommended solution
            
            Reasoning:"""

            reasoning = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=600,
                temperature=0.2
            )

            return {
                "reasoning": reasoning.strip(),
                "problem": problem,
                "context": context
            }

        except Exception as e:
            return {"error": str(e)}