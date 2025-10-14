#!/usr/bin/env python3
"""
MCP Client for LangChain Integration using langchain-mcp-adapters
Supports multiple MCP servers (CoinGecko + Local Unified Server)
"""
import asyncio
import logging
import os
from typing import Any, Dict, List, Optional
from pathlib import Path
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

class MCPToolClient:
    """Client to interact with multiple MCP servers using langchain-mcp-adapters."""
    
    def __init__(self, server_script: str = "unified_mcp_server.py"):
        self.server_script = server_script
        self.client: Optional[MultiServerMCPClient] = None
        self.tools: List[BaseTool] = []
        
    async def initialize(self) -> bool:
        """Initialize connections to all MCP servers and load tools."""
        try:
            logger.info("Initializing MCP clients for multiple servers...")
            
            # Define all server connections
            connections = {}
            
            # 1. CoinGecko MCP Server (External) - for crypto tools
            # Using npx with @coingecko/coingecko-mcp as per official docs
            coingecko_api_key = os.getenv("COINGECKO_DEMO_API_KEY") or os.getenv("COINGECKO_PRO_API_KEY")
            coingecko_env = "demo" if os.getenv("COINGECKO_DEMO_API_KEY") else "pro"
            
            if coingecko_api_key:
                env_key = "COINGECKO_DEMO_API_KEY" if coingecko_env == "demo" else "COINGECKO_PRO_API_KEY"
                connections["coingecko_mcp_local"] = {
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "@coingecko/coingecko-mcp"],
                    "env": {
                        env_key: coingecko_api_key,
                        "COINGECKO_ENVIRONMENT": coingecko_env
                    }
                }
                logger.info(f"✓ CoinGecko MCP server configured ({coingecko_env} tier)")
            else:
                logger.warning("⚠ COINGECKO_DEMO_API_KEY or COINGECKO_PRO_API_KEY not found, skipping CoinGecko server")
            
            # 2. Local Unified MCP Server (Stock + Portfolio + Knowledge)
            script_path = Path(__file__).parent / self.server_script
            if script_path.exists():
                connections["unified_server"] = {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [str(script_path)],
                    "cwd": str(Path(__file__).parent)
                }
                logger.info(f"✓ Unified MCP server configured: {script_path}")
            else:
                logger.error(f"Unified MCP server script not found: {script_path}")
            
            if not connections:
                logger.error("No MCP servers configured!")
                return False
            
            # Initialize MultiServerMCPClient with all connections
            self.client = MultiServerMCPClient(connections=connections)
            
            # Load tools from all servers
            self.tools = []
            for server_name in connections.keys():
                try:
                    server_tools = await self.client.get_tools(server_name=server_name)
                    self.tools.extend(server_tools)
                    logger.info(f"✓ Loaded {len(server_tools)} tools from '{server_name}' server")
                    
                    # Log tool names
                    for tool in server_tools:
                        logger.debug(f"  - {tool.name}: {tool.description[:50]}...")
                        
                except Exception as e:
                    logger.error(f"✗ Failed to load tools from '{server_name}': {e}")
                    # Continue with other servers even if one fails
            
            logger.info(f"✅ Total tools loaded across all servers: {len(self.tools)}")
            return len(self.tools) > 0
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP clients: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_tools(self) -> List[BaseTool]:
        """Get the list of LangChain tools."""
        return self.tools
    
    def get_tool_by_name(self, tool_name: str) -> Optional[BaseTool]:
        """Get a specific tool by name."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool by name with the given arguments."""
        tool = self.get_tool_by_name(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")
        
        try:
            # Invoke the tool using LangChain's tool interface
            result = await tool.ainvoke(arguments)
            
            # Handle different result types
            if isinstance(result, tuple):
                # If result is a tuple (content, artifacts), return the content
                return str(result[0])
            else:
                return str(result)
                
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            raise e
    
    async def cleanup(self):
        """Cleanup resources."""
        # The MultiServerMCPClient manages its own cleanup
        # through the session context managers
        logger.info("MCP client cleanup completed")


# Singleton instance for global access
_mcp_client: Optional[MCPToolClient] = None

async def get_mcp_client() -> MCPToolClient:
    """Get or create the MCP client singleton."""
    global _mcp_client
    
    if _mcp_client is None:
        _mcp_client = MCPToolClient()
        
    return _mcp_client

async def initialize_mcp_client() -> MCPToolClient:
    """Initialize and connect the MCP client."""
    client = await get_mcp_client()
    
    if not client.tools:
        success = await client.initialize()
        if not success:
            raise RuntimeError("Failed to initialize MCP client")
    
    return client

async def cleanup_mcp_client():
    """Cleanup the MCP client."""
    global _mcp_client
    
    if _mcp_client:
        await _mcp_client.cleanup()
        _mcp_client = None
