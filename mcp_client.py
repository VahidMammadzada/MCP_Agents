#!/usr/bin/env python3
"""
MCP Client for LangChain Integration
Provides a bridge between LangChain agents and the unified MCP server
"""
import asyncio
import json
import logging
import subprocess
import sys
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class MCPToolClient:
    """Client to interact with the unified MCP server."""
    
    def __init__(self, server_script: str = "unified_mcp_server.py"):
        self.server_script = server_script
        self.server_process = None
        self.session = None
        self.available_tools = {}
        self._client_context = None
        
    async def start_server(self) -> bool:
        """Start the MCP server process."""
        try:
            script_path = Path(__file__).parent / self.server_script
            
            if not script_path.exists():
                logger.error(f"MCP server script not found: {script_path}")
                return False
            
            logger.info(f"Starting MCP server: {script_path}")
            
            # Start the server process
            self.server_process = await asyncio.create_subprocess_exec(
                sys.executable, str(script_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path(__file__).parent
            )
            
            logger.info(f"Started MCP server: {self.server_script}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            return False
    
    async def connect(self) -> bool:
        """Connect to the MCP server and initialize session."""
        try:
            # Import MCP client components
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            
            # Start the server if it's not already running
            if not self.server_process:
                if not await self.start_server():
                    logger.error("Failed to start MCP server")
                    return False
                
                # Give the server a moment to start
                await asyncio.sleep(2)
                
            # Using the stdio_client function with our own server parameters
            script_path = Path(__file__).parent / self.server_script
            server_params = StdioServerParameters(
                command=sys.executable,
                args=[str(script_path)]
            )
            
            # Create client session and store the context manager
            self._client_context = stdio_client(server_params)
            read_stream, write_stream = await self._client_context.__aenter__()
            
            # Create a ClientSession with the streams
            self.session = ClientSession(read_stream, write_stream)
                
            # Initialize the session
            init_result = await self.session.initialize()
            logger.info(f"MCP session initialized: {init_result}")
            
            # List available tools
            tools_result = await self.session.list_tools()
            self.available_tools = {
                tool.name: {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                }
                for tool in tools_result.tools
            }
            
            logger.info(f"Connected to MCP server with {len(self.available_tools)} tools")
            return True
                
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            return False
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool on the MCP server."""
        if not self.session:
            raise ValueError("Not connected to MCP server")
            
        if tool_name not in self.available_tools:
            raise ValueError(f"Tool not found: {tool_name}")
        
        try:
            result = await self.session.call_tool(tool_name, arguments)
            
            # Extract text content from the result
            if result.content and len(result.content) > 0:
                content = result.content[0]
                if hasattr(content, 'text'):
                    return content.text
                else:
                    return str(content)
            else:
                return "No content returned from tool"
                
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            raise e
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools."""
        return list(self.available_tools.values())
    
    async def stop_server(self):
        """Stop the MCP server process."""
        # Close the client context properly
        if self._client_context and self.session:
            try:
                await self._client_context.__aexit__(None, None, None)
                logger.info("MCP client session closed")
            except Exception as e:
                logger.error(f"Error closing MCP client session: {e}")
        
        if self.server_process:
            try:
                self.server_process.terminate()
                try:
                    await asyncio.wait_for(self.server_process.wait(), timeout=5)
                    logger.info("MCP server stopped")
                except asyncio.TimeoutError:
                    self.server_process.kill()
                    await self.server_process.wait()
                    logger.warning("MCP server killed (timeout)")
            except Exception as e:
                logger.error(f"Error stopping MCP server: {e}")
        
        self.server_process = None
        self.session = None
        self._client_context = None

# Singleton instance for global access
_mcp_client = None

async def get_mcp_client() -> MCPToolClient:
    """Get or create the MCP client singleton."""
    global _mcp_client
    
    if _mcp_client is None:
        _mcp_client = MCPToolClient()
        # Note: Connection should be established separately
        
    return _mcp_client

async def initialize_mcp_client() -> MCPToolClient:
    """Initialize and connect the MCP client."""
    client = await get_mcp_client()
    
    if not client.session:
        success = await client.connect()
        if not success:
            raise RuntimeError("Failed to connect to MCP server")
    
    return client

async def cleanup_mcp_client():
    """Cleanup the MCP client."""
    global _mcp_client
    
    if _mcp_client:
        await _mcp_client.stop_server()
        _mcp_client = None
