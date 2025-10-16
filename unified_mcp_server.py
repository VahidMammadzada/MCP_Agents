#!/usr/bin/env python3
"""
Unified MCP Server - Stock, Portfolio, and Knowledge Tools
This server provides stock market data and portfolio management tools.
Crypto tools are handled by the external CoinGecko MCP server.
"""
import asyncio
import json
import logging
import os
from typing import Any, Dict, List
from datetime import datetime, timedelta
import pandas as pd

import yfinance as yf
import httpx
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unified-mcp-server")

class UnifiedMCPServer:
    """Unified MCP Server hosting all financial tools."""
    
    def __init__(self):
        self.server = Server("unified-financial-tools")
        
        # Initialize external connections
        # Note: ccxt exchange not needed since crypto is handled by CoinGecko MCP
        self.mongo_client = None
        self.db = None
        
        # Setup MongoDB connection (with timeout to prevent hanging)
        self.setup_mongodb()
        
        # Setup handlers
        self.setup_handlers()

    def setup_mongodb(self):
        """Initialize MongoDB connection."""
        try:
            mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
            # Add timeout to prevent hanging
            self.mongo_client = MongoClient(
                mongo_uri,
                serverSelectionTimeoutMS=2000  # 2 second timeout
            )
            self.db = self.mongo_client[os.getenv("MONGODB_DB_NAME", "financial_data")]
            
            # Test connection with timeout
            self.mongo_client.admin.command('ping')
            logger.info("✓ MongoDB connected successfully")
            
        except Exception as e:
            logger.warning(f"⚠ MongoDB connection failed: {e}")
            logger.warning("⚠ Portfolio/knowledge tools will have limited functionality")
            self.mongo_client = None
            self.db = None

    def setup_handlers(self):
        """Setup MCP server handlers."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> ListToolsResult:
            """List all available financial tools (Stock, Portfolio, Knowledge)."""
            return ListToolsResult(
                tools=[
                    # Stock/NASDAQ Tools
                    Tool(
                        name="get_stock_price",
                        description="Get current stock price and basic information",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string", "description": "Stock symbol (e.g., AAPL, MSFT)"}
                            },
                            "required": ["symbol"]
                        }
                    ),
                    Tool(
                        name="get_stock_info",
                        description="Get comprehensive stock information and metrics",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string", "description": "Stock symbol"}
                            },
                            "required": ["symbol"]
                        }
                    ),
                    Tool(
                        name="get_stock_history",
                        description="Get historical stock price data",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string", "description": "Stock symbol"},
                                "period": {"type": "string", "description": "Period (1d, 5d, 1mo, 3mo, 6mo, 1y)", "default": "1mo"}
                            },
                            "required": ["symbol"]
                        }
                    ),
                    Tool(
                        name="get_market_movers",
                        description="Get top market gainers and losers",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "market": {"type": "string", "description": "Market type (nasdaq, sp500)", "default": "nasdaq"},
                                "limit": {"type": "integer", "description": "Number of stocks to return", "default": 10}
                            }
                        }
                    ),
                    Tool(
                        name="get_stock_financials",
                        description="Get financial statements for a stock",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string", "description": "Stock symbol"},
                                "statement_type": {"type": "string", "description": "Type (income, balance, cashflow)", "default": "income"}
                            },
                            "required": ["symbol"]
                        }
                    ),
                    
                    # Portfolio/Knowledge Tools
                    Tool(
                        name="search_knowledge",
                        description="Search investment knowledge base using RAG",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query for investment knowledge"},
                                "limit": {"type": "integer", "description": "Number of results", "default": 5}
                            },
                            "required": ["query"]
                        }
                    ),
                    Tool(
                        name="add_knowledge",
                        description="Add investment knowledge to the database",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "title": {"type": "string", "description": "Knowledge title"},
                                "content": {"type": "string", "description": "Knowledge content"},
                                "category": {"type": "string", "description": "Category (strategy, analysis, etc.)"},
                                "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags"}
                            },
                            "required": ["title", "content"]
                        }
                    ),
                    Tool(
                        name="analyze_portfolio",
                        description="Analyze a portfolio composition and provide insights",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "portfolio": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "symbol": {"type": "string"},
                                            "weight": {"type": "number"},
                                            "shares": {"type": "number"}
                                        }
                                    },
                                    "description": "Portfolio holdings"
                                }
                            },
                            "required": ["portfolio"]
                        }
                    ),
                    Tool(
                        name="get_risk_assessment",
                        description="Assess portfolio risk metrics",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "symbols": {"type": "array", "items": {"type": "string"}, "description": "Stock symbols"},
                                "timeframe": {"type": "string", "description": "Analysis timeframe", "default": "1y"}
                            },
                            "required": ["symbols"]
                        }
                    )
                ]
            )

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
            """Handle tool calls."""
            tool_name = name
            args = arguments or {}
            
            try:
                # Route to appropriate tool handler
                if tool_name.startswith("get_stock") or tool_name == "get_market_movers":
                    result = await self._handle_stock_tool(tool_name, args)
                    return result.content
                elif tool_name in ["search_knowledge", "add_knowledge", "analyze_portfolio", "get_risk_assessment"]:
                    result = await self._handle_portfolio_tool(tool_name, args)
                    return result.content
                else:
                    return [TextContent(type="text", text=f"Unknown tool: {tool_name}")]
                    
            except Exception as e:
                logger.error(f"Error in tool call {tool_name}: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    # Stock Tool Handlers
    async def _handle_stock_tool(self, tool_name: str, args: Dict[str, Any]) -> CallToolResult:
        """Handle stock-related tools."""
        
        if tool_name == "get_stock_price":
            symbol = args.get("symbol", "").upper()
            
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                history = stock.history(period="1d")
                
                if not history.empty:
                    current_price = history['Close'].iloc[-1]
                    open_price = history['Open'].iloc[-1]
                    
                    price_data = {
                        "symbol": symbol,
                        "current_price": round(current_price, 2),
                        "open": round(open_price, 2),
                        "high": round(history['High'].iloc[-1], 2),
                        "low": round(history['Low'].iloc[-1], 2),
                        "volume": int(history['Volume'].iloc[-1]),
                        "change": round(current_price - open_price, 2),
                        "change_percent": round(((current_price - open_price) / open_price) * 100, 2),
                        "company_name": info.get("longName", "N/A"),
                        "market_cap": info.get("marketCap", "N/A"),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    return CallToolResult(
                        content=[TextContent(type="text", text=json.dumps(price_data, indent=2))]
                    )
                
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Could not fetch price data for {symbol}")]
                )
                
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error fetching stock price: {str(e)}")]
                )

        elif tool_name == "get_stock_info":
            symbol = args.get("symbol", "").upper()
            
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                
                stock_data = {
                    "symbol": symbol,
                    "company_name": info.get("longName"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "market_cap": info.get("marketCap"),
                    "enterprise_value": info.get("enterpriseValue"),
                    "pe_ratio": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "price_to_book": info.get("priceToBook"),
                    "dividend_yield": info.get("dividendYield"),
                    "beta": info.get("beta"),
                    "52_week_high": info.get("fiftyTwoWeekHigh"),
                    "52_week_low": info.get("fiftyTwoWeekLow"),
                    "average_volume": info.get("averageVolume"),
                    "business_summary": info.get("longBusinessSummary"),
                    "employees": info.get("fullTimeEmployees"),
                    "website": info.get("website"),
                    "headquarters": f"{info.get('city', '')}, {info.get('state', '')}, {info.get('country', '')}".strip(", ")
                }
                
                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(stock_data, indent=2))]
                )
                
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error fetching stock info: {str(e)}")]
                )

        # Add other stock tools...
        return CallToolResult(
            content=[TextContent(type="text", text=f"Stock tool {tool_name} not implemented yet")]
        )

    # Portfolio/Knowledge Tool Handlers
    async def _handle_portfolio_tool(self, tool_name: str, args: Dict[str, Any]) -> CallToolResult:
        """Handle portfolio and knowledge-related tools."""
        
        if tool_name == "search_knowledge":
            query = args.get("query", "")
            limit = args.get("limit", 5)
            
            if not self.db:
                return CallToolResult(
                    content=[TextContent(type="text", text="MongoDB not available for knowledge search")]
                )
            
            try:
                # Simple text search in MongoDB
                collection = self.db.knowledge_base
                
                # Create text index if it doesn't exist
                try:
                    collection.create_index([("title", "text"), ("content", "text")])
                except:
                    pass  # Index might already exist
                
                # Search using text index
                results = list(collection.find(
                    {"$text": {"$search": query}},
                    {"score": {"$meta": "textScore"}}
                ).sort([("score", {"$meta": "textScore"})]).limit(limit))
                
                if not results:
                    # Fallback to regex search
                    results = list(collection.find({
                        "$or": [
                            {"title": {"$regex": query, "$options": "i"}},
                            {"content": {"$regex": query, "$options": "i"}}
                        ]
                    }).limit(limit))
                
                knowledge_results = []
                for doc in results:
                    knowledge_results.append({
                        "title": doc.get("title"),
                        "content": doc.get("content", "")[:200] + "...",  # Truncate for preview
                        "category": doc.get("category"),
                        "tags": doc.get("tags", []),
                        "created_at": doc.get("created_at")
                    })
                
                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(knowledge_results, indent=2))]
                )
                
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error searching knowledge: {str(e)}")]
                )

        elif tool_name == "add_knowledge":
            title = args.get("title", "")
            content = args.get("content", "")
            category = args.get("category", "general")
            tags = args.get("tags", [])
            
            if not self.db:
                return CallToolResult(
                    content=[TextContent(type="text", text="MongoDB not available for adding knowledge")]
                )
            
            try:
                collection = self.db.knowledge_base
                
                doc = {
                    "title": title,
                    "content": content,
                    "category": category,
                    "tags": tags,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
                
                result = collection.insert_one(doc)
                
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Knowledge added successfully with ID: {result.inserted_id}")]
                )
                
            except Exception as e:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error adding knowledge: {str(e)}")]
                )

        # Add other portfolio tools...
        return CallToolResult(
            content=[TextContent(type="text", text=f"Portfolio tool {tool_name} not implemented yet")]
        )

    async def run(self):
        """Run the unified MCP server."""
        from mcp.server.stdio import stdio_server
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="unified-financial-tools",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )

if __name__ == "__main__":
    server = UnifiedMCPServer()
    asyncio.run(server.run())
