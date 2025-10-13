#!/usr/bin/env python3
"""
Specialized LangChain Agents using MCP Tools
Each agent focuses on a specific domain and uses relevant MCP tools
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool

from langchain_mcp_tools import (
    get_crypto_tools, 
    get_stock_tools, 
    get_portfolio_tools,
    initialize_tools
)

logger = logging.getLogger(__name__)

class BaseSpecializedAgent:
    """Base class for specialized financial agents."""
    
    def __init__(self, agent_name: str, description: str, llm_model: str = "gemini-2.5-flash"):
        self.agent_name = agent_name
        self.description = description
        self.llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0.1)
        self.tools = []
        self.agent_executor = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    async def initialize(self, tools: List[BaseTool]):
        """Initialize the agent with tools."""
        self.tools = tools
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self._get_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )
        
        logger.info(f"{self.agent_name} initialized with {len(self.tools)} tools")
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for the agent."""
        return f"""You are {self.agent_name}, a specialized financial AI assistant.

{self.description}

Your role:
1. Analyze queries in your domain of expertise
2. Use available tools to gather accurate, real-time data
3. Provide clear, actionable insights based on the data
4. Explain your reasoning and methodology

Guidelines:
- Be accurate and factual
- Use current market data when available
- Provide specific actionable recommendations
- Explain risks and limitations
- Cite data sources when relevant

Available tools: {[tool.name for tool in self.tools]}

Current time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"""

    async def process_query(self, query: str, context: Dict[str, Any] = None) -> str:
        """Process a query and return analysis."""
        try:
            result = await self.agent_executor.ainvoke({
                "input": query,
                "chat_history": self.memory.chat_memory.messages
            })
            
            return result["output"]
            
        except Exception as e:
            logger.error(f"Error in {self.agent_name}: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities summary."""
        return {
            "name": self.agent_name,
            "description": self.description,
            "tools": [tool.name for tool in self.tools],
            "tool_count": len(self.tools)
        }

class CryptoAgent(BaseSpecializedAgent):
    """Specialized agent for cryptocurrency analysis."""
    
    def __init__(self, llm_model: str = "gemini-1.5-pro"):
        super().__init__(
            agent_name="CryptoAgent",
            description="""I specialize in cryptocurrency market analysis, price tracking, and blockchain insights.
            
My expertise includes:
- Real-time cryptocurrency prices and market data
- Market capitalization and volume analysis
- Price trend analysis and historical data
- Top cryptocurrency rankings
- Market movement insights
- Technical analysis of crypto assets

I provide accurate, up-to-date information about the cryptocurrency market to help with investment decisions.""",
            llm_model=llm_model
        )
    
    async def initialize(self):
        """Initialize with crypto-specific tools."""
        crypto_tools = get_crypto_tools()
        await super().initialize(crypto_tools)

class StockAgent(BaseSpecializedAgent):
    """Specialized agent for stock market analysis."""
    
    def __init__(self, llm_model: str = "gemini-1.5-pro"):
        super().__init__(
            agent_name="StockAgent", 
            description="""I specialize in stock market analysis, company research, and equity investment insights.

My expertise includes:
- Real-time stock prices and market data
- Company financial information and metrics
- Historical price analysis and trends
- Market movers (gainers and losers)
- Financial statement analysis
- Sector and industry analysis
- Investment recommendations based on fundamentals

I provide comprehensive stock market analysis to support investment decision-making.""",
            llm_model=llm_model
        )
    
    async def initialize(self):
        """Initialize with stock-specific tools."""
        stock_tools = get_stock_tools()
        await super().initialize(stock_tools)

class PortfolioAgent(BaseSpecializedAgent):
    """Specialized agent for portfolio management and investment advice."""
    
    def __init__(self, llm_model: str = "gemini-1.5-pro"):
        super().__init__(
            agent_name="PortfolioAgent",
            description="""I specialize in portfolio management, investment strategy, and financial planning.

My expertise includes:
- Portfolio composition analysis and optimization
- Risk assessment and management
- Investment strategy development
- Asset allocation recommendations
- Diversification analysis
- Knowledge base search for investment insights
- Custom investment advice based on goals and risk tolerance

I use a comprehensive knowledge base and analytical tools to provide personalized investment guidance.""",
            llm_model=llm_model
        )
    
    async def initialize(self):
        """Initialize with portfolio-specific tools."""
        portfolio_tools = get_portfolio_tools()
        await super().initialize(portfolio_tools)

class AgentFactory:
    """Factory for creating and managing specialized agents."""
    
    @staticmethod
    async def create_crypto_agent(llm_model: str = "gemini-1.5-pro") -> CryptoAgent:
        """Create and initialize a crypto agent."""
        agent = CryptoAgent(llm_model)
        await agent.initialize()
        return agent
    
    @staticmethod
    async def create_stock_agent(llm_model: str = "gemini-1.5-pro") -> StockAgent:
        """Create and initialize a stock agent."""
        agent = StockAgent(llm_model)
        await agent.initialize()
        return agent
    
    @staticmethod
    async def create_portfolio_agent(llm_model: str = "gemini-1.5-pro") -> PortfolioAgent:
        """Create and initialize a portfolio agent."""
        agent = PortfolioAgent(llm_model)
        await agent.initialize()
        return agent
    
    @staticmethod
    async def create_all_agents(llm_model: str = "gemini-1.5-pro") -> Dict[str, BaseSpecializedAgent]:
        """Create and initialize all specialized agents."""
        # First initialize tools to ensure MCP connection
        await initialize_tools()
        
        agents = {
            "crypto": await AgentFactory.create_crypto_agent(llm_model),
            "stock": await AgentFactory.create_stock_agent(llm_model),
            "portfolio": await AgentFactory.create_portfolio_agent(llm_model)
        }
        
        logger.info(f"Created {len(agents)} specialized agents")
        return agents
