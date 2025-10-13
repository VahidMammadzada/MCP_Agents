#!/usr/bin/env python3
"""
Hierarchical LLM Router using LangChain and MCP Tools
This agent orchestrates multiple specialized agents and provides comprehensive summaries.
"""
import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import get_openai_callback

from pydantic import BaseModel, Field
from dotenv import load_dotenv

from specialized_agents import AgentFactory, BaseSpecializedAgent
from mcp_client import initialize_mcp_client, cleanup_mcp_client

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("hierarchical-llm-router")

@dataclass
class AgentResponse:
    """Response from a specialized agent."""
    agent_name: str
    query: str
    response: str
    confidence: float = 1.0
    execution_time: float = 0.0
    error: Optional[str] = None

class QueryRoutingDecision(BaseModel):
    """Model for routing decisions."""
    agents_to_use: List[str] = Field(description="List of agents to use (crypto, stock, portfolio)")
    reasoning: str = Field(description="Explanation of routing decision")
    requires_summary: bool = Field(default=True, description="Whether to provide a summary")
    priority_order: List[str] = Field(default=[], description="Order of agent execution")

class HierarchicalLLMRouter:
    """Main hierarchical router that orchestrates multiple agents and provides summaries."""
    
    def __init__(self, llm_model: str = "gemini-1.5-pro"):
        self.session_id = str(uuid.uuid4())
        self.llm_model = llm_model
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Specialized agents
        self.agents: Dict[str, BaseSpecializedAgent] = {}
        
        # Create routing prompt
        self.routing_prompt = self._create_routing_prompt()
        self.summary_prompt = self._create_summary_prompt()
        
    def _initialize_llm(self):
        """Initialize the LLM based on environment configuration."""
        llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        
        if llm_provider == "openai":
            return ChatOpenAI(
                model=self.llm_model,
                temperature=0.1,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        elif llm_provider == "anthropic":
            return ChatAnthropic(
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
                temperature=0.1,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        elif llm_provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
                temperature=0.1,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        else:
            # Default to Gemini
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0.1,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )

    def _create_routing_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for routing decisions."""
        system_message = """You are an intelligent financial query router that decides which specialized agents to use.

Available Agents:
1. CRYPTO AGENT: Cryptocurrency analysis, prices, market data, blockchain insights
2. STOCK AGENT: Stock market analysis, company research, financial metrics
3. PORTFOLIO AGENT: Portfolio management, investment advice, risk assessment, knowledge search

Instructions:
1. Analyze the user's query to understand their intent
2. Determine which agents are needed (can be multiple)
3. Consider if the query requires data from multiple domains
4. Decide if a summary is needed when using multiple agents
5. Set priority order if multiple agents are used

Examples:

Query: "What's the current price of Bitcoin and Apple stock?"
Decision: {
  "agents_to_use": ["crypto", "stock"],
  "reasoning": "Query requires both cryptocurrency and stock data",
  "requires_summary": true,
  "priority_order": ["crypto", "stock"]
}

Query: "How should I diversify my portfolio between crypto and stocks?"
Decision: {
  "agents_to_use": ["crypto", "stock", "portfolio"],
  "reasoning": "Need current market data from both domains plus portfolio advice",
  "requires_summary": true,
  "priority_order": ["crypto", "stock", "portfolio"]
}

Query: "What's Tesla's P/E ratio?"
Decision: {
  "agents_to_use": ["stock"],
  "reasoning": "Simple stock information query",
  "requires_summary": false,
  "priority_order": ["stock"]
}

Respond with a JSON object following the QueryRoutingDecision schema."""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "Query: {query}")
        ])

    def _create_summary_prompt(self) -> ChatPromptTemplate:
        """Create prompt for summarizing multiple agent responses."""
        system_message = """You are a financial analysis summarizer. Your job is to synthesize responses from multiple specialized agents into a comprehensive, coherent summary.

Guidelines:
1. Combine insights from all agents into a unified response
2. Highlight key findings and recommendations
3. Identify correlations and connections between different data points
4. Provide actionable insights based on the combined analysis
5. Maintain the technical accuracy from each agent
6. Present information in a logical, easy-to-understand format

Structure your summary as:
1. **Key Findings** - Main insights from each agent
2. **Market Analysis** - Current market conditions and trends
3. **Investment Implications** - What this means for investors
4. **Recommendations** - Specific actionable advice
5. **Risk Considerations** - Important risks to consider

Original Query: {original_query}

Agent Responses:
{agent_responses}

Please provide a comprehensive summary that synthesizes all the information."""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "Please summarize the following agent responses for the query: '{original_query}'\n\n{agent_responses}")
        ])

    async def initialize(self):
        """Initialize the router and all specialized agents."""
        logger.info("Initializing Hierarchical LLM Router...")
        
        # Initialize MCP client first
        await initialize_mcp_client()
        
        # Create all specialized agents
        self.agents = await AgentFactory.create_all_agents(self.llm_model)
        
        logger.info(f"Router initialized with {len(self.agents)} specialized agents")

    async def route_and_process_query(self, query: str) -> str:
        """Route query to appropriate agents and provide comprehensive response."""
        try:
            with get_openai_callback() as cb:
                # Step 1: Determine routing
                routing_decision = await self._get_routing_decision(query)
                logger.info(f"Routing decision: {routing_decision.reasoning}")
                
                # Step 2: Execute agents in order
                agent_responses = await self._execute_agents(query, routing_decision)
                
                # Step 3: Generate summary if needed
                if routing_decision.requires_summary and len(agent_responses) > 1:
                    final_response = await self._generate_summary(query, agent_responses)
                elif agent_responses:
                    # Single agent response
                    final_response = agent_responses[0].response
                else:
                    final_response = "I apologize, but I couldn't process your query at this time."
                
                # Log token usage
                logger.info(f"Total tokens used: {cb.total_tokens}, Cost: ${cb.total_cost:.4f}")
                
                return final_response
                
        except Exception as e:
            logger.error(f"Error in route_and_process_query: {e}")
            return f"I apologize, but I encountered an error processing your query: {str(e)}"

    async def _get_routing_decision(self, query: str) -> QueryRoutingDecision:
        """Determine which agents to use for the query."""
        try:
            messages = self.routing_prompt.format_messages(query=query)
            response = await self.llm.ainvoke(messages)
            
            # Parse JSON response
            response_text = response.content
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text.strip()
            
            decision_data = json.loads(json_text)
            return QueryRoutingDecision(**decision_data)
            
        except Exception as e:
            logger.error(f"Error in routing decision: {e}")
            # Fallback to simple keyword-based routing
            return self._fallback_routing(query)

    def _fallback_routing(self, query: str) -> QueryRoutingDecision:
        """Fallback routing using keyword matching."""
        query_lower = query.lower()
        agents_to_use = []
        
        # Check for crypto keywords
        crypto_keywords = ["crypto", "cryptocurrency", "bitcoin", "ethereum", "btc", "eth", "coin", "token"]
        if any(keyword in query_lower for keyword in crypto_keywords):
            agents_to_use.append("crypto")
        
        # Check for stock keywords
        stock_keywords = ["stock", "share", "equity", "nasdaq", "company", "ticker", "financial", "earnings"]
        if any(keyword in query_lower for keyword in stock_keywords):
            agents_to_use.append("stock")
        
        # Check for portfolio keywords
        portfolio_keywords = ["portfolio", "investment", "advice", "allocation", "risk", "diversify", "recommend"]
        if any(keyword in query_lower for keyword in portfolio_keywords):
            agents_to_use.append("portfolio")
        
        # Default to portfolio if no clear match
        if not agents_to_use:
            agents_to_use = ["portfolio"]
        
        return QueryRoutingDecision(
            agents_to_use=agents_to_use,
            reasoning=f"Fallback routing based on keywords: {agents_to_use}",
            requires_summary=len(agents_to_use) > 1,
            priority_order=agents_to_use
        )

    async def _execute_agents(self, query: str, routing_decision: QueryRoutingDecision) -> List[AgentResponse]:
        """Execute the selected agents in priority order."""
        responses = []
        
        for agent_name in routing_decision.priority_order:
            if agent_name in self.agents:
                try:
                    start_time = asyncio.get_event_loop().time()
                    
                    # Execute agent
                    agent = self.agents[agent_name]
                    response = await agent.process_query(query)
                    
                    execution_time = asyncio.get_event_loop().time() - start_time
                    
                    agent_response = AgentResponse(
                        agent_name=agent_name,
                        query=query,
                        response=response,
                        execution_time=execution_time
                    )
                    
                    responses.append(agent_response)
                    logger.info(f"{agent_name} agent completed in {execution_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Error executing {agent_name} agent: {e}")
                    error_response = AgentResponse(
                        agent_name=agent_name,
                        query=query,
                        response=f"Error in {agent_name} agent: {str(e)}",
                        error=str(e)
                    )
                    responses.append(error_response)
        
        return responses

    async def _generate_summary(self, original_query: str, agent_responses: List[AgentResponse]) -> str:
        """Generate a comprehensive summary from multiple agent responses."""
        try:
            # Format agent responses for summary
            formatted_responses = []
            for response in agent_responses:
                formatted_responses.append(f"**{response.agent_name.upper()} AGENT:**\n{response.response}")
            
            agent_responses_text = "\n\n".join(formatted_responses)
            
            messages = self.summary_prompt.format_messages(
                original_query=original_query,
                agent_responses=agent_responses_text
            )
            
            summary_response = await self.llm.ainvoke(messages)
            
            # Add execution metadata
            total_execution_time = sum(r.execution_time for r in agent_responses)
            agents_used = [r.agent_name for r in agent_responses]
            
            summary_with_metadata = f"{summary_response.content}\n\n---\n"
            summary_with_metadata += f"*Analysis provided by: {', '.join(agents_used)} | "
            summary_with_metadata += f"Total execution time: {total_execution_time:.2f}s*"
            
            return summary_with_metadata
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            # Return concatenated responses as fallback
            return "\n\n---\n\n".join(r.response for r in agent_responses)

    async def chat(self, message: str) -> str:
        """Process a chat message with memory."""
        try:
            # Add user message to memory
            self.memory.chat_memory.add_user_message(message)
            
            # Process query
            response = await self.route_and_process_query(message)
            
            # Add assistant response to memory
            self.memory.chat_memory.add_ai_message(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        status = {
            "session_id": self.session_id,
            "llm_model": self.llm_model,
            "agents": {}
        }
        
        for name, agent in self.agents.items():
            status["agents"][name] = agent.get_capabilities()
        
        return status

    async def cleanup(self):
        """Cleanup resources."""
        await cleanup_mcp_client()
        logger.info("Router cleanup completed")

async def main():
    """Main function to run the hierarchical financial agent."""
    print("🤖 Initializing Hierarchical Financial Intelligence System...")
    
    router = HierarchicalLLMRouter()
    await router.initialize()
    
    print("✅ System Ready!")
    print("💡 This system orchestrates multiple AI agents for comprehensive financial analysis")
    print("🎯 Agents: CryptoAgent, StockAgent, PortfolioAgent")
    print("📊 Features: Real-time data, intelligent routing, comprehensive summaries")
    print("Type 'status' for system info, 'quit' to exit\n")
    
    try:
        while True:
            user_input = input("💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'status':
                status = router.get_agent_status()
                print(f"\n📊 System Status:\n{json.dumps(status, indent=2)}\n")
                continue
            
            if not user_input:
                continue
                
            print("🔄 Analyzing query and routing to appropriate agents...")
            response = await router.chat(user_input)
            print(f"\n🤖 Financial Intelligence System:\n{response}\n")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    finally:
        await router.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
