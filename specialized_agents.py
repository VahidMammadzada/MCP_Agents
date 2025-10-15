#!/usr/bin/env python3
"""
Specialized LangChain Agents using MCP Tools
Each agent focuses on a specific domain and uses relevant MCP tools
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool

from mcp_client import get_mcp_client, initialize_mcp_client

logger = logging.getLogger(__name__)

class BaseSpecializedAgent:
    """Base class for specialized financial agents."""
    
    def __init__(self, agent_name: str, description: str, llm_model: str = "gemini-2.5-pro"):
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
        
        if not tools:
            logger.warning(f"{self.agent_name} initialized with no tools!")
            return
        
        # Create system prompt with tool instructions
        # Note: We don't include tool descriptions in the prompt because
        # create_tool_calling_agent automatically handles tool information
        system_message = f"""{self._get_system_prompt()}

IMPORTANT: You MUST use the available tools to get real-time data. Never make up information.

When a user asks a question:
1. Identify the appropriate tool to use
2. Call the tool with the correct parameters
3. Use the tool's response to answer the question accurately"""

        # Create prompt template for tool calling agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent using tool calling (works with Gemini's native tool calling)
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            return_intermediate_steps=True
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
- You have access to all available tools (crypto, stock, portfolio, knowledge)
- Choose the most relevant tools for the query based on tool descriptions

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
    
    def __init__(self, llm_model: str = "gemini-2.5-pro"):
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
        mcp_client = await get_mcp_client()
        all_tools = mcp_client.get_tools()
        await super().initialize(all_tools)

# COMMENTED OUT - ChromaDB integration disabled
# class DocumentAgent(BaseSpecializedAgent):
#     """Specialized agent for document analysis and Q&A using ChromaDB."""

#     def __init__(self, llm_model: str = "gemini-2.5-pro"):
#         super().__init__(
#             agent_name="DocumentAgent",
#             description="""I specialize in document analysis, storage, and question answering using semantic search.

# My expertise includes:
# - Processing and storing uploaded documents in vector database
# - Semantic search across document collections
# - Question answering based on document content
# - Document summarization and key insights extraction
# - RAG (Retrieval Augmented Generation) for accurate responses
# - Managing document collections by topic or project

# I use ChromaDB for persistent vector storage and semantic search to provide accurate answers based on your documents.""",
#             llm_model=llm_model
#         )
    
#     async def initialize(self):
#         """Initialize with all available tools (including ChromaDB tools)."""
#         mcp_client = await get_mcp_client()
#         all_tools = mcp_client.get_tools()
#         await super().initialize(all_tools)
    
#     def _get_system_prompt(self) -> str:
#         """Get system prompt for the document agent."""
#         return """You are a Document Analysis Expert with deep expertise in:

# **Core Capabilities:**
# - Document processing and vector storage
# - Semantic search and information retrieval
# - Question answering based on document content
# - Document summarization and analysis
# - RAG (Retrieval Augmented Generation)

# **Available Chroma Tools:**
# - chroma_create_collection: Create new collections for organizing documents
# - chroma_add_documents: Store documents with embeddings for semantic search
# - chroma_query_documents: Search documents semantically to find relevant content
# - chroma_get_documents: Retrieve specific documents by ID or filters
# - chroma_list_collections: See all available document collections
# - chroma_peek_collection: View a sample of documents in a collection
# - chroma_get_collection_info: Get detailed collection information
# - chroma_delete_collection: Remove collections when no longer needed

# **Workflow for Document Q&A:**
# 1. **First Query**: If no collection exists, create one: `chroma_create_collection(name="user_documents")`
# 2. **Document Upload**: When user provides documents, use `chroma_add_documents()` to store them
# 3. **Questions**: Use `chroma_query_documents()` to find relevant passages
# 4. **Answer**: Synthesize information from search results to provide accurate answers
# 5. **Cite Sources**: Always reference which documents you're using

# **Best Practices:**
# - Create semantic, descriptive collection names
# - Include metadata (filename, date, source) when adding documents
# - Use natural language queries for better semantic search
# - Combine multiple search results for comprehensive answers
# - Always cite which documents contain the information
# - Explain if documents don't contain the answer

# **Response Format:**
# When answering questions:
# 1. Search relevant documents
# 2. Provide clear, accurate answer
# 3. Cite sources: "According to [document_name]..."
# 4. If answer not found: "Based on the documents provided, I couldn't find information about..."

# Remember: Your strength is accurate, source-based answers from uploaded documents."""

class AgentFactory:
    """Factory for creating and managing specialized agents."""
    
    @staticmethod
    async def create_crypto_agent(llm_model: str = "gemini-2.5-pro") -> CryptoAgent:
        """Create and initialize a crypto agent."""
        agent = CryptoAgent(llm_model)
        await agent.initialize()
        return agent
    
    # COMMENTED OUT - ChromaDB integration disabled
    # @staticmethod
    # async def create_document_agent(llm_model: str = "gemini-2.5-pro") -> DocumentAgent:
    #     """Create and initialize a document agent."""
    #     agent = DocumentAgent(llm_model)
    #     await agent.initialize()
    #     return agent
    
    @staticmethod
    async def create_all_agents(llm_model: str = "gemini-2.5-pro") -> Dict[str, BaseSpecializedAgent]:
        """Create and initialize all specialized agents."""
        # First initialize MCP client to ensure server connections
        await initialize_mcp_client()
        
        agents = {
            "crypto": await AgentFactory.create_crypto_agent(llm_model),
            # "document": await AgentFactory.create_document_agent(llm_model)  # COMMENTED OUT
        }
        
        logger.info(f"Created {len(agents)} specialized agents")
        return agents
