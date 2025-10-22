# Financial AI System - Gemini API Integration

This system has been adapted to use Google's Gemini API as the primary LLM provider. The system provides specialized financial agents for cryptocurrency, stock market, and portfolio analysis.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Gemini API key

### Setup

1. **Clone and navigate to the project directory:**
   ```powershell
   cd c:\Users\VahidPC\MCP_Agents
   ```

2. **Run the setup script (optional):**
   ```powershell
   .\setup_gemini.ps1
   ```

3. **Get your Gemini API key:**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key

4. **Configure environment variables:**
   - Edit the `.env` file
   - Replace `your_google_api_key_here` with your actual API key:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

5. **Test the integration:**
   ```powershell
   python test_gemini.py
   ```

6. **Start the system:**
   ```powershell
   python launch_financial_ai.py
   ```

## 🤖 Available Models

### Gemini Models (Primary)
- `gemini-1.5-pro` (default) - Best performance, recommended
- `gemini-1.5-flash` - Faster responses, good for quick queries
- `gemini-pro` - Standard model

### Alternative Providers
The system also supports:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Ollama (local models)

To switch providers, change the `LLM_PROVIDER` in your `.env` file.

## 🏦 Specialized Agents

### 1. CryptoAgent
- Real-time cryptocurrency prices
- Market data and analysis
- Historical price trends
- Top cryptocurrency rankings

### 2. StockAgent
- Stock prices and market data
- Company financial information
- Market movers analysis
- Investment insights

### 3. PortfolioAgent
- Portfolio optimization
- Risk assessment
- Asset allocation recommendations
- Investment strategy guidance

## 📁 Project Structure

```
MCP_Agents/
├── specialized_agents.py      # Main agent classes (updated for Gemini)
├── main_llm_router.py        # Router with Gemini support
├── langchain/                # LangChain implementations
│   ├── hierarchical_llm_router.py
│   ├── base_langchain_agent.py
│   └── ...
├── requirements.txt          # Dependencies (includes Gemini)
├── .env.example             # Environment template
├── setup_gemini.ps1         # Setup script
├── test_gemini.py           # Integration test
└── README_gemini.md         # This file
```

## 🔧 Configuration

### Environment Variables (.env)
```
# Primary LLM Provider
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-1.5-pro

# Alternative providers (optional)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Database (for portfolio data)
MONGODB_URI=mongodb://localhost:27017
```

## 🧪 Testing

Run the test suite to ensure everything is working:

```powershell
# Test Gemini API integration
python test_gemini.py

# Test individual agents
python test_agents.py

# Test the complete system
python test_hierarchical_system.py
```

## 📊 Usage Examples

### Using Specialized Agents

```python
import asyncio
from specialized_agents import AgentFactory

async def main():
    # Create agents
    crypto_agent = await AgentFactory.create_crypto_agent()
    stock_agent = await AgentFactory.create_stock_agent()
    
    # Query crypto agent
    crypto_response = await crypto_agent.process_query(
        "What's the current price of Bitcoin and Ethereum?"
    )
    
    # Query stock agent
    stock_response = await stock_agent.process_query(
        "How is Apple stock performing today?"
    )
    
    print(f"Crypto: {crypto_response}")
    print(f"Stocks: {stock_response}")

asyncio.run(main())
```

### Using the Hierarchical Router

```python
from langchain.hierarchical_llm_router import HierarchicalLLMRouter

router = HierarchicalLLMRouter()
await router.initialize()

response = await router.process_query(
    "I want to invest $10000. What's the best strategy considering current crypto and stock markets?"
)
```

## 🔍 Key Changes for Gemini

### 1. Dependencies
- Added `langchain-google-genai>=1.0.0`
- Added `google-generativeai>=0.3.0`

### 2. LLM Initialization
```python
# Old (OpenAI)
from langchain_openai import ChatOpenAI
self.llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.1)

# New (Gemini)
from langchain_google_genai import ChatGoogleGenerativeAI
self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.1)
```

### 3. Default Models
- Changed default from `gpt-4-turbo-preview` to `gemini-1.5-pro`
- Updated all agent constructors and factory methods

### 4. Environment Configuration
- Primary provider is now `gemini`
- Added `GOOGLE_API_KEY` configuration
- Updated example environment file

## 🚨 Troubleshooting

### Common Issues

1. **Import Error: langchain_google_genai**
   ```powershell
   pip install langchain-google-genai google-generativeai
   ```

2. **API Key Not Set**
   - Check your `.env` file
   - Ensure `GOOGLE_API_KEY` is set correctly
   - Verify the key is valid at [Google AI Studio](https://makersuite.google.com/app/apikey)

3. **Rate Limiting**
   - Gemini has rate limits; the system will retry automatically
   - Consider upgrading your Gemini API quota if needed

4. **Model Not Found**
   - Ensure you're using a valid Gemini model name
   - Check the latest model names in Google AI Studio

### Getting Help

1. Run the test script: `python test_gemini.py`
2. Check logs in the `logs/` directory
3. Verify your API key has the necessary permissions
4. Ensure all dependencies are installed correctly

## 🔄 Switching Back to OpenAI

If you need to switch back to OpenAI:

1. Update `.env`:
   ```
   LLM_PROVIDER=openai
   OPENAI_API_KEY=your_openai_key_here
   ```

2. The system will automatically use OpenAI models

## 📈 Performance Notes

- **Gemini 1.5 Pro**: Best for complex financial analysis
- **Gemini 1.5 Flash**: Good for quick price checks
- **Response Time**: Typically 2-5 seconds for financial queries
- **Cost**: Generally more cost-effective than GPT-4

## 🔗 Resources

- [Google AI Studio](https://makersuite.google.com/)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [LangChain Google GenAI Integration](https://python.langchain.com/docs/integrations/llms/google_ai)
- [Project Repository](https://github.com/your-repo/financial-ai)
