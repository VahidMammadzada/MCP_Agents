#!/usr/bin/env python3
"""
Simplified Financial AI System Launcher
A simpler approach to launching the hierarchical LLM router
"""
import sys
import os
from pathlib import Path
import time

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import required modules
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify required environment variables
if not os.getenv("GOOGLE_API_KEY"):
    print("❌ Missing GOOGLE_API_KEY environment variable.")
    print("Please set it in your .env file.")
    sys.exit(1)

print("🌟 Simplified Financial AI Launcher")
print("🧠 Powered by Google Gemini AI")
print("=" * 60)

async def main():
    """Main entry point."""
    try:
        # Import hierarchical router directly
        sys.path.append(str(Path(__file__).parent / "langchain"))
        from hierarchical_llm_router import HierarchicalLLMRouter
        
        print("🤖 Initializing Financial Intelligence System...")
        print("🧠 Powered by Google Gemini AI")
        
        # Initialize the router
        router = HierarchicalLLMRouter()
        await router.initialize()
        
        print("\n✅ System Ready!")
        print("💡 Type 'help' for commands, 'quit' to exit\n")
        
        # Interactive loop
        while True:
            try:
                user_input = input("💬 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    show_help()
                    continue
                
                if not user_input:
                    continue
                    
                print("🔄 Processing your request...")
                response = await router.chat(user_input)
                print(f"\n🤖 AI Response:\n{response}\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
        # Cleanup
        await router.cleanup()
        
    except Exception as e:
        print(f"❌ Failed to start system: {e}")
        import traceback
        traceback.print_exc()

def show_help():
    """Show help information."""
    help_text = """
📚 Financial AI System - Help
🚀 Powered by Google Gemini AI

🎯 COMMANDS:
  help     - Show this help message
  quit     - Exit the system

💡 EXAMPLE QUERIES:
  - "What's the current price of Bitcoin and Tesla stock?"
  - "How should I diversify my portfolio between crypto and stocks?"
  - "Analyze Apple's financial performance and give investment advice"
  - "What are the top performing cryptocurrencies this week?"
"""
    print(help_text)

if __name__ == "__main__":
    asyncio.run(main())
