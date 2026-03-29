import asyncio
import os
from dotenv import load_dotenv

# Ensure we test with the .env settings
load_dotenv()

from core.llm.llm_manager import LLMManager

async def main():
    manager = LLMManager()
    await manager.initialize()
    
    print(f"Provider Information: {manager.provider_info}")
    print(f"Is LLM Active: {manager.is_llm_active}")
    print("--- Testing Generation ---")
    
    response = await manager.generate("Hello, who are you and what model is powering you?")
    print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())
