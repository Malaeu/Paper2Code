#!/usr/bin/env python3
"""
Simple .env file loader for Paper2Code
Loads environment variables from .env file if it exists.
"""
import os
from pathlib import Path

def load_env(env_file=".env"):
    """Load environment variables from .env file."""
    env_path = Path(env_file)
    if not env_path.exists():
        print(f"⚠️  {env_file} not found. Copy .env.example to .env and add your API keys.")
        return False
    
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"\'')
                if value:  # Only set non-empty values
                    os.environ[key] = value
                    print(f"✅ {key} loaded")
    
    return True

if __name__ == "__main__":
    print("🔑 Loading API keys from .env file...")
    success = load_env()
    
    if success:
        # Check which APIs are configured
        apis = []
        if os.getenv("OPENAI_API_KEY"):
            apis.append("OpenAI")
        if os.getenv("ANTHROPIC_API_KEY"):
            apis.append("Anthropic")  
        if os.getenv("GEMINI_API_KEY"):
            apis.append("Gemini")
            
        if apis:
            print(f"🚀 Ready to use: {', '.join(apis)}")
        else:
            print("❌ No API keys found in .env file")
    else:
        print("❌ Failed to load .env file")