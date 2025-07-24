#!/usr/bin/env python3
"""
Test OpenAI API directly
"""

import os
import json
from openai import OpenAI

def test_openai_api():
    """Test if OpenAI API works correctly"""
    try:
        # Get API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY environment variable is not set")
            return False
        
        # Initialize client
        client = OpenAI(api_key=api_key)
        print(f"Using API key: {api_key[:5]}...{api_key[-5:]}")
        
        # Make a simple request
        completion = client.chat.completions.create(
            model="o3-mini-2025-04-16",  # Use a valid model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello world"}
            ],
            max_tokens=10
        )
        
        # Get the response
        response = completion.choices[0].message.content
        print(f"OpenAI API Response: {response}")
        
        # Check if response is valid
        if response and len(response) > 0:
            print("✅ OpenAI API is working correctly")
            return True
        else:
            print("❌ OpenAI API returned an empty response")
            return False
    
    except Exception as e:
        print(f"❌ Error testing OpenAI API: {e}")
        return False

if __name__ == "__main__":
    print("Testing OpenAI API...")
    test_openai_api()