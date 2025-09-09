from typing import Dict, Any, Optional
import json
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

def generate_underwriting_summary(narrative: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    """Use OpenAI to generate a brief professional underwriting summary."""
    prompt = f"""
You are a real estate underwriting analyst. 
Given the following property narrative and financial metrics, create a concise, professional underwriting summary:

Narrative:
{json.dumps(narrative, indent=2)}

Metrics:
{json.dumps(metrics, indent=2)}

Summary:
"""
    try:
        response = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
        )
        messages=[{"role": "user", "content": prompt}]
        response = response.invoke(messages)
        summary = response.content.strip()
        return summary
    except Exception as e:
        return f"AI summary generation failed: {e}"