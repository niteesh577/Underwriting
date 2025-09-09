from typing import Dict, Any, Optional
import json
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def generate_underwriting_analysis(narrative: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, str]:
    """
    Use OpenAI to generate a professional underwriting analysis including:
    1. Investment Recommendation
    2. Key Investment Highlights
    3. Risk Considerations
    """
    prompt = f"""
You are a real estate underwriting analyst. 

Given the property narrative and financial metrics below, provide a structured professional underwriting analysis including:

1. Investment Recommendation (PASS, CONSIDER, BUY)
2. Key Investment Highlights (bullet points)
3. Risk Considerations (bullet points)

Narrative:
{json.dumps(narrative, indent=2)}

Metrics:
{json.dumps(metrics, indent=2)}

Please respond in JSON format like:
{{
  "investment_recommendation": "...",
  "key_investment_highlights": ["...","..."],
  "risk_considerations": ["...","..."]
}}
"""
    try:
        chat = ChatOpenAI(model="gpt-4", temperature=0.3)
        response = chat.invoke([{"role": "user", "content": prompt}])
        # Ensure JSON parsing
        analysis = json.loads(response.content)
        return analysis
    except Exception as e:
        return {
            "investment_recommendation": f"AI generation failed: {e}",
            "key_investment_highlights": [],
            "risk_considerations": []
        }