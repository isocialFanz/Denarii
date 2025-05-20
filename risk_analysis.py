import openai
import pandas as pd
import os

def get_llm_analysis(simulation_df: pd.DataFrame) -> str:
    sim_json = simulation_df.head(12).to_json(orient="records")

    prompt = f"""
You are a Senior Tokenomics Advisor.

Analyze this tokenomics simulation output and identify any risks or concerns:
- Inflation pressure
- Vesting cliffs
- Centralization
- Ineffective burns
- Utility gaps

Suggest improvements in simple language for a Web3 founder.

Simulation Data:
{sim_json}
"""

    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a Tokenomics Risk Advisor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return response["choices"][0]["message"]["content"]

