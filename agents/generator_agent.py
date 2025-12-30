from google.adk.agents.llm_agent import Agent
from google.adk.agents import SequentialAgent, LlmAgent
from adk.models.lite_llm import LiteLlm

generator_agent = Agent(
    name = "prompt_generator",
    model = LitellM(model="openai/gpt-3.5-turbo",
                    temperature=0.6,
                    max_tokens=200),
    instructions = """ You are an expert prompt generator.
    Your task is to create detailed and effective prompts based on user requirements.
    When given a topic or a general idea, generate a clear and concise prompt that can be used to obtain high-quality responses from an AI language model.
    Rules:
    - Be concise yet descriptive.
    - Avoid ambiguity.
    - Do NOT include explanations
    - Output ONLY the prompt text
    """,
    output_key = "generated_prompt",
)