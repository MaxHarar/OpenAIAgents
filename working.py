import os
from dotenv import load_dotenv
from agents import Agent, ModelSettings

#Load Env variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
	raise ValueError("key not found")


basic_agent = Agent(
	name="My First Agent",
	instructions="You are a helpful assistant that provides factual information.",
	model="gpt-4o",
)

advanced_agent = Agent(
	name="Advanced Assistant",
	instructions="""You are a professional, concise assistant who always provides accurate information. When you don't know something, clearly state that. Focus on giving actionable insights when answering questions.""",
	model="gpt-4o",
	model_settings=ModelSettings(
		temperature=0.3, # Lower for more deterministic outputs
		max_tokens=1024, #Max length of response
	),
	tools=[],
)
