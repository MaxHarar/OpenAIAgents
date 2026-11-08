import os
from dotenv import load_dotenv
from agents import Agent, Runner

#Load Env variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
	raise ValueError("key not found")


from agents import Agent

basic_agnet = Agent(
	name="My First Agent",
	instructions= "You are a helpful assistant that provides factual information.",
	model = "gpt-4o"
)

from agents impot ModelSettings

advanced_agnet = Agent(
	name="Advanced Assistant",
	instructions =""YOu are a professional, concise assistant who alyways provides accurate information. When you don't know something, clearly state that. Focus on giving actionable insights when answer questions."",
model="gpt-40",
model_settings=ModelSettings(
	temperature=0.3, # Lower for more deterministic outputs 
	max_tokens=1024, #Max length of response
),
tools[]
)

