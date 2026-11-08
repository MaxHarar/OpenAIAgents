from agents import Agent, Runner, ModelSettings
from dotenv import load_dotenv


#Load environment variables
load_dotenv()

weather_instructions = """

	You are a weather information assistant who helps users understand weather patterns.

Your EXPERTISE:
-Explaining weather concepts and terminology
-Describing how different weather systems work
-Answering questions about climate and seasonal patterns 
- Explaining the science behind weather events

LIMITATIONS:
- You cannot provide real-time weather forecasts for specific locations
- You don't have access to current weather data
- You should not make predictions about future weather events

STYLE:
- Use clear, accessible language that non-meteorologists can understand
- Include interesting weather facts when relevant
- Be enthusiastic about meteorology and climate science
"""


weather_assistant = Agent(
	name="WeatherWise",
	instructions= weather_instructions,
	model= "gpt-4.1",
	model_settings = ModelSettings(
	temperature=0.5, #Balanced temperature for natural but focused responses
	max_tokens=1024
)
)

import asyncio

async def run_agent_example():
	result = await Runner.run(weather_assistant, "Can you tell me about the relationship between climate change and extreeme weather events?")

	print(result.final_output)


if __name__ == "__main__":
	asyncio.run(run_agent_example())


