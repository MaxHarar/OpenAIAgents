import os
from dotenv import load_dotenv
from agents import Agent, Runner

#Load Env variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
        raise ValueError("key not found")

import asyncio 
from agents import Agent, Runner

async def main():
	agent = Agent(
		name = "Test Agent",
		instructions= "YOu are a helpful assistant that provides concise responses."
)
	result = await Runner.run(agent, "Hello! Are you working correctly?")
	print(result.final_output)





if __name__ == "__main__":
	asyncio.run(main())


