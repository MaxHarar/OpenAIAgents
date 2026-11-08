# OpenAI Agents

Practice scripts I'm writing while working through DataCamp's OpenAI Agents SDK tutorial.

## Scripts

| File | What it does |
|------|--------------|
| `async.py` | Runs a minimal agent with `Runner.run` to check the setup works. |
| `working.py` | Notes on defining basic and advanced agents. It doesn't run yet because it has syntax errors. |
| `weatherassistant.py` | Runs a weather explainer agent with custom instructions and model settings. It has no live data. |
| `liveWeatherAssistant.py` | Asks which location you want, then looks up its current weather through OpenWeatherMap function tools. |

## Setup

Requires Python 3.13 (the version these scripts were written on).

```bash
python3 -m venv agents-env
source agents-env/bin/activate
pip install openai-agents python-dotenv requests
```

Create a `.env` file in the project root. Git ignores it, so your keys stay out of the repo.

```
OPENAI_API_KEY=your-openai-key
OPENWEATHERMAP_API_KEY=your-openweathermap-key
```

`OPENWEATHERMAP_API_KEY` is only needed for `liveWeatherAssistant.py`.

## Usage

```bash
python liveWeatherAssistant.py
```

The script prints the assistant's capabilities, then asks for a location. After each answer you can reply, for example to say which Paris you meant, or press Enter to quit.

## Credit

These scripts follow this tutorial:

Bexruz (Bex) Tuychiev, [OpenAI Agents SDK Tutorial: Building AI Systems That Take Action](https://www.datacamp.com/tutorial/openai-agents-sdk-tutorial), DataCamp, March 2025.
