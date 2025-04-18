# import gradio as gr
import random
from smolagents import GradioUI, CodeAgent, HfApiModel, LiteLLMModel

# Import our custom tools from their modules
from tools import DuckDuckGoSearchTool, WeatherInfoTool, HubStatsTool, LocationInfoTool
from retriever import load_guest_dataset
import litellm
litellm._turn_on_debug()

# Initialize the Hugging Face model
# model = HfApiModel()
model = LiteLLMModel(
    model_id="ollama_chat/qwen2.5:7b-instruct", #Can try diffrent model here I am using qwen2.5 7B model
    api_base="http://127.0.0.1:11434",
    num_ctx=8192
)


# Initialize the web search tool
search_tool = DuckDuckGoSearchTool()

# Initialize the location tool
location_info_tool = LocationInfoTool()

# Initialize the weather tool
weather_info_tool = WeatherInfoTool()

# Initialize the Hub stats tool
hub_stats_tool = HubStatsTool()

# Load the guest dataset and initialize the guest info tool
guest_info_tool = load_guest_dataset()

# Create Alfred with all the tools
alfred = CodeAgent(
    # tools=[guest_info_tool, weather_info_tool, hub_stats_tool, search_tool],
    tools=[guest_info_tool, search_tool, location_info_tool, weather_info_tool, hub_stats_tool],
    model=model,
    add_base_tools=True,  # Add any additional base tools
    planning_interval=3   # Enable planning every 3 steps
)

if __name__ == "__main__":
    GradioUI(alfred).launch()