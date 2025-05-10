from langchain.agents import Tool, initialize_agent, AgentType
from tools.data_retrieval import get_market_summary

# Define a tool for fetching market summary
market_tool = Tool(
    name="MarketSummaryFetcher",
    func=lambda _: get_market_summary(),
    description="Provides a summary of current market index performance."
)

def create_agent(llm):
    """
    Create an agent for providing general market summaries.
    """
    agent = initialize_agent(
        tools=[market_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )
    return agent
