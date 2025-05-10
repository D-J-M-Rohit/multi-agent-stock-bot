from langchain.agents import Tool, initialize_agent, AgentType
from tools.data_retrieval import get_recent_news

# Define a tool for fetching recent news
news_tool = Tool(
    name="NewsFetcher",
    func=lambda ticker: get_recent_news(ticker),
    description="Fetches recent news headlines for a given stock ticker symbol."
)

def create_agent(llm):
    """
    Create an agent for retrieving financial news related to a specific stock.
    """
    agent = initialize_agent(
        tools=[news_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )
    return agent
