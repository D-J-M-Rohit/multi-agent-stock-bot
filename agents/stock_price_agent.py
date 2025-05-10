from langchain.agents import Tool, initialize_agent, AgentType
from tools.data_retrieval import get_stock_price

# Define a tool for fetching stock price
stock_price_tool = Tool(
    name="StockPriceFetcher",
    func=lambda ticker: get_stock_price(ticker),
    description="Fetches the current stock price (and daily change) for a given stock ticker symbol."
)

def create_agent(llm):
    """
    Create a LangChain agent specialized in retrieving stock prices.
    """
    agent = initialize_agent(
        tools=[stock_price_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )
    return agent
