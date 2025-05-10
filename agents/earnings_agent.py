from langchain.agents import Tool, initialize_agent, AgentType
from tools.data_retrieval import get_financial_statements

# Define a tool for fetching financial statements/earnings info
financials_tool = Tool(
    name="FinancialsFetcher",
    func=lambda ticker: get_financial_statements(ticker),
    description="Fetches financial highlights (revenue, net income) for a given company ticker."
)

def create_agent(llm):
    """
    Create an agent for retrieving financial statement data or earnings info for a given stock.
    """
    agent = initialize_agent(
        tools=[financials_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )
    return agent
