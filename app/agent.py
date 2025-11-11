"""
Module: agent.py
----------------
Implements a Pandas DataFrame Agent powered by Mistral (via Ollama) that can reason dynamically
over e-commerce operational data (products + orders).

Objective:
Provide the LLM with a structured dataset that merges product information and
delivery performance, allowing it to generate custom analytics queries (group by, mean, etc.)
on demand, without predefining every possible aggregation.
"""

import warnings
from typing import Any

import logging
import os
import pandas as pd

from app import constants


logger = logging.getLogger(__name__)


# Configure basic logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/agent_errors.log",
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
warnings.filterwarnings("ignore", category=UserWarning)


def load_data() -> pd.DataFrame:
    """
    Load and merge product and order datasets into a single analytical DataFrame.

    The function combines product and order data to expose a structured,
    analysis-ready view for the LLM. It computes lightweight derived features
    such as `late_rate` (percentage of late deliveries) to provide interpretable
    metrics for reasoning and analytics.

    Returns:
        pd.DataFrame: Merged DataFrame containing product information and
        delivery performance metrics, including `late_rate`.
    """
    df_products = pd.read_csv(constants.PRODUCTS_PATH)
    df_orders = pd.read_csv(constants.ORDERS_PATH)

    late_rate = (
        df_orders.groupby("product_id")["delivered_late"]
        .mean()
        .reset_index(name="late_rate")
    )

    df = pd.merge(df_products, late_rate, on="product_id", how="left")
    return df


def get_pandas_agent() -> Any:
    """
    Create and configure a Pandas DataFrame agent powered by Mistral (via Ollama).

    The agent allows natural-language analytical queries over tabular data.
    It loads the merged dataset from `load_data()`, initializes an Ollama LLM,
    and wraps it with a Pandas agent for structured reasoning on the data.

    Returns:
        AgentExecutor: Configured LangChain agent capable of executing
        natural-language analytical queries on the e-commerce dataset.
    """
    from langchain.agents import AgentExecutor
    from langchain_community.llms import Ollama
    from langchain_experimental.agents import create_pandas_dataframe_agent

    llm = Ollama(
        model=constants.LLM_MODEL_AGENT, temperature=constants.TEMPERATURE_AGENT
    )

    df = load_data()
    base_agent = create_pandas_dataframe_agent(llm, df, verbose=True)

    agent = AgentExecutor.from_agent_and_tools(
        base_agent.agent,
        base_agent.tools,
        handle_parsing_errors=lambda e: f"Handled parsing error: {str(e)}",
        verbose=True,
    )

    return agent


def ask_agent(question: str) -> str:
    """
    Query the Pandas Agent with a natural language question.

    Args:
        question (str): Example - "Which categories have the highest return rate?"

    Returns:
        str: The LLM-generated answer.
    """
    agent = get_pandas_agent()
    response = agent.run(question)
    logger.info(f"\nQuestion: {question}")
    logger.info(f"Answer: {response}")
    return response


if __name__ == "__main__":
    ask_agent("Which product categories have the highest average return rate?")
