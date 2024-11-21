from crewai_tools import BaseTool
import openai
import os
from dotenv import load_dotenv
from langchain.tools import tool

# Load environment variables for API
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = "ruslandev/llama-3-8b-gpt-4o"


class CalculatorTools:
    """
    A utility for performing mathematical operations.
    """

    @tool("Make a calculation")
    def calculate(operation: str) -> str:
        """
        Perform a mathematical calculation.

        Args:
            operation (str): A mathematical expression (e.g., '200*7', '5000/2*10').

        Returns:
            str: The result of the calculation or an error message.
        """
        try:
            return str(eval(operation))
        except SyntaxError:
            return "Error: Invalid syntax in mathematical expression."
        except Exception as e:
            return f"Error: {str(e)}"


class SearchFilterTool(BaseTool):
    """
    Tool for performing filtered searches using GPT models.
    """

    name: str = "Search Filter Tool"
    description: str = (
        "A tool that searches for content based on user queries, filters, and date ranges. "
        "It leverages GPT models to simulate search results."
    )

    def _run(self, inputs: dict) -> dict:
        query = inputs.get("search_query", "")
        filters = ", ".join(inputs.get("filters", []))
        date_range = inputs.get("date_range", "last_30_days")

        prompt = (
            f"Search for the following query: '{query}', with filters: '{filters}', "
            f"within the date range: '{date_range}'."
        )

        try:
            response = openai.Completion.create(
                model=os.getenv("OPENAI_MODEL_NAME"),
                prompt=prompt,
                max_tokens=200,
                temperature=0.7,
            )

            if response and "choices" in response:
                search_results = response["choices"][0]["text"].strip().split("\n")
                return {"search_results": search_results}
            else:
                return {"search_results": []}
        except Exception as e:
            return {"error": str(e)}


class RecipeDatabaseTool(BaseTool):
    """
    Tool for fetching detailed recipe information based on result IDs.
    """

    name: str = "Recipe Database Tool"
    description: str = "Fetches detailed information about specific result IDs using GPT."

    def _run(self, inputs: dict) -> dict:
        result_ids = inputs.get("result_ids", [])

        if not result_ids:
            return {"result_details": []}

        result_details = []
        for result_id in result_ids:
            prompt = f"Provide detailed information about the result ID: {result_id}."
            try:
                response = openai.Completion.create(
                    model=os.getenv("OPENAI_MODEL_NAME"),
                    prompt=prompt,
                    max_tokens=150,
                    temperature=0.7,
                )

                if response and "choices" in response:
                    result_details.append(response["choices"][0]["text"].strip())
            except Exception as e:
                result_details.append(
                    f"Error fetching data for result ID {result_id}: {str(e)}"
                )

        return {"result_details": result_details}


class RecipeFormatterTool(BaseTool):
    """
    Tool for formatting recipes into a clean, readable structure.
    """

    name: str = "Recipe Formatter Tool"
    description: str = (
        "Formats search results into a clean, readable structure using GPT."
    )

    def _run(self, inputs: dict) -> dict:
        result_details = inputs.get("result_details", [])

        if not result_details:
            return {"formatted_results": []}

        formatted_results = []
        for result in result_details:
            prompt = (
                f"Format the following recipe into a clear and structured format: {result}"
            )
            try:
                response = openai.Completion.create(
                    model=os.getenv("OPENAI_MODEL_NAME"),
                    prompt=prompt,
                    max_tokens=200,
                    temperature=0.7,
                )

                if response and "choices" in response:
                    formatted_results.append(response["choices"][0]["text"].strip())
            except Exception as e:
                formatted_results.append(f"Error formatting result: {str(e)}")

        return {"formatted_results": formatted_results}
