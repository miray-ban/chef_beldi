from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv

# Load environment variables for API
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = "gpt-4-0125-preview"

# Ensure that Tool is based on pydantic.BaseModel for validation (if needed)
class Tool(BaseModel):
    name: str = "Default Tool Name"  # Default value for name
    description: str = "Default Tool Description"  # Default value for description

    class Config:
        arbitrary_types_allowed = True

class SearchFilterTool(Tool):
    """
    Tool for searching content using GPT.
    Uses GPT-3 or GPT-4 to generate search results based on the user's query.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, inputs):
        """
        Executes the search using GPT based on the search query.
        """
        query = inputs.get("search_query", "")
        filters = ",".join(inputs.get("filters", []))
        date_range = inputs.get("date_range", "last_30_days")
        
        # Generate a prompt for GPT to simulate a search response
        prompt = f"Search for the following query: {query}, with filters: {filters} within the date range of {date_range}."

        # Query OpenAI GPT model to simulate search results
        response = openai.Completion.create(
            model="gpt-4",  # You can use 'gpt-3.5-turbo' or 'gpt-4'
            prompt=prompt,
            max_tokens=200,
            temperature=0.7
        )

        if response and "choices" in response:
            # Simulate search results from GPT's response
            search_results = response["choices"][0]["text"].strip().split("\n")
            return {"search_results": search_results}
        else:
            return {"search_results": []}

class RecipeDatabaseTool(Tool):
    """
    Tool for fetching detailed information from GPT about a specific result.
    This simulates fetching detailed metadata or descriptions for a result.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, inputs):
        """
        Fetches detailed data for a given result.
        """
        result_ids = inputs.get("result_ids", [])
        
        if not result_ids:
            return {"result_details": []}
        
        result_details = []
        for result_id in result_ids:
            # Use GPT to generate detailed data for each result ID
            prompt = f"Provide detailed information about the result ID: {result_id}."
            response = openai.Completion.create(
                model="gpt-4",
                prompt=prompt,
                max_tokens=150,
                temperature=0.7
            )
            
            if response and "choices" in response:
                result_details.append(response["choices"][0]["text"].strip())
        
        return {"result_details": result_details}

class RecipeFormatterTool(Tool):
    """
    Tool for formatting the search results into a clean, readable format.
    Uses GPT to organize the search results into a user-friendly structure.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, inputs):
        """
        Formats the results into a clean, readable structure using GPT.
        """
        result_details = inputs.get("result_details", [])
        
        formatted_results = []
        for result in result_details:
            # Use GPT to format the result details
            prompt = f"Format the following search result in a user-friendly structure:\n{result}"
            response = openai.Completion.create(
                model="gpt-4",
                prompt=prompt,
                max_tokens=200,
                temperature=0.7
            )

            if response and "choices" in response:
                formatted_results.append(response["choices"][0]["text"].strip())
        
        return {"formatted_results": "\n".join(formatted_results)}

