from crewai import Crew, Process
from tasks import search_recipes_task, fetch_recipe_details_task, generate_custom_recipe_task, format_recipe_task, main_task
from agents import recipe_researcher, recipe_creator, recipe_formatter
from dotenv import load_dotenv
import os

# Load environment variables for API
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = "gpt-4-0125-preview"

# Forming the recipe generation crew with agents, tasks, and tools
crew = Crew(
    agents=[recipe_researcher, recipe_creator, recipe_formatter],  # List agents handling tasks like research, creation, and formatting
    tasks=[search_recipes_task, fetch_recipe_details_task, generate_custom_recipe_task, format_recipe_task],  # Define the tasks
    process=Process.sequential,  # Sequential task execution (tasks are executed one after the other)
    memory=True,  # Enable memory for retaining previous task outputs
    cache=True,  # Enable caching for repeated tasks
    max_rpm=100,  # Set the maximum requests per minute (adjust as necessary)
    share_crew=True  # Share the crew between users, or set to False for isolated use
)

# Start the crew's task execution process
def generate_recipe(user_preferences, ingredient_filters, dish_type):
    # Define inputs for the tasks
    inputs = {
        "user_preferences": user_preferences,
        "ingredient_filters": ingredient_filters,
        "dish_type": dish_type
    }

    # Run the main task to generate the recipe
    result = crew.kickoff(inputs=inputs)
    
    # Return the formatted recipe
    return result['formatted_recipe']

# Example usage
if __name__ == "__main__":
    user_preferences = {
        "dietary_restrictions": "vegetarian",
        "preferred_cuisine": "Italian",
        "avoid_ingredients": ["gluten"],
        "servings": 4
    }
    ingredient_filters = ["tomato", "basil", "cheese"]
    dish_type = "main course"
    
    formatted_recipe = generate_recipe(user_preferences, ingredient_filters, dish_type)
    print("Formatted Recipe:\n", formatted_recipe)
