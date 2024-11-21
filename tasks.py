from crewai import Task
from tools import SearchFilterTool, RecipeDatabaseTool, RecipeFormatterTool
from dotenv import load_dotenv
from agents import RecipeAgents
from textwrap import dedent
import os

# Load environment variables for API
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL_NAME", "ruslandev/llama-3-8b-gpt-4o")

if not api_key or not model_name:
    raise EnvironmentError("Missing OPENAI_API_KEY or OPENAI_MODEL_NAME in environment variables.")

llm = {"model": model_name, "api_key": api_key}

# Initialize tools
search_filter_tool = SearchFilterTool(name="Search Filter", description="Filter recipe searches based on criteria.")
recipe_database_tool = RecipeDatabaseTool(name="Recipe Database", description="Search in recipe database.")
recipe_formatter_tool = RecipeFormatterTool(name="Recipe Formatter", description="Format recipes into easy-to-follow instructions.")

# Initialize agents
recipe_agents = RecipeAgents()
recipe_researcher = recipe_agents.recipe_researcher()
recipe_creator = recipe_agents.recipe_creator()
recipe_formatter = recipe_agents.recipe_formatter()

class RecipeTasks:
    def search_recipes(self, agent, user_preferences, ingredient_filters, dish_type):
        return Task(
            description=dedent(f"""
            **Task**: Search for Recipes
            **Description**: Search for recipes in the database based on user preferences, such as dish type, 
                main ingredients, or dietary restrictions. Use the SearchFilterTool to look up recipes in the database 
                that match the criteria provided by the user.

            **Parameters**:
            - User Preferences: {user_preferences}
            - Ingredient Filters: {ingredient_filters}
            - Dish Type: {dish_type}

            **Note**: Ensure results are highly relevant by accurately applying filters.
            """),
            agent=agent,
            tool=SearchFilterTool,
            inputs={"user_preferences": user_preferences, "ingredient_filters": ingredient_filters, "dish_type": dish_type},
            outputs=["recipe_ids"],    
            expected_output="A list of recipe IDs matching the search criteria.",
            instructions="Use the SearchFilterTool to look up recipes and return a list of recipe IDs."
        )

    def fetch_recipe_details(self, agent, recipe_ids):
        return Task(
            description=dedent(f"""
            **Task**: Fetch Recipe Details
            **Description**: Retrieve detailed information about the identified recipes, including the name, ingredients, 
                preparation steps, cooking time, and nutritional information.

            **Parameters**:
            - Recipe IDs: {recipe_ids}

            **Note**: Ensure data integrity by cross-verifying recipe details.
            """),
            agent=agent,
            tool=RecipeDatabaseTool,
            inputs={"recipe_ids": recipe_ids},
            outputs=["recipe_details"],
            expected_output="Detailed information for the provided recipe IDs (name, ingredients, steps, cooking time).",

            instructions="Query the database using the RecipeDatabaseTool to get full details of the recipes."
        )

    def generate_custom_recipe(self, agent, user_preferences, ingredient_filters):
        return Task(
            description=dedent(f"""
            **Task**: Generate a Custom Recipe
            **Description**: Create a personalized recipe based on the user's preferences, including adjustments 
                for dietary restrictions or specific ingredients.

            **Parameters**:
            - User Preferences: {user_preferences}
            - Ingredient Filters: {ingredient_filters}

            **Note**: Leverage creativity to design a recipe that is both practical and appealing.
            """),
            agent=agent,
            tool=None,  # No specific tool, as this task uses the LLM directly
            inputs={"user_preferences": user_preferences, "ingredient_filters": ingredient_filters},
            outputs=["custom_recipe"],
            expected_output="A fully generated recipe with ingredients, steps, and additional notes.",

            instructions="Use the LLM to generate a complete recipe with detailed instructions."
        )

    def format_recipe(self, agent, recipe_details, custom_recipe):
        return Task(
            description=dedent(f"""
            **Task**: Format the Recipe
            **Description**: Format the final recipe to make it ready for display or sharing with the user. 
                This includes structuring it clearly, with well-defined sections and suggestions.

            **Parameters**:
            - Recipe Details: {recipe_details}
            - Custom Recipe: {custom_recipe}

            **Note**: Ensure the formatted recipe is user-friendly and visually appealing.
            """),
            agent=agent,
            tool=RecipeFormatterTool,
            inputs={"recipe_details": recipe_details, "custom_recipe": custom_recipe},
            outputs=["formatted_recipe"],
            expected_output="A user-friendly and visually appealing formatted recipe.",

            instructions=(
                "Use the RecipeFormatterTool to structure the recipe. Ensure the result includes sections like: "
                "Name, Ingredients, Step-by-step Instructions, Cooking Time, Servings, and Additional Notes."
            )
        )

    def main_task(self, agent, user_preferences, ingredient_filters, dish_type):
       
        return Task(
            description=dedent(f"""
            **Task**: Generate and Format a Complete Recipe
            **Description**: Orchestrate all the necessary steps to search, generate, and format a complete recipe. 
                This includes searching, fetching details, generating, and formatting.

            **Parameters**:
            - User Preferences: {user_preferences}
            - Ingredient Filters: {ingredient_filters}
            - Dish Type: {dish_type}

            **Note**: Ensure seamless execution of subtasks and deliver a high-quality final recipe.
            """),
            agent=agent,
            subtasks=[
                self.search_recipes(agent, user_preferences, ingredient_filters, dish_type),
                self.fetch_recipe_details(agent, "recipe_ids"),
                self.generate_custom_recipe(agent, user_preferences, ingredient_filters),
                self.format_recipe(agent, "recipe_details", "custom_recipe")
            ],
            inputs={"user_preferences": user_preferences, "ingredient_filters": ingredient_filters, "dish_type": dish_type},
            outputs=["formatted_recipe"],
            expected_output="A fully generated and formatted recipe, ready for display.",

            instructions=(
                "Coordinate the steps by calling each subtask. Ensure the inputs and outputs of each step "
                "are properly connected to produce a complete and accurate final result."
            )
        )

# Example usage
recipe_tasks = RecipeTasks()

main_task = recipe_tasks.main_task(
    agent=recipe_researcher,
    user_preferences={"diet": "vegan", "cuisine": "Italian"},
    ingredient_filters=["tomatoes", "basil"],
    dish_type="main_course"
)
