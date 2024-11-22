from crewai import Task
from crewai import LLM
from tools import SearchFilterTool, RecipeDatabaseTool, RecipeFormatterTool
from dotenv import load_dotenv
from agents import RecipeAgents
from textwrap import dedent
import os

# Load environment variables for API
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROG_MODEL_NAME")

if not api_key or not model_name:
    raise EnvironmentError("Missing OPENAI_API_KEY or OPENAI_MODEL_NAME in environment variables.")

llm = LLM(
    model=model_name,
    api_key=groq_api_key
)

# Initialize agents
recipe_agents = RecipeAgents()
recipe_researcher = recipe_agents.recipe_researcher()
recipe_formatter = recipe_agents.recipe_formatter()

class RecipeTasks:
    def search_recipes(self, agent, dietary_restrictions, preferred_cuisine, avoid_ingredients, ingredient_filters):
        return Task(
            description=dedent("""
                Find and describe a recipe that matches the following criteria:
                1. Dietary Restrictions: {dietary_restrictions}.
                2. Preferred Cuisine: {preferred_cuisine}.
                3. Avoid Ingredients: {avoid_ingredients}.
                4. Must Include Ingredients: {ingredient_filters}.
                Provide a detailed description of the recipe, including the title, list of ingredients, and step-by-step instructions.
                Your response must follow this exact format
            """),
            agent=agent,
            inputs={"dietary_restrictions": dietary_restrictions, "preferred_cuisine": preferred_cuisine, "avoid_ingredients": avoid_ingredients, "ingredient_filters": ingredient_filters},
            outputs=["recipe"],
            expected_output="A list of recipe IDs matching the search criteria.",
            instructions="Use the SearchFilterTool to look up recipes and return a list of recipe IDs."
        )


    def format_recipe(self, agent, recipe_details):
        return Task(
            description=dedent(
                """
                Format the recipe into easy-to-follow instructions, including cooking time, servings, and ingredients.
                Ensure the recipe is structured with clear headings for the title, ingredients, and instructions.
                """
            ),
            expected_output="A polished, user-friendly format of the recipe, including title, ingredients, and steps.",
            agent=agent,
            inputs={"recipe": recipe_details},
        )

    def main_task(self, agent_recipe_researcher, agent_recipe_formatter, dietary_restrictions, preferred_cuisine, avoid_ingredients, ingredient_filters):

        return Task(
            description=dedent(f"""
            **Task**: Generate and Format a Complete Recipe
            **Description**: Orchestrate all the necessary steps to search, generate, and format a complete recipe.
                This includes searching, fetching details, generating, and formatting.

            **Parameters**:
            - Dietary restrictions: {dietary_restrictions}
            - Ingredient Filters: {ingredient_filters}
            - Preferred cuisine: {preferred_cuisine}
            - Avoid ingredients: {avoid_ingredients}

            **Note**: Ensure seamless execution of subtasks and deliver a high-quality final recipe.
            """),
            agent=agent_recipe_formatter,
            subtasks=[
                self.search_recipes(agent_recipe_researcher, dietary_restrictions, preferred_cuisine, avoid_ingredients, ingredient_filters),
                self.format_recipe(agent_recipe_formatter, "search_recipes.recipe_ids")
            ],
            expected_output="A fully generated and formatted recipe, ready for display.",

        )

# Example usage
recipe_tasks = RecipeTasks()

main_task = recipe_tasks.main_task(
    agent_recipe_researcher=recipe_researcher,
    agent_recipe_formatter=recipe_formatter,
    dietary_restrictions="vegetarian",
    ingredient_filters=["tomatoes", "basil"],
    avoid_ingredients= ["gluten"],
    preferred_cuisine="Italian"
)

# Define the Crew
recipe_crew = Crew(
    agents=[recipe_researcher, recipe_formatter],
    tasks=[main_task],  # Pass the Task object, not the method
    process=Process.sequential,
)

inputs = {
    "dietary_restrictions": "vegetarian",
    "preferred_cuisine": "Italian",
    "avoid_ingredients": ["gluten"],
    "ingredient_filters": ["tomato", "basil", "cheese"],
}

# Execute the crew
result = recipe_crew.kickoff()
