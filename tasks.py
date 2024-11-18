from crewai import Task
from tools import SearchFilterTool, RecipeDatabaseTool, RecipeFormatterTool
from dotenv import load_dotenv
import os

# Load environment variables for API
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = "gpt-4-0125-preview"
llm = {
    "model": os.environ["OPENAI_MODEL_NAME"],
    "api_key": os.environ["OPENAI_API_KEY"]
}
# Define tasks

## 1. Task: Search for recipes based on user preferences
search_recipes_task = Task(
    name="SearchRecipes",
    description=(
        "Search for recipes in the database based on user preferences, such as dish type, "
        "main ingredients, or dietary restrictions."
    ),
    tool=SearchFilterTool,
    inputs=["user_preferences", "ingredient_filters", "dish_type"],
    outputs=["recipe_ids"],
    instructions=(
        "Use the SearchFilterTool to look up recipes in the database "
        "that match the criteria provided by the user."
    ),
    llm=llm,
    allow_subtasks=False,
    expected_output="recipe_ids"  # Explicitly define the expected output type here
)

## 2. Task: Fetch detailed information about the selected recipes
fetch_recipe_details_task = Task(
    name="FetchRecipeDetails",
    description=(
        "Retrieve detailed information about the identified recipes, including the name, ingredients, "
        "preparation steps, cooking time, and nutritional information."
    ),
    tool=RecipeDatabaseTool,
    inputs=["recipe_ids"],
    outputs=["recipe_details"],
    instructions=(
        "Query the database using the RecipeDatabaseTool to get full details "
        "of the recipes corresponding to the provided IDs."
    ),
    llm=llm,
    allow_subtasks=False,
    expected_output="recipe_details"  # Explicitly define the expected output type here
)

## 3. Task: Generate a customized recipe
generate_custom_recipe_task = Task(
    name="GenerateCustomRecipe",
    description=(
        "Create a personalized recipe based on the user's preferences, "
        "including adjustments for dietary restrictions or specific ingredients."
    ),
    tool=None,  # No specific tool, as this task uses the LLM directly
    inputs=["user_preferences", "ingredient_filters"],
    outputs=["custom_recipe"],
    instructions=(
        "Use the LLM to generate a complete recipe with detailed instructions. "
        "Take into account dietary preferences, restrictions, and provided ingredients."
    ),
    llm=llm,
    allow_subtasks=False,
    expected_output="custom_recipe" # Explicitly define the expected output type here
)

## 4. Task: Format the recipe for display or sharing
format_recipe_task = Task(
    name="FormatRecipe",
    description=(
        "Format the final recipe to make it ready for display or sharing with the user. "
        "This includes structuring it clearly, with well-defined sections and suggestions."
    ),
    tool=RecipeFormatterTool,
    inputs=["recipe_details", "custom_recipe"],
    outputs=["formatted_recipe"],
    instructions=(
        "Use the RecipeFormatterTool to structure the recipe. Ensure the result includes sections like: "
        "Name, Ingredients, Step-by-step Instructions, Cooking Time, Servings, and Additional Notes."
    ),
    llm=llm,
    allow_subtasks=False,
    expected_output="formatted_recipe"  # Explicitly define the expected output type here
)

## 5. Main Task: Orchestrate steps to produce the final recipe
main_task = Task(
    name="GenerateAndFormatRecipe",
    description=(
        "Orchestrate all the necessary steps to search, generate, and format a complete recipe. "
        "This includes searching, fetching details, generating, and formatting."
    ),
    subtasks=[search_recipes_task, fetch_recipe_details_task, generate_custom_recipe_task, format_recipe_task],
    inputs=["user_preferences", "ingredient_filters", "dish_type"],
    outputs=["formatted_recipe"],
    instructions=(
        "Coordinate the steps by calling each subtask. Ensure the inputs and outputs of each step "
        "are properly connected to produce a complete and accurate final result."
    ),
    llm=llm,
    allow_subtasks=True,
    expected_output="formatted_recipe"  # Explicitly define the expected output type here
)
