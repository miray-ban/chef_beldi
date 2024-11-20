from crewai import Agent
from tools import SearchFilterTool, RecipeDatabaseTool, RecipeFormatterTool
from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv()

# Configuration de l'environnement
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = "gpt-4-0125-preview"

# Initialisation du LLM
llm = {
    "model": os.environ["OPENAI_MODEL_NAME"],
    "api_key": os.environ["OPENAI_API_KEY"]
}

# Initialisation des outils
search_filter_tool = SearchFilterTool(name="Search Filter", description="Filter recipe searches based on criteria.") 
recipe_database_tool = RecipeDatabaseTool(name="Recipe Database", description="Search in recipe database.") 
recipe_formatter_tool = RecipeFormatterTool(name="Recipe Formatter", description="Format recipes into easy-to-follow instructions.")

# Création des agents
recipe_researcher = Agent(
    role='Recipe Researcher',
    goal='Find recipes based on ingredient {ingredient_filters} and type {dish_type}. Filter by preparation time or diet if provided.',
    verbose=True,
    memory=True,
    backstory="An expert in culinary research, specializing in finding recipes from a database and applying user-specific filters like diet and preparation time.",
    tools=[recipe_database_tool, search_filter_tool],  # Pass the instantiated tools
    llm=llm,
    allow_delegation=True
)

recipe_creator = Agent(
    role='Recipe Creator',
    goal='Create a custom recipe based on user preferences: {user_preferences}.',
    verbose=True,
    memory=True,
    backstory="A creative chef who designs unique recipes tailored to user preferences, ensuring the recipe is practical and easy to follow.",
    tools=[recipe_formatter_tool],  # Only use the formatter tool here
    llm=llm,
    allow_delegation=False
)

recipe_formatter = Agent(
    role='Recipe Formatter',
    goal='Format the recipe {recipe} into easy-to-follow instructions, including cooking time, servings, and ingredients.',
    verbose=True,
    memory=False,
    backstory="An assistant specialized in formatting recipes into a polished format that is easy to read and follow.",
    tools=[recipe_formatter_tool],  # Only use the formatter tool here
    llm=llm,
    allow_delegation=False
)
