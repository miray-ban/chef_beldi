from crewai import Agent
from textwrap import dedent
from dotenv import load_dotenv
import os

from tools import SearchFilterTool, RecipeDatabaseTool, RecipeFormatterTool

"""
Creating Agents Cheat Sheet:
- Think like a boss. Work backwards from the goal and think which employee 
    you need to hire to get the job done.
- Define the Captain of the crew who orient the other agents towards the goal. 
- Define which experts the captain needs to communicate with and delegate tasks to.
    Build a top down structure of the crew.

Goal:
- Build an intelligent recipe assistant capable of researching, creating, and formatting recipes.

Captain/Manager/Boss:
- Recipe Researcher

Employees/Experts to hire:
- Recipe Creator
- Recipe Formatter

Notes:
- Agents should be results-driven and have a clear goal in mind
- Role is their job title
- Goals should be actionable
- Backstory should be their resume
"""


class RecipeAgents:
    def __init__(self):
        # Charger les variables d'environnement
        load_dotenv()

        self.llm = {
            "model": os.getenv("OPENAI_MODEL_NAME", "gpt-4-0125-preview"),
            "api_key": os.getenv("OPENAI_API_KEY")
        }

        # Initialisation des outils
        self.search_filter_tool = SearchFilterTool(
            name="Search Filter",
            description="Filter recipe searches based on criteria."
        )
        self.recipe_database_tool = RecipeDatabaseTool(
            name="Recipe Database",
            description="Search in recipe database."
        )
        self.recipe_formatter_tool = RecipeFormatterTool(
            name="Recipe Formatter",
            description="Format recipes into easy-to-follow instructions."
        )

    def recipe_researcher(self):
        return Agent(
            role="Recipe Researcher",
            backstory=dedent(
                """An expert in culinary research, specializing in finding recipes 
                from a database and applying user-specific filters like diet and preparation time."""
            ),
            goal=dedent(
                """Find recipes based on ingredient {ingredient_filters} and type {dish_type}. 
                Filter by preparation time or diet if provided."""
            ),
            tools=[self.recipe_database_tool, self.search_filter_tool],
            verbose=True,
            memory=True,
            llm=self.llm,
            allow_delegation=True,
        )

    def recipe_creator(self):
        return Agent(
            role="Recipe Creator",
            backstory=dedent(
                """A creative chef who designs unique recipes tailored to user preferences, 
                ensuring the recipe is practical and easy to follow."""
            ),
            goal=dedent(
                """Create a custom recipe based on user preferences: {user_preferences}."""
            ),
            tools=[self.recipe_formatter_tool],
            verbose=True,
            memory=True,
            llm=self.llm,
            allow_delegation=False,
        )

    def recipe_formatter(self):
        return Agent(
            role="Recipe Formatter",
            backstory=dedent(
                """An assistant specialized in formatting recipes into a polished format 
                that is easy to read and follow."""
            ),
            goal=dedent(
                """Format the recipe {recipe} into easy-to-follow instructions, including 
                cooking time, servings, and ingredients."""
            ),
            tools=[self.recipe_formatter_tool],
            verbose=True,
            memory=False,
            llm=self.llm,
            allow_delegation=False,
        )
