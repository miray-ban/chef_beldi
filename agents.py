from crewai import Agent
from textwrap import dedent
from dotenv import load_dotenv
import os

"""
Creating Agents Cheat Sheet:
- Think like a boss. Work backwards from the goal and think which employee 
  you need to hire to get the job done.
- Define the Captain of the crew who orients the other agents towards the goal. 
- Define which experts the captain needs to communicate with and delegate tasks to.
  Build a top-down structure of the crew.

Goal:
- Build an intelligent recipe assistant capable of researching, creating, and formatting recipes.

Captain/Manager/Boss:
- Recipe Researcher

Employees/Experts to hire:
- Recipe Creator
- Recipe Formatter

Notes:
- Agents should be results-driven and have a clear goal in mind.
- Role is their job title.
- Goals should be actionable.
- Backstory should be their resume.
"""


class RecipeAgents:
    def __init__(self):
        # Load environment variables
        load_dotenv()
      
        groq_api_key = os.getenv("GROQ_API_KEY")
        model_name = os.getenv("GROG_MODEL_NAME")

        if not api_key or not model_name:
            raise EnvironmentError("Missing OPENAI_API_KEY or OPENAI_MODEL_NAME in environment variables.")

        self.llm = LLM(
            model=model_name,
            api_key=groq_api_key
        )

    def recipe_researcher(self):
        """
        Creates an agent for researching recipes based on user-specific filters.

        Returns:
            Agent: Configured recipe researcher agent.
        """
        return Agent(
          role="Recipe Expert",
          goal="Find and provide detailed descriptions of recipes tailored to specific criteria.",
          backstory="ou are an expert chef specializing in finding recipes that fit dietary needs.",
          llm=llm,
          allow_delegation=True, 
          memory=True,
      )


    def recipe_formatter(self):
        """
        Creates an agent for formatting recipes into polished, user-friendly formats.

        Returns:
            Agent: Configured recipe formatter agent.
        """
        return Agent(
            role="Recipe Formatter",
            backstory=dedent(
                """You are an assistant specialized in formatting recipes into a polished format
                that is easy to read and follow."""
            ),
            goal=dedent(
                """Format the recipe into easy-to-follow instructions, including
                cooking time, servings, and ingredients."""
            ),
            verbose=False,
            memory=True,
            llm=llm,
            allow_delegation=False,
        )
