import logging
from crewai import Crew, Process
from tasks import RecipeTasks
from agents import RecipeAgents
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL_NAME", "ruslandev/llama-3-8b-gpt-4o")

if not api_key or not model_name:
    raise EnvironmentError("Missing OPENAI_API_KEY or OPENAI_MODEL_NAME in environment variables.")


class RecipeCrew:
    def __init__(self, user_preferences, ingredient_filters, dish_type):
        # Validate input keys
        required_keys = ["dietary_restrictions", "preferred_cuisine", "avoid_ingredients", "servings"]
        for key in required_keys:
            if key not in user_preferences:
                raise KeyError(f"Missing required key in user_preferences: '{key}'")

        self.user_preferences = user_preferences
        self.ingredient_filters = ingredient_filters
        self.dish_type = dish_type

        # Initialize agents
        recipe_agents = RecipeAgents()
        self.recipe_researcher = recipe_agents.recipe_researcher()
        self.recipe_creator = recipe_agents.recipe_creator()
        self.recipe_formatter = recipe_agents.recipe_formatter()

        # Initialize tasks
        recipe_tasks = RecipeTasks()
        self.tasks = [
            recipe_tasks.search_recipes(
                agent=self.recipe_researcher,
                user_preferences=self.user_preferences,
                ingredient_filters=self.ingredient_filters,
                dish_type=self.dish_type,
            ),
            recipe_tasks.fetch_recipe_details(
                agent=self.recipe_researcher,
                recipe_ids=[],  # Dynamically updated later
            ),
            recipe_tasks.generate_custom_recipe(
                agent=self.recipe_creator,
                user_preferences=self.user_preferences,
                ingredient_filters=self.ingredient_filters,
            ),
            recipe_tasks.format_recipe(
                agent=self.recipe_formatter,
                recipe_details=[],  # Dynamically updated later
                custom_recipe=None,  # Dynamically updated later
            ),
        ]

        # Setup the crew configuration
        self.crew = Crew(
            agents=[self.recipe_researcher, self.recipe_creator, self.recipe_formatter],
            tasks=self.tasks,
            process=Process.sequential,
            memory=True,
            cache=True,
            max_rpm=100,
            verbose=True,
            share_crew=True,
        )

    def run(self):
        logging.info("Starting the recipe generation process...")

        try:
            # Step 1: Search for recipes
            search_inputs = {
                "user_preferences": self.user_preferences,
                "ingredient_filters": self.ingredient_filters,
                "dish_type": self.dish_type,
            }
            search_result = self.crew.kickoff(inputs=search_inputs)
            logging.info("Search Result: %s", search_result)

            if not search_result or "recipe_ids" not in search_result:
                raise KeyError("Search task failed. 'recipe_ids' key is missing.")

            recipe_ids = search_result["recipe_ids"]
            logging.info("Recipe IDs Retrieved: %s", recipe_ids)

            # Step 2: Fetch recipe details
            fetch_inputs = {"recipe_ids": recipe_ids}
            fetch_result = self.crew.kickoff(inputs=fetch_inputs)
            logging.info("Fetch Result: %s", fetch_result)

            if not fetch_result or "recipe_details" not in fetch_result:
                raise KeyError("Fetch task failed. 'recipe_details' key is missing.")

            recipe_details = fetch_result["recipe_details"]
            logging.info("Fetched Recipe Details: %s", recipe_details)

            # Step 3: Generate a custom recipe
            generate_inputs = {
                "user_preferences": self.user_preferences,
                "ingredient_filters": self.ingredient_filters,
            }
            generate_result = self.crew.kickoff(inputs=generate_inputs)
            logging.info("Generate Result: %s", generate_result)

            if not generate_result or "custom_recipe" not in generate_result:
                raise KeyError("Generate task failed. 'custom_recipe' key is missing.")

            custom_recipe = generate_result["custom_recipe"]
            logging.info("Custom Recipe Generated: %s", custom_recipe)

            # Step 4: Format the recipe
            format_inputs = {
                "recipe_details": recipe_details,
                "custom_recipe": custom_recipe,
            }
            formatted_result = self.crew.kickoff(inputs=format_inputs)
            logging.info("Formatted Result: %s", formatted_result)

            if not formatted_result or "formatted_recipe" not in formatted_result:
                raise KeyError("Format task failed. 'formatted_recipe' key is missing.")

            formatted_recipe = formatted_result["formatted_recipe"]
            logging.info("Final Formatted Recipe: %s", formatted_recipe)

            return formatted_recipe

        except KeyError as e:
            logging.error("KeyError: %s", e)
        except Exception as e:
            logging.error("An unexpected error occurred: %s", e)

        return None


if __name__ == "__main__":
    logging.info("## Welcome to the Recipe Generator Crew ##")

    # Define user preferences and inputs
    user_preferences = {
        "dietary_restrictions": "vegetarian",
        "preferred_cuisine": "Italian",
        "avoid_ingredients": ["gluten"],
        "servings": 4,
    }
    ingredient_filters = ["tomato", "basil", "cheese"]
    dish_type = "main course"

    try:
        recipe_crew = RecipeCrew(user_preferences, ingredient_filters, dish_type)
        formatted_recipe = recipe_crew.run()

        if formatted_recipe:
            logging.info("Formatted Recipe:\n%s", formatted_recipe)
        else:
            logging.warning("Failed to generate a formatted recipe.")
    except KeyError as e:
        logging.error("KeyError: %s", e)
    except EnvironmentError as e:
        logging.error("EnvironmentError: %s", e)
    except Exception as e:
        logging.error("An unexpected error occurred: %s", e)
