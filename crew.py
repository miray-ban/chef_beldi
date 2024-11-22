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
groq_api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROG_MODEL_NAME")

if not groq_api_key or not model_name:
    raise EnvironmentError("Missing OPENAI_API_KEY or OPENAI_MODEL_NAME in environment variables.")

class RecipeCrew:
    def __init__(self, user_preferences):
        # Validate input keys
        required_keys = ["dietary_restrictions", "preferred_cuisine", "avoid_ingredients", "ingredient_filters"]
        for key in required_keys:
            if key not in user_preferences:
                raise KeyError(f"Missing required key in user_preferences: '{key}'")

        self.user_preferences = user_preferences

        # Initialize agents
        recipe_agents = RecipeAgents()
        recipe_researcher = recipe_agents.recipe_researcher()
        recipe_formatter = recipe_agents.recipe_formatter()

        # Initialize tasks
        recipe_tasks = RecipeTasks()

        main_task = recipe_tasks.main_task(
            agent_recipe_researcher=recipe_researcher,
            agent_recipe_formatter=recipe_formatter,
            dietary_restrictions=self.user_preferences["dietary_restrictions"],
            ingredient_filters=self.user_preferences["ingredient_filters"],
            avoid_ingredients= self.user_preferences["avoid_ingredients"],
            preferred_cuisine=self.user_preferences["preferred_cuisine"]
        )

        # Setup the crew configuration
        self.recipe_crew = Crew(
            agents=[recipe_researcher, recipe_formatter],
            tasks=[main_task],
            process=Process.sequential,
        )


    def run(self):
        logging.info("Starting the recipe generation process...")
        
        try:
            search_inputs = {
                "dietary_restrictions": self.user_preferences["dietary_restrictions"],
                "preferred_cuisine": self.user_preferences["preferred_cuisine"],
                "avoid_ingredients": self.user_preferences["avoid_ingredients"],
                "ingredient_filters": self.user_preferences["ingredient_filters"],
            }
            search_result = self.recipe_crew.kickoff()
            
            logging.info("Search Result: %s", search_result)
            return search_result

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
        "ingredient_filters": ["tomato", "basil", "cheese"],
    }


    try:
        recipe_crew = RecipeCrew(user_preferences)
        formatted_recipe = recipe_crew.run()

        if formatted_recipe:
            print(formatted_recipe)
            logging.info("Formatted Recipe:\n%s", formatted_recipe)
        else:
            logging.warning("Failed to generate a formatted recipe.")
    except KeyError as e:
        logging.error("KeyError: %s", e)
    except EnvironmentError as e:
        logging.error("EnvironmentError: %s", e)
    except Exception as e:
        logging.error("An unexpected error occurred: %s", e)        
