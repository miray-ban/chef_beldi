from crewai import Crew, Process
from tasks import search_recipes_task, fetch_recipe_details_task, generate_custom_recipe_task, format_recipe_task
from agents import recipe_researcher, recipe_creator, recipe_formatter
from dotenv import load_dotenv
import os
from textwrap import dedent

# Load environment variables for API
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] = "ruslandev/llama-3-8b-gpt-4o"

class RecipeCrew:
    def __init__(self, user_preferences, ingredient_filters, dish_type):
        self.user_preferences = user_preferences
        self.ingredient_filters = ingredient_filters
        self.dish_type = dish_type

        # Initialize agents and tasks
        self.agents = [recipe_researcher, recipe_creator, recipe_formatter]
        self.tasks = [
            search_recipes_task,
            fetch_recipe_details_task,
            generate_custom_recipe_task,
            format_recipe_task
        ]
        # Setup the crew configuration
        self.crew = Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            memory=True,
            cache=True,
            max_rpm=100,
            verbose=True,
            share_crew=True
        )

    def run(self):
        inputs = {
            "user_preferences": self.user_preferences,
            "ingredient_filters": self.ingredient_filters,
            "dish_type": self.dish_type
        }
        
        print("Task Inputs:", inputs)  # Debug log for inputs

        try:
            # Step 1: Kickoff the crew for initial recipe search and filtering
            result = self.crew.kickoff(inputs=inputs)
            print("Crew Result:", result)  # Debug log for results
            
            if 'recipe_ids' not in result:
                raise KeyError("Error: 'recipe_ids' key is missing in the result.")
            recipe_ids = result['recipe_ids']
            print("Recipe IDs:", recipe_ids)  # Debug log for recipe IDs

            # Step 2: Fetch detailed recipe data
            fetch_result = self.crew.kickoff({"recipe_ids": recipe_ids})
            print("Fetch Result:", fetch_result)  # Debug log for fetch result
            
            if 'recipe_details' not in fetch_result:
                raise KeyError("Error: 'recipe_details' key is missing in the fetch result.")
            recipe_details = fetch_result['recipe_details']
            print("Fetched Recipe Details:", recipe_details)  # Debug log for recipe details

            # Step 3: Generate a custom recipe based on preferences and filters
            custom_recipe_result = self.crew.kickoff({
                "user_preferences": self.user_preferences,
                "ingredient_filters": self.ingredient_filters
            })
            print("Custom Recipe Result:", custom_recipe_result)  # Debug log for custom recipe result
            
            if 'custom_recipe' not in custom_recipe_result:
                raise KeyError("Error: 'custom_recipe' key is missing in the custom recipe result.")
            custom_recipe = custom_recipe_result['custom_recipe']
            print("Generated Custom Recipe:", custom_recipe)  # Debug log for custom recipe

            # Step 4: Prepare inputs for the formatting task
            format_inputs = {
                "recipe_details": recipe_details,
                "custom_recipe": custom_recipe
            }
            print("Formatting Inputs:", format_inputs)  # Debug log for formatting inputs

            # Step 5: Format the final recipe
            formatted_recipe_result = self.crew.kickoff(format_inputs)
            print("Formatted Recipe Result:", formatted_recipe_result)  # Debug log for formatted recipe result
            
            if 'formatted_recipe' not in formatted_recipe_result:
                raise KeyError("Error: 'formatted_recipe' key is missing in the result.")
            
            formatted_recipe = formatted_recipe_result['formatted_recipe']
            print("Formatted Recipe:", formatted_recipe)  # Debug log for the formatted recipe
            
            return formatted_recipe

        except KeyError as e:
            print(str(e))  # Log the error if any key is missing
            return None  # Exit gracefully on error

# Main function for interacting with the user and executing the recipe generation
if __name__ == "__main__":
    print("## Welcome to the Recipe Generator Crew")
    print('-------------------------------')

    user_preferences = {
        "dietary_restrictions": "vegetarian",
        "preferred_cuisine": "Italian",
        "avoid_ingredients": ["gluten"],
        "servings": 4
    }
    ingredient_filters = ["tomato", "basil", "cheese"]
    dish_type = "main course"

    recipe_crew = RecipeCrew(user_preferences, ingredient_filters, dish_type)
    formatted_recipe = recipe_crew.run()
    
    if formatted_recipe:
        print("Formatted Recipe:\n", formatted_recipe)
    else:
        print("Error: Failed to generate a formatted recipe.")
