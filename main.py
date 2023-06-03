# These are the import statements. You're importing several modules that your script will use.
import os
from dotenv import load_dotenv
import cohere
from cohere.responses.classify import Example
import pprint
from halo import Halo
from colorama import Fore, Style, init
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import pandas as pd


load_dotenv()  # This loads environment variables from a .env file, which is good for sensitive info like API keys

# PrettyPrinter makes dictionary output easier to read
pp = pprint.PrettyPrinter(indent=4)


# Initializes the Cohere API key from the environment variables. Raises an error if the key isn't found.
COHERE_KEY = os.getenv("COHERE_KEY")
if COHERE_KEY is None:
    raise ValueError("Cohere API key not found in the environment variables.")

# Initializes the ChromaDB client with certain settings. These settings specify that the client should use DuckDB with Parquet for storage, 
# and it should store its data in a directory named 'database'.
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="database"
))

# Initializes a CohereEmbeddingFunction, which is a specific function that generates embeddings using the Cohere model.
# These embeddings will be used to add and retrieve examples in the ChromaDB database.
cohere_ef = embedding_functions.CohereEmbeddingFunction(
    api_key=COHERE_KEY,  model_name=os.getenv('COHERE_MODEL_NAME')
)

# Gets or creates a ChromaDB collection named 'help_desk', using the Cohere embedding function.
example_collection = chroma_client.get_or_create_collection(
    name="help_desk", embedding_function=cohere_ef)

# Reads the CSV data into pandas DataFrames.
df_department = pd.read_csv('training_data_department.csv')
df_mood = pd.read_csv('training_data_mood.csv')

# Converts the DataFrames to lists of dictionaries.
department_dict = df_department.to_dict('records')
mood_dict = df_mood.to_dict('records')

# If the number of examples in the collection is less than the number of examples in the department data,
# adds the examples to the collection.
if example_collection.count() < len(department_dict):
    for id, item in enumerate(department_dict):
        index = example_collection.count() if example_collection.count() is not None else 0
        example_collection.add(
            documents=[item['text']],
            metadatas=[{"department": item['label'],
                        "mood": mood_dict[id]['label']}],
            ids=[f"id_{index}"]
        )

#  Creates the function generate_response. This function takes a list of messages, generates a response using the Cohere API,
# classifies the mood and department using the examples in the ChromaDB collection, and then returns the messages, mood, and department.
def generate_response(messages):
    # Creates a loading animation
    spinner = Halo(text='Loading...', spinner='dots')
    spinner.start()

    # Initializes the Cohere API client with your API key
    co = cohere.Client(COHERE_KEY)

    # Gets the mood and department classifications for the messages.
    mood = get_mood_classification(messages, co)
    department = get_department_classification(messages, co)

    spinner.stop()  # Stops the loading animation after receiving the response

    # Defines a priority level for each mood.
    mood_priority = {
        'Despair': 1,
        'Sorrowful': 2,
        'Frustrated': 3,
        'Anxious': 4,
        'Irritated': 5,
        'Neutral': 6,
        'Satisfied': 7,
        'Joyful': 8
    }

    # Prints the user's mood, its priority level, and the responsible department
    print(
        f"\n{Fore.CYAN}Question Received: {Fore.WHITE}{Style.BRIGHT}{messages}{Style.RESET_ALL}"
        f"\n{Fore.GREEN}Mood Detected: {Fore.YELLOW}{Style.BRIGHT}{mood}{Style.RESET_ALL}"
        f"\n{Fore.GREEN}Priority Level: {Fore.RED if mood_priority[mood] <= 2 else Fore.YELLOW if mood_priority[mood] <= 4 else Fore.CYAN}{Style.BRIGHT}{mood_priority[mood]}{Style.RESET_ALL}"
        f"\n{Fore.GREEN}Department to handle your request: {Fore.MAGENTA}{Style.BRIGHT}{department}{Style.RESET_ALL}"
    )

    return messages, mood, department


# Two helper functions to get mood and department classification. They query examples from the ChromaDB collection,
# send a classification request to the Cohere API, and extract the prediction from the response.
def get_department_classification(messages, co):

    department_examples = []
    results = example_collection.query(
        query_texts=[messages],
        n_results=90
    )

    for doc, md in zip(results['documents'][0], results['metadatas'][0]):
        department_examples.append(Example(doc, md['department']))

    department_response = co.classify(
        model=os.getenv("COHERE_MODEL_NAME"),
        inputs=[messages],
        examples=department_examples
    )  # Sends the classification request to the Cohere model

    # Extracts the prediction from the response
    department = department_response.classifications[0].prediction
    return department


def get_mood_classification(messages, co):

    mood_examples = []
    results = example_collection.query(
        query_texts=[messages],
        n_results=90
    )

    for doc, md in zip(results['documents'][0], results['metadatas'][0]):
        mood_examples.append(Example(doc, md['mood']))

    mood_response = co.classify(
        model=os.getenv("COHERE_MODEL_NAME"),
        inputs=[messages],
        examples=mood_examples
    )  # Sends the classification request to the Cohere model

    # Extracts the prediction from the response
    mood = mood_response.classifications[0].prediction
    return mood


def main():
    # A message to inform the user that they can type 'quit' to end the conversation.
    print("Type 'quit' at any time to end the conversation.")

    # An infinite loop that prompts the user for input, generates a response, and adds the response to the ChromaDB collection.
    # The loop breaks when the user types 'quit'.
    while True:
        input_text = input("You: ")

        if input_text.lower() == "quit":
            print("Goodbye!")
            break  # This breaks the infinite loop, ending the script.

        response, mood, department = generate_response(input_text)


        # Adds the response to the ChromaDB collection.
        index = example_collection.count() if example_collection.count() is not None else 0
        example_collection.add(
            documents=[response],
            metadatas=[{"department": department,
                        "mood": mood}],
            ids=[f"id_{index}"]
        )


if __name__ == "__main__":  # Ensures that the main function only runs if this script is the main entry point. If this script is imported by another script, the main function won't automatically run.
    main()
