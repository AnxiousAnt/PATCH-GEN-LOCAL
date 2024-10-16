import os
import re
import random
import string
import pandas as pd
import csv
from ollama import generate
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

console = Console()

# Directories for storing the outputs
PD_PATCH_DIR = "Generated-Patches"
DECODED_OUTPUT_DIR = "decoded-outputs"

# Ensure directories exist
os.makedirs(PD_PATCH_DIR, exist_ok=True)
os.makedirs(DECODED_OUTPUT_DIR, exist_ok=True)

# Load the local CSV file containing prompts using pandas
def load_prompts_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df['prompt'].tolist()

# Function to generate a random hex code for filenames
def generate_random_hex(length=8):
    return ''.join(random.choices(string.hexdigits, k=length)).lower()

# Alpaca prompt format
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Function to load valid objects from a CSV file
def load_valid_objects(csv_file):
    valid_objects = set()
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            valid_objects.add(row['objects'].strip())
    return valid_objects

# Function to validate a Pure Data patch for invalid objects
def validate_pd_patch(pd_file, valid_objects):
    invalid_objects = set()
    with open(pd_file, 'r') as f:
        content = f.read()
        objects = re.findall(r'#X obj \d+ \d+ (\S+)(?:;|\s|$)', content)
        for obj in objects:
            obj = obj.rstrip(';')
            if obj not in valid_objects:
                invalid_objects.add(obj)
    return invalid_objects

# Function to generate a Pure Data patch using the Ollama model
def generate_pure_data_patch(prompt, hex_code, valid_objects):
    # Format the prompt
    instruction = "create a Pd patch that matches the following request."
    formatted_prompt = alpaca_prompt.format(instruction, prompt, "")

    # Initialize an empty string to store the response
    response = ""

    # Generate the response from the local model using Ollama
    for part in generate('patch-gen', formatted_prompt, stream=True):
        response += part['response']
        print(part['response'], end='', flush=True)

    # Save the decoded output to a text file in the decoded outputs directory
    text_file_name = os.path.join(DECODED_OUTPUT_DIR, f"decoded_output_{hex_code}.txt")
    with open(text_file_name, "w") as text_file:
        text_file.write(f"Prompt: {prompt}\n\n{response}")
    console.print(f"\n[bright_green]Decoded output saved as '{text_file_name}'[/bright_green]")

    # Extract the Pure Data patch using regex
    pattern = r"```([\s\S]*?)```"
    match = re.search(pattern, response)

    if match:
        pd_patch = match.group(1).strip()

        # Save the extracted patch to a .pd file in the patches directory
        pd_file_name = os.path.join(PD_PATCH_DIR, f"generated_patch_{hex_code}.pd")
        with open(pd_file_name, "w") as pd_file:
            pd_file.write(pd_patch)
        console.print(f"[bright_green]Patch saved as '{pd_file_name}'[/bright_green]")

        console.print(f"[bright_magenta]Validating patch (checking for invalid objects) ...[/bright_magenta]")
        # Validate the patch for invalid objects
        invalid_objects = validate_pd_patch(pd_file_name, valid_objects)
        if invalid_objects:
            console.print(f"[bright_red]WARNING: The generated patch might contain objects that don't exist, please retry with a different prompt.[/bright_red]")
        else:
            console.print("[bright_green]All objects in the Pd patch are valid.[/bright_green]")
    else:
        console.print(f"[bright_red]No patch found in the response.[/bright_red]")

# Menu-based interface to accept user input
def menu(prompts, valid_objects):
    console.print(Panel(Text("Welcome to the Patch Generator!", justify="center", style="bold bright_cyan")))

    counter = 1
    while True:
        table = Table(show_header=True, header_style="bold bright_magenta")
        table.add_column("Option", justify="center")
        table.add_column("Description", justify="left")
        table.add_row("1", "Generate a patch")
        table.add_row("2", "Generate a patch with a random prompt")
        table.add_row("3", "Exit")

        console.print(table)

        choice = Prompt.ask("\nEnter your choice", choices=["1", "2", "3"], default="1")

        if choice == "1":
            prompt = Prompt.ask("[bright_cyan]Enter a prompt to generate a Pure Data patch:[/bright_cyan]")
            console.print("[bright_green]\nResponse:\n[/bright_green]")
            hex_code = generate_random_hex()
            generate_pure_data_patch(prompt, hex_code, valid_objects)
            counter += 1
        elif choice == "2":
            random_prompt = random.choice(prompts)
            console.print(f"[bright_cyan]Random prompt selected:[/bright_cyan] {random_prompt}")
            console.print("[bright_green]\nResponse:\n[/bright_green]")
            hex_code = generate_random_hex()
            generate_pure_data_patch(random_prompt, hex_code, valid_objects)
            counter += 1
        elif choice == "3":
            console.print(Panel(Text("Exiting the program. Goodbye!", style="bold bright_red"), expand=False))
            break

# Main script
if __name__ == "__main__":
    # Load the prompts from the local CSV file (for random)
    csv_file_path = "patch-gen-dataset-v0.8.7_prompts.csv"  
    prompts = load_prompts_from_csv(csv_file_path)

    # Load the valid Pd objects from the CSV file
    valid_objects_file = "objects.csv"
    valid_objects = load_valid_objects(valid_objects_file)
    
    # Start the menu interface
    menu(prompts, valid_objects)
