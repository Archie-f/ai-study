from pathlib import Path

print("\n----------------- Part-2 ------------------\n")

# From a string
p = Path('src/ai_chatbot/chat.py')

# Current working directory (where you ran Python from)
cwd = Path.cwd()

# User's home directory (/Users/akif on Mac, /home/akif on Linux)
home = Path.home()

# The folder containing the current .py file
# __file__ is a special variable Python sets to the path of the current file
here = Path(__file__).parent

print(f"p -> {p}")
print(f"cwd -> {cwd}")
print(f"home -> {home}")
print(f"here -> {here}")

# What's the difference between Path.cwd() and Path(__file__).parent?
# => The core difference is that Path.cwd() changes depending on where you run the python command from,
#    while Path(__file__).parent is locked to where the actual code file is saved on your computer.

print("\n----------------- 0 ------------------\n")
print(f".parent -> {p.parent}")
print(f".name -> {p.name}")
print(f".stem -> {p.stem}")
print(f".suffix -> {p.suffix}")
print("----------------- 0 ------------------")
print(f".exists() -> {p.exists()}")
print(f".exists() -> {here.exists()}")
print(f"is file() -> {p.is_file()}")
print(f"is_file() -> {here.is_file()}")
print(f"is dir() -> {p.is_dir()}")
print(f"is_dir() -> {here.is_dir()}")
print("----------------- 0 ------------------")
this_file = Path(__file__)
print(f"this_file -> {this_file}")
print(f".exists() -> {this_file.exists()}")
print(f".is_file() -> {this_file.is_file()}")
print(f".is_dir() -> {this_file.is_dir()}")
print("----------------- 0 ------------------")

# Exercise 3
# Create a file called pathlib_practice.py in your project.
# Write a function called list_python_files(directory: Path) -> list[Path]
#   that returns all .py files in the given directory (use rglob).
# Write a function called read_file_safely(path: Path) -> str
#   that returns the file contents if the file exists, or an empty string if it does not.
# In the main block, call list_python_files on your src/ai_chatbot directory
#   and print each filename using .name.
# Run it and verify the output.

def list_python_files(directory) -> list[Path]:
    return list(Path(directory).rglob("*.py"))

python_files = list_python_files(Path.cwd())
for f in python_files:
    print(f.name)

def read_file_safely(path: Path) -> str:
    if not path.exists():
        return f"Given path: '{path}' is not a file."

    if not path.is_file():
        return f"File '{path}' does not exist."

    return path.read_text()

print("----------------- 0 ------------------")
target_directory = Path('../ai_chatbot/src/ai_chatbot').resolve()
python_files = list_python_files(target_directory)
for python_file in python_files:
    print(python_file.name)




