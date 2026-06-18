# Exercise 1 — Write a system prompt for a travel assistant
# You are building a travel booking assistant. Write a system prompt that:
#
#   Step 1: Frames the role (travel domain, friendly tone).
#   Step 2: Adds two constraints: only discuss destinations in Europe; if asked about other regions,
#           respond with a specific fallback phrase.
#   Step 3: Adds a format directive: always respond with Destination, Best time to visit,
#           and One must-see attraction.
#
# Do not write any Python yet. Write the system prompt string only.
# Test it mentally: if a user asks about Tokyo, what should the model say?

SYSTEM_PROMPT: str = """
You are an experienced travel advisor, specializing in destinations in Europe, answering all questions with a friendly tone. 

# Constraints
Only discuss destinations in Europe. If asked about any destination out of Europe, say:
"I'm sorry. I can only help you with the destinations in Europe. For this destination, you should ask an other assistant."

# Format
For cities, include both the city name and country. For countries, use the country name only. 
Always answer strictly in this format:
Destination: (Destination name)
Best time to visit: (Best time of the year to visit the destination.)
Must-see: (One sentence: A must-see attraction)"""

# Exercise 2 — Build a messages list by hand
# Without writing any API call code, construct a messages list for this scenario:
#
# A user is asking a recipe assistant three questions:
#   Turn 1: user asks what ingredients are in carbonara.
#   Turn 1: assistant lists: eggs, guanciale, Pecorino Romano, black pepper.
#   Turn 2: user asks if they can substitute bacon for guanciale.
#   Turn 2: assistant says yes, with a note about flavour difference.
#   Turn 3: user asks how long to cook the pasta.
#
# Write the messages list as a Python list of dicts, exactly as you would pass it to the API.
# Do not make an API call — just write the data structure.
# How many tokens do you expect this list to consume approximately?

messages_list: list[dict[str, str]] = [
    {"role": "user", "content": "What are the ingredients in carbonara?"},
    {"role": "assistant", "content": "The ingredients in carbonara are: eggs, guanciale, Pecorino Romano, black pepper."},
    {"role": "user", "content": "Can we substitute bacon for guanciale?"},
    {"role": "assistant", "content": "Yes. But substituting bacon for guanciale will cause a difference in flavour."},
    {"role": "user", "content": "How long does it take to cook pasta?"}
]