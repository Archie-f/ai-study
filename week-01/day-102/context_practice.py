import time
from contextlib import contextmanager

# Exercise 1
# Type the DatabaseConnection class into a Python file called context_practice.py.
# Run it and observe the output — verify that 'Closing connection' prints even when an exception is raised.
# Modify __exit__ to print the exception type: print(f'Exception type: {exc_type}')
# Run it again. What does exc_type show when there is no exception?
# Try returning True from __exit__. What happens to the ValueError?

print("\n----------------- Exercise-1 ------------------\n")

class DatabaseConnection:
    def __init__(self, host: str):
        self.host = host
        self.connection = None

    def __enter__(self):
        print(f"Opening connection to: {self.host}")
        self.connection = f"conn_to_{self.host}"
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Closing connection to: {self.host}")
        print(f"Exception type: {exc_type}")
        self.connection = None
        return True           # True = suppress (swallow) the exceptions
        # return None           # None = do not suppress exceptions (let them propagate)
        # return False          # False = do not suppress exceptions (let them propagate)

with DatabaseConnection('localhost') as db:
    print(f"Connection: {db.connection}")

print("\n----------------- 0 ------------------\n")

with DatabaseConnection('localhost') as db_conn:
    print("Raising exception. About to crash...")
    raise ValueError("This is an intentionally raised error.")
    print("This line will never run.")

# Exercise 2
# Write a @contextmanager function called log_section(title: str).
# It should print '=== START: {title} ===' before the block runs.
# It should print '=== END: {title} ===' after the block runs.
# Use it like this:  with log_section('loading model'):  and put some code inside.
# Bonus: Make it also print how long the section took (use time.time()).

print("\n----------------- Exercise-2 ------------------\n")

@contextmanager
def log_section(title: str):
    print(f'=== START: {title} ===')
    yield
    print(f'=== END: {title} ===')

with log_section('loading model'):
    print("Loading model...")
    start_time = time.time()
    time.sleep(0.846)
    duration = time.time() - start_time
    print(f"Total loading time: {duration:.2f} seconds.")
