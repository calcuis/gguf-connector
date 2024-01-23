print("Please select a connector:\n1. ctransformers\n2. llama.cpp")
choice = input("Enter your choice (1 to 2): ")

if choice=="1":
    from .c import *
elif choice=="2":
    from .cpp import *
else:
    print("Not a valid number.")

__version__ = "0.3.1"