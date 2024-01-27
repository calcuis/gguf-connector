print("Please select a connector:\n1. llama.cpp\n2. ctransformers")
choice = input("Enter your choice (1 to 2): ")

if choice=="1":
    from .cpp import *
elif choice=="2":
    from .c import *
else:
    print("Not a valid number.")
