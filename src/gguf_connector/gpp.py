import os

gguf_files = [file for file in os.listdir() if file.endswith('.gguf')]

if gguf_files:
    print("GGUF file(s) available. Select which one to use:")
    
    for index, file_name in enumerate(gguf_files, start=1):
        print(f"{index}. {file_name}")

    choice = input(f"Enter your choice (1 to {len(gguf_files)}): ")
    
    try:
        choice_index=int(choice)-1
        selected_file=gguf_files[choice_index]
        print(f"Model file: {selected_file} is selected!")
        ModelPath=selected_file

        from llama_cpp import Llama
        llm = Llama(model_path=ModelPath)

        while True:
            ask = input("Enter a Question (Q for quit): ")

            if ask.lower() == 'q':
                  break

            output = llm("Q: "+ask, max_tokens=2048, echo=True)
            answer = output['choices'][0]['text']
            print(answer+"\n")

    except (ValueError, IndexError):
        print("Invalid choice. Please enter a valid number.")
else:
    print("No GGUF files are available in the current directory.")
    input("--- Press ENTER To Exit ---")

print("Goodbye!")
