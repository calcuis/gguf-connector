import os

gguf_files = [file for file in os.listdir() if file.endswith('.gguf')]

if gguf_files:
    print("GGUF file(s) available. Select which one to use as Clip Handler:")
    
    for index, file_name in enumerate(gguf_files, start=1):
        print(f"{index}. {file_name}")
    choice1 = input(f"Enter your choice (1 to {len(gguf_files)}): ")

    try:
        choice_index=int(choice1)-1
        selected_file=gguf_files[choice_index]
        print(f"Model file: {selected_file} is selected!")
        clip_model_path=selected_file

        from llama_cpp.llama_chat_format import Llava15ChatHandler
        chat_handler = Llava15ChatHandler(clip_model_path)
        print("GGUF file(s) available. Select which one to use as Vision Model:")
        
        for index, file_name in enumerate(gguf_files, start=1):
            print(f"{index}. {file_name}")
        choice2 = input(f"Enter your choice (1 to {len(gguf_files)}): ")

        try:
            choice_index=int(choice2)-1
            selected_file=gguf_files[choice_index]
            print(f"Model file: {selected_file} is selected!")
            model_path=selected_file

            from llama_cpp import Llama
            llm = Llama(
                model_path=model_path,
                chat_handler=chat_handler,
                n_ctx=2048,
                )
            
            while True:
                ask = input("Provide a picture URL (Q for quit): ")
                # sample prompt: https://raw.githubusercontent.com/calcuis/gguf-connector/master/gguf.gif
              
                if ask.lower() == 'q':
                    break

                from rich.progress import Progress
                with Progress(transient=True) as progress:
                    task = progress.add_task("Processing", total=None)
                    response = llm.create_chat_completion(
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type" : "text", "text": "What's in this image?"},
                                    {"type": "image_url", "image_url": {"url": ask } }
                                ]
                            }
                        ]
                    )
                    print(response["choices"][0]["message"]["content"])
        except (ValueError, IndexError):
            print("Invalid choice. Please enter a valid number.")
    except (ValueError, IndexError):
            print("Invalid choice. Please enter a valid number.")
else:
    print("No GGUF files are available in the current directory.")
    input("--- Press ENTER To Exit ---")
print("Goodbye!")
