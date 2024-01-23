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

        from tkinter import *
        import tkinter.scrolledtext as st

        root = Tk()
        root.title("chatGPT")
        root.columnconfigure([0, 1, 2], minsize=150)
        root.rowconfigure(0, weight=2)
        root.rowconfigure(1, weight=1)
        
        # if os.path.isfile("logo.png"):
        #     icon = PhotoImage(file = "logo.png")
        #     root.iconphoto(False, icon)

        icon = PhotoImage(file = os.path.join(os.path.dirname(__file__), "logo.png"))
        root.iconphoto(False, icon)

        i = Entry()
        o = st.ScrolledText()

        def submit(i):
            root.title("Processing...")
            output = llm("Q: "+str(i.get()), max_tokens=4096, echo=True)
            answer = output['choices'][0]['text']
            print(answer)
            o.insert(INSERT, answer+"\n\n")
            i.delete(0, END)
            root.title("chatGPT")

        btn = Button(text = "Submit", command = lambda: submit(i))
        i.grid(row=1, columnspan=2, sticky="nsew")
        btn.grid(row=1, column=2, sticky="nsew")
        o.grid(row=0, columnspan=3, sticky="nsew")
        root.mainloop()

    except (ValueError, IndexError):
        print("Invalid choice. Please enter a valid number.")
else:
    print("No GGUF files are available in the current directory.")
    input("--- Press ENTER To Exit ---")