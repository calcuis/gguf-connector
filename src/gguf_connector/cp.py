
def pdf_handler():
    import os

    pdf_files = [file for file in os.listdir() if file.endswith('.pdf')]

    def join_text(input_text):
        joined_text = ' '.join(input_text.splitlines())
        return joined_text

    if pdf_files:
        print("PDF file(s) available. Select which one to use:")
        
        for index, file_name in enumerate(pdf_files, start=1):
            print(f"{index}. {file_name}")

        choice = input(f"Enter your choice (1 to {len(pdf_files)}): ")
        
        try:
            choice_index=int(choice)-1
            selected_file=pdf_files[choice_index]
            print(f"PDF file: {selected_file} is selected!")

            from pypdf import PdfReader
            reader = PdfReader(selected_file)

            text=""
            number_of_pages = len(reader.pages)
            for i in range(number_of_pages):
                page = reader.pages[i]
                text += page.extract_text()
            
            output_text = join_text(text)
            inject = f"analyze the content below: "+output_text

            print(f"\nPDF cotent extracted as below:\n\n"+text)
            input("---Enter to analyze the PDF content above---")

            print("Processing...")
          
            ans = llm(inject)
            print(inject+ans)

        except (ValueError, IndexError):
            print("Invalid choice. Please enter a valid number.")
    else:
        print("No PDF files are available in the current directory.")
        input("--- Press ENTER To Exit ---")

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

        print("Processing...")

        from ctransformers import AutoModelForCausalLM
        llm = AutoModelForCausalLM.from_pretrained(ModelPath)

        while True:
            ask = input("---Enter to select a PDF file (Q for quit)---")

            if ask.lower() == "q":
                  break
            
            pdf_handler()

    except (ValueError, IndexError):
        print("Invalid choice. Please enter a valid number.")
else:
    print("No GGUF files are available in the current directory.")
    input("--- Press ENTER To Exit ---")

print("Goodbye!")
