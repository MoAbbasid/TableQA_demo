from transformers import pipeline
import pandas as pd
import gradio as gr

# Define the models
models = {
    "GTQA (google/tapas-large-finetuned-wtq)": pipeline(task="table-question-answering", model="google/tapas-large-finetuned-wtq"),
    "GTSQA (google/tapas-large-finetuned-sqa)": pipeline(task="table-question-answering", model="google/tapas-large-finetuned-sqa"),
    "MSWTQA (microsoft/tapex-large-finetuned-wtq)": pipeline(task="table-question-answering", model="microsoft/tapex-large-finetuned-wtq"),
    "MSTQA (microsoft/tapex-large-finetuned-wikisql)": pipeline(task="table-question-answering", model="microsoft/tapex-large-finetuned-wikisql")
}

def main(model_choice, file_path, text):
    # Read the Excel file
    table_df = pd.read_excel(file_path, engine='openpyxl').astype(str)
    
    # Prepare the input for the model
    tqa_pipeline_input = {
        "table": table_df,
        "query": text
    }
    
    # Get the selected model
    model = models.get(model_choice)
    if model is None:
        return f"Model choice '{model_choice}' not found."

    
    # Run the model
    result = model(tqa_pipeline_input)["answer"]
    return result


iface = gr.Interface(
    fn=main,
    inputs=[
        gr.Dropdown(choices=list(models.keys()), label="Select Model", value=list(models.keys())[0]),
        gr.File(type="filepath", label="Upload XLSX file"),
        gr.Textbox(type="text", label="Enter text"),
    ],
    outputs=[gr.Textbox(type="text", label="Text Input Output")],
    title="Multi-input Processor",
    description="Upload an XLSX file and/or enter text, and the processed output will be displayed.",
    examples=[
      ["","Literature_review_Test.xlsx", "How many papers are before the year 2020?"],
      ["","Literature_review_Test.xlsx", "How many papers are after the year 2020?"],
      ["","Literature_review_Test.xlsx", "what is the paper with NISIT in the title?"],
    ],
)

# Launch the Gradio interface
iface.launch(debug=True)
