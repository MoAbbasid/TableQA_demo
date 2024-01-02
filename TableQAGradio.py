#!/usr/bin/env python
# coding: utf-8

# ## Using Gradio to create a simple interface.
# 
# Check out the library on [github](https://github.com/gradio-app/gradio-UI) and see the [getting started](https://gradio.app/getting_started.html) page for more demos.

# We'll start with a basic function that greets an input name.

# In[1]:


get_ipython().system('pip install -q gradio')


# Now we'll wrap this function with a Gradio interface.

# In[2]:


from transformers import pipeline
import pandas as pd

tqa = pipeline(task="table-question-answering", model="google/tapas-large-finetuned-wtq")


# In[ ]:


tsqa = pipeline(task="table-question-answering", model="google/tapas-large-finetuned-sqa")


# In[ ]:


mstqa = pipeline(task="table-question-answering", model="microsoft/tapex-large-finetuned-wikisql")


# In[ ]:


mswtqa = pipeline(task="table-question-answering", model="microsoft/tapex-large-finetuned-wtq")


# In[6]:


table2 = pd.read_excel("/content/Sample.xlsx").astype(str)
table3 = table2.head(20)


# In[7]:


table3


# In[ ]:


#t4 = table3.reset_index()
table4


# In[9]:


query = "what is the highest delta onu rx power?"
query2 = "what is the lowest delta onu rx power?"
query3 = "what is the most frequent login id?"
query4 = "how many rows with nan values are there?"
query5 = "how many S2 values are there"


# In[11]:


result = tsqa(table=table3, query=query5)["answer"]
result


# In[12]:


from collections import Counter
Counter(result)


# In[13]:


#mstqa(table=table4, query=query1)["answer"]


# In[14]:


mswtqa(table=table3, query=query5)["answer"]


# In[15]:


def main(filepath, query):

  table5 = pd.read_excel(filepath).head(20).astype(str)
  result = tsqa(table=table5, query=query)["answer"]
  return result

#greet("World")


# In[16]:


import gradio as gr

iface = gr.Interface(
    fn=main,
    inputs=[
        gr.File(type="filepath", label="Upload XLSX file"),
        gr.Textbox(type="text", label="Enter text"),
    ],
    outputs=[gr.Textbox(type="text", label="Text Input Output")],
    title="Multi-input Processor",
    description="Upload an XLSX file and/or enter text, and the processed output will be displayed.",
)

# Launch the Gradio interface
iface.launch()


# In[21]:


get_ipython().system('pip install notebook')


# In[34]:


import os
import subprocess

# Use subprocess to execute the shell command
subprocess.run(["jupyter", "nbconvert", "--to", "script", "--format", "script", "--output", "/content/", "/content/drive/MyDrive/Colab Notebooks/NEW TableQA-GRADIO: Hello World.ipynb"])


# In[19]:


get_ipython().system('gradio deploy')


# In[32]:


from google.colab import drive
drive.mount('/content/drive')


# That's all! Go ahead and open that share link in a new tab. Check out our [getting started](https://gradio.app/getting_started.html) page for more complicated demos.
