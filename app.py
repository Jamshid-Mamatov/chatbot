from langchain.vectorstores import Epsilla
from pyepsilla import vectordb
from sentence_transformers import SentenceTransformer
import streamlit as st

import subprocess
from typing import List

import os

os.environ["TOKENIZERS_PARALLELISM"] = "False"

# Local embedding model for embedding the question.
model = SentenceTransformer('all-MiniLM-L6-v2')

class LocalEmbeddings():
  def embed_query(self, text: str) -> List[float]:
    return model.encode(text).tolist()

embeddings = LocalEmbeddings()

# Connect to Epsilla as knowledge base.
client = vectordb.Client()
vector_store = Epsilla(
  client,
  embeddings,
  db_path="/tmp/localchatdb",
  db_name="LocalChatDB"
)
vector_store.use_collection("LocalChatCollection")

# The 1st welcome message
st.title("💬 Chatbot")

if "messages" not in st.session_state:

  st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

# A fixture of chat history
for msg in st.session_state.messages:
  st.chat_message(msg["role"]).write(msg["content"])

# Answer user question upon receiving
if question := st.chat_input():
  st.session_state.messages.append({"role": "user", "content": question})

  context = '\n'.join(map(lambda doc: doc.page_content, vector_store.similarity_search(question, k = 5)))

  st.chat_message("user").write(question)

  # Here we use prompt engineering to ingest the most relevant pieces of chunks from knowledge into the prompt.
  prompt = f'''
    Answer the Question based on the given Context. Try to understand the Context and rephrase them.
    Please don't make things up or say things not mentioned in the Context. Ask for more information when needed.

    Context:
    {context}

    Question:
    {question}

    Answer:
    '''
  print(prompt)

  # Call the local LLM and wait for the generation to finish. This is just a quick demo and we can improve it
  # with better ways in the future.
  # command = ['/llama2/llama.cpp/main', '-m', '~/llama2/llama.cpp/models/7B/ggml-model-q4_0.bin', '-n','1024','--repeat_penalty','1.0','--color','-i','-r "User:"',"-f",prompt]
  command = [
    "/Users/jamshid/llama2/llama.cpp/main","-m", 
    "/Users/jamshid/llama2/llama.cpp/models/7B/ggml-model-q4_0.bin",
    "-n", "1024",
    "--repeat_penalty", "1.0",
    "--color",
    "-i",
    "-r", "user:",
    "-f", prompt
]
  process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  # print(process)
  content = ''
  while True:
    output = process.stdout.readline()
    # print(output)
    if output:
      content = content + output
    return_code = process.poll()
    # print(return_code)
    if return_code is not None:
      break

  # Append the response
  msg = { 'role': 'assistant', 'content': content }
  st.session_state.messages.append(msg)
  st.chat_message("assistant").write(msg['content'])