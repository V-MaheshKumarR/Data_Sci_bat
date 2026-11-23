# RAG Application — Project Architecture & Line-by-Line Explanation

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    RAG Pipeline                         │
│                                                         │
│  ┌───────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │  PDF File  │───>│ Text Splitter │───>│  Embeddings  │ │
│  │ (Source)   │    │  (Chunks)     │    │  (Vectors)   │ │
│  └───────────┘    └──────────────┘    └──────┬───────┘ │
│                                              │         │
│                                              ▼         │
│                                     ┌──────────────┐   │
│                                     │   MongoDB     │   │
│                                     │ Atlas (Store) │   │
│                                     └──────┬───────┘   │
│                                            │           │
│  ┌───────────┐    ┌──────────────┐         │           │
│  │   User     │───>│ Query        │────────>│           │
│  │  Question  │    │ Embedding    │  Vector Search      │
│  └───────────┘    └──────────────┘         │           │
│                                            ▼           │
│                   ┌──────────────┐   ┌──────────────┐  │
│                   │   LLM (GPT)  │<──│  Relevant    │  │
│                   │   Answer     │   │  Documents   │  │
│                   └──────────────┘   └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Tool Used | Purpose |
|---|---|---|
| Embedding Model | OpenRouter (Nvidia Nemotron) | Converts text → numerical vectors |
| Document Loader | LangChain PyPDFLoader | Reads PDF files |
| Text Splitter | LangChain RecursiveCharacterTextSplitter | Breaks document into small chunks |
| Vector Database | MongoDB Atlas | Stores chunks + their embeddings |
| Vector Search | MongoDB Atlas Vector Search | Finds similar documents by vector |
| LLM | OpenRouter (GPT) | Generates final answer from context |

---

## Cell-by-Cell Explanation

---

### Cell 1 — Create OpenAI Client (via OpenRouter)

```python
from openai import OpenAI
# Import the OpenAI library — we use it to talk to LLMs and embedding models

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  # Instead of calling OpenAI directly, we route through OpenRouter
  # OpenRouter gives access to many models (Nvidia, Meta, OpenAI, etc.) via one API

  api_key="sk-or-v1-...",
  # Your personal API key to authenticate with OpenRouter
)
# `client` is now an object we'll use to call embedding models and LLMs
```

---

### Cell 2 — Define Embedding Function

```python
model = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
# Choose which embedding model to use (this one is free on OpenRouter)
# An embedding model converts text into a list of numbers (a "vector")

def get_embedding(text, input_type="document"):
# Define a reusable function that takes any text and returns its embedding vector

  response = client.embeddings.create(
  # Call the embedding API using our OpenRouter client
    model=model,
    # Use the Nvidia embedding model specified above
    input=[
      {
        "content": [
          {"type": "text", "text": text}
          # Pass the input text in the format the API expects
        ]
      }
    ],
    encoding_format="float"
    # Return the embedding as a list of floating-point numbers
  )
  return response.data[0].embedding
  # Extract and return just the list of numbers (the vector)
  # e.g., [0.012, -0.034, 0.056, ...] — a list of 2048 numbers
```

---

### Cell 3 — Test the Embedding Function

```python
embed = get_embedding("AI Technology")
# Call our function with sample text "AI Technology"
# This sends the text to the API and gets back a vector (list of numbers)
```

---

### Cell 4 — View the Embedding

```python
embed
# Print the full vector to see what it looks like
# Output: [0.0123, -0.0456, 0.0789, ...] — a long list of decimal numbers
```

---

### Cell 5 — Check Embedding Dimensions

```python
len(embed)
# How many numbers are in the vector?
# Output: 2048 — this model produces 2048-dimensional vectors
# This number is important — MongoDB needs to know it when creating the search index
```

---

### Cell 6 — Load and Split the PDF

```python
from langchain_community.document_loaders import PyPDFLoader
# PyPDFLoader reads PDF files and extracts text from each page

from langchain_text_splitters import RecursiveCharacterTextSplitter
# RecursiveCharacterTextSplitter breaks long text into smaller pieces ("chunks")

loader = PyPDFLoader(r"C:\Users\v-maheshkur\Downloads\war_pdf_RAG.pdf")
# Point the loader to your local PDF file

data = loader.load()
# Read the PDF — `data` is now a list of Document objects, one per page
# Each Document has .page_content (the text) and .metadata (page number, etc.)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
# Create a splitter that cuts text into chunks of max 400 characters
# chunk_overlap=20 means consecutive chunks share 20 characters
# WHY? So that sentences at chunk boundaries aren't cut off mid-thought

documents = text_splitter.split_documents(data)
# Split all pages into small chunks
# `documents` is now a list of many small Document objects
```

---

### Cell 7 — Concept Illustration (comment only)

```python
"Hi i am Mahesh your DS trainer i will be taking topics ahead"
# This is just a comment illustrating how chunking + embedding works:
# Chunk 1: "Hi i am Mahesh your DS trainer"       → embedding [1,43,6,7]
# Chunk 2: "DS trainer i will be taking topics ahead" → embedding [4,71,43,6]
# Notice the overlap: "DS trainer" appears in both chunks
```

---

### Cell 8 — Inspect Chunks

```python
len(documents)
# How many chunks were created from the PDF? e.g., 50

documents[0].page_content
# View the text of the first chunk to verify it looks right
```

---

### Cell 9 — Generate Embeddings for All Chunks

```python
docs_to_insert = [{
    "text": doc.page_content,
    # Store the original text (so we can read it later)
    "embedding": get_embedding(doc.page_content)
    # Convert each chunk's text to a vector using our function
} for doc in documents]
# List comprehension — loops through every chunk and creates a dictionary with:
#   "text": the readable text
#   "embedding": the numerical vector
# This may take a while since it calls the API for each chunk
```

---

### Cell 10 — Preview the Data

```python
docs_to_insert
# View the list of dictionaries we're about to insert into MongoDB
# Each item has "text" (string) and "embedding" (list of 2048 floats)
```

---

### Cell 11 — Connect to MongoDB and Insert Data

```python
from pymongo import MongoClient
# PyMongo is the Python driver for MongoDB

mongo_client = MongoClient("mongodb+srv://...@ragappcluster.p8mndqc.mongodb.net/...")
# Connect to MongoDB Atlas (cloud-hosted MongoDB)
# The connection string includes username, password, and cluster address

collection = mongo_client["rag_db"]["war_documents"]
# mongo_client["rag_db"]       → select (or create) a database called "rag_db"
# ["war_documents"]            → select (or create) a collection called "war_documents"
# A "collection" in MongoDB is like a "table" in SQL

result = collection.insert_many(docs_to_insert)
# Insert ALL our chunk documents (text + embedding) into MongoDB at once
# Each dictionary becomes one document in the collection
```

---

### Cell 12 — Confirm Insertion

```python
result
# Shows the insertion result, including the list of generated _id values
# One _id per document inserted
```

---

### Cell 13 — Create Vector Search Index

```python
from pymongo.operations import SearchIndexModel
import time

index_name = "vector_index"
# Name for our search index — we'll reference this name when querying

search_index_model = SearchIndexModel(
  definition={
    "fields": [
      {
        "type": "vector",          # This field contains vector data
        "numDimensions": 2048,     # Each vector has 2048 numbers (must match our model)
        "path": "embedding",       # The field name in our documents that holds the vector
        "similarity": "cosine"     # Use cosine similarity to compare vectors
        # Cosine similarity measures the angle between two vectors
        # Closer to 1.0 = more similar meaning
      }
    ]
  },
  name=index_name,
  type="vectorSearch"              # This is a vector search index (not a regular text index)
)

collection.create_search_index(model=search_index_model)
# Tell MongoDB Atlas to build the vector search index
# This enables fast similarity search over our embeddings

print("Polling to check if the index is ready. This may take up to a minute.")
predicate = None
if predicate is None:
   predicate = lambda index: index.get("queryable") is True
   # A small function that checks if the index is ready to accept queries

while True:
   indices = list(collection.list_search_indexes(index_name))
   # Ask MongoDB for the current status of our index
   if len(indices) and predicate(indices[0]):
      break
      # If the index exists and is queryable, stop waiting
   time.sleep(5)
   # Otherwise wait 5 seconds and check again

print(index_name + " is ready for querying.")
```

---

### Cell 14 — Quick Vector Search Test

```python
query_embedding = get_embedding("January 2025, President")
# Convert our search query into a vector (same way we did for documents)

result = collection.aggregate([
  # aggregate() runs a MongoDB aggregation pipeline (a series of data processing steps)
  {
    "$vectorSearch": {
      "index": "vector_index",        # Use the index we created
      "path": "embedding",            # Search in the "embedding" field
      "queryVector": query_embedding,  # Compare against our query vector
      "numCandidates": 2048,          # Consider this many candidates internally
      "limit": 5                      # Return top 5 most similar documents
    }
  }
])
```

---

### Cell 15 — View Search Results

```python
result.to_list()
# Convert the cursor (lazy result) into a Python list to see the matching documents
```

---

### Cell 16 — Reusable Vector Search Function

```python
def get_query_results(query):
  """Gets results from a vector search query."""

  query_embedding = get_embedding(query, input_type="query")
  # Step 1: Convert the user's question into a vector

  pipeline = [
      {
        "$vectorSearch": {
          "index": "vector_index",
          "queryVector": query_embedding,
          "path": "embedding",
          "numCandidates": 2048,
          "limit": 5
          # Step 2: Find the 5 most similar chunks in MongoDB
        }
      },
      {
        "$project": {
          "_id": 0,      # Exclude the MongoDB ID from results
          "text": 1      # Include only the "text" field
          # Step 3: Return just the text, not the embedding vectors
        }
      }
  ]

  results = collection.aggregate(pipeline)
  # Run the pipeline on MongoDB

  array_of_results = []
  for doc in results:
      array_of_results.append(doc)
  # Convert cursor to a Python list of dictionaries

  return array_of_results
  # Returns something like: [{"text": "chunk1..."}, {"text": "chunk2..."}, ...]
```

---

### Cell 17 — Test the Search Function

```python
get_query_results("who are presidents involved during the war")
# Ask a question → get back the 5 most relevant text chunks from our PDF
```

---

### Cell 18 — Full RAG: Retrieve + Generate Answer

```python
query = "what are the Negotiations during the war"
# The user's question

context_docs = get_query_results(query)
# Step 1 (Retrieve): Find relevant chunks from MongoDB using vector search

context_string = " ".join([doc["text"] for doc in context_docs])
# Step 2: Combine all retrieved chunks into one big string
# e.g., "chunk1 text chunk2 text chunk3 text ..."

prompt = f"""Use the following pieces of context to answer the question at the end.
    {context_string}
    Question: {query}
"""
# Step 3: Build a prompt that gives the LLM the context + the question
# This is the "Augmented" part of RAG — we augment the LLM's knowledge
# with our own documents

model_name = "openai/gpt-oss-120b:free"
# Choose which LLM to use for generating the answer

completion = client.chat.completions.create(
  model=model_name,
  messages=[{"role": "user", "content": prompt}]
)
# Step 4 (Generate): Send the prompt to the LLM and get a response

print(completion.choices[0].message.content)
# Step 5: Print the LLM's answer
# The answer is grounded in YOUR PDF, not just the LLM's training data
```

---

## Summary of the RAG Flow

```
1. LOAD      →  Read PDF file
2. SPLIT     →  Break into small chunks (400 chars each)
3. EMBED     →  Convert each chunk to a 2048-dim vector
4. STORE     →  Save text + vector into MongoDB Atlas
5. INDEX     →  Create a vector search index on MongoDB
6. QUERY     →  User asks a question → convert to vector
7. SEARCH    →  Find top 5 similar chunks via cosine similarity
8. AUGMENT   →  Combine retrieved chunks into a prompt
9. GENERATE  →  Send prompt to LLM → get grounded answer
```
