# Academic Assistant using Llama Model and Chroma

This repository provides an implementation of an academic assistant that utilizes the Llama-3.2-3B model and Chroma for document retrieval and response generation. The assistant helps users with academic planning by providing answers based on a custom dataset stored in Chroma. It performs semantic search over the dataset to retrieve relevant documents and then generates context-aware responses using a language model.

## Overview

The application integrates the Llama model with Chroma to create an academic assistant. It supports:

    Embedding generation: Convert text into embeddings using a pre-trained model.
    Semantic search: Search for relevant documents based on a query.
    Response generation: Use the retrieved documents as context to generate answers.

The assistant is designed to help users with academic-related queries, retrieving answers from a custom dataset of text stored in Chroma.

## Model Architecture

### Llama Model

The assistant uses the Llama-3.2-3B-Instruct model from Hugging Face for generating responses. It is a causal language model, capable of generating contextually appropriate responses based on the provided input.

    Tokenizer: The AutoTokenizer is used to tokenize the input and output text.
    Model: The AutoModelForCausalLM is used to generate responses from the input text and retrieved documents.

### Chroma

Chroma is used as the vector database for storing and retrieving text embeddings. The text data is split into chunks and each chunk is converted into an embedding for efficient semantic search.
Functions
create_collection(collection_name="text_data")

Creates a Chroma collection to store text data and their embeddings. It persists data to disk for future use.
generate_embeddings(text)

Generates an embedding for the input text using the Llama model. The embedding is calculated by averaging the hidden states of the last layer.
add_text_to_chroma(text, collection_name="text_data")

Adds the text and its embedding to a Chroma collection. The text is tokenized, and its embedding is stored in the collection for future retrieval.
semantic_search(query, collection_name="text_data")

Performs semantic search using the query embedding. It retrieves the top n documents from the Chroma collection that are most relevant to the query.
generate_response(query, retrieved_docs)

Generates a response based on the user query and the documents retrieved during semantic search. It combines the query with the retrieved documents as context and feeds them to the Llama model to generate a context-aware answer.
read_and_process_large_file(file_path, chunk_size=1024, overlap=20)

Reads a large text file and splits it into smaller chunks with overlap. This ensures that the chunks retain context across splits, which is important for accurate embedding generation.

## Usage

    Set up the Chroma collection: The first time you run the script, it will create a new Chroma collection to store the embeddings. You can modify the create_collection function to use a different collection name.

    Input file: Place your large academic text in a file (e.g., output_file.txt). The script will process and split it into chunks before adding it to Chroma.

    Querying: After the text is added to the Chroma collection, you can ask the assistant any academic-related question. The assistant will retrieve relevant documents and generate an answer.

## Example Interaction

![image](https://github.com/user-attachments/assets/5485566c-638e-4bed-bf1a-29825c4789bb)
