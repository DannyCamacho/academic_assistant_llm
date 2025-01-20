import torch
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList


# Initialize Chroma client and specify the directory for persistent storage
persist_directory = "chroma_db"  # Directory where Chroma will store the data
client = chromadb.PersistentClient(path=persist_directory)

# Initialize Llama model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# Check if CUDA is available and move model to GPU if so
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Set the padding token to be the same as eos_token (if eos_token exists)
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token
stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])


# Define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


# Create a Chroma collection if not already exists
def create_collection(collection_name="text_data"):
    try:
        # Create collection and persist it
        collection = client.create_collection(name=collection_name)
        print(f"Collection '{collection_name}' created.")
    except Exception as e:
        print(f"Collection '{collection_name}' already exists or an error occurred: {e}")


# Function to generate embeddings for input text
def generate_embeddings(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Move the tokenized inputs to the same device as the model (GPU or CPU)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Ensure inputs are on the correct device

    # Pass the input through the model and get the output
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Extract the hidden states from the model's output
    hidden_states = outputs.hidden_states  # This is a tuple of hidden states from all layers

    # We use the hidden state from the last layer (it is the last element in the tuple)
    last_hidden_state = hidden_states[-1]  # Shape: [batch_size, sequence_length, hidden_size]

    # Get the mean of the last layer hidden states for each token to create the embedding
    embedding = last_hidden_state.mean(dim=1)  # Averaging across tokens (sequence length)

    return embedding[0].cpu().numpy().tolist()  # Move embedding to CPU before returning


# Function to add text and its embedding to Chroma
def add_text_to_chroma(text, collection_name="text_data"):
    embedding = generate_embeddings(text)

    # Access the Chroma collection
    collection = client.get_or_create_collection(name=collection_name)

    # Generate a unique ID for each document (e.g., using the hash of the text)
    doc_id = str(hash(text))  # You can use any method to generate unique IDs

    # Add the text and its embedding
    collection.add(
        ids=[doc_id],  # Add unique ID
        documents=[text],
        metadatas=[{"source": "user_input"}],
        embeddings=[embedding]
    )
    print(f"Text added to Chroma: {text[:30]}...")  # Print first 30 characters for brevity


# Function to perform semantic search in Chroma
def semantic_search(query, collection_name="text_data"):
    query_embedding = generate_embeddings(query)

    # Access the Chroma collection
    collection = client.get_collection(name=collection_name)

    # Perform the search using the query embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5  # Limit the number of results
    )

    return results['documents']


# Function to generate a response using the model with retrieved documents
def generate_response(query, retrieved_docs):
    # print(retrieved_docs)
    context = " ".join(retrieved_docs)

    # Combine the query with the retrieved context to form the input for the model
    input_text = (f"You are an academic assistant specializing in the computer science department. "
                  f"Please help me with my academic planning. "
                  f"I need you to provide clear, well-supported responses using only the provided context information."
                  f"Question: {query}"
                  f"Context: {context}"
                  f"Answer:")

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=4096)

    # Move the tokenized inputs to the same device as the model (GPU or CPU)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Ensure inputs are on the correct device

    # Generate a response using the model
    with torch.no_grad():
        stopping_criteria = StoppingCriteriaList([StopOnTokens()])
        attention_mask = inputs["attention_mask"]

        outputs = model.generate(inputs['input_ids'],
                                 num_return_sequences=1,
                                 stopping_criteria=stopping_criteria,  # without this model will ramble
                                 temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
                                 top_p=0.15,  # select from top tokens whose probability add up to 15%
                                 top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
                                 max_new_tokens=600,  # max number of tokens to generate in the output
                                 repetition_penalty=1.1,  # without this output begins repeating
                                 attention_mask=attention_mask,
                                 pad_token_id=tokenizer.pad_token_id,
                                 )

    # Decode the generated output text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


# Function to read a large text file and split it into chunks with overlap
def read_and_process_large_file(file_path, chunk_size=1024, overlap=20):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().replace("\n", " ")

    # Split the large text into chunks with overlap
    chunks = [text[i:i + chunk_size] for i in range(0, len(text) - chunk_size + 1, chunk_size - overlap)]

    return chunks


if __name__ == "__main__":
    # Set up the Chroma collection (it will load from persistent storage if already created)
    create_collection()

    # Read and process the text file
    file_path = 'input_file.txt'  # Replace with your actual file path
    chunks = read_and_process_large_file(file_path, chunk_size=512, overlap=20)  # Adjust as needed

    # Add text chunks to Chroma
    for chunk in chunks:
        add_text_to_chroma(chunk)

    print("Welcome to the Academic Advisor. Type 'exit' to quit.")

    while True:
        # Get user input (academic query)
        query = input("Ask your academic question: ")

        if query.lower() == "exit":
            print("Goodbye!")
            break

        # Perform a semantic search
        [retrieved_docs] = semantic_search(query)

        # Generate a response using the retrieved documents
        response = generate_response(query, retrieved_docs)

        if "Answer:" in response:
            start_index = response.index("Answer:")
            print(response[start_index:])
        else:
            print(f"Answer: {response}")
