import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_chroma import Chroma

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Check if the OpenAI API key is loaded correctly
if openai_api_key is None:
    raise ValueError("OpenAI API key not found. Please set it in your .env file.")

# Define a simple GNN layer
class SimpleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleGNN, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, node_features, adjacency_matrix):
        # Perform a simple GNN operation: aggregation and transformation
        aggregated = torch.matmul(adjacency_matrix, node_features)
        transformed = self.relu(self.linear(aggregated))
        return transformed

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Initialize Chroma vector store
vector_store = Chroma(persist_directory="chroma_index", embedding_function=embeddings)
retriever = vector_store.as_retriever()

# Initialize language model
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

# Initialize GNN
gnn = SimpleGNN(input_dim=768, hidden_dim=512)  # Example dimensions

# Define the RAG with GNN
def gnn_rag(query):
    # Retrieve documents using the new invoke method
    retrieved_docs = retriever.invoke(query)
    
    # Check if documents were retrieved
    if not retrieved_docs:
        print("No relevant documents found.")
        return "No relevant documents found for the query."

    # Construct adjacency matrix based on some relationship criteria
    # For simplicity, assume fully connected graph
    num_nodes = len(retrieved_docs)
    adjacency_matrix = torch.ones((num_nodes, num_nodes))
    
    # Encode documents
    doc_embeddings = torch.stack([torch.tensor(doc.embedding) for doc in retrieved_docs])
    
    # Apply GNN
    gnn_output = gnn(doc_embeddings, adjacency_matrix)
    
    # Aggregate GNN outputs
    aggregated_embedding = gnn_output.mean(dim=0).unsqueeze(0)
    
    # Generate answer using aggregated embedding
    # (This part would require integrating the embedding into the generation process)
    # For simplicity, concatenate with query
    combined_input = query + " " + " ".join([doc.page_content for doc in retrieved_docs])
    answer = llm(combined_input)
    return answer


# Example usage
# query = "Explain the role of attention mechanisms in transformers."
query = "What is the sum of one and five"
answer = gnn_rag(query)
print(answer)
