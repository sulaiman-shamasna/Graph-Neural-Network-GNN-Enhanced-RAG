import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Load and process the PDF
pdf_path = 'data/Understanding_Climate_Change.pdf'
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"Number of pages loaded: {len(documents)}")

# Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
print(f"Number of document chunks after splitting: {len(docs)}")

# Initialize or load the vector store
persist_directory = "chroma_index"
if os.path.exists(persist_directory):
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    vector_store = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    vector_store.persist()

# Initialize retriever
retriever = vector_store.as_retriever()

# Initialize language model
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

# Initialize GNN
gnn = SimpleGNN(input_dim=1536, hidden_dim=512)  # OpenAI embeddings are 1536-dimensional

# Define the RAG with GNN
def gnn_rag(query):
    # Retrieve documents using the new invoke method
    retrieved_docs = retriever.invoke(query)
    
    # Check if documents were retrieved
    if not retrieved_docs:
        print("No relevant documents found.")
        return "No relevant documents found for the query."

    # Extract embeddings for the retrieved documents
    doc_texts = [doc.page_content for doc in retrieved_docs]
    doc_embeddings = embeddings.embed_documents(doc_texts)
    doc_embeddings = torch.tensor(doc_embeddings)

    # Construct adjacency matrix (fully connected graph)
    num_nodes = len(retrieved_docs)
    adjacency_matrix = torch.ones((num_nodes, num_nodes))

    # Apply GNN
    gnn_output = gnn(doc_embeddings, adjacency_matrix)

    # Aggregate GNN outputs
    aggregated_embedding = gnn_output.mean(dim=0).unsqueeze(0)

    # Generate answer using the aggregated information
    combined_input = query + "\n\n" + "\n\n".join(doc_texts)
    answer = llm(combined_input)
    return answer

# Example usage
query = "Explain the impact of greenhouse gases on climate change."
answer = gnn_rag(query)
print(answer)
