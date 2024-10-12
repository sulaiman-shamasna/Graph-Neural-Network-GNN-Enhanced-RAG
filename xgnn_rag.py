import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.nn.functional as F
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Check if the OpenAI API key is loaded correctly
if openai_api_key is None:
    raise ValueError("OpenAI API key not found. Please set it in your .env file.")

# Define a Graph Attention Network (GAT) layer
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable weight matrix
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Attention coefficients
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(p=dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, adj):
        # Linear transformation
        Wh = torch.matmul(h, self.W)  # (N, out_features)

        # Self-attention on the nodes
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # (N, N)

        # Masked attention - only consider connected nodes
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        # Compute the final output
        h_prime = torch.matmul(attention, Wh)

        return F.elu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # Number of nodes

        # Repeat Wh N times and concatenate
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        # Concatenate to shape (N*N, 2*out_features)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # Reshape to (N, N, 2*out_features)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

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

# Define the RAG with GNN
def gnn_rag(query):
    # Retrieve documents using the invoke method
    retrieved_docs = retriever.invoke(query)

    # Check if documents were retrieved
    if not retrieved_docs:
        print("No relevant documents found.")
        return "No relevant documents found for the query."

    # Extract embeddings for the retrieved documents
    doc_texts = [doc.page_content for doc in retrieved_docs]
    doc_embeddings = embeddings.embed_documents(doc_texts)  # List of embeddings
    doc_embeddings = torch.tensor(doc_embeddings)  # Convert to tensor of shape (N, 1536)

    # Build the adjacency matrix based on cosine similarity
    similarity_matrix = cosine_similarity(doc_embeddings)
    # Set a threshold to determine edges
    threshold = 0.7
    adjacency_matrix = (similarity_matrix > threshold).astype(float)
    adjacency_matrix = torch.tensor(adjacency_matrix)

    # Initialize GAT layer
    gat_layer = GraphAttentionLayer(in_features=1536, out_features=512)

    # Apply GAT layer
    gnn_output = gat_layer(doc_embeddings, adjacency_matrix)

    # Aggregate GNN outputs (e.g., mean pooling)
    aggregated_embedding = gnn_output.mean(dim=0).unsqueeze(0)

    # Generate answer using the aggregated information
    combined_input = query + "\n\n" + "\n\n".join(doc_texts)
    answer = llm(combined_input)
    return answer

# Example usage
query = "Explain the impact of greenhouse gases on climate change."
answer = gnn_rag(query)
print(answer)
