from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from PyPDF2 import PdfReader


from dotenv import load_dotenv
load_dotenv()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Path to the uploaded document
pdf_path = r"C:\Users\sfaiz\OneDrive\Desktop\Official ALY 6080 Website\ALY_6080_Experential_learning_Group_1_Module_12_Capstone_Sponsor_Deliverable.pdf"

# Extract text from the PDF
document_text = extract_text_from_pdf(pdf_path)

# Split the document into smaller chunks for FAISS
chunk_size = 1000  # Adjust based on requirements
document_chunks = [document_text[i:i+chunk_size] for i in range(0, len(document_text), chunk_size)]

# Generate embeddings
embeddings = OpenAIEmbeddings()

# Create FAISS index
vector_store = FAISS.from_texts(document_chunks, embeddings)

# Save the index locally in the 'db' folder
vector_store.save_local("db")
