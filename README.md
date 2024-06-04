# PDF-chatbot-using-llama3-on-groq-api
 This chatbot leverages Meta's Llama-3 8B language model to interactively engage with PDF documents. By integrating  natural language processing capabilities, the chatbot allows users to query and receive detailed responses from PDF  content seamlessly
## Key features include:
- PDF Upload and Preview: Users can upload PDFs and preview them within the app. 
- Document Processing: PDFs are split into chunks and embedded for efficient retrieval.
- Vector Store and Retrieval: FAISS vector database is used to create a vector store from the document embeddings.
- Language Model Integration: Powered by Groq's Llama-3 8B model for generating context-aware responses.
- Retrieval Chain: Combines document retrieval and response generation for accurate answers.

# Run the main.py file.

## Also create a .env file with the keys for groq_api and open_ai
GROQ_API_KEY = ""
OPENAI_API_KEY = ""

### Open AI key is required as we are using the Open AI embeddings for the document embeddings. Any open source embeddings can be used for the same
