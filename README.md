# PDF-chatbot-using-llama3-on-groq-api
 This chatbot leverages Meta's Llama-3 8B language model to interactively engage with PDF documents. By integrating  natural language processing capabilities, the chatbot allows users to query and receive detailed responses from PDF  content seamlessly
## Key features include:
- PDF Upload and Preview: Users can upload PDFs and preview them within the app. 
- Document Processing: PDFs are split into chunks and embedded for efficient retrieval.
- Vector Store and Retrieval: FAISS vector database is used to create a vector store from the document embeddings.
- Language Model Integration: Powered by Groq's Llama-3 8B model for generating context-aware responses.
- Retrieval Chain: Combines document retrieval and response generation for accurate answers.


## In the .env file add the keys for groq_api and open_ai

```sh
GROQ_API_KEY = "" 
OPENAI_API_KEY = ""
```

Link for Groq API: [https://console.groq.com/keys](https://console.groq.com/keys)  
Link for Open_AI Key: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)


### The Open AI embeddings were used because they provided the most accurate results. However, any open-source model embeddings can be used as they can be easily changed in the code.
### Update: The Cohere Embeddings also works really. (pip install langchain-cohere)

## How to Run

Run the `main.py` file with the following command:

```sh
streamlit run main.py
