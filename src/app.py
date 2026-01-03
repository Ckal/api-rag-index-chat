from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup

app = FastAPI()

# Middleware to allow cross-origin communications
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'], 
    allow_credentials=True, 
    allow_methods=['*'], 
    allow_headers=['*'],
)



# Function to crawl all URLs from a domain
def get_all_links_from_domain(domain_url):
    print("domain url " + domain_url)
    visited_urls = set()
    domain_links = set()
    parsed_initial_url = urlparse(domain_url)
    base_domain = parsed_initial_url.netloc
    get_links_from_page(domain_url, visited_urls, domain_links, domain_url)
    return domain_links

# Function to crawl links from a page within the same domain
def get_links_from_page(url, visited_urls, all_links, base_domain):
    print("url " + url)
    print("base_domain " + base_domain)
    if not url.startswith(base_domain):
        return
    
    if url in visited_urls:
        return

    visited_urls.add(url)
    print("Getting next " + url)
    response = requests.get(url)
    print(response)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        base_url = urlparse(url).scheme + '://' + urlparse(url).netloc
        links = soup.find_all('a', href=True)

        for link in links:
            href = link.get('href')
            absolute_url = urljoin(base_url, href)
            parsed_url = urlparse(absolute_url)
            if absolute_url not in visited_urls: 
              if absolute_url.startswith(base_domain):
                  print("hrefe " +absolute_url)
                  all_links.add(absolute_url)
                  get_links_from_page(absolute_url, visited_urls, all_links, base_domain)

    else:
        print(f"Failed to retrieve content from {url}. Status code: {response.status_code}")


from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain.chains import RetrievalQA

from langchain.memory import ConversationBufferMemory

from langchain_community.document_transformers import BeautifulSoupTransformer


# Function to index URLs in RAG
def index_urls_in_rag(urls=[]):
      # Load the RAG model
    rag_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    #rag_model = "intfloat/e5-mistral-7b-instruct"
    encode_kwargs = {
        "normalize_embeddings": False
    }  # set True to compute cosine similarity
    embeddings = HuggingFaceEmbeddings(
        model_name=rag_model, encode_kwargs=encode_kwargs, model_kwargs={"device": "cpu"}
    )

    # Create a vector store for storing embeddings of documents
    vector_store = Chroma(persist_directory="/home/user/.cache/chroma_db", embedding_function=embeddings)
    
   # print("Embedding " +urls)


    for url in urls:
        # Get text from the URL
        loader = WebBaseLoader(url)
        document = loader.load()

        # Transform
        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = bs_transformer.transform_documents(
            document, class_to_extract=["p", "li", "div", "a"]
        )
        
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(docs_transformed)
        print(document_chunks)
        # Index document chunks into the vector store
        vector_store.add_documents(document_chunks)

    
    # Convert vector store to retriever
    retriever = vector_store.as_retriever()
 
    return retriever



# Function to load the RAG model
def load_model():
    model =  HuggingFaceHub(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        model_kwargs={"max_length": 1048, "temperature":0.1, "max_new_tokens":512, "top_p":0.95, "repetition_penalty":1.0},
    )
    return model

def get_conversational_rag_chain(retriever_chain): 
    
    llm = load_model()
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Du bist eine freundlicher Mitarbeiterin Namens Susie und arbeitest in einenm Call Center. Nutze immer und nur den CONTEXT für die Antwort auf folgende Frage. Antworte mit: 'Ich bin mir nicht sicher. Wollen Sie eine Mitarbeiter sprechen' Wenn die Antwort nicht aus dem Context hervorgeht. Antworte bitte immer auf Deutsch? CONTEXT:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

    
def get_response(message, history=[]): 
    retriever_chain = index_urls_in_rag()
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": history,
        "input": message  + " Assistant: ",
        "chat_message": message + " Assistant: "
    })
    #print("get_response " +response)
    res = response['answer']
    parts = res.split(" Assistant: ")
    last_part = parts[-1]
    return last_part#[-1]['generation']['content']   


# Index URLs on app startup
@app.on_event("startup")
async def startup():
    print("donee.... ")
    domain_url = 'https://ki-fusion-labs.de/blog.html'
    links = get_all_links_from_domain(domain_url)
    print(links)
    retriever_chain = index_urls_in_rag(links)
    
    retriever_chain.invoke("Was ist bofrost*")
    get_response("Was kosten Schoko Osterhasen?")

# Define API endpoint to receive queries and provide responses
@app.post("/generate/")
def generate(user_input): 
    
    return get_response(user_input, [])

# Define API endpoint to receive queries and provide responses
@app.post("/update/")
def generate(index_url): 
    retriever_chain = index_urls_in_rag([index_url])
    retriever_chain.invoke("Was ist bofrost*")
    get_response("Was kosten Schoko Osterhasen?")