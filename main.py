import os  # file paths, environment variables
import uuid  # generate unique session IDs
from datetime import datetime  # store timestamps

import docx  # to ready .docx file
import PyPDF2  # to ready PDF file
import chromadb  # vector database
from chromadb.utils import embedding_functions  # sentence transformer embeddings
from dotenv import load_dotenv  # to load .env file
from openai import OpenAI  # to call Groq API

# =========================================================
# 0. CONFIGURATION + ENVIRONMENT
# =========================================================

# Load environment variables from .env file
load_dotenv()

# Folder where your documents are stored (folder where documents are stored)
DOCS_FOLDER = "docs"

# ChromaDB configuration
# ChromaDB folder decides collection name and embedding model
CHROMA_DB_PATH = "chroma_db"
CHROMA_COLLECTION_NAME = "documents_collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Groq / OpenAI-compatible configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")


# =========================================================
# 1. FILE READING FUNCTIONS
# =========================================================

# open .txt file / returns all the text
def read_text_file(file_path: str) -> str:
    """
    Read content from a plain text (.txt) file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


# open PDF in binary mode / extract each page text / pages which return None, it will skip those pages / combines everything and return a string
def read_pdf_file(file_path: str) -> str:
    """
    Read content from a PDF file using PyPDF2.
    Handles pages that return None for text.
    """
    text = ""
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            # Some PDF pages may return None (no extractable text)
            if page_text:
                text += page_text + "\n"
    return text


# load .docx file / take paragraph-wise text and join into a single string
def read_docx_file(file_path: str) -> str:
    """
    Read content from a Word document (.docx) using python-docx.
    """
    doc = docx.Document(file_path)
    # Join all paragraph texts into a single string with newlines
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])


# pick out the file extension (e.g. .txt, .pdf)
def read_document(file_path: str) -> str:
    """
    Detect file type by extension and route to the correct reader.
    Supports: .txt, .pdf, .docx
    """
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    # call the correct read function on the basis of extension
    if file_extension == ".txt":
        return read_text_file(file_path)
    elif file_extension == ".pdf":
        return read_pdf_file(file_path)
    elif file_extension == ".docx":
        return read_docx_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


# =========================================================
# 2. SPLITTING TEXT INTO CHUNKS
# =========================================================

# break a large text into chunks of 500 characters
def split_text(text: str, chunk_size: int = 500):
    """
    Split large text into smaller chunks while trying
    to preserve sentence boundaries.

    - Replaces newlines with spaces.
    - Splits on '. ' to get sentence-like pieces.
    - Accumulates sentences until the chunk_size is exceeded.
    """
    # Normalize newlines to spaces / convert all the newlines into spaces / splits according to the sentence
    sentences = text.replace("\n", " ").split(". ")
    # variables to create new chunk
    chunks = []
    current_chunk = []
    current_size = 0

    # skip empty sentences
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Ensure each sentence ends with a period for consistency (add "." at the end of the sentences)
        if not sentence.endswith("."):
            sentence += "."

        sentence_size = len(sentence)  # size of the sentence

        # If adding this sentence would exceed chunk_size,
        # start a new chunk
        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size

    # Add the final chunk if it has content
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# =========================================================
# 3. SETUP CHROMADB (VECTOR STORE)
# =========================================================

def init_chroma_collection():
    """
    Initialize a persistent ChromaDB client and collection
    with a SentenceTransformer embedding function.
    """
    # Create a persistent ChromaDB client (stored on disk)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Configure the embedding model used by ChromaDB
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    # Get or create a collection (like a table in a DB)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=sentence_transformer_ef,
    )

    return collection


# =========================================================
# 4. DOCUMENT PROCESSING + INDEXING
# =========================================================

def process_document(file_path: str):
    """
    Read a single document, split it into chunks,
    and generate metadata + IDs for each chunk.

    Returns:
        ids: list of unique IDs for each chunk
        chunks: list of text chunks
        metadatas: list of metadata dicts for each chunk
    """
    try:
        # read file and create chunks
        content = read_document(file_path)
        chunks = split_text(content)

        file_name = os.path.basename(file_path)  # take out the file name
        # generates metadata + ID for every chunk
        metadatas = [{"source": file_name, "chunk": i} for i in range(len(chunks))]
        ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]

        return ids, chunks, metadatas

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return [], [], []


# adds 100-100 chunks at a time into the database
def add_to_collection(collection, ids, texts, metadatas, batch_size: int = 100):
    """
    Add chunked documents to the Chroma collection in batches.

    - Helps avoid sending too much data at once.
    - Skips if there is nothing to add.
    """
    if not texts:
        return

    for i in range(0, len(texts), batch_size):
        end_idx = min(i + batch_size, len(texts))
        # actual insert into the vector DB
        collection.add(
            documents=texts[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx],
        )


def process_and_add_documents(collection, folder_path: str):
    """
    Walk through all files in a folder and index them into ChromaDB.
    """
    if not os.path.exists(folder_path):
        print(f"⚠️ Docs folder '{folder_path}' not found. Skipping indexing.")
        return

    files = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, file))
    ]

    if not files:
        print(f"⚠️ No files found in '{folder_path}'.")
        return

    for file_path in files:
        print(f"📄 Processing: {os.path.basename(file_path)} ...")
        ids, texts, metadatas = process_document(file_path)
        add_to_collection(collection, ids, texts, metadatas)
        print(f"✅ Added {len(texts)} chunks from {os.path.basename(file_path)}\n")


# =========================================================
# 5. SEMANTIC SEARCH HELPERS
# =========================================================

def semantic_search(collection, query: str, n_results: int = 2):
    """
    Perform a semantic search in the Chroma collection
    for the given query text.
    """
    # convert query into embeddings and returns the most relevant chunk in the Database
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )
    return results


def get_context_with_sources(results):
    """
    Convert Chroma query results into:
    - a single context string (all top documents joined),
    - a list of source descriptions for each chunk.

    Handles the case where no documents are returned.
    """
    if not results or not results.get("documents") or not results["documents"][0]:
        # No results found, return empty context and sources
        return "", []

    context = "\n\n".join(results["documents"][0])  # join the top chunks and create a context

    # convert chunk into meta-data list
    sources = [
        f"{meta['source']} (chunk {meta['chunk']})"
        for meta in results["metadatas"][0]
    ]

    return context, sources


def print_search_results(results):
    """
    Utility function to debug and see search results in a readable way.
    Not required for main RAG flow.
    """
    print("\nSearch Results:\n" + "-" * 50)

    if not results or not results.get("documents") or not results["documents"][0]:
        print("No results found.")
        return

    for i in range(len(results["documents"][0])):
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        print(f"\nResult {i + 1}")
        print(f"Source: {meta['source']}, Chunk {meta['chunk']}")
        print(f"Distance: {distance}")
        print(f"Content: {doc}\n")


# =========================================================
# 6. GROQ LLM CLIENT + PROMPTING
# =========================================================

def init_groq_client():
    """
    Initialize a Groq-compatible OpenAI client.
    """
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
    )
    return client


# generate prompts containing - Context, Conversation history, Current question
def build_rag_prompt(context: str, conversation_history: str, query: str) -> str:
    """
    Build the final prompt that will be sent to the LLM.

    - Includes retrieved document context
    - Includes conversation history
    - Includes the current user question
    """
    prompt = f"""Based on the following context and conversation history, 
please provide a relevant and contextual response. 
If the answer cannot be derived from the context, only use the conversation 
history or say "I cannot answer this based on the provided information."

Context from documents:
{context}

Previous conversation:
{conversation_history}

Human: {query}

Assistant:"""
    return prompt


# call the Groq model and returns LLM answer
def generate_response(
        client: OpenAI,
        query: str,
        context: str,
        conversation_history: str = "",
) -> str:
    """
    Call the Groq LLM to generate a response using:
    - retrieved context
    - conversation history
    - user query

    LLM is instructed to answer ONLY from context and history.
    """
    prompt = build_rag_prompt(context, conversation_history, query)

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that answers questions "
                        "only based on the provided context and conversation history. "
                        "If the answer is not present, say 'I don't know'."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,  # 0 → deterministic, factual style
            max_tokens=500,
        )

        # Groq returns message objects, so we use .content
        return response.choices[0].message.content

    except Exception as e:
        # In production you would log this properly
        return f"Error generating response: {str(e)}"


# converts follow-up question into standalone question
def contextualize_query(
        client: OpenAI,
        query: str,
        conversation_history: str,
) -> str:
    """
    Convert a follow-up question into a standalone question using the LLM.

    Example:
      History: "We were talking about RAG".
      Query: "When is it used?"
      → Standalone: "When is RAG used in real-world applications?"
    """
    contextualize_prompt = """
Given a chat history and the latest user question 
which might reference context in the chat history, formulate a standalone 
question which can be understood without the chat history. 
Do NOT answer the question. Just rewrite it in standalone form.
"""

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[
                {"role": "system", "content": contextualize_prompt},
                {
                    "role": "user",
                    "content": f"Chat history:\n{conversation_history}\n\nQuestion:\n{query}",
                },
            ],
            temperature=0,
        )

        return completion.choices[0].message.content

    except Exception as e:
        print(f"Error contextualizing query: {str(e)}")
        # If contextualization fails, just return the original query
        return query


# =========================================================
# 7. CONVERSATION MEMORY (IN-MEMORY DICT)
# =========================================================

# Simple in-memory store for all sessions.
# You can replace this with a DB later (Redis, Mongo, etc.).
conversations = {}


def create_session() -> str:
    """
    Create a new conversation session and return its unique ID.
    """
    session_id = str(uuid.uuid4())
    conversations[session_id] = []
    return session_id


def add_message(session_id: str, role: str, content: str):
    """
    Add a single message to a session's conversation history.

    role: "user" or "assistant"
    """
    if session_id not in conversations:
        conversations[session_id] = []

    conversations[session_id].append(
        {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
    )


def get_conversation_history(session_id: str, max_messages: int | None = None):
    """
    Return a list of messages for a given session.
    Optionally limit to the last N messages.
    """
    if session_id not in conversations:
        return []

    history = conversations[session_id]
    if max_messages:
        history = history[-max_messages:]

    return history


def format_history_for_prompt(session_id: str, max_messages: int = 5) -> str:
    """
    Convert the last few messages of a session into a text format
    that can be inserted into the LLM prompt.

    Example:
      Human: ...
      Assistant: ...
    """
    history = get_conversation_history(session_id, max_messages)
    formatted_history = ""

    for msg in history:
        role = "Human" if msg["role"] == "user" else "Assistant"
        formatted_history += f"{role}: {msg['content']}\n\n"

    return formatted_history.strip()


# =========================================================
# 8. RAG PIPELINE FUNCTIONS
# =========================================================

def rag_query(
        client: OpenAI,
        collection,
        query: str,
        n_chunks: int = 3,
        conversation_history: str = "",
):
    """
    Perform a basic RAG query:
    1. Retrieve top n_chunks from ChromaDB.
    2. Build context.
    3. Call LLM to generate answer.
    """
    # 1. Semantic search
    results = semantic_search(collection, query, n_chunks)

    # 2. Convert results to context + sources
    context, sources = get_context_with_sources(results)

    # 3. Generate final answer from LLM
    answer = generate_response(client, query, context, conversation_history)

    return answer, sources


def conversational_rag_query(
        client: OpenAI,
        collection,
        query: str,
        session_id: str,
        n_chunks: int = 3,
):
    """
    Full conversational RAG:
    - Reads conversation history.
    - Contextualizes follow-up question.
    - Runs RAG retrieval.
    - Generates answer using Groq.
    - Updates conversation memory.
    """
    # Get formatted conversation history for prompt
    conversation_history = format_history_for_prompt(session_id)

    # Use LLM to contextualize the query (make it standalone if needed)
    standalone_query = contextualize_query(client, query, conversation_history)
    print(f"\n🧠 Standalone Query: {standalone_query}")

    # Retrieve relevant chunks from Chroma
    results = semantic_search(collection, standalone_query, n_chunks)
    context, sources = get_context_with_sources(results)

    print("\n📚 Retrieved Context:\n", context)
    print("\n📎 Sources:\n", sources)

    # Generate answer using RAG response function
    response = generate_response(client, standalone_query, context, conversation_history)

    # Store the interaction in conversation memory
    add_message(session_id, "user", query)  # original user question
    add_message(session_id, "assistant", response)

    return response, sources


# =========================================================
# 9. DEMO / MAIN EXECUTION
# =========================================================

if __name__ == "__main__":
    # 1) Initialize Chroma collection and Groq client
    collection = init_chroma_collection()
    client = init_groq_client()
    print("✅ Groq client initialized successfully!")

    # 2) Index documents (run once or whenever docs change)
    #    Be careful: running repeatedly with same IDs can cause conflicts.
    print("\n🔍 Indexing documents from 'docs' folder...")
    process_and_add_documents(collection, DOCS_FOLDER)

    # 3) Simple one-shot RAG query demo (no conversation memory)
    # print("\n=== Simple RAG Demo ===")
    # query = "Real world applications of RAG"
    # answer, sources = rag_query(client, collection, query)
    # print("\nQuery:", query)
    # print("\nAnswer:", answer)
    # print("\nSources used:")
    # for src in sources:
    #     print(f"- {src}")

    # 4) Conversational RAG demo with session memory
    print("\n=== Conversational RAG Demo ===")
    session_id = create_session()

    # First question
    q1 = ""
    resp1, src1 = conversational_rag_query(client, collection, q1, session_id)
    print("\nUser:", q1)
    print("\nAssistant:", resp1)

    # Follow-up question (will be contextualized)
    q2 = ""
    resp2, src2 = conversational_rag_query(client, collection, q2, session_id)
    print("\nUser:", q2)
    print("\nAssistant:", resp2)
