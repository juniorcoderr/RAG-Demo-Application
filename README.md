# ⭐ **📘 From-Scratch RAG Pipeline using ChromaDB + Groq (Llama 3.3-70B)**

A fully custom **Retrieval-Augmented Generation (RAG)** system built **without LangChain**, designed to show how RAG works internally — from document loading → chunking → embeddings → vector storage → semantic search → LLM reasoning → conversation memory.

This project is perfect for:

* Developers learning RAG from the core
* People who want *full control* without hidden abstractions
* Beginners who want a clean, easy-to-understand pipeline
* Groq users who want ultra-fast LLM inference
* Interview / demo / portfolio use

---

# 🚀 **What This Project Does**

This project builds a complete RAG system from scratch that can:

- ✅ Load `.txt`, `.pdf`, `.docx` documents  
- ✅ Split large documents into small chunks  
- ✅ Convert chunks into embeddings using SentenceTransformer  
- ✅ Store embeddings in ChromaDB (persistent vector database)  
- ✅ Perform semantic search for the most relevant chunks  
- ✅ Feed context into **Groq Llama-3.3-70B** for high-quality answers  
- ✅ Handle **conversation memory**  
- ✅ Convert follow-up questions into standalone queries  
- ✅ Fully simulate ChatGPT-style chat with your own documents  

---

# 🔥 **Key Features**

### **1️⃣ Multi-format Document Loader**

* Reads **PDFs**, **Word documents**, and **plain text**
* Automatically detects file type
* Cleans and normalizes text

### **2️⃣ Smart Text Chunking**

* Sentence-aware splitting
* Prevents cutting in the middle of ideas
* Default chunk size = 500 characters

### **3️⃣ ChromaDB Vector Storage**

* Persistent local vector DB
* MiniLM-L6-v2 embedding model
* Fast semantic search
* Stores metadata like file names + chunk numbers

### **4️⃣ Groq Llama-3.3-70B Integration**

* Extremely fast & accurate responses
* Prompt includes:

  * Retrieved document context
  * Chat history
  * User question

### **5️⃣ Follow-up Question Understanding**

Example:
User: *"What is RAG?"*
User: *"Where is it used?"* → converted into → *"Where is RAG used?"*

### **6️⃣ Conversation Memory System**

Stores messages with timestamps:

```
conversations = {
    session_id: [
        { "role": "user", "content": "...", "timestamp": "..." },
    ]
}
```

### **7️⃣ Full Conversational RAG**

You get:

* Contextual retrieval
* Memory-aware answers
* Chat-like experience

---

# 🏗 **High-Level Architecture**

```
                ┌──────────────────────────┐
                │     Documents (docs/)    │
                │  .txt / .pdf / .docx     │
                └─────────────┬────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │    Text Chunking Engine  │
                │    (500-char chunks)     │
                └─────────────┬────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │  Embedding Generator     │
                │ (MiniLM-L6-v2)           │
                └─────────────┬────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │   ChromaDB Vector Store  │
                │ (Persistent Collection)  │
                └─────────────┬────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │   Semantic Search Layer  │
                └─────────────┬────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │   Groq LLM (70B)         │
                │   + Context + Memory     │
                └─────────────┬────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │     Final Answer         │
                └──────────────────────────┘
```

---

# 📂 **Project Structure**

```
📦 RAG-from-scratch
 ┣ 📂 docs/               # Your input documents
 ┣ 📂 chroma_db/          # Vector DB (auto-created)
 ┣ 📜 main.py             # Full RAG pipeline
 ┣ 📜 requirements.txt
 ┣ 📜 .env                # GROQ_API_KEY=your_key_here
```

---

# 🧠 **How the Pipeline Works (Simple Explanation)**

### **1. Read Documents**

* PDF → extract text per page
* DOCX → read paragraphs
* TXT → direct read

### **2. Chunk Documents**

Break text into clean pieces → easier for LLM to understand.

### **3. Embed + Index into ChromaDB**

Every chunk gets:

* A unique ID
* Embedding vector
* Metadata (file + chunk number)

### **4. Semantic Search**

User question → embedding → find top similar chunks.

### **5. Build RAG Prompt**

LLM receives:

```
Retrieved Context
Conversation History
User Question
```

### **6. Groq Llama 3.3-70B Generates Final Answer**
AI generates the final answer.

### **7. Conversation Saved**

Used for follow-up questions.

---

# 🖥 **Technologies Used**

* **Python**
* **ChromaDB** - Vector Store
* **Sentence Transformers (MiniLM-L6-v2)** - Embeddings
* **Groq API** - Llama 3.3-70B
* **PyPDF2** - PDF reading
* **python-docx** - DOCX reading
* **dotenv** - Environment variables

---

# 📌 Example Capabilities

### Ask questions like:

```
"What is RAG?"
"Explain chunking."
"Where is retrieval used?"
"Give real-world examples."
```

### And the system answers from YOUR documents:

```
"According to RAG_(Retrieval-Augmented_Generation).pdf, RAG is..."
```

---

# ✅ **Why This Project Is Special**

Unlike LangChain/LlamaIndex, this project gives:

✔ Full visibility
✔ Full control
✔ Zero abstraction
✔ Better debugging
✔ Production-level transparency

Perfect for:

* Learning
* Interviews
* Real-world integrations
* Custom enterprise RAG designs

---

# 🛠 Setup Instructions

```
pip install -r requirements.txt
```

Add `.env`:

```
GROQ_API_KEY=your_key_here
```

Run the project:

```
python main.py
```

---

# ⭐ **Conclusion**

This repo is a fully working **end-to-end RAG system** built from scratch —
no frameworks, no shortcuts, complete transparency.

Perfect for anyone who wants to understand **how real RAG works internally**.
