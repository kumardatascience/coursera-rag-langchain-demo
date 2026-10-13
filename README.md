# RAG with LangChain — Company Policy Q&A Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about a private company-policy document, with conversational memory for follow-up questions.

Built while completing the IBM / Coursera lab **"Summarize Private Documents Using RAG, LangChain, and LLMs"**, but rewritten with **modern LangChain (LCEL)** instead of the deprecated `RetrievalQA`, `ConversationalRetrievalChain` and `ConversationBufferMemory` classes.

## Features

- **Document ingestion** — load, chunk, embed and store a text document in a vector database
- **Semantic retrieval** — find the most relevant chunks for any question
- **Grounded answers** — the LLM answers only from the retrieved context and says "I don't know" when the answer isn't in the document
- **Conversational memory** — follow-up questions like *"What is the aim of it?"* work, thanks to a question-rewriting step that turns them into standalone queries before retrieval
- **Interactive chat loop** in the terminal

## Tech Stack

| Component | Tool |
|---|---|
| Framework | LangChain (LCEL) |
| LLM | Google Gemini (`langchain-google-genai`) |
| Embeddings | Hugging Face `sentence-transformers` (`langchain-huggingface`) |
| Vector store | Chroma (`langchain-chroma`) |
| Document loading / splitting | `langchain-community`, `langchain-text-splitters` |

## How It Works

**Phase 1 — Ingestion (runs once)**
```
companyPolicies.txt → TextLoader → CharacterTextSplitter → HuggingFaceEmbeddings → Chroma
```

**Phase 2 — Question answering (every question)**
```
Question
  → rewrite with chat history (only for follow-ups)
  → retriever finds the top 4 similar chunks
  → chunks + chat history + question go into the prompt
  → Gemini answers
  → question and answer are saved to chat history
```

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/kumardatascience/coursera-rag-langchain-demo.git
   cd coursera-rag-langchain-demo
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install langchain-core langchain-community langchain-text-splitters \
               langchain-huggingface langchain-chroma langchain-google-genai \
               sentence-transformers python-dotenv wget
   ```

4. **Add your API key** — create a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

5. **Run**
   ```bash
   python main.py
   ```
   Type your questions in the chat; type `quit`, `exit` or `bye` to stop.

## Example

```
Question: What is the mobile policy?
Answer: The mobile policy states that ...

Question: What is the aim of it?
Searching for: What is the aim of the mobile policy?
Answer: The aim is ...

Question: Can I eat in company vehicles?
Answer: I don't know based on the provided document.
```

## Project Structure

```
.
├── data/
│   └── companyPolicies.txt   # source document
├── main.py                   # ingestion, RAG chain, memory, chat loop
├── .gitignore
└── README.md
```

## Modern vs. Deprecated Approach

| Coursera lab (deprecated) | This project (modern) |
|---|---|
| `RetrievalQA.from_chain_type(...)` | LCEL pipeline: `retriever \| format_docs \| prompt \| llm \| StrOutputParser()` |
| `ConversationBufferMemory` | A plain `chat_history` list of `HumanMessage` / `AIMessage` |
| `ConversationalRetrievalChain` | Explicit question-rewrite chain + answer chain |

## Acknowledgements

Based on the IBM Skills Network lab on Coursera. The sample policy document is provided by the course.