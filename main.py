# ============================================================
# RAG over company policies - Gemini + LangChain (modern LCEL)
# Same lab as the Coursera notebook, without deprecated classes
# ============================================================

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import wget
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Conveyor-belt (LCEL) pieces
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Memory pieces  (CHUNK 1 - imports)
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


# ============================================================
# SETUP: secrets + the brain (Gemini)
# ============================================================
load_dotenv()

filename = 'data/companyPolicies.txt'
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'
# wget.download(url, out=filename)   # uncomment only the first time to download the file

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# Optional connection test (costs one API call - delete once it works)
# print(llm.invoke("Explain RAG in one sentence").text)


# ============================================================
# PHASE 1 - INGESTION (runs once): load -> split -> embed -> store
# ============================================================
loader = TextLoader(filename)
docs = loader.load()                       # list with 1 Document (the whole file)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)   # list of chunk Documents
print(len(texts))

embeddings = HuggingFaceEmbeddings()          # load the embedding model
docsearch = Chroma.from_documents(texts, embeddings)   # embed every chunk + store in Chroma
print('document ingested')


# ============================================================
# PHASE 2 - RETRIEVAL HELPERS (shared by Part A and Part B)
# ============================================================
retriever = docsearch.as_retriever()          # question -> 4 closest chunks

def format_docs(docs):
    # open each envelope, take the text, staple them with a blank line between
    return "\n\n".join(d.page_content for d in docs)


# ============================================================
# PART A - BASIC RAG, NO MEMORY  (lab: "Integrating LangChain" + "Using prompt template")
# ============================================================
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Use the following context to answer the question. "
     "If you don't know the answer, just say you don't know.\n\n{context}"),
    ("human", "{input}"),
])

qa = (
    {
        "context": retriever | format_docs,   # lane 1: question -> chunks -> stapled text
        "input": RunnablePassthrough(),       # lane 2: question passes through unchanged
    }
    | prompt              # fill {context} and {input}
    | llm                 # Gemini answers
    | StrOutputParser()   # plain text answer
)

print(qa.invoke("what is mobile policy?"))
print(qa.invoke("Can I eat in company vehicles?"))   # not in the doc -> should say "don't know"


# ============================================================
# PART B - RAG WITH MEMORY  (lab: "Make the conversation have memory")
# ============================================================

# ---- CHUNK 1: the notebook (empty list that will hold the conversation) ----
chat_history = []

# ---- CHUNK 2: the question fixer ("secretary") - runs BEFORE the retriever ----
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and the latest question, rewrite the question "
               "so it makes sense on its own. Do NOT answer it, only rewrite it. "
               "If it is already clear, return it unchanged."),
    MessagesPlaceholder("chat_history"),      # space where the notebook gets pasted
    ("human", "{input}"),                     # space for the new question
])
rewrite_chain = rewrite_prompt | llm | StrOutputParser()

# ---- CHUNK 3: the answerer - runs AFTER the retriever ----
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Use the following context to answer the question. "
               "If you don't know the answer, just say you don't know.\n\n{context}"),
    MessagesPlaceholder("chat_history"),      # old questions + answers
    ("human", "{input}"),                     # current question
])
answer_chain = answer_prompt | llm | StrOutputParser()

# ---- CHUNK 4: one full turn = fix -> retrieve -> answer -> save ----
def ask(question):
    # Step 1: fix the question (skip on the first turn - nothing to fix yet)
    if chat_history:
        search_query = rewrite_chain.invoke({"chat_history": chat_history, "input": question})
    else:
        search_query = question
    print("Searching for:", search_query)

    # Step 2: retrieve chunks using the fixed question
    context = format_docs(retriever.invoke(search_query))

    # Step 3: answer using chunks + notebook + original question
    answer = answer_chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "input": question,
    })

    # Step 4: write this turn into the notebook (for next time)
    chat_history.append(HumanMessage(question))
    chat_history.append(AIMessage(answer))
    return answer

print(ask("What is mobile policy?"))
print(ask("List points in it?"))
print(ask("What is the aim of it?"))


# ============================================================
# PART C - THE CHAT LOOP  (lab: "Wrap up and make it an agent")
# ============================================================
def chat():
    chat_history.clear()   # fresh notebook for a new conversation
    while True:
        question = input("Question: ")
        if question.lower() in ["quit", "exit", "bye"]:
            print("Answer: Goodbye!")
            break
        print("Answer:", ask(question))

# chat()   # uncomment to start chatting; type quit / exit / bye to stop