from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PERSIST_DIR = "chroma_db"


def format_docs(docs):
    """Turns retrieved chunks into one text block, with sources marked."""
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata['source']} | {d.metadata['name']}]\n{d.page_content}"
        for d in docs
    )


def format_history(history, max_turns=3):
    """Turns the last few (question, answer) pairs into a text block."""
    recent = history[-max_turns:]
    if not recent:
        return "No previous conversation."
    return "\n\n".join(f"Q: {q}\nA: {a}" for q, a in recent)

def needs_rewrite(question: str, history: list) -> bool:
    """
    Only the very first question in a conversation is guaranteed to have
    nothing to rewrite against. For every question after that, we let the
    rewrite LLM itself judge whether the question is already standalone
    (the REWRITE_PROMPT already instructs it to return questions UNCHANGED
    when no rewriting is needed) -- rather than guessing with a keyword
    list, which would be inconsistent with using semantic/LLM judgment
    everywhere else in this pipeline.
    """
    return bool(history)

# --- REWRITE PROMPT ---
# Turns a follow-up question ("what about errors?") into a standalone question
# ("does client.py handle errors?") using conversation history, BEFORE retrieval runs.
REWRITE_PROMPT = ChatPromptTemplate.from_template("""Given the conversation history and a new question, 
rewrite the new question into a standalone question that makes sense without needing the history.

- If the new question is already standalone (doesn't reference anything from before), return it UNCHANGED.
- If it's a follow-up (e.g. uses "it", "that", "the other one", "what about..."), rewrite it to explicitly 
  name what it's referring to, based on the history.
- Output ONLY the rewritten question, nothing else — no explanation, no quotes.

Conversation history:
{history}

New question: {question}

Standalone question:""")


# --- MAIN ANSWERING PROMPT ---
MAIN_PROMPT = ChatPromptTemplate.from_template("""You are a code assistant that answers questions about a specific GitHub repository, using ONLY the retrieved code context provided below. You never use outside knowledge about libraries, frameworks, or general programming concepts unless it's needed to explain what the retrieved code itself is doing.

HOW TO HANDLE DIFFERENT QUESTION TYPES:

1. SPECIFIC questions (e.g. "how does X work", "what does function Y do"):
   Answer precisely using the matching chunk(s). Quote key logic in your own words, not verbatim code dumps.

2. BROAD/VAGUE questions (e.g. "what is this project", "explain this repo", "what tech stack is used"):
   Synthesize an answer by combining clues across ALL retrieved chunks — file names, docstrings, 
   imports, class/function names, and any README content — into one coherent summary. 
   Do not require a single chunk to fully answer the question; piece it together.

3. COMPARISON questions (e.g. "difference between A and B", "how do X and Y relate"):
   Address each side using its own relevant chunk(s), then explicitly state the relationship or difference.

4. QUESTIONS PARTIALLY COVERED by the context:
   Give the best answer you can from what's available. Explicitly note which part of the 
   question isn't covered by the retrieved context, rather than refusing to answer at all.

5. QUESTIONS ASKING FOR YOUR OPINION/RECOMMENDATION (e.g. "should I use X", "which is better", 
   "what do you suggest", "is this a good approach"):
   First state clearly what the codebase itself shows or supports, if anything relevant exists in the context.
   Then, clearly separated and labeled as "My suggestion (not from this repo):", you may offer your own 
   reasoned opinion. Never present your own opinion as if it were something the codebase states.

6. QUESTIONS COMPLETELY UNRELATED to the retrieved context, and not asking for your opinion either:
   Say: "I don't have enough information in this repo to answer that."

RULES:
- Always cite the file path and function/class name for every claim, in the format (File: path | Name: identifier).
- Never invent function names, arguments, or behavior not present in the context.
- If multiple chunks describe related pieces (e.g. a client and a server, or a class used across files), 
  explain how they connect to each other, not just each in isolation.
- Keep answers concise and structured (use bullet points for multi-part answers), but complete.

Context:
{context}

Question: {question}

Answer:""")


def get_rag_chain():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()
    answer_chain = MAIN_PROMPT | llm | StrOutputParser()

    def full_pipeline(inputs: dict) -> dict:
        """
        inputs = {"question": <new question>, "history": <list of (q,a) tuples>}
        Returns {"answer": ..., "standalone_question": ..., "sources": [...]}
        """
        question = inputs["question"]
        history = inputs["history"]

        # STEP A: only rewrite if there's actual history to rewrite against.
        # Skips one LLM call (saves latency + quota) on every conversation's
        # first question, where rewriting would always return it unchanged anyway.
        if needs_rewrite(question, history):
            history_text = format_history(history)
            standalone_question = rewrite_chain.invoke({
                "question": question,
                "history": history_text
            }).strip()
        else:
            standalone_question = question

        # STEP B: retrieve using the standalone question
        retrieved_docs = retriever.invoke(standalone_question)
        context_text = format_docs(retrieved_docs)

        # STEP C: generate the final answer using retrieved context
        answer = answer_chain.invoke({
            "context": context_text,
            "question": standalone_question
        })

        return {
            "answer": answer,
            "standalone_question": standalone_question,
            "sources": retrieved_docs
        }

    return full_pipeline