import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import fitz  # PyMuPDF
import requests
from duckduckgo_search import DDGS
import os
import hashlib
import re

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchMind — RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Outfit:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
    background: #070712 !important;
    color: #e8e8f5 !important;
    font-family: 'Outfit', sans-serif !important;
}

.main .block-container { padding: 2rem 2rem 4rem !important; max-width: 1000px !important; margin: 0 auto !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d0d1a !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

/* Input fields */
.stTextInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    color: #e8e8f5 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 14px !important;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: rgba(100,200,255,0.3) !important;
    box-shadow: none !important;
}

/* Buttons */
.stButton > button {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: 1px !important;
    border-radius: 10px !important;
    border: none !important;
    font-size: 14px !important;
    transition: all 0.2s !important;
}

.primary-btn .stButton > button {
    background: linear-gradient(135deg, #64c8ff, #a78bfa) !important;
    color: #070712 !important;
    font-weight: 700 !important;
    width: 100% !important;
    padding: 14px !important;
}

.primary-btn .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(100,200,255,0.25) !important;
}

/* Answer card */
.answer-card {
    background: rgba(100,200,255,0.04);
    border: 1px solid rgba(100,200,255,0.15);
    border-radius: 16px;
    padding: 24px;
    margin: 16px 0;
    line-height: 1.8;
    font-size: 15px;
    font-weight: 300;
}

.answer-card h1, .answer-card h2, .answer-card h3 {
    font-family: 'Syne', sans-serif !important;
    color: #64c8ff !important;
    margin-bottom: 12px !important;
}

/* Source cards */
.source-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 3px solid #a78bfa;
    border-radius: 10px;
    padding: 14px 16px;
    margin: 8px 0;
    font-size: 13px;
    line-height: 1.6;
}

.source-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #a78bfa;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 6px;
}

/* Section headers */
.section-head {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.3);
    margin: 24px 0 12px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-head::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.05);
}

/* Mode pills */
.mode-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    padding: 4px 12px;
    border-radius: 99px;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 16px;
}

.mode-pdf { background: rgba(167,139,250,0.1); color: #a78bfa; border: 1px solid rgba(167,139,250,0.2); }
.mode-web { background: rgba(100,200,255,0.1); color: #64c8ff; border: 1px solid rgba(100,200,255,0.2); }
.mode-kb  { background: rgba(52,211,153,0.1);  color: #34d399; border: 1px solid rgba(52,211,153,0.2); }

/* History items */
.history-item {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 12px;
    margin: 6px 0;
    font-size: 13px;
    cursor: pointer;
    transition: border-color 0.2s;
}

.history-item:hover { border-color: rgba(100,200,255,0.2); }

.stSelectbox select, [data-testid="stSelectbox"] {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.08) !important;
    color: #e8e8f5 !important;
}

.stFileUploader {
    background: rgba(255,255,255,0.02) !important;
    border: 1px dashed rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
}

.stSpinner > div { border-color: #64c8ff transparent transparent transparent !important; }

.mono { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: rgba(255,255,255,0.4); }
</style>
""", unsafe_allow_html=True)

# ── Initialize session state ──────────────────────────────────
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = None
if 'api_configured' not in st.session_state:
    st.session_state.api_configured = False

# ── Load embedding model ──────────────────────────────────────
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_chroma_client():
    return chromadb.Client()

embedder = load_embedder()
chroma_client = get_chroma_client()

# ── Helper functions ──────────────────────────────────────────
def extract_pdf_text(pdf_file):
    """Extract text from uploaded PDF."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    chunks = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        # Split into chunks of ~500 chars with overlap
        words = text.split()
        chunk_size = 100  # words
        overlap = 20
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk) > 100:  # skip very short chunks
                chunks.append({
                    'text': chunk,
                    'source': f"PDF Page {page_num + 1}",
                    'page': page_num + 1
                })
    return chunks

def search_web(query, max_results=5):
    """Search DuckDuckGo for relevant results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        chunks = []
        for r in results:
            chunks.append({
                'text': f"{r.get('title', '')}. {r.get('body', '')}",
                'source': r.get('href', 'Web'),
                'title': r.get('title', 'Web Result')
            })
        return chunks
    except Exception as e:
        return []

def get_kb_chunks():
    """Built-in AI/ML knowledge base."""
    return [
        {'text': 'Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.', 'source': 'AI/ML Knowledge Base'},
        {'text': 'Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.', 'source': 'AI/ML Knowledge Base'},
        {'text': 'Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.', 'source': 'AI/ML Knowledge Base'},
        {'text': 'Transformer models are a type of neural network architecture that has revolutionized NLP. The attention mechanism allows the model to focus on different parts of the input when producing an output. BERT, GPT, and T5 are popular transformer models.', 'source': 'AI/ML Knowledge Base'},
        {'text': 'Retrieval Augmented Generation (RAG) is a technique that combines retrieval of relevant documents with generative AI to produce more accurate and grounded responses. It reduces hallucinations by providing real context to the LLM.', 'source': 'AI/ML Knowledge Base'},
        {'text': 'Large Language Models (LLMs) are AI systems trained on vast amounts of text data. They can generate human-like text, answer questions, summarize content, and perform many language tasks. Examples include GPT-4, Gemini, Claude, and LLaMA.', 'source': 'AI/ML Knowledge Base'},
        {'text': 'Vector databases store data as high-dimensional vectors (embeddings) and enable fast similarity search. They are essential for RAG systems. Popular options include ChromaDB, Pinecone, Weaviate, and FAISS.', 'source': 'AI/ML Knowledge Base'},
        {'text': 'Fine-tuning is the process of taking a pre-trained model and training it further on a specific dataset for a specific task. It allows models to adapt to domain-specific knowledge while retaining general capabilities.', 'source': 'AI/ML Knowledge Base'},
        {'text': 'Embeddings are dense vector representations of text that capture semantic meaning. Similar texts have similar embeddings. They are used in search, recommendation systems, and RAG pipelines.', 'source': 'AI/ML Knowledge Base'},
        {'text': 'Prompt engineering is the practice of designing and optimizing prompts to effectively communicate with AI language models. It involves crafting instructions that guide the model to produce desired outputs.', 'source': 'AI/ML Knowledge Base'},
        {'text': 'Data Science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data.', 'source': 'AI/ML Knowledge Base'},
        {'text': 'Computer Vision is a field of AI that trains computers to interpret and understand the visual world. Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects.', 'source': 'AI/ML Knowledge Base'},
    ]

def store_in_vectordb(chunks, collection_name):
    """Store chunks in ChromaDB."""
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass

    collection = chroma_client.create_collection(collection_name)

    texts  = [c['text'] for c in chunks]
    sources = [c['source'] for c in chunks]
    ids    = [hashlib.md5(t.encode()).hexdigest()[:16] + str(i) for i, t in enumerate(texts)]

    embeddings = embedder.encode(texts).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=[{'source': s} for s in sources],
        ids=ids
    )
    return collection

def retrieve_context(query, collection, n_results=4):
    """Retrieve most relevant chunks for the query."""
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_results, collection.count())
    )
    chunks = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        chunks.append({'text': doc, 'source': meta['source']})
    return chunks

def generate_answer(query, context_chunks, gemini_model):
    """Generate answer using Gemini with retrieved context."""
    context_text = "\n\n".join([
        f"[Source {i+1}: {c['source']}]\n{c['text']}"
        for i, c in enumerate(context_chunks)
    ])

    prompt = f"""You are ResearchMind, an expert AI research assistant. 
Answer the question below using ONLY the provided context. 
Be comprehensive, accurate, and cite sources by number [1], [2], etc.
If the context doesn't contain enough information, say so honestly.

CONTEXT:
{context_text}

QUESTION: {query}

INSTRUCTIONS:
- Give a detailed, well-structured answer
- Cite sources inline like [1], [2]
- Use bullet points or numbered lists where appropriate
- End with a brief summary
- Be honest if information is limited

ANSWER:"""

    client = st.session_state.groq_client
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )
    return response.choices[0].message.content

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;
         letter-spacing:2px;margin-bottom:4px;">
        🧠 RESEARCH<span style="color:#64c8ff;">MIND</span>
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
         color:rgba(255,255,255,0.3);letter-spacing:2px;margin-bottom:24px;">
        RAG-POWERED ASSISTANT
    </div>
    """, unsafe_allow_html=True)

    # API Key
    st.markdown('<div class="section-head">Configuration</div>', unsafe_allow_html=True)
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="gsk_...",
        help="Get free key at console.groq.com"
    )

    if api_key:
        try:
            groq_client = Groq(api_key=api_key)
            st.session_state.api_configured = True
            st.session_state.groq_client = groq_client
            st.success("✅ API Connected")
        except:
            st.error("❌ Invalid API key")
            st.session_state.api_configured = False

    # Mode selection
    st.markdown('<div class="section-head">Knowledge Source</div>', unsafe_allow_html=True)
    mode = st.selectbox(
        "Source",
        ["📄 PDF Document", "🌐 Web Search", "📚 AI/ML Knowledge Base", "🔀 All Sources"],
        label_visibility="collapsed"
    )

    # PDF upload
    if "PDF" in mode or "All" in mode:
        st.markdown('<div class="section-head">Upload PDF</div>', unsafe_allow_html=True)
        uploaded_pdf = st.file_uploader(
            "Upload PDF",
            type=['pdf'],
            label_visibility="collapsed"
        )

        if uploaded_pdf and uploaded_pdf.name != st.session_state.pdf_name:
            with st.spinner("Processing PDF..."):
                chunks = extract_pdf_text(uploaded_pdf)
                if chunks:
                    store_in_vectordb(chunks, "pdf_collection")
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_name = uploaded_pdf.name
                    st.session_state.pdf_sample_questions = None  # reset questions for new PDF
                    st.success(f"✅ {len(chunks)} chunks extracted")
                else:
                    st.error("Could not extract text from PDF")

    # Stats
    st.markdown('<div class="section-head">Stats</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
         color:rgba(255,255,255,0.3);line-height:2;">
        Questions asked: {len(st.session_state.chat_history)}<br>
        PDF loaded: {'Yes' if st.session_state.pdf_processed else 'No'}<br>
        Model: Gemini 1.5 Flash<br>
        Embeddings: MiniLM-L6-v2
    </div>
    """, unsafe_allow_html=True)

    # Clear history
    if st.button("🗑️ Clear History"):
        st.session_state.chat_history = []
        st.rerun()

# ── Main area ─────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-bottom:40px;">
    <div style="font-family:'Syne',sans-serif;font-size:56px;font-weight:800;
         letter-spacing:-2px;line-height:1;margin-bottom:12px;">
        Research<span style="color:#64c8ff;">Mind</span>
    </div>
    <p style="font-size:16px;font-weight:300;color:rgba(255,255,255,0.4);
       max-width:500px;margin:0 auto;">
        Ask anything. Get AI-powered answers with cited sources.
        Upload PDFs, search the web, or query the knowledge base.
    </p>
</div>
""", unsafe_allow_html=True)

# Mode indicator
mode_labels = {
    "📄 PDF Document": ("mode-pdf", "📄 PDF MODE"),
    "🌐 Web Search": ("mode-web", "🌐 WEB SEARCH MODE"),
    "📚 AI/ML Knowledge Base": ("mode-kb", "📚 KNOWLEDGE BASE MODE"),
    "🔀 All Sources": ("mode-web", "🔀 ALL SOURCES MODE")
}
mode_class, mode_text = mode_labels.get(mode, ("mode-kb", "MODE"))
st.markdown(f'<div class="mode-pill {mode_class}">{mode_text}</div>', unsafe_allow_html=True)

# Sample questions
st.markdown('<div class="section-head">Sample Questions</div>', unsafe_allow_html=True)

# Generate PDF-specific questions if PDF is loaded
if "PDF" in mode and st.session_state.pdf_processed:
    if 'pdf_sample_questions' not in st.session_state:
        st.session_state.pdf_sample_questions = None
    
    if st.session_state.pdf_sample_questions is None and st.session_state.api_configured:
        with st.spinner("Generating questions from PDF..."):
            try:
                # Get some text from PDF
                collection = chroma_client.get_collection("pdf_collection")
                results = collection.get(limit=3)
                sample_text = " ".join(results['documents'][:3])[:1000]
                
                prompt = f"""Based on this document excerpt, generate exactly 4 short, specific questions a researcher would ask.
Return ONLY the 4 questions, one per line, no numbering, no extra text.

Document: {sample_text}

Questions:"""
                
                client = st.session_state.groq_client
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200
                )
                questions_text = response.choices[0].message.content
                questions = [q.strip() for q in questions_text.strip().split('\n') if q.strip()][:4]
                st.session_state.pdf_sample_questions = questions
            except:
                st.session_state.pdf_sample_questions = None

    samples = st.session_state.pdf_sample_questions or [
        "What is the main topic of this document?",
        "What are the key findings?",
        "What methodology was used?",
        "What are the conclusions?"
    ]
else:
    # Reset PDF questions when switching modes
    if 'pdf_sample_questions' in st.session_state:
        st.session_state.pdf_sample_questions = None
    samples = [
        "What is Retrieval Augmented Generation?",
        "Explain transformer architecture",
        "What are vector databases used for?",
        "How does fine-tuning work?"
    ]

cols = st.columns(4)
for i, (col, sample) in enumerate(zip(cols, samples)):
    with col:
        if st.button(sample, key=f"sample_{i}"):
            st.session_state.query = sample

# Query input
st.markdown('<div class="section-head">Ask Your Question</div>', unsafe_allow_html=True)
query = st.text_area(
    "Question",
    value=st.session_state.get('query', ''),
    height=100,
    placeholder="Ask anything... e.g. 'What is the main contribution of this paper?' or 'Explain how RAG works'",
    label_visibility="collapsed"
)

st.markdown(f"<p class='mono'>{len(query)} characters</p>", unsafe_allow_html=True)

st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
ask_btn = st.button("🔍  SEARCH & ANSWER", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── Process query ─────────────────────────────────────────────
if ask_btn:
    if not query.strip():
        st.error("Please enter a question!")
    elif not st.session_state.api_configured:
        st.error("Please enter your Gemini API key in the sidebar!")
    else:
        with st.spinner("🔍 Retrieving context and generating answer..."):
            all_chunks = []

            # Get chunks based on mode
            if "PDF" in mode or "All" in mode:
                if st.session_state.pdf_processed:
                    try:
                        collection = chroma_client.get_collection("pdf_collection")
                        pdf_chunks = retrieve_context(query, collection, n_results=3)
                        all_chunks.extend(pdf_chunks)
                    except:
                        pass

            if "Web" in mode or "All" in mode:
                web_chunks = search_web(query, max_results=4)
                all_chunks.extend(web_chunks[:3])

            if "Knowledge" in mode or "All" in mode:
                kb_chunks_all = get_kb_chunks()
                kb_collection = store_in_vectordb(kb_chunks_all, "kb_collection")
                kb_chunks = retrieve_context(query, kb_collection, n_results=3)
                all_chunks.extend(kb_chunks)

            if not all_chunks:
                # Fallback to web search
                all_chunks = search_web(query, max_results=4)

            if all_chunks:
                # Generate answer
                answer = generate_answer(query, all_chunks[:6], None)

                # Save to history
                st.session_state.chat_history.append({
                    'query': query,
                    'answer': answer,
                    'sources': all_chunks[:6],
                    'mode': mode
                })

                # Display answer
                st.markdown('<div class="section-head">Answer</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

                # Display sources
                st.markdown('<div class="section-head">Sources Used</div>', unsafe_allow_html=True)
                for i, chunk in enumerate(all_chunks[:6]):
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-num">SOURCE {i+1} · {chunk['source']}</div>
                        {chunk['text'][:200]}...
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Could not retrieve relevant context. Try a different question or source.")

# ── Chat history ──────────────────────────────────────────────
if st.session_state.chat_history:
    st.markdown('<div class="section-head">Previous Questions</div>', unsafe_allow_html=True)
    for i, item in enumerate(reversed(st.session_state.chat_history[:-1])):
        with st.expander(f"Q: {item['query'][:60]}..."):
            st.markdown(f'<div class="answer-card">{item["answer"]}</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center;margin-top:60px;padding-top:20px;
     border-top:1px solid rgba(255,255,255,0.05);
     font-family:'JetBrains Mono',monospace;font-size:11px;
     color:rgba(255,255,255,0.2);">
    RESEARCHMIND v1.0 · BUILT BY VISHAL · GEMINI + RAG + CHROMADB · PORTFOLIO PROJECT #2
</div>
""", unsafe_allow_html=True)
