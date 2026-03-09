# The web interface for our dictionary
# This is what users actually see and interact with

import streamlit as st
import json
import os
from rag_engine import TfidfSearchEngine, SwahiliDictionaryRAG, prepare_search_documents

# Set up the page
st.set_page_config(
    page_title="Kamusi ya Kiswahili - RAG Dictionary",
    page_icon="📚",
    layout="wide"
)


st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #0066cc, #0099ff);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .dictionary-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #0066cc;
        margin-bottom: 1rem;
    }
    .error-card {
        background-color: #fff3f3;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #cc0000;
        margin-bottom: 1rem;
    }
    .warning-card {
        background-color: #fff9e6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffaa00;
        margin-bottom: 1rem;
    }
    .suggestion-card {
        background-color: #e6f3ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #0066cc;
        margin-bottom: 1rem;
    }
    .suggestion-item {
        background-color: white;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        border: 1px solid #ddd;
        cursor: pointer;
        transition: all 0.2s;
    }
    .suggestion-item:hover {
        background-color: #f0f8ff;
        border-color: #0066cc;
        transform: translateX(5px);
    }
    .citation {
        font-size: 0.9rem;
        color: #666;
        font-style: italic;
        margin-top: 1rem;
        padding-top: 0.5rem;
        border-top: 1px solid #ddd;
    }
    .stButton > button {
        width: 100%;
    }
    .suggestion-button {
        text-align: left !important;
        background-color: white !important;
        color: #0066cc !important;
        border: 1px solid #0066cc !important;
    }
    .suggestion-button:hover {
        background-color: #e6f3ff !important;
        border-color: #0066cc !important;
    }
    div[data-testid="column"] {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📚 Smart Swahili Dictionary</h1>
    <p>Smart Swahili Dictionary with RAG | Ask in English or Swahili</p>
</div>
""", unsafe_allow_html=True)

# Keep track of state between interactions
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'dictionary_loaded' not in st.session_state:
    st.session_state.dictionary_loaded = False
if 'example_query' not in st.session_state:
    st.session_state.example_query = ""
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'last_response' not in st.session_state:
    st.session_state.last_response = ""
if 'last_retrieval' not in st.session_state:
    st.session_state.last_retrieval = None
if 'error_type' not in st.session_state:
    st.session_state.error_type = None
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = []
if 'auto_search' not in st.session_state:
    st.session_state.auto_search = False

# Function to perform search
def perform_search(query):
    if st.session_state.rag_system and query:
        with st.spinner("Searching..."):
            # Find relevant entries
            retrieval_result = st.session_state.rag_system.retrieve(query)
            
            # Generate an answer
            response = st.session_state.rag_system.generate_response(retrieval_result)
            
            # Save for later
            st.session_state.last_query = query
            st.session_state.last_response = response
            st.session_state.last_retrieval = retrieval_result
            st.session_state.suggestions = retrieval_result.get('spelling_suggestions', [])
            
            # Determine error type for styling
            if retrieval_result.get('spelling_suggestions'):
                st.session_state.error_type = "suggestions"
            elif not retrieval_result['retrieved']:
                st.session_state.error_type = "not_found"
            elif retrieval_result['retrieved'][0]['relevance'] < 0.2 and not retrieval_result['retrieved'][0].get('direct_match', False):
                st.session_state.error_type = "weak_match"
            else:
                st.session_state.error_type = "success"
            
            # Clear the example and auto_search flag
            st.session_state.example_query = ""
            st.session_state.auto_search = False

# Sidebar - where users load dictionaries and see examples
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("📖 Load Dictionary")
    
    # Give users options for loading dictionary
    dict_source = st.radio(
        "Dictionary source:",
        ["Use default dictionary", "Upload my own file"]
    )
    
    dictionary_data = None
    
    if dict_source == "Upload my own file":
        uploaded_file = st.file_uploader(
            "Choose a JSON file",
            type=['json'],
            help="Upload a JSON file with dictionary entries"
        )
        
        if uploaded_file is not None:
            try:
                dictionary_data = json.load(uploaded_file)
                st.success(f"✅ Loaded {len(dictionary_data)} entries")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        # Try to load the default dictionary
        default_dict_path = "data/dictionary.json"
        if os.path.exists(default_dict_path):
            try:
                with open(default_dict_path, 'r', encoding='utf-8') as f:
                    dictionary_data = json.load(f)
                st.success(f"✅ Loaded default dictionary ({len(dictionary_data)} entries)")
            except Exception as e:
                st.error(f"❌ Couldn't load default dictionary: {str(e)}")
        else:
            st.warning("⚠️ No default dictionary found. Please upload a file.")
            st.info("💡 Create a 'data' folder and put 'dictionary.json' in it.")
    
    # Initialize the RAG system if we have dictionary data
    if dictionary_data and not st.session_state.dictionary_loaded:
        try:
            # Prepare documents for searching
            search_docs, search_metadata = prepare_search_documents(dictionary_data)
            
            # Set up the search engine
            search_engine = TfidfSearchEngine()
            search_engine.fit(search_docs, search_metadata)
            
            # Create the RAG system
            st.session_state.rag_system = SwahiliDictionaryRAG(search_engine, dictionary_data)
            st.session_state.dictionary_loaded = True
            st.session_state.error_type = None
            
            # Show a preview of the dictionary
            with st.expander("👀 Preview entries"):
                for i, entry in enumerate(dictionary_data[:5], 1):
                    st.write(f"{i}. **{entry['kichwa']}** - {entry['maana'][:50]}...")
                if len(dictionary_data) > 5:
                    st.write(f"... and {len(dictionary_data)-5} more")
                    
        except Exception as e:
            st.error(f"❌ Error setting up RAG: {str(e)}")
            st.session_state.dictionary_loaded = False
    
    # Some example questions to get users started
    st.subheader("💡 Try these examples")
    example_queries = [
        "Maana ya rafiki ni nini?",
        "What does mwalimu mean?",
        "Who teaches in a school?",
        "Define kitabu",
        "Tafsiri ya chakula",
        "Nani anafundisha shuleni?"
    ]
    
    for ex in example_queries:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            if st.session_state.rag_system:
                # Set the query and trigger auto-search
                st.session_state.example_query = ex
                st.session_state.auto_search = True
                st.rerun()
            else:
                st.warning("⚠️ Please load a dictionary first")

# Main area - where the magic happens
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔍 Ask a Question")
    
    # Question input box
    query = st.text_input(
        "Type your question:",
        value=st.session_state.example_query,
        placeholder="e.g., Maana ya rafiki ni nini? or What does 'rafiki' mean?",
        key="query_input"
    )
    
    # Search button
    search_clicked = st.button("🔎 Search", type="primary", use_container_width=True)
    
    # Auto-search if triggered by example button
    if st.session_state.auto_search and st.session_state.example_query:
        perform_search(st.session_state.example_query)
        st.rerun()
    
    # Manual search
    if search_clicked and query:
        perform_search(query)
        st.rerun()
    
    # Show the answer if we have one
    if st.session_state.last_response:
        # Choose card style based on result type
        if st.session_state.error_type == "not_found":
            card_class = "error-card"
            icon = "❌"
        elif st.session_state.error_type == "weak_match":
            card_class = "warning-card"
            icon = "⚠️"
        elif st.session_state.error_type == "suggestions":
            card_class = "suggestion-card"
            icon = "💡"
        else:
            card_class = "dictionary-card"
            icon = "✅"
        
        st.markdown(f"### Answer {icon}")
        
        st.markdown(f"""
        <div class="{card_class}">
            {st.session_state.last_response}
        </div>
        """, unsafe_allow_html=True)
        
        # Show clickable suggestions if available
        if st.session_state.suggestions:
            st.markdown("### 🔍 Try these suggestions:")
            
            # Create columns for suggestions
            cols = st.columns(min(len(st.session_state.suggestions), 3))
            for i, sugg in enumerate(st.session_state.suggestions[:3]):
                col_idx = i % 3
                with cols[col_idx]:
                    if st.button(f"📖 {sugg}", key=f"sugg_{sugg}_{i}", use_container_width=True):
                        st.session_state.example_query = sugg
                        st.session_state.auto_search = True
                        st.rerun()
        
        # Simple feedback buttons
        col_a, col_b, col_c = st.columns([1, 1, 4])
        with col_a:
            if st.button("👍", key="feedback_up", help="This was helpful"):
                st.toast("Thanks for the feedback! Asante!")
        with col_b:
            if st.button("👎", key="feedback_down", help="Not helpful"):
                st.toast("Thanks, we'll try to improve! Tutajitahidi kuboresha!")

with col2:
    st.header("Details")
    
    if st.session_state.last_retrieval:
        retrieval = st.session_state.last_retrieval
        
        # Show what language we detected
        lang_map = {'sw': '🇹🇿 Swahili', 'en': '🇬🇧 English'}
        detected_lang = lang_map.get(retrieval['language'], 'Unknown')
        confidence = retrieval['confidence']
        
        st.info(f"**Detected Language:** {detected_lang}")
        st.progress(min(confidence, 1.0), text=f"Confidence: {confidence:.1%}")
        
        # Show spelling suggestions if available
        if retrieval.get('spelling_suggestions'):
            st.success("**💡 Spelling Suggestions Found**")
            for sugg in retrieval['spelling_suggestions']:
                # Make suggestions in details panel clickable too
                if st.button(f"📌 {sugg}", key=f"detail_sugg_{sugg}", use_container_width=True):
                    st.session_state.example_query = sugg
                    st.session_state.auto_search = True
                    st.rerun()
        
        # Show possible word if it's valid
        if retrieval.get('possible_word') and retrieval['possible_word'] in st.session_state.rag_system.all_headwords:
            st.info(f"**🔍 Did you mean:** {retrieval['possible_word']}?")
        
        # Show which entries we found
        if retrieval['retrieved']:
            st.subheader("📚 Retrieved Entries")
            
            for i, item in enumerate(retrieval['retrieved'], 1):
                headword = item['headword']
                entry = st.session_state.rag_system.get_entry(headword)
                
                if entry:
                    with st.container():
                        st.markdown(f"**{i}. {headword}**")
                        st.markdown(f"*{entry.get('aina', 'noun')}*")
                        st.markdown(f"_{entry['maana'][:80]}..._")
                        
                        if item.get('direct_match', False):
                            st.markdown("🔵 **Direct match**")
                        
                        relevance = item.get('relevance', 0)
                        if relevance > 0:
                            # Color code relevance
                            if relevance > 0.5:
                                st.progress(relevance, text=f"Relevance: {relevance:.1%} ✅")
                            elif relevance > 0.2:
                                st.progress(relevance, text=f"Relevance: {relevance:.1%} ⚠️")
                            else:
                                st.progress(relevance, text=f"Relevance: {relevance:.1%} ❌")
                        
                        if i < len(retrieval['retrieved']):
                            st.divider()
        else:
            st.warning("No entries retrieved")
    
    # Show some stats if dictionary is loaded
    if st.session_state.dictionary_loaded:
        st.subheader("📊 Dictionary Stats")
        total_entries = len(st.session_state.rag_system.dictionary_data)
        total_examples = sum(len(e.get('mfano', [])) for e in st.session_state.rag_system.dictionary_data)
        
        st.metric("Total entries", total_entries)
        st.metric("Total examples", total_examples)
        
        # Show first few words for reference
        with st.expander("📖 Available words"):
            words = st.session_state.rag_system.all_headwords[:15]
            st.write("First 15 words:")
            for word in words:
                st.markdown(f"• {word}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>📖 Smart Swahili Dictionary | Ask questions naturally in English or Swahili<br>
    All entries come from an actual Swahili dictionary</p>
    <p style="font-size: 0.8rem;">Made for Swahili learners and language enthusiasts</p>
</div>
""", unsafe_allow_html=True)