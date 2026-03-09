# This file handles all the dictionary search and retrieval logic
# It's basically the brain of our application

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# Helper function to prepare dictionary entries for searching
# -----------------------------------------------------------------------------
def prepare_search_documents(entries):
    """Turn dictionary entries into text we can search through"""
    documents = []
    metadata = []
    
    for entry in entries:
        # Start with empty list for this entry's text
        doc_parts = []
        
        # Repeat the headword a few times so it carries more weight in search
        doc_parts.extend([entry['kichwa']] * 3)
        
        # Add the definition
        doc_parts.append(entry['maana'])
        
        # Add the Swahili parts of examples
        for ex in entry.get('mfano', []):
            # Split at the parenthesis to get just the Swahili part
            sw_part = ex.split('(')[0].strip()
            doc_parts.append(sw_part)
        
        # Add usage notes if they exist
        if 'matumizi' in entry:
            doc_parts.append(entry['matumizi'])
        
        # Add synonyms if they exist
        if 'kisawe' in entry:
            doc_parts.extend(entry['kisawe'])
        
        # Grab English translations from examples
        for ex in entry.get('mfano', []):
            eng_match = re.search(r'\((.*?)\)', ex)
            if eng_match:
                doc_parts.append(eng_match.group(1))
        
        # Combine everything into one big string
        full_text = ' '.join(doc_parts)
        documents.append(full_text)
        
        # Keep some basic info about this entry
        metadata.append({
            'kichwa': entry['kichwa'],
            'aina': entry.get('aina', 'nomino'),
            'maana': entry['maana']
        })
    
    return documents, metadata


# -----------------------------------------------------------------------------
# Our search engine - uses TF-IDF to find relevant entries
# -----------------------------------------------------------------------------
class TfidfSearchEngine:
    """Simple but effective search using TF-IDF - no fancy AI models needed"""
    
    def __init__(self):
        # Set up the vectorizer with some sensible defaults
        self.vectorizer = TfidfVectorizer(
            max_features=1000,      # Keep it manageable
            stop_words=None,         # Don't remove any words, Swahili stop words might be important
            lowercase=True,          # Convert everything to lowercase
            ngram_range=(1, 2),      # Look at single words and pairs of words
            analyzer='word'
        )
        self.documents = None
        self.metadata = None
        self.tfidf_matrix = None
        self.feature_names = None
    
    def fit(self, documents, metadata):
        """Train the search engine on our dictionary entries"""
        print("Building TF-IDF matrix...")
        self.documents = documents
        self.metadata = metadata
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        print(f"Done! Matrix shape: {self.tfidf_matrix.shape}")
        return self
    
    def search(self, query, k=5, min_relevance=0.1):
        """Find the k most relevant entries for a query with minimum relevance threshold"""
        # Turn the query into the same format as our documents
        query_vector = self.vectorizer.transform([query])
        
        # Calculate how similar the query is to each document
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get indices of the top k matches
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Package up the results
        results = []
        for idx in top_indices:
            if similarities[idx] > min_relevance:  # Only include if above threshold
                results.append({
                    'index': idx,
                    'metadata': self.metadata[idx],
                    'relevance': float(similarities[idx]),
                    'document': self.documents[idx][:200] + '...' if len(self.documents[idx]) > 200 else self.documents[idx]
                })
        return results
    
    def search_by_headword(self, headword):
        """Quick lookup by exact headword match"""
        for i, meta in enumerate(self.metadata):
            if meta['kichwa'].lower() == headword.lower():
                return [{
                    'index': i,
                    'metadata': meta,
                    'relevance': 1.0,
                    'document': self.documents[i][:200] + '...'
                }]
        return []


# -----------------------------------------------------------------------------
# The main RAG system that ties everything together
# -----------------------------------------------------------------------------
class SwahiliDictionaryRAG:
    """This is where retrieval meets generation - we find entries and format answers"""
    
    def __init__(self, search_engine, dictionary_data):
        self.search_engine = search_engine
        self.dictionary_data = dictionary_data
        # Create a quick lookup index by headword
        self.index = {entry['kichwa']: entry for entry in dictionary_data}
        
        # Build a cache of English translations from examples
        self.english_cache = {}
        for entry in dictionary_data:
            for ex in entry.get('mfano', []):
                eng_match = re.search(r'\((.*?)\)', ex)
                if eng_match:
                    self.english_cache[entry['kichwa']] = eng_match.group(1)
        
        # Keep a list of all headwords for quick checking
        self.all_headwords = [entry['kichwa'].lower() for entry in dictionary_data]
    
    def detect_language(self, text):
        """Figure out if someone's asking in Swahili or English"""
        text = text.lower()
        
        # Common Swahili words and question markers
        swahili_markers = [
            'ni', 'ya', 'wa', 'na', 'cha', 'vya', 'nini', 'maana',
            'tafsiri', 'ufafanuzi', 'gani', 'je', 'nani', 'wapi', 'lini',
            'kwa', 'cha', 'vya', 'mwa', 'ndani', 'katika'
        ]
        
        # Swahili noun class prefixes
        swahili_prefixes = ['m', 'wa', 'ki', 'vi', 'n', 'u', 'i', 'ji', 'ma']
        
        words = text.split()
        
        # Count Swahili indicators
        sw_count = 0
        for word in words:
            if word in swahili_markers:
                sw_count += 2  # Give extra weight to exact matches
            elif any(word.startswith(prefix) and len(word) > 2 for prefix in swahili_prefixes):
                sw_count += 1
        
        # Common English question words
        en_markers = ['what', 'how', 'define', 'meaning', 'definition', 'call', 'word']
        en_count = sum(1 for marker in en_markers if marker in text)
        
        # Decide based on which count is higher
        if sw_count > en_count:
            return 'sw', sw_count / max(len(words), 1)
        else:
            return 'en', en_count / max(len(words), 1)
    
    def extract_possible_word(self, query):
        """Try to extract a possible dictionary word from the query"""
        query_lower = query.lower()
        words = query_lower.split()
        
        # First, check if any whole word matches a headword
        for word in words:
            # Clean the word (remove punctuation)
            clean_word = re.sub(r'[^\w\s]', '', word)
            if clean_word in self.all_headwords:
                return clean_word
        
        # Check for partial matches (like 'rafiki' in 'maana ya rafiki')
        for headword in self.all_headwords:
            if headword in query_lower:
                return headword
        
        return None
    
    def retrieve(self, query, k=3):
        """Pull the most relevant entries for a query"""
        # First, figure out what language they're using
        lang, confidence = self.detect_language(query)
        
        # Try to extract a possible word
        possible_word = self.extract_possible_word(query)
        
        # Check for direct headword match first
        direct_matches = []
        for headword in self.index:
            if headword.lower() in query.lower():
                direct_matches.append({
                    'headword': headword,
                    'metadata': {'kichwa': headword},
                    'relevance': 0.95,  # High relevance for direct matches
                    'direct_match': True
                })
        
        # If we found direct matches, use those
        if direct_matches:
            results = direct_matches[:k]
            found_exact = True
        else:
            # Otherwise do a proper search with a minimum relevance threshold
            search_results = self.search_engine.search(query, k=k*2, min_relevance=0.15)
            results = []
            seen = set()
            for res in search_results:
                headword = res['metadata']['kichwa']
                if headword not in seen:
                    results.append({
                        'headword': headword,
                        'metadata': res['metadata'],
                        'relevance': res['relevance'],
                        'direct_match': False
                    })
                    seen.add(headword)
                    if len(results) >= k:
                        break
            found_exact = bool(results)
        
        return {
            'query': query,
            'language': lang,
            'confidence': confidence,
            'retrieved': results,
            'found_exact': found_exact,
            'possible_word': possible_word
        }
    
    def get_entry(self, headword):
        """Look up a full dictionary entry by its headword"""
        # Try exact match first
        if headword in self.index:
            return self.index[headword]
        
        # Try case-insensitive
        for key in self.index:
            if key.lower() == headword.lower():
                return self.index[key]
        
        # Try without noun class prefixes (like removing 'm' from 'mtoto')
        for key in self.index:
            if len(key) > 1 and (key[1:] == headword or (len(key) > 2 and key[2:] == headword)):
                return self.index[key]
        
        return None
    
    def generate_response(self, retrieval_result):
        """Create a natural-sounding answer from retrieved entries"""
        query = retrieval_result['query']
        retrieved = retrieval_result['retrieved']
        lang = retrieval_result['language']
        found_exact = retrieval_result['found_exact']
        possible_word = retrieval_result['possible_word']
        
        # If no results found at all
        if not retrieved:
            return self._generate_not_found_response(query, lang, possible_word)
        
        # If results are below relevance threshold (very weak matches)
        if retrieved[0]['relevance'] < 0.2 and not retrieved[0].get('direct_match', False):
            return self._generate_weak_match_response(query, retrieved, lang, possible_word)
        
        # Grab the best match
        best = retrieved[0]
        entry = self.get_entry(best['headword'])
        
        if not entry:
            return self._generate_not_found_response(query, lang, possible_word)
        
        # Respond in the same language they asked in
        if lang == 'sw':
            response = self._generate_swahili_response(entry, best)
        else:
            response = self._generate_english_response(entry, best)
        
        return response
    
    def _generate_not_found_response(self, query, lang, possible_word):
        """Generate a helpful message when word isn't found"""
        if lang == 'sw':
            response = f"""
❌ **Neno '{query}' halipatikani kwenye kamusi**
The word '{query}' was not found in the dictionary

**Mapendekezo:**
• Angalia tahajia - maybe you misspelled it
• Jaribu neno sawa - try a similar word
"""
            if possible_word:
                response += f"\n• Je, ulimaanisha '{possible_word}'? (Did you mean '{possible_word}'?)"
            
            response += """

**Mifano ya maswali sahihi:**
• Maana ya rafiki ni nini?
• Tafsiri ya chakula
• Mwalimu anafundisha nini?
"""
        else:
            response = f"""
❌ **Word '{query}' not found in the dictionary**
Neno '{query}' halipatikani kwenye kamusi

**Suggestions:**
• Check your spelling
• Try a similar word
"""
            if possible_word:
                response += f"\n• Did you mean '{possible_word}'?"
            
            response += """

**Examples of valid questions:**
• What does rafiki mean?
• Define chakula
• Who is a mwalimu?
"""
        
        return response
    
    def _generate_weak_match_response(self, query, retrieved, lang, possible_word):
        """Generate response when matches are weak"""
        if lang == 'sw':
            response = f"""
⚠️ **Neno '{query}' halilingani vizuri na maingizo yoyote**
The word '{query}' doesn't closely match any dictionary entries

**Mapendekezo:**
"""
            if possible_word:
                response += f"\n• Je, ulimaanisha '{possible_word}'? (Did you mean '{possible_word}'?)"
            else:
                response += """
• Angalia tahajia - check your spelling
• Tumia neno moja tu - use a single word
• Jaribu neno la msingi - try the basic word form
"""
            
            # Show closest matches
            response += "\n\n**Maingizo yaliyo karibu zaidi:**\n"
            for i, item in enumerate(retrieved[:2], 1):
                response += f"{i}. **{item['headword']}** - {item['metadata']['maana'][:80]}...\n"
        
        else:
            response = f"""
⚠️ **Word '{query}' doesn't closely match any dictionary entries**
Neno '{query}' halilingani vizuri na maingizo yoyote

**Suggestions:**
"""
            if possible_word:
                response += f"\n• Did you mean '{possible_word}'?"
            else:
                response += """
• Check your spelling
• Use a single word
• Try the basic word form
"""
            
            # Show closest matches
            response += "\n\n**Closest matches:**\n"
            for i, item in enumerate(retrieved[:2], 1):
                response += f"{i}. **{item['headword']}** - {item['metadata']['maana'][:80]}...\n"
        
        return response
    
    def _generate_swahili_response(self, entry, retrieval_info):
        """Put together a Swahili answer"""
        headword = entry['kichwa']
        definition = entry['maana']
        examples = entry.get('mfano', [])
        
        response = f"""
### 📖 {headword}
**Aina:** {entry.get('aina', 'nomino')}

**Maana:** {definition}
"""
        if examples:
            response += "\n**Mifano:**\n"
            for ex in examples[:2]:
                response += f"• {ex}\n"
        
        if 'matumizi' in entry:
            response += f"\n**Matumizi:** {entry['matumizi']}\n"
        
        if 'kisawe' in entry and entry['kisawe']:
            response += f"\n**Visawe:** {', '.join(entry['kisawe'])}\n"
        
        if not retrieval_info.get('direct_match', False):
            response += f"\n*Uhusiano: {retrieval_info['relevance']:.1%}*"
        
        response += f"\n---\n*Chanzo: Kamusi ya Kiswahili*"
        
        return response
    
    def _generate_english_response(self, entry, retrieval_info):
        """Put together an English answer"""
        headword = entry['kichwa']
        definition = entry['maana']
        examples = entry.get('mfano', [])
        
        response = f"""
### 📖 {headword}
**Part of Speech:** {entry.get('aina', 'noun')}

**Meaning:** {definition}
"""
        if examples:
            response += "\n**Examples:**\n"
            for ex in examples[:2]:
                response += f"• {ex}\n"
        
        if 'matumizi' in entry:
            response += f"\n**Usage:** {entry['matumizi']}\n"
        
        if 'kisawe' in entry and entry['kisawe']:
            response += f"\n**Synonyms:** {', '.join(entry['kisawe'])}\n"
        
        if not retrieval_info.get('direct_match', False):
            response += f"\n*Relevance: {retrieval_info['relevance']:.1%}*"
        
        response += f"\n---\n*Source: Swahili Dictionary*"
        
        return response


def load_dictionary_from_json(filepath):
    """Just a helper to load JSON files"""
    import json
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)