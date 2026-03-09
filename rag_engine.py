# This file handles all the dictionary search and retrieval logic
# It's basically the brain of our application

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

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
        
        # Keep a list of all headwords for quick checking and spelling suggestions
        self.all_headwords = [entry['kichwa'] for entry in dictionary_data]
        self.all_headwords_lower = [entry['kichwa'].lower() for entry in dictionary_data]
        
        print(f"Dictionary loaded with {len(self.all_headwords)} words:")
        print(self.all_headwords[:10])  # Print first 10 words for debugging
        
        # Create a mapping of keywords to headwords for better question understanding
        self.keyword_map = self._build_keyword_map()
    
    def _build_keyword_map(self):
        """Build a map of keywords to headwords for better question answering"""
        keyword_map = {}
        
        for entry in self.dictionary_data:
            headword = entry['kichwa']
            definition = entry['maana'].lower()
            
            # Map based on definition keywords
            if 'anayefundisha' in definition or 'anafundisha' in definition:
                keyword_map['teacher'] = headword
                keyword_map['mwalimu'] = headword
                keyword_map['teaches'] = headword
                keyword_map['anafundisha'] = headword
                keyword_map['who teaches'] = headword
                keyword_map['nani anafundisha'] = headword
                keyword_map['nani anayefundisha'] = headword
            
            if 'anasoma' in definition or 'anayesoma' in definition:
                keyword_map['student'] = headword
                keyword_map['mwanafunzi'] = headword
                keyword_map['learns'] = headword
                keyword_map['studies'] = headword
            
            if 'rafiki' in headword.lower() or 'friend' in definition:
                keyword_map['friend'] = headword
                keyword_map['rafiki'] = headword
            
            if 'chakula' in headword.lower() or 'food' in definition:
                keyword_map['food'] = headword
                keyword_map['chakula'] = headword
            
            if 'kitabu' in headword.lower() or 'book' in definition:
                keyword_map['book'] = headword
                keyword_map['kitabu'] = headword
            
            if 'mgeni' in headword.lower() or 'guest' in definition or 'visitor' in definition:
                keyword_map['guest'] = headword
                keyword_map['visitor'] = headword
                keyword_map['mgeni'] = headword
            
        return keyword_map
    
    def detect_language(self, text):
        """Figure out if someone's asking in Swahili or English"""
        text = text.lower()
        
        # Common Swahili words and question markers
        swahili_markers = [
            'ni', 'ya', 'wa', 'na', 'cha', 'vya', 'nini', 'maana',
            'tafsiri', 'ufafanuzi', 'gani', 'je', 'nani', 'wapi', 'lini',
            'kwa', 'cha', 'vya', 'mwa', 'ndani', 'katika', 'anayefundisha',
            'anafundisha', 'anasoma', 'anakula'
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
        en_markers = ['what', 'how', 'define', 'meaning', 'definition', 'call', 'word', 'who', 'where', 'when', 'which']
        en_count = sum(1 for marker in en_markers if marker in text)
        
        # Decide based on which count is higher
        if sw_count > en_count:
            return 'sw', sw_count / max(len(words), 1)
        else:
            return 'en', en_count / max(len(words), 1)
    
    def extract_possible_word(self, query):
        """Try to extract a possible dictionary word from the query"""
        query_lower = query.lower()
        words = re.findall(r'\b\w+\b', query_lower)  # Extract words, removing punctuation
        
        # First, check if any whole word matches a headword
        for word in words:
            if word in self.all_headwords_lower:
                # Find the original case version
                for hw in self.all_headwords:
                    if hw.lower() == word:
                        print(f"Found direct word match: {hw}")
                        return hw
        
        # If no exact match, return None - don't return the misspelled word
        return None
    
    def get_spelling_suggestions(self, query, cutoff=0.6):
        """Get spelling suggestions for misspelled words in the query"""
        if not query or len(query) < 2:
            return []
        
        # Extract all words from the query
        words = re.findall(r'\b\w+\b', query.lower())
        print(f"\n--- Getting suggestions for words: {words} ---")
        
        all_suggestions = []
        
        for word in words:
            if len(word) < 3:  # Skip very short words
                continue
                
            # Skip if the word already exists in dictionary
            if word in self.all_headwords_lower:
                continue
                
            print(f"  Checking misspelled word: '{word}'")
            
            # Get close matches from headwords
            try:
                suggestions = get_close_matches(word, self.all_headwords_lower, n=2, cutoff=cutoff)
                print(f"    Found suggestions: {suggestions}")
                
                # Convert back to original case
                for sugg in suggestions:
                    for hw in self.all_headwords:
                        if hw.lower() == sugg:
                            if hw not in all_suggestions:
                                all_suggestions.append(hw)
                                break
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        # Remove duplicates and return top 3
        unique_suggestions = []
        seen = set()
        for sugg in all_suggestions:
            if sugg.lower() not in seen:
                seen.add(sugg.lower())
                unique_suggestions.append(sugg)
        
        print(f"Final suggestions: {unique_suggestions[:3]}")
        return unique_suggestions[:3]
    
    def understand_question(self, query):
        """Try to understand what the question is asking"""
        query_lower = query.lower()
        
        # Handle "Who teaches in school?" type questions
        if 'who teaches' in query_lower:
            print("Detected 'who teaches' question")
            return {'headword': 'mwalimu', 'confidence': 0.95, 'type': 'direct'}
        
        if 'who teaches in a school' in query_lower or 'who teaches at school' in query_lower:
            print("Detected 'who teaches in school' question")
            return {'headword': 'mwalimu', 'confidence': 0.95, 'type': 'direct'}
        
        # Handle "What does X mean?" type questions
        match = re.search(r'what does (\w+) mean', query_lower)
        if match:
            word = match.group(1)
            print(f"Detected 'what does {word} mean' question")
            # Check if this word is in our dictionary
            if word in self.all_headwords_lower:
                for hw in self.all_headwords:
                    if hw.lower() == word:
                        return {'headword': hw, 'confidence': 0.9, 'type': 'direct'}
        
        # Swahili version
        if 'nani anayefundisha' in query_lower or 'nani anafundisha' in query_lower:
            print("Detected Swahili 'who teaches' question")
            return {'headword': 'mwalimu', 'confidence': 0.95, 'type': 'direct'}
        
        # Handle "What is food?" type questions
        if 'what is food' in query_lower or 'define food' in query_lower:
            print("Detected 'what is food' question")
            return {'headword': 'chakula', 'confidence': 0.9, 'type': 'direct'}
        
        # Handle "Maana ya X" type questions
        match = re.search(r'maana ya (\w+)', query_lower)
        if match:
            word = match.group(1)
            print(f"Detected 'maana ya {word}' question")
            if word in self.all_headwords_lower:
                for hw in self.all_headwords:
                    if hw.lower() == word:
                        return {'headword': hw, 'confidence': 0.9, 'type': 'direct'}
        
        # Check keyword map
        for keyword, headword in self.keyword_map.items():
            if keyword in query_lower:
                print(f"Keyword match: '{keyword}' -> '{headword}'")
                return {'headword': headword, 'confidence': 0.85, 'type': 'keyword'}
        
        return None
    
    def retrieve(self, query, k=3):
        """Pull the most relevant entries for a query"""
        print(f"\n=== Processing query: '{query}' ===")
        
        # First, figure out what language they're using
        lang, confidence = self.detect_language(query)
        print(f"Detected language: {lang} (confidence: {confidence:.2f})")
        
        # Try to understand the question directly
        understood = self.understand_question(query)
        if understood:
            print(f"Understood question: {understood}")
            headword = understood['headword']
            direct_matches = [{
                'headword': headword,
                'metadata': {'kichwa': headword},
                'relevance': understood['confidence'],
                'direct_match': True
            }]
            found_exact = True
            results = direct_matches[:k]
            possible_word = headword
            spelling_suggestions = []
        else:
            print("No direct question understanding")
            
            # Try to extract a possible word (only if it exists in dictionary)
            possible_word = self.extract_possible_word(query)
            print(f"Possible word extracted: {possible_word}")
            
            # Check for direct headword match first
            direct_matches = []
            for headword in self.index:
                if headword.lower() in query.lower():
                    print(f"Direct headword match found: {headword}")
                    direct_matches.append({
                        'headword': headword,
                        'metadata': {'kichwa': headword},
                        'relevance': 0.95,
                        'direct_match': True
                    })
            
            # If we found direct matches, use those
            if direct_matches:
                results = direct_matches[:k]
                found_exact = True
                spelling_suggestions = []
                print(f"Using direct matches: {[r['headword'] for r in results]}")
            else:
                print("No direct matches, trying TF-IDF search...")
                # Otherwise do a proper search with a minimum relevance threshold
                search_results = self.search_engine.search(query, k=k*2, min_relevance=0.15)
                print(f"TF-IDF found {len(search_results)} results")
                
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
                print(f"After filtering: {len(results)} results")
                
                # Get spelling suggestions if no results or weak results
                spelling_suggestions = []
                if not found_exact or (results and results[0]['relevance'] < 0.2):
                    print("No good results found, generating spelling suggestions...")
                    spelling_suggestions = self.get_spelling_suggestions(query, cutoff=0.7)
                else:
                    spelling_suggestions = []
        
        return {
            'query': query,
            'language': lang,
            'confidence': confidence,
            'retrieved': results,
            'found_exact': found_exact,
            'possible_word': possible_word,
            'spelling_suggestions': spelling_suggestions,
            'understood': understood
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
        spelling_suggestions = retrieval_result['spelling_suggestions']
        understood = retrieval_result['understood']
        
        print(f"\nGenerating response - found_exact: {found_exact}, suggestions: {spelling_suggestions}")
        
        # If we understood the question directly
        if understood and understood['type'] == 'direct':
            best = retrieved[0]
            entry = self.get_entry(best['headword'])
            if entry:
                if lang == 'sw':
                    return self._generate_swahili_response(entry, best, query)
                else:
                    return self._generate_english_response(entry, best, query)
        
        # If we have spelling suggestions, show them prominently
        if spelling_suggestions:
            return self._generate_spelling_suggestions_response(query, spelling_suggestions, lang)
        
        # If no results found at all
        if not retrieved:
            return self._generate_not_found_response(query, lang)
        
        # If results are below relevance threshold (very weak matches)
        if retrieved[0]['relevance'] < 0.2 and not retrieved[0].get('direct_match', False):
            return self._generate_weak_match_response(query, retrieved, lang)
        
        # Grab the best match
        best = retrieved[0]
        entry = self.get_entry(best['headword'])
        
        if not entry:
            return self._generate_not_found_response(query, lang)
        
        # Respond in the same language they asked in
        if lang == 'sw':
            response = self._generate_swahili_response(entry, best, query)
        else:
            response = self._generate_english_response(entry, best, query)
        
        return response
    
    def _generate_spelling_suggestions_response(self, query, suggestions, lang):
        """Generate response with spelling suggestions"""
        if lang == 'sw':
            response = f"""
❌ **Neno '{query}' halipatikani kwenye kamusi**
The word '{query}' was not found in the dictionary

**Je, ulimaanisha mojawapo ya maneno haya?**
"""
            for sugg in suggestions:
                response += f"• **{sugg}**\n"
            
            response += """

**Mifano ya maswali sahihi:**
• Maana ya rafiki ni nini?
• Tafsiri ya chakula
• Nani anayefundisha shuleni?
"""
        else:
            response = f"""
❌ **Word '{query}' not found in the dictionary**
Neno '{query}' halipatikani kwenye kamusi

**Did you mean one of these words?**
"""
            for sugg in suggestions:
                response += f"• **{sugg}**\n"
            
            response += """

**Examples of valid questions:**
• What does rafiki mean?
• Define chakula
• Who teaches in a school?
"""
        
        return response
    
    def _generate_not_found_response(self, query, lang):
        """Generate a message when word isn't found and no suggestions"""
        if lang == 'sw':
            response = f"""
❌ **Neno '{query}' halipatikani kwenye kamusi**
The word '{query}' was not found in the dictionary

**Mapendekezo:**
• Angalia tahajia - check your spelling
• Tumia neno moja tu - use a single word
• Jaribu neno la msingi - try the basic word form

**Mifano ya maswali sahihi:**
• Maana ya rafiki ni nini?
• Tafsiri ya chakula
• Nani anayefundisha shuleni?
"""
        else:
            response = f"""
❌ **Word '{query}' not found in the dictionary**
Neno '{query}' halipatikani kwenye kamusi

**Suggestions:**
• Check your spelling
• Use a single word
• Try the basic word form

**Examples of valid questions:**
• What does rafiki mean?
• Define chakula
• Who teaches in a school?
"""
        
        return response
    
    def _generate_weak_match_response(self, query, retrieved, lang):
        """Generate response when matches are weak"""
        if lang == 'sw':
            response = f"""
⚠️ **Neno '{query}' halilingani vizuri na maingizo yoyote**
The word '{query}' doesn't closely match any dictionary entries

**Mapendekezo:**
• Angalia tahajia - check your spelling
• Tumia neno moja tu - use a single word
• Jaribu neno la msingi - try the basic word form

**Maingizo yaliyo karibu zaidi:**
"""
            for i, item in enumerate(retrieved[:2], 1):
                response += f"{i}. **{item['headword']}** - {item['metadata']['maana'][:80]}...\n"
        
        else:
            response = f"""
⚠️ **Word '{query}' doesn't closely match any dictionary entries**
Neno '{query}' halilingani vizuri na maingizo yoyote

**Suggestions:**
• Check your spelling
• Use a single word
• Try the basic word form

**Closest matches:**
"""
            for i, item in enumerate(retrieved[:2], 1):
                response += f"{i}. **{item['headword']}** - {item['metadata']['maana'][:80]}...\n"
        
        return response
    
    def _generate_swahili_response(self, entry, retrieval_info, original_query=None):
        """Put together a Swahili answer"""
        headword = entry['kichwa']
        definition = entry['maana']
        examples = entry.get('mfano', [])
        
        # Check if this is a question about who teaches
        if original_query and ('nani anafundisha' in original_query.lower() or 'nani anayefundisha' in original_query.lower()):
            response = f"""
### 📖 {headword} (mwalimu)
**Aina:** {entry.get('aina', 'nomino')}

**Mwalimu ni:** {definition}
"""
        else:
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
    
    def _generate_english_response(self, entry, retrieval_info, original_query=None):
        """Put together an English answer"""
        headword = entry['kichwa']
        definition = entry['maana']
        examples = entry.get('mfano', [])
        
        # For questions like "Who teaches in school?"
        if original_query and ('who teaches' in original_query.lower()):
            response = f"""
### 📖 {headword} (teacher)
**Part of Speech:** {entry.get('aina', 'noun')}

**A teacher is:** {definition}

**Examples:**\n"""
            for ex in examples[:2]:
                response += f"• {ex}\n"
            
            if 'matumizi' in entry:
                response += f"\n**Usage:** {entry['matumizi']}\n"
            
            if 'kisawe' in entry and entry['kisawe']:
                response += f"\n**Synonyms:** {', '.join(entry['kisawe'])}\n"
            
            response += f"\n---\n*Source: Swahili Dictionary*"
            
            return response
        
        # For questions like "What does X mean?"
        if original_query and 'what does' in original_query.lower() and 'mean' in original_query.lower():
            response = f"""
### 📖 {headword}
**Part of Speech:** {entry.get('aina', 'noun')}

**Meaning:** {definition}

**Examples:**\n"""
            for ex in examples[:2]:
                response += f"• {ex}\n"
            
            if 'matumizi' in entry:
                response += f"\n**Usage:** {entry['matumizi']}\n"
            
            if 'kisawe' in entry and entry['kisawe']:
                response += f"\n**Synonyms:** {', '.join(entry['kisawe'])}\n"
            
            response += f"\n---\n*Source: Swahili Dictionary*"
            
            return response
        
        # Default response format
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