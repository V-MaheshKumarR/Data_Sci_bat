"""Append properly-structured Tokenization → Word2Vec cells to the NLP notebook."""
import json
import uuid
from pathlib import Path

NB_PATH = Path(r"d:\Data Science_bat\NLP\Deeplearning_NLP_notebook.ipynb")

def code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in src.rstrip("\n").split("\n")],
    }

def md_cell(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "source": [line + "\n" for line in src.rstrip("\n").split("\n")],
    }

new_cells = [
    md_cell("""# ── Step: Tokenization ──────────────────────────────────────────
# Tokenization = splitting raw text into smaller units called **tokens**.
#   • **Word tokenization**     → break a sentence into individual words
#   • **Sentence tokenization** → break a paragraph into individual sentences
# NLTK tokenizers are smarter than `.split()` — they handle punctuation,
# contractions like "don't", abbreviations, etc."""),

    code_cell("""# Download the tokenizer models (one-time download)
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize, sent_tokenize

# Use the FIRST raw review from our corpus as the demo document
sample_text = corpus[0]
sample_text"""),

    code_cell("""# Sentence tokenization → list of sentences
sentences = sent_tokenize(sample_text)
print("Number of sentences:", len(sentences))
sentences"""),

    code_cell("""# Word tokenization → list of word/punctuation tokens
words = word_tokenize(sample_text)
print("Number of word tokens:", len(words))
words"""),

    md_cell("""# ── Step: Normalization (Stemming & Lemmatization) ──────────────
# Normalization reduces different forms of a word to a single base form
# so the model treats them as the same token.
#
#   • **Stemming**      → rule-based suffix chopping
#                         "playing", "played", "plays" → "play"
#                         Fast but may produce non-real words ("studies" → "studi").
#
#   • **Lemmatization** → dictionary-based, returns the actual root word (lemma)
#                         "better" → "good",  "running" → "run"
#                         Slower but more linguistically accurate."""),

    code_cell("""# Stemming with the Porter algorithm (most common stemmer)
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

# Clean the sample first (lowercase + remove special chars + stopwords),
# then split into words so we stem only meaningful tokens.
cleaned_words = text_cleaner(sample_text).split()

stemmed_words = [stemmer.stem(w) for w in cleaned_words]
print("Original :", cleaned_words)
print("Stemmed  :", stemmed_words)"""),

    code_cell("""# Lemmatization with WordNet (returns real dictionary words)
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# pos='v' → treat tokens as verbs, gives better results for
# words like "running" → "run", "studies" → "study".
lemmatized_words = [lemmatizer.lemmatize(w, pos='v') for w in cleaned_words]
print("Original    :", cleaned_words)
print("Lemmatized  :", lemmatized_words)"""),

    md_cell("""# ── Step: POS Tagging (Parts-of-Speech) ─────────────────────────
# POS tagging assigns a grammatical category to each token:
#   NN  → Noun        VB → Verb       JJ → Adjective
#   RB  → Adverb      PRP → Pronoun   IN → Preposition   ... etc.
# Useful for understanding sentence structure and is a pre-requisite
# for tasks like NER, chunking, and better lemmatization."""),

    code_cell("""# Download the POS tagger model (one-time)
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

from nltk import pos_tag

# Tag each token from our word_tokenize output with its POS label
pos_tags = pos_tag(words)
pos_tags"""),

    md_cell("""# ── Step: Named Entity Recognition (NER) ────────────────────────
# NER identifies and classifies "named entities" inside text:
#   • PERSON       → "Elon Musk", "Kumar"
#   • ORGANIZATION → "Google", "NASA"
#   • GPE          → Geo-political entities ("India", "New York")
#   • DATE / TIME / MONEY / PERCENT  ... etc.
# We use NLTK's `ne_chunk` which works on the POS-tagged tokens."""),

    code_cell("""# Download NER models (one-time)
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

from nltk import ne_chunk

# ne_chunk takes POS-tagged tokens and returns a Tree where named
# entities are grouped sub-trees labelled with their entity type.
ner_tree = ne_chunk(pos_tags)
ner_tree"""),

    code_cell("""# Extract just the named entities (sub-trees) from the parse tree
named_entities = []
for chunk in ner_tree:
    # A sub-tree means NLTK detected a named entity
    if hasattr(chunk, 'label'):
        entity_text = " ".join(token for token, pos in chunk.leaves())
        named_entities.append((entity_text, chunk.label()))

print("Detected named entities:")
named_entities"""),

    md_cell("""# ── (Optional) Better NER with spaCy ────────────────────────────
# spaCy ships with a pre-trained statistical NER model that is usually
# more accurate than NLTK's. Run the next cell only if spaCy is available."""),

    code_cell("""# Uncomment if spaCy / the English model isn't installed yet.
# !pip install spacy
# !python -m spacy download en_core_web_sm

import spacy

# Load the small English pipeline (tokenizer + tagger + parser + NER)
nlp = spacy.load("en_core_web_sm")

doc = nlp(sample_text)

# Print every detected entity along with its type
for ent in doc.ents:
    print(f"{ent.text:<25} → {ent.label_}")"""),

    md_cell("""# ── Word2Vec — Teaching a Computer the *Meaning* of Words ───────
#
# **The big idea (in one line):**
# > Words that appear in **similar contexts** should have **similar vectors**.
#
# Computers don't understand text — they only understand numbers.
# Word2Vec converts every word into a list of numbers (a **vector**, e.g. 100 numbers long)
# such that words with related meanings end up close together in that number-space.
#
# ### A tiny intuition example
# Look at these sentences:
# ```
# 1)  The king ruled the kingdom.
# 2)  The queen ruled the kingdom.
# 3)  The man walked to the market.
# 4)  The woman walked to the market.
# ```
# - "king" and "queen" appear with similar neighbours (`ruled`, `kingdom`)
#   → Word2Vec gives them similar vectors.
# - "man" and "woman" appear with similar neighbours (`walked`, `market`)
#   → similar vectors too.
#
# ### The famous magic trick (vector arithmetic)
# Once trained, Word2Vec vectors can do **math on meaning**:
#
# $$ \\text{king} - \\text{man} + \\text{woman} \\approx \\text{queen} $$
#
# That's the moment most beginners say *"whoa"* — the model learned the concept
# of gender / royalty purely from reading sentences, without anyone telling it.
#
# ### Two flavours of Word2Vec
# | Variant   | What it predicts                                  | Analogy                          |
# |-----------|---------------------------------------------------|----------------------------------|
# | **CBOW**  | Given the surrounding words, predict the middle word | Fill-in-the-blank               |
# | **Skip-gram** | Given the middle word, predict the surrounding words | Guess the neighbours            |
#
# We'll train a tiny Skip-gram model below using **gensim**."""),

    code_cell("""# Install gensim if it's not already available (Colab usually has it)
# !pip install gensim

from gensim.models import Word2Vec"""),

    code_cell("""# ── Step 1: Build a tiny toy corpus ─────────────────────────────
# Each sentence is a list of words (already tokenized & lowercased).
# Notice how "king/queen" share contexts, and "man/woman" share contexts.
toy_sentences = [
    ["the", "king",  "ruled", "the", "kingdom"],
    ["the", "queen", "ruled", "the", "kingdom"],
    ["the", "king",  "loves", "the", "queen"],
    ["the", "man",   "walked", "to", "the", "market"],
    ["the", "woman", "walked", "to", "the", "market"],
    ["the", "man",   "loves", "the", "woman"],
    ["a",   "boy",   "is", "a", "young", "man"],
    ["a",   "girl",  "is", "a", "young", "woman"],
    ["the", "prince", "will", "be", "the", "king"],
    ["the", "princess", "will", "be", "the", "queen"],
]
toy_sentences[:3]"""),

    code_cell("""# ── Step 2: Train the Word2Vec model ────────────────────────────
# Key hyper-parameters explained:
#   sentences  → the training data (list of token lists)
#   vector_size→ how many numbers to use per word (the "embedding size")
#   window     → how many neighbour words count as "context" on each side
#   min_count  → ignore rare words that appear fewer than this many times
#   sg=1       → use Skip-gram (sg=0 would be CBOW)
#   epochs     → number of passes over the corpus
w2v = Word2Vec(
    sentences=toy_sentences,
    vector_size=50,
    window=2,
    min_count=1,
    sg=1,
    epochs=200,
    seed=42,
)

print("Vocabulary size:", len(w2v.wv))
print("First 10 words :", list(w2v.wv.index_to_key)[:10])"""),

    code_cell("""# ── Step 3: Look at the vector for a single word ────────────────
# This is the "embedding" — just an array of 50 numbers that
# encodes the meaning of the word "king" in this tiny universe.
king_vec = w2v.wv["king"]
print("Shape of king's vector:", king_vec.shape)
king_vec"""),

    code_cell("""# ── Step 4: Find the most similar words ─────────────────────────
# Word2Vec measures similarity using cosine similarity between vectors.
# Words used in similar contexts should rank at the top.
print("Most similar to 'king':")
for word, score in w2v.wv.most_similar("king", topn=5):
    print(f"   {word:<10} → similarity = {score:.3f}")

print("\\nMost similar to 'woman':")
for word, score in w2v.wv.most_similar("woman", topn=5):
    print(f"   {word:<10} → similarity = {score:.3f}")"""),

    code_cell("""# ── Step 5: Direct similarity between two words ─────────────────
# Higher number (closer to 1.0) = more similar in meaning.
print("similarity(king, queen)  =", round(w2v.wv.similarity("king", "queen"), 3))
print("similarity(king, market) =", round(w2v.wv.similarity("king", "market"), 3))"""),

    code_cell("""# ── Step 6: The famous analogy — vector arithmetic ──────────────
# king - man + woman  ≈  queen
# In gensim:
#   positive = words to ADD
#   negative = words to SUBTRACT
result = w2v.wv.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=3,
)

print("king - man + woman ≈ ?")
for word, score in result:
    print(f"   {word:<10} → {score:.3f}")"""),

    code_cell("""# ── Step 7: Spot the odd one out ────────────────────────────────
# Word2Vec can pick the word that doesn't belong in a group
# because its vector sits far from the others.
print("Odd one out from [king, queen, prince, market]:",
      w2v.wv.doesnt_match(["king", "queen", "prince", "market"]))"""),

    md_cell("""# ── Step 8: Visualise the word vectors in 2D ────────────────────
# Our vectors live in 50-dimensional space — impossible to draw.
# We use **PCA** to squash them down to 2 dimensions so we can plot them.
# Watch how related words cluster together!"""),

    code_cell("""from sklearn.decomposition import PCA

# Pick a subset of interesting words to visualise
words_to_plot = ["king", "queen", "prince", "princess",
                 "man", "woman", "boy", "girl",
                 "market", "kingdom"]

# Stack their vectors into a matrix of shape (10, 50)
vectors = np.array([w2v.wv[w] for w in words_to_plot])

# Reduce 50-D → 2-D using PCA
coords_2d = PCA(n_components=2, random_state=42).fit_transform(vectors)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(coords_2d[:, 0], coords_2d[:, 1], s=80, color="steelblue")

for word, (x, y) in zip(words_to_plot, coords_2d):
    plt.annotate(word, (x, y), fontsize=12,
                 xytext=(6, 4), textcoords="offset points")

plt.title("Word2Vec embeddings (PCA → 2D)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()"""),

    md_cell("""# ── Recap ───────────────────────────────────────────────────────
# 1. Word2Vec turns each word into a fixed-size vector of numbers.
# 2. It learns by reading sentences and noticing **which words appear together**.
# 3. After training:
#    - Similar words → similar vectors (`king` ≈ `queen`).
#    - Vector math captures relationships (`king − man + woman ≈ queen`).
#    - Plotting the vectors shows meaningful clusters.
#
# Real-world Word2Vec models are trained on **billions of words** (Wikipedia,
# Google News, etc.) and produce vectors of size 100–300. The intuition you
# just saw on a 10-sentence corpus scales up to those models exactly the same way."""),
]

with NB_PATH.open("r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Existing cells: {len(nb['cells'])}")
nb["cells"].extend(new_cells)
print(f"After append   : {len(nb['cells'])}")

with NB_PATH.open("w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done.")
