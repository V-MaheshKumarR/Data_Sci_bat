# 📘 Introduction to Natural Language Processing (NLP)
###  Notes — Class 1

---

## 🎯 Learning Objectives
By the end of this class, you will be able to:
1. Understand what NLP is and where it is used.
2. Explain the difference between **Traditional NLP** and **Deep Learning NLP**.
3. Understand the basic building blocks of NLP (tokens, stopwords, stemming, etc.).
4. Understand the difference between **ANN** and **RNN** with diagrams.

---

## 1. 🤔 What is NLP?

**Natural Language Processing (NLP)** is a branch of Artificial Intelligence that helps computers **understand, interpret, and generate human language** (English, Hindi, Tamil, etc.).

> 👉 In simple words: **"Teaching computers to read, write, listen, and talk like humans."**

### Why is NLP hard?
Human language is:
- **Ambiguous** → "I saw her duck" (bird? or bending down?)
- **Context dependent** → "Apple" (fruit? or company?)
- **Full of sarcasm, slang, emojis** → "Great job 🙄"
- **Different in every language** → grammar, scripts, tone

---

## 2. 🌍 Real-World Applications of NLP

| Application | Example |
|---|---|
| 🔍 Search Engines | Google, Bing |
| 💬 Chatbots | ChatGPT, Alexa, Siri |
| 📧 Spam Detection | Gmail spam filter |
| 🌐 Translation | Google Translate |
| 📝 Autocomplete | Gmail "Smart Compose" |
| 😊 Sentiment Analysis | Amazon/Flipkart reviews |
| 📰 Text Summarization | News apps like Inshorts |
| 🎙️ Speech Recognition | Voice typing |

---

## 3. 🧱 Basic Building Blocks of NLP

Before any model can process text, we need to **preprocess** it.

```
Raw Text  →  Clean Text  →  Tokens  →  Numbers (Vectors)  →  Model
```

### 🔤 Key Terms

| Term | Meaning | Example |
|---|---|---|
| **Corpus** | A large collection of text | All Wikipedia articles |
| **Document** | One piece of text | One review, one email |
| **Token** | A single word / unit | "I love NLP" → ["I", "love", "NLP"] |
| **Vocabulary** | Unique tokens in corpus | {"I", "love", "NLP", "hate"} |
| **Stopwords** | Common words (remove them) | "is", "the", "a", "an" |
| **Stemming** | Cut word to root (rough) | "playing" → "play" |
| **Lemmatization** | Cut to dictionary root | "better" → "good" |
| **POS Tagging** | Label part-of-speech | "run" → verb |
| **NER** | Identify names/places | "Elon works at Tesla" → PERSON, ORG |

---

## 4. 🛤️ Two Approaches to NLP

There are **two major approaches** to solving NLP problems:

```
                    ┌────────────────────────────┐
                    │   NLP Approaches           │
                    └────────────┬───────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
     ┌────────▼─────────┐                ┌──────────▼──────────┐
     │  Traditional NLP │                │  Deep Learning NLP  │
     │  (Rule + ML)     │                │  (Neural Networks)  │
     └──────────────────┘                └─────────────────────┘
```

---

### 🏛️ A. Traditional Approach (Classical NLP)

Uses **hand-crafted rules** + **classical Machine Learning** algorithms.

#### Steps:
1. **Text Cleaning** — lowercase, remove punctuation, stopwords
2. **Tokenization** — split into words
3. **Stemming / Lemmatization**
4. **Feature Extraction** — convert text to numbers using:
   - **Bag of Words (BoW)**
   - **TF-IDF** (Term Frequency – Inverse Document Frequency)
   - **One-Hot Encoding**
5. **ML Model** — Naive Bayes, Logistic Regression, SVM, Random Forest

#### ✅ Pros
- Easy to understand and implement
- Works well on small datasets
- Fast to train
- Interpretable results

#### ❌ Cons
- Cannot understand **context** or **word order**
- "Dog bites man" = "Man bites dog" in BoW 😮
- Cannot capture meaning or semantics
- Poor with long sentences

---

### 🧠 B. Deep Learning Approach (Modern NLP)

Uses **Neural Networks** to learn language patterns automatically.

#### Steps:
1. Basic cleaning (minimal)
2. Tokenization
3. **Word Embeddings** — Word2Vec, GloVe, FastText, BERT
   - Words become **dense vectors** capturing meaning
   - Example: `king - man + woman ≈ queen`
4. **Deep Learning Model**:
   - **RNN / LSTM / GRU** — handle sequence
   - **Transformers** — BERT, GPT (state-of-the-art)

#### ✅ Pros
- Understands **context** and **meaning**
- Handles long text well
- State-of-the-art accuracy
- No manual feature engineering

#### ❌ Cons
- Needs **huge data**
- Needs **GPU / high compute**
- Hard to interpret (black box)
- Slow to train

---

### ⚖️ Traditional vs Deep Learning — Quick Comparison

| Feature | Traditional NLP | Deep Learning NLP |
|---|---|---|
| Data required | Small | Very large |
| Feature engineering | Manual (BoW, TF-IDF) | Automatic |
| Context understanding | ❌ No | ✅ Yes |
| Word order matters? | ❌ No | ✅ Yes |
| Speed | Fast | Slow |
| Hardware | CPU | GPU |
| Example Models | Naive Bayes, SVM | LSTM, BERT, GPT |
| Accuracy | Moderate | Very High |

---

## 5. 🧩 ANN vs RNN — The Key Difference

To understand modern NLP, we must know **why normal neural networks (ANN) don't work well for text**, and why **RNNs were invented**.

---

### 🧠 A. ANN (Artificial Neural Network)

- Input → Hidden Layers → Output
- Processes **all inputs at once**
- Has **no memory** of previous inputs
- Great for **images, tabular data**
- ❌ **Bad for sequences** (text, audio, time-series)

#### 🖼️ ANN Visualization

```
      Input Layer        Hidden Layer        Output
       (x1, x2, x3)      (neurons)           (y)

        x1 ●──────────┐
                       \
        x2 ●────────────●─────────● ───► y
                       /
        x3 ●──────────┘

   → All inputs go in at once
   → No concept of "time" or "sequence"
   → No memory
```

---

### 🔁 B. RNN (Recurrent Neural Network)

- Designed for **sequential data** (text, speech, time-series)
- Has a **memory** (hidden state) that carries info from previous steps
- Processes inputs **one at a time**, in order
- Perfect for NLP → word order matters!

#### 🖼️ RNN Visualization (Unrolled Over Time)

```
   Time →   t=1          t=2          t=3          t=4
            "I"          "love"       "learning"   "NLP"
             │             │            │            │
             ▼             ▼            ▼            ▼
           ┌───┐  h1    ┌───┐  h2    ┌───┐  h3    ┌───┐
   h0 ───► │RNN│──────► │RNN│──────► │RNN│──────► │RNN│────► Output
           └───┘        └───┘        └───┘        └───┘
             │             │            │            │
             ▼             ▼            ▼            ▼
            y1            y2           y3           y4

   → Same RNN cell reused at every time step
   → Hidden state (h) carries MEMORY of previous words
   → Word order is preserved
```

---

### 🔑 Core Difference — ANN vs RNN

| Feature | ANN | RNN |
|---|---|---|
| Data type | Fixed size (image, table) | Sequential (text, speech) |
| Memory | ❌ No memory | ✅ Has memory (hidden state) |
| Input processing | All at once | One step at a time |
| Word order | Ignored | Preserved |
| Parameter sharing | No | Yes (same weights across time) |
| Good for | Images, tabular | NLP, time-series, audio |
| Example use | Digit recognition | Language translation |

---

### 💡 Intuition with an Example

**Sentence:** _"I am not happy"_

- **ANN (with BoW):** Sees words {I, am, not, happy} → may predict **positive** because of "happy" ❌
- **RNN:** Reads "I" → "am" → "not" → "happy" in order → remembers "not" → correctly predicts **negative** ✅

> 👉 This is why RNNs (and later LSTMs, Transformers) revolutionized NLP!

---

## 6. 🚀 Evolution of NLP (Big Picture)

```
1950s ─── Rule-based systems (if-else rules)
        │
1990s ─── Statistical NLP (Naive Bayes, HMM)
        │
2000s ─── Classical ML + TF-IDF
        │
2013  ─── Word2Vec (word embeddings)
        │
2014  ─── RNN / LSTM (sequence models)
        │
2017  ─── Transformers ("Attention is All You Need")
        │
2018  ─── BERT (Google)
        │
2020+ ─── GPT-3, GPT-4, ChatGPT, LLaMA (Large Language Models)
```

---

## 7. 🛠️ Popular NLP Libraries in Python

| Library | Use |
|---|---|
| **NLTK** | Learning, classical NLP |
| **spaCy** | Production-grade NLP |
| **TextBlob** | Simple sentiment analysis |
| **Gensim** | Word2Vec, topic modeling |
| **scikit-learn** | ML models + TF-IDF |
| **Hugging Face Transformers** | BERT, GPT, modern models |
| **TensorFlow / PyTorch** | Deep learning models |

---

## 8. 📌 Summary (One-Liner Recap)

- **NLP** = Teaching computers human language.
- **Traditional NLP** = Rules + BoW/TF-IDF + classical ML → fast but no context.
- **Deep Learning NLP** = Embeddings + Neural Nets → understands context.
- **ANN** = No memory, for fixed data.
- **RNN** = Has memory, for sequences — perfect for text.
- **Modern NLP** = Transformers (BERT, GPT) → state of the art.

---

## 🎓 Homework / Practice

1. Write down 5 real-world apps where NLP is used daily.
2. Take the sentence `"The movie was not good at all"` and explain why BoW would fail.
3. Draw an RNN diagram for the sentence `"I love Python"`.
4. List 3 differences between ANN and RNN in your own words.

---

### 📚 Next Class Preview
➡️ **Text Preprocessing in Python** — Tokenization, Stopwords, Stemming, Lemmatization using NLTK and spaCy with hands-on code.

---
_Happy Learning! 🚀_
