# Transformers Architecture: An End-to-End Story for Students

## Class Goal

By the end of this lesson, students should be able to explain:

1. Why transformers were invented.
2. How text becomes numbers.
3. What self-attention does.
4. Why Query, Key, and Value exist.
5. Why multi-head attention is powerful.
6. What happens inside an encoder block and decoder block.
7. How models like BERT, GPT, and T5 are built from the same idea.
8. How a small transformer-style project flows from input sentence to prediction.

---

# 1. The Story Begins: A Student Reading a Sentence

Imagine a student reads this sentence:

```text
The movie was not good
```

A weak model may see the word `good` and say: positive sentiment.

But a smart student notices the word `not` before `good`, and understands:

```text
not good = negative meaning
```

This is the real NLP problem:

A word does not have complete meaning alone. A word gets meaning from the other words around it.

Earlier models struggled with this.

---

# 2. Before Transformers: Why Older Models Were Not Enough

## Bag of Words

Bag of Words counts words but forgets order.

```text
Dog bites man
Man bites dog
```

Bag of Words may see the same words in both sentences, but the meaning is completely different.

Problem:

```text
It knows which words exist, but not how they are connected.
```

## RNN / LSTM

RNNs read one word at a time.

```text
The -> movie -> was -> not -> good
```

This is closer to how humans read, but it has problems:

1. Slow, because words are processed one after another.
2. Long sentences are hard because early information can fade.
3. It is difficult to train very large RNNs efficiently.

Then came the transformer.

---

# 3. Transformer Big Idea

The transformer asks a beautiful question:

```text
For every word, which other words should I pay attention to?
```

In the sentence:

```text
The movie was not good
```

When understanding `good`, the model should pay strong attention to `not`.

So the transformer does not simply read left to right like an RNN. It allows every token to look at other tokens directly.

This mechanism is called:

```text
Self-Attention
```

Self-attention means:

```text
Words inside the same sentence attend to each other.
```

---

# 4. Full Transformer Pipeline

At a high level:

```text
Raw Text
   -> Tokenization
   -> Token IDs
   -> Token Embeddings
   -> Positional Encoding
   -> Transformer Blocks
   -> Output Layer
   -> Prediction
```

Now we will walk through every piece.

---

# 5. Tokenization: Breaking Text into Pieces

Computers cannot directly understand text. First we split text into tokens.

Sentence:

```text
Transformers are powerful
```

Tokens:

```text
["Transformers", "are", "powerful"]
```

Modern models often use subword tokens.

Example:

```text
unhappiness -> ["un", "happiness"]
playing -> ["play", "ing"]
```

Why subwords?

1. Handles unknown words better.
2. Reduces vocabulary size.
3. Works across many word forms.

After tokenization, tokens are mapped to IDs.

```text
"Transformers" -> 45
"are"          -> 12
"powerful"     -> 89
```

So the sentence becomes:

```text
[45, 12, 89]
```

---

# 6. Embeddings: Turning Tokens into Meaningful Numbers

A token ID is only a label. It does not contain meaning.

So each token ID is converted into a vector called an embedding.

Example:

```text
"king"  -> [0.21, 0.80, -0.33, ...]
"queen" -> [0.25, 0.78, -0.29, ...]
"car"   -> [-0.71, 0.10, 0.44, ...]
```

Words with related meanings get similar vectors.

Important idea:

```text
Embedding = meaning in number form
```

it like a location on a map:

```text
king and queen are nearby
car and petrol are nearby
pizza and cheese are nearby
```

---

# 7. Positional Encoding: Giving Order to the Model

Self-attention can compare all words at once. That is powerful, but it creates a new problem.

If the model only sees tokens together, it may not know the order.

These two sentences use the same words:

```text
Dog bites man
Man bites dog
```

So we add position information.

```text
Token Embedding + Positional Encoding = Final Input Vector
```

Position tells the model:

```text
This is word 1
This is word 2
This is word 3
```

There are two common approaches:

1. Fixed sinusoidal positional encoding.
2. Learned positional embeddings.

Student-friendly version:

```text
Embedding tells what the word means.
Position tells where the word stands.
```

---

# 8. Self-Attention: The Heart of Transformers

Self-attention answers this question for each token:

```text
Which other tokens are important for understanding me?
```

Sentence:

```text
The animal did not cross the road because it was tired
```

What does `it` refer to?

The model should attend to `animal`.

Sentence:

```text
The movie was not good
```

When understanding `good`, the model should attend to `not`.

So attention creates relationships between words.

---

# 9. Query, Key, and Value: The Classroom Analogy

Every token creates three vectors:

```text
Query
Key
Value
```

Use this analogy:

A student enters a library.

1. Query = what the student is looking for.
2. Key = labels on books that say what each book contains.
3. Value = the actual information inside each book.

In transformer language:

```text
Query asks a question.
Key answers: do I match this question?
Value gives the information if matched.
```

For every word:

```text
Query = what am I searching for?
Key   = what information do I offer?
Value = my actual content
```

---

# 10. Attention Score

For a word, the model compares its Query with every other word's Key.

```text
Attention Score = Query dot Key
```

If Query and Key match strongly, attention score is high.

Example:

```text
good -> attends strongly to -> not
```

Then scores are scaled:

```text
score / sqrt(dimension)
```

Why scale?

Large vector dimensions can create very large dot products. Scaling keeps training stable.

Then softmax converts scores into probabilities.

```text
Raw scores       -> [2.0, 1.0, 0.1]
After softmax    -> [0.66, 0.24, 0.10]
```

Now the model knows how much attention to give each token.

---

# 11. Weighted Sum of Values

After softmax, the model multiplies attention weights with Value vectors.

```text
Output = attention weights x Value vectors
```

Student-friendly sentence:

```text
The model mixes information from important words into the current word.
```

For `good`, the final representation may include strong information from `not`.

That is how meaning changes from:

```text
good
```

to:

```text
not good
```

---

# 12. Self-Attention Formula

The famous formula is:

```text
Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k)) V
```

Break it down:

1. QK^T compares every Query with every Key.
2. Divide by sqrt(d_k) to stabilize values.
3. Softmax converts scores into attention weights.
4. Multiply by V to collect useful information.

This is the mathematical center of transformers.

---

# 13. Multi-Head Attention: Many Students Looking Differently

One attention head may learn one type of relationship.

Examples:

1. Head 1: subject-verb relationship.
2. Head 2: negation relationship.
3. Head 3: pronoun-reference relationship.
4. Head 4: adjective-noun relationship.

Instead of one attention mechanism, transformers use many attention heads in parallel.

```text
Multi-Head Attention = many attention views at the same time
```

Class analogy:

One student reads for grammar.
One student reads for sentiment.
One student reads for who refers to whom.
One student reads for important keywords.

Then all students combine their understanding.

---

# 14. Feed Forward Network: Thinking After Attention

After attention mixes information, each token goes through a small neural network.

This is called:

```text
Feed Forward Network
```

It usually has two linear layers with an activation function in between.

```text
Linear -> ReLU/GELU -> Linear
```

Attention allows tokens to talk to each other.
Feed Forward Network processes each token's updated meaning more deeply.

Simple explanation:

```text
Attention gathers context.
Feed Forward Network thinks about the gathered context.
```

---

# 15. Residual Connections: Do Not Forget the Original

Deep networks can lose useful information while passing through many layers.

So transformers use residual connections.

```text
output = layer(input) + input
```

This means:

```text
Keep the old information and add the new information.
```

Why useful?

1. Helps gradients flow during training.
2. Makes deep models easier to train.
3. Prevents useful original signal from disappearing.

---

# 16. Layer Normalization: Keeping Values Stable   

As data passes through layers, values can become too large or unstable.

Layer normalization keeps values balanced.

Student-friendly analogy:

```text
Before moving to the next classroom, everyone lines up in a neat range.
```

In model terms:

```text
LayerNorm stabilizes activations so training is smoother.
```

---

# 17. One Encoder Block

A transformer encoder block looks like this:

```text
Input
  -> Multi-Head Self-Attention
  -> Add + LayerNorm
  -> Feed Forward Network
  -> Add + LayerNorm
  -> Output
```

Encoder blocks are stacked many times.

```text
Encoder Block 1
Encoder Block 2
Encoder Block 3
...
```

The final output contains context-aware representations of every token.

Used in:

```text
BERT-style models
```

Best for:

1. Text classification.
2. Named entity recognition.
3. Semantic search.
4. Question answering over given text.

---

# 18. One Decoder Block

A decoder block is used for text generation.

It has:

```text
Masked Self-Attention
Add + LayerNorm
Cross-Attention
Add + LayerNorm
Feed Forward Network
Add + LayerNorm
```

The important new piece is masked attention.

---

# 19. Masked Self-Attention: No Looking Into the Future

When generating text, the model predicts the next token.

Example:

```text
I love machine ____
```

Input --> I Love this ____   output -->  Machine   [model should predict this]

I love this Machine
I Love
I love this 


The model should not see the answer before predicting it.

So decoder self-attention uses a mask:

```text
Token 1 can see token 1
Token 2 can see token 1 and 2
Token 3 can see token 1, 2, and 3
Token 4 cannot see future tokens
```

This prevents cheating during training.

Used in:

```text
GPT-style models
```

Best for:

1. Chatbots.
2. Story generation.
3. Code generation.
4. Autocomplete.

---

# 20. Cross-Attention: Decoder Looking at Encoder

In encoder-decoder models, the encoder reads the input and the decoder generates output.

Example translation:

```text
Input:  I love India
Output: J'aime l'Inde
```

Encoder reads the English sentence.
Decoder generates the French sentence.

Cross-attention allows decoder tokens to look at encoder outputs.

```text
Decoder Query attends to Encoder Keys and Values
```

Used in:

```text
T5, original Transformer, translation models
```

---

# 21. Output Layer and Softmax

After transformer blocks, the model produces final vectors.

For classification:

```text
Final vector -> classifier -> class probabilities
```

Example sentiment:

```text
Positive: 0.10
Negative: 0.90
```

For generation:

```text
Final vector -> vocabulary scores -> softmax -> next token probability
```

Example:

```text
I love machine ____

learning: 0.72
translation: 0.12
pizza: 0.01
```

The model chooses the next token and continues.

---

# 22. Training a Transformer

Transformers learn by prediction.

## GPT-style training

Predict the next token.

```text
Input:  I love machine
Target: learning

Input --> I   output --> love
Input --> I love   output --> machine
Input --> I love machine  output --> learning

```

## BERT-style training

Predict masked words.

```text
Input:  I love [MASK] learning
Target: machine

Input --> I [MASK] machine  output --> love
Input --> I love [MASK] learning output --> machine
Input --> I love machine [MASK] output --> learning


```
## Encoder-decoder training

Generate target sequence from input sequence.

```text
Input:  English sentence
Target: French sentence
```

Training loop:

```text
Prediction -> Loss -> Backpropagation -> Update weights
```

---

# 23. BERT vs GPT vs T5

| Model family | Architecture | Main behavior | Example tasks |
|---|---|---|---|
| BERT | Encoder only | Understand text | Classification, NER, QA |
| GPT | Decoder only | Generate text | Chat, writing, code |
| T5 | Encoder-decoder | Convert input text to output text | Translation, summarization |

Simple memory trick:

```text
BERT reads deeply.
GPT writes forward.
T5 reads then writes.
```

---

# 24. Complete End-to-End Example

Sentence:

```text
The movie was not good
```

Step by step:

1. Tokenization

```text
["The", "movie", "was", "not", "good"]
```

2. Token IDs

```text
[10, 52, 31, 80, 44]
```

3. Embeddings

```text
Each token becomes a vector.

[[1,2,3,4,4],[1,2,3,4,4],[1,2,3,3,4],[1,2,3,4,4],[1,2,3,4,4]]

```

4. Positional encoding

```text
The model learns token order.
```

5. Self-attention

```text
"good" attends strongly to "not".
```

6. Multi-head attention

```text
Different heads capture grammar, sentiment, relationships.
```

7. Feed forward network

```text
Each token representation is processed deeply.
```

8. Classification head

```text
The model predicts sentiment.
```

9. Output

```text
Negative
```

---

# 25. Teaching Board Diagram

Use this diagram on the board:

```text
Sentence
  |
  v
Tokenizer
  |
  v
Token IDs
  |
  v
Embedding + Position
  |
  v
+----------------------------------+
| Transformer Block                |
|                                  |
| Multi-Head Self-Attention        |
| Add + LayerNorm                  |
| Feed Forward Network             |
| Add + LayerNorm                  |
+----------------------------------+
  |
  v
Output Head
  |
  v
Prediction
```

---

# 26. Classroom Explanation Script

You can explain it like this:

A transformer is like a classroom discussion. Every word is a student. Each student first knows only itself. Then self-attention lets every student ask: who should I listen to so I can understand my role in this sentence?

For the sentence `The movie was not good`, the word `good` listens carefully to `not`. The meaning changes. Multi-head attention is like having several groups of students. One group studies grammar, one group studies sentiment, one group studies long-distance relationships, and another group studies important keywords.

After the discussion, each student writes a better version of their understanding. The feed forward network refines that understanding. Residual connections make sure old knowledge is not lost, and layer normalization keeps the values stable.

After many such blocks, the model has a rich understanding of the sentence. Then a final output layer uses that understanding to classify, generate, translate, summarize, or answer questions.

---

# 27. Key Takeaways

1. Tokenization changes text into pieces.
2. Embeddings change pieces into vectors.
3. Positional encoding adds word order.
4. Self-attention connects words to other important words.
5. Query, Key, and Value are the search system inside attention.
6. Multi-head attention learns multiple relationship types.
7. Feed forward layers refine each token.
8. Residual connections preserve information.
9. Layer normalization stabilizes training.
10. Encoder models understand text.
11. Decoder models generate text.
12. Encoder-decoder models transform one sequence into another.

---

# 28. Mini Project Idea

Use the included class demo project:

```text
transformer_attention_class_demo.html
```

Students type a sentence, click through the pipeline, and observe:

1. Tokens.
2. Embedding plus position vectors.
3. Attention weights.
4. Multi-head attention behavior.
5. Final class prediction.

Suggested class activity:

1. Type: `the movie was good`
2. Type: `the movie was not good`
3. Compare attention on `good`.
4. Ask students why `not` changes the prediction.
5. Show how different heads focus on different relationships.
