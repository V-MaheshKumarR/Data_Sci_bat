# Transformer Architecture Classroom Mini Project

This folder contains a complete teaching package for explaining transformer architecture to students.

## Files

| File | Purpose |
|---|---|
| `Transformers_End_to_End_Story.md` | Full beginner-friendly story explanation of every major transformer component |
| `Beginner_Transformer_Attention_Colab.ipynb` | Colab-ready beginner notebook that demonstrates self-attention with a tiny sentiment example |
| `transformer_attention_class_demo.html` | Interactive mini project for class demonstration |

## Mini Project: Explainable Toy Transformer

The HTML demo simulates a tiny transformer-style text classifier. It is intentionally small and visual, so students can see what happens at each stage.

The Colab notebook explains the same idea in code using NumPy, Pandas, and Matplotlib. It walks students through tokenization, embeddings, positional encoding, Query/Key/Value, self-attention, attention visualization, and a tiny sentiment output head.

Pipeline shown in the demo:

```text
Sentence
  -> Tokens
  -> Embedding + Position
  -> Self-Attention
  -> Multi-Head Attention
  -> Feed Forward / Output Head
  -> Sentiment Prediction
```

## Best Class Flow

1. Open `transformer_attention_class_demo.html` in a browser.
2. Start with this sentence:

```text
the movie was good
```

3. Then compare it with:

```text
the movie was not good
```

4. Ask students to observe which token receives attention from `good`.
5. Explain that this is the core reason transformers understand context better than Bag of Words.

## Important Teaching Note

This project is a toy educational simulator, not a trained production transformer. The weights are intentionally simple and visible so students can understand the idea before seeing PyTorch or Hugging Face code.

## Suggested Discussion Questions

1. Why does word order matter?
2. Why is `not good` different from `good`?
3. What does a Query search for?
4. What does a Key advertise?
5. What does a Value carry?
6. Why do we need multiple attention heads?
7. Which architecture is better for classification: encoder, decoder, or encoder-decoder?
8. Which architecture is better for chatbot-style generation?
