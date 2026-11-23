"""Generate an Introduction-to-Generative-AI PPT for students with basic ML background."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
import os

OUT = os.path.join(os.path.dirname(__file__), "Introduction_to_GenAI.pptx")

# Theme colors
PRIMARY = RGBColor(0x0F, 0x3D, 0x6E)   # deep blue
ACCENT = RGBColor(0x18, 0xA9, 0x99)    # teal
DARK = RGBColor(0x1F, 0x1F, 0x1F)
LIGHT = RGBColor(0xF4, 0xF6, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

BLANK = prs.slide_layouts[6]


def add_background(slide, color=LIGHT):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid(); bg.fill.fore_color.rgb = color
    bg.line.fill.background()
    bg.shadow.inherit = False
    return bg


def add_header_bar(slide, title, subtitle=None):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.0))
    bar.fill.solid(); bar.fill.fore_color.rgb = PRIMARY
    bar.line.fill.background()
    tb = slide.shapes.add_textbox(Inches(0.4), Inches(0.15), Inches(12.5), Inches(0.7))
    tf = tb.text_frame; tf.margin_left = 0
    p = tf.paragraphs[0]; p.text = title
    p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = WHITE
    if subtitle:
        sb = slide.shapes.add_textbox(Inches(0.4), Inches(1.05), Inches(12.5), Inches(0.4))
        sp = sb.text_frame.paragraphs[0]; sp.text = subtitle
        sp.font.size = Pt(14); sp.font.italic = True; sp.font.color.rgb = ACCENT


def add_bullets(slide, bullets, left=0.6, top=1.7, width=12.0, height=5.4, size=18):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame; tf.word_wrap = True
    for i, item in enumerate(bullets):
        if isinstance(item, tuple):
            text, level = item
        else:
            text, level = item, 0
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("• " if level == 0 else "   – ") + text
        p.font.size = Pt(size - level * 2)
        p.font.color.rgb = DARK
        p.space_after = Pt(6)
        if level == 0:
            p.font.bold = False


def add_footer(slide, page_no, total):
    tb = slide.shapes.add_textbox(Inches(11.5), Inches(7.05), Inches(1.7), Inches(0.3))
    p = tb.text_frame.paragraphs[0]
    p.text = f"Intro to Gen AI  |  {page_no}/{total}"
    p.font.size = Pt(10); p.font.color.rgb = PRIMARY


# -------- Slide 1: Title --------
s = prs.slides.add_slide(BLANK)
add_background(s, PRIMARY)
# Decorative band
band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(5.3), prs.slide_width, Inches(0.15))
band.fill.solid(); band.fill.fore_color.rgb = ACCENT; band.line.fill.background()

tb = s.shapes.add_textbox(Inches(0.7), Inches(2.3), Inches(12), Inches(1.5))
p = tb.text_frame.paragraphs[0]
p.text = "Introduction to Generative AI"
p.font.size = Pt(54); p.font.bold = True; p.font.color.rgb = WHITE

tb2 = s.shapes.add_textbox(Inches(0.7), Inches(3.6), Inches(12), Inches(0.8))
p2 = tb2.text_frame.paragraphs[0]
p2.text = "From Machine Learning to Models that Create"
p2.font.size = Pt(26); p2.font.color.rgb = ACCENT; p2.font.italic = True

tb3 = s.shapes.add_textbox(Inches(0.7), Inches(5.7), Inches(12), Inches(0.6))
p3 = tb3.text_frame.paragraphs[0]
p3.text = "A beginner-friendly session for ML students"
p3.font.size = Pt(18); p3.font.color.rgb = WHITE


slides_spec = [
    # 2
    ("Agenda", "What we'll cover today", [
        "What is Generative AI?",
        "Discriminative vs Generative models — the key difference",
        "A quick tour of generative model families",
        "Inside Large Language Models (LLMs)",
        "Prompting, embeddings & RAG",
        "Real-world applications",
        "Limitations, ethics & risks",
        "How to start learning + tools",
    ]),
    # 3
    ("Quick Recap: What is ML?", "You already know this — let's anchor it", [
        "Machine Learning = algorithms that learn patterns from data instead of being explicitly programmed.",
        ("Supervised learning: labeled data → predict y from x (regression, classification).", 1),
        ("Unsupervised learning: find structure (clustering, dimensionality reduction).", 1),
        ("Reinforcement learning: agent learns by reward signals.", 1),
        "Most classical ML is discriminative — it learns P(y | x).",
        "Generative AI extends ML to learn P(x) or P(x, y) — i.e., it learns the data itself, well enough to create new samples.",
    ]),
    # 4
    ("What is Generative AI?", "Models that create new content", [
        "Generative AI = models that produce new data (text, images, audio, code, video) similar to what they were trained on.",
        "It learns the underlying distribution of the training data and samples from it.",
        "Examples you've used:",
        ("ChatGPT / Gemini / Claude → text & code", 1),
        ("DALL·E / Stable Diffusion / Midjourney → images", 1),
        ("GitHub Copilot → code completion", 1),
        ("Suno / ElevenLabs → music & speech", 1),
        "Core idea: instead of choosing a label, the model generates the next token / pixel / sample.",
    ]),
    # 5
    ("Discriminative vs Generative", "The mental model shift", [
        "Discriminative model — learns the boundary between classes.",
        ("Question it answers: 'Given this email, is it spam?' → P(y|x)", 1),
        ("Examples: Logistic Regression, SVM, Random Forest, most CNN classifiers.", 1),
        "Generative model — learns how the data itself is produced.",
        ("Question it answers: 'What does a typical spam email look like?' → P(x) or P(x|y)", 1),
        ("Examples: Naive Bayes (classical), GANs, VAEs, Diffusion, LLMs.", 1),
        "Rule of thumb: Discriminative → predicts. Generative → creates.",
    ]),
    # 6
    ("A Brief History", "How we got here", [
        "1950s–1980s: Rule-based AI, early neural nets.",
        "2014: GANs (Goodfellow) — two networks compete to generate realistic images.",
        "2013–2017: VAEs and sequence-to-sequence models.",
        "2017: 'Attention Is All You Need' → the Transformer architecture.",
        "2018–2020: BERT, GPT-2, GPT-3 — scale starts to matter a lot.",
        "2022: ChatGPT brings Gen AI to mainstream.",
        "2023–2026: Multimodal models (text + image + audio + video), agents, on-device LLMs.",
    ]),
    # 7
    ("Families of Generative Models", "Know the main types", [
        "GANs (Generative Adversarial Networks): generator vs discriminator. Great for sharp images.",
        "VAEs (Variational Autoencoders): encode data into a latent space, then decode/sample. Smooth interpolation.",
        "Autoregressive models: predict next token given previous ones. Power LLMs (GPT family).",
        "Diffusion models: start from noise, gradually denoise into an image. Power Stable Diffusion, DALL·E 3.",
        "Flow-based models: invertible transformations, exact likelihoods.",
        "Transformers underlie most modern LLMs and many image/audio models.",
    ]),
    # 8
    ("Inside an LLM (1/2)", "Tokens, embeddings, attention", [
        "Text is split into tokens (sub-words). Example: 'generative' → ['gener', 'ative'].",
        "Each token is mapped to an embedding — a dense vector capturing meaning.",
        "Self-attention lets the model weigh the importance of every other token when processing one token.",
        "Stacked transformer blocks = deep contextual understanding.",
        "Training objective (most LLMs): predict the next token. Simple objective, huge data → emergent capabilities.",
    ]),
    # 9
    ("Inside an LLM (2/2)", "From pretraining to a helpful assistant", [
        "Stage 1 — Pretraining: next-token prediction on trillions of tokens (web, books, code).",
        "Stage 2 — Supervised fine-tuning (SFT): teach it to follow instructions using curated examples.",
        "Stage 3 — RLHF / RLAIF: human (or AI) feedback aligns answers with helpfulness & safety.",
        "Result: a base model becomes a chat assistant.",
        "Inference is just sampling tokens one at a time, conditioned on your prompt.",
    ]),
    # 10
    ("Prompting Basics", "How you 'program' an LLM", [
        "Prompt = the input you give the model. Better prompts → better outputs.",
        "Zero-shot: just ask. 'Summarize this paragraph.'",
        "Few-shot: show 2–3 examples of the task before asking.",
        "Chain-of-thought: ask the model to 'think step by step'.",
        "Role prompting: 'You are a data science tutor…'.",
        "Practical tips: be specific, give context, specify format (JSON, bullet points), set constraints.",
    ]),
    # 11
    ("Embeddings & Vector Search", "Meaning as geometry", [
        "Embedding = a vector representation of text/image/audio where similar things are close together.",
        "Cosine similarity measures how related two pieces of content are.",
        "Vector databases (FAISS, Pinecone, Chroma, Milvus) store millions of embeddings for fast similarity search.",
        "Use cases: semantic search, recommendations, deduplication, clustering documents.",
        "This is the foundation of RAG (next slide).",
    ]),
    # 12
    ("RAG — Retrieval-Augmented Generation", "Give LLMs access to your data", [
        "Problem: LLMs don't know your private documents and can hallucinate.",
        "Solution: Retrieve relevant chunks → inject them into the prompt → let the LLM answer.",
        "Pipeline:",
        ("1. Chunk documents and create embeddings.", 1),
        ("2. Store in a vector DB.", 1),
        ("3. At query time, retrieve top-k similar chunks.", 1),
        ("4. Pass them as context to the LLM along with the user's question.", 1),
        "Benefits: up-to-date answers, source citations, fewer hallucinations, no retraining needed.",
    ]),
    # 13
    ("Fine-tuning vs Prompting vs RAG", "When to use what", [
        "Prompting: fastest, no training, good for general tasks.",
        "RAG: when you need the model to use your own/up-to-date knowledge base.",
        "Fine-tuning: when you need a specific style, format, or domain behavior consistently.",
        ("LoRA / QLoRA: efficient fine-tuning that updates small adapter layers — cheap and fast.", 1),
        "Order of preference for most projects: Prompt → RAG → Fine-tune.",
    ]),
    # 14
    ("Multimodal & Agentic AI", "Beyond plain text", [
        "Multimodal models accept/produce multiple modalities: text + image + audio + video.",
        ("Examples: GPT-4o, Gemini, Claude with vision.", 1),
        "Agents = LLMs that can plan, use tools (search, code, APIs), and take actions to complete goals.",
        ("Frameworks: LangChain, LlamaIndex, AutoGen, CrewAI.", 1),
        "Function/tool calling lets the model decide when to call your Python functions or APIs.",
    ]),
    # 15
    ("Real-World Applications", "Where Gen AI shines", [
        "Chatbots, virtual assistants, customer support automation.",
        "Code generation, autocomplete, code review (GitHub Copilot).",
        "Content creation: marketing copy, summaries, translations.",
        "Image & video generation, design assistance.",
        "Healthcare: drug discovery, medical report drafting.",
        "Education: personalized tutors, explanation generation.",
        "Data tasks: synthetic data generation, document Q&A, BI assistants.",
    ]),
    # 16
    ("Limitations & Risks", "Use responsibly", [
        "Hallucinations: confidently wrong answers — always verify critical outputs.",
        "Bias: models reflect biases in training data.",
        "Privacy: never paste sensitive data into public models without policy checks.",
        "Cost & latency: large models are expensive to run at scale.",
        "Context window limits: models can only 'see' so many tokens at once.",
        "Intellectual property & copyright: still an evolving legal area.",
        "Security: prompt injection, data exfiltration, jailbreaks.",
    ]),
    # 17
    ("Ethics & Responsible AI", "Build with care", [
        "Transparency: tell users when they're interacting with AI.",
        "Fairness: test outputs across demographics; mitigate bias.",
        "Human-in-the-loop for high-stakes decisions (health, finance, legal).",
        "Guardrails: content filters, rate limits, output validation.",
        "Attribution: cite sources when using RAG; respect data licenses.",
        "Sustainability: be aware of compute & energy costs.",
    ]),
    # 18
    ("Tools & Ecosystem to Explore", "Hands-on starter pack", [
        "APIs / Models: OpenAI, Anthropic, Google Gemini, Mistral, Meta Llama, Hugging Face.",
        "Local: Ollama, LM Studio, llama.cpp — run models on your laptop.",
        "Frameworks: LangChain, LlamaIndex, Haystack.",
        "Vector DBs: FAISS, Chroma, Pinecone, Weaviate, Milvus.",
        "Fine-tuning: Hugging Face Transformers + PEFT (LoRA), Unsloth.",
        "UI: Streamlit, Gradio, Next.js for quick demos.",
    ]),
    # 19
    ("Suggested Learning Path", "From here, in order", [
        "1. Solidify Python + NumPy + basic deep learning (PyTorch).",
        "2. Learn the Transformer architecture (read 'The Illustrated Transformer').",
        "3. Use the OpenAI / Hugging Face APIs — build a simple chatbot.",
        "4. Build a RAG app over your own PDFs.",
        "5. Try fine-tuning a small open-source model (LoRA on a 7B model).",
        "6. Explore agents and tool use.",
        "7. Read papers: Attention Is All You Need, GPT-3, InstructGPT, LLaMA, RAG.",
    ]),
    # 20
    ("Mini Project Ideas", "Try one this week", [
        "Personal Q&A bot over your class notes (RAG with Chroma + a small LLM).",
        "Resume tailoring assistant (prompt engineering + structured output).",
        "Image caption generator using a vision-language model.",
        "Code explainer extension that summarizes selected code.",
        "SQL assistant that converts natural language to queries with validation.",
        "Study-flashcard generator from a textbook chapter.",
    ]),
    # 21
    ("Key Takeaways", "Remember these", [
        "Gen AI = ML models that create new data, not just classify it.",
        "Transformers + scale + alignment = today's LLMs.",
        "Prompting → RAG → Fine-tuning: pick the simplest tool that works.",
        "Always evaluate, validate, and watch for hallucinations & bias.",
        "The fastest way to learn is to build small projects end-to-end.",
    ]),
]

# Closing slide is added after
total = len(slides_spec) + 2  # title + content + closing

for idx, (title, subtitle, bullets) in enumerate(slides_spec, start=2):
    s = prs.slides.add_slide(BLANK)
    add_background(s, LIGHT)
    add_header_bar(s, title, subtitle)
    add_bullets(s, bullets)
    add_footer(s, idx, total)

# -------- Closing slide --------
s = prs.slides.add_slide(BLANK)
add_background(s, PRIMARY)
band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(5.3), prs.slide_width, Inches(0.15))
band.fill.solid(); band.fill.fore_color.rgb = ACCENT; band.line.fill.background()

tb = s.shapes.add_textbox(Inches(0.7), Inches(2.6), Inches(12), Inches(1.5))
p = tb.text_frame.paragraphs[0]
p.text = "Thank You!"
p.font.size = Pt(64); p.font.bold = True; p.font.color.rgb = WHITE

tb2 = s.shapes.add_textbox(Inches(0.7), Inches(3.9), Inches(12), Inches(0.8))
p2 = tb2.text_frame.paragraphs[0]
p2.text = "Questions, ideas, project pitches — let's discuss."
p2.font.size = Pt(24); p2.font.color.rgb = ACCENT

prs.save(OUT)
print(f"Saved: {OUT}")
