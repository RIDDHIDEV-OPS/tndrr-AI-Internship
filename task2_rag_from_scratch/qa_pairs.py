"""
qa_pairs.py
-----------
15 ground-truth QA pairs derived from the Wikipedia "Artificial Intelligence"
article (and related pages fetched by document_loader.py).

Each entry contains:
  question        – the natural-language question
  expected_answer – a concise correct answer
  source_passage  – the verbatim text excerpt that supports the answer
                    (used to compute recall@k during evaluation)
  topic           – thematic tag for analysis
"""

QA_PAIRS: list[dict] = [
    # -----------------------------------------------------------------------
    # 1. Definition
    # -----------------------------------------------------------------------
    {
        "id": 1,
        "topic": "definition",
        "question": "What is artificial intelligence?",
        "expected_answer": (
            "Artificial intelligence is the simulation of human intelligence processes "
            "by machines, especially computer systems, including learning, reasoning, "
            "and self-correction."
        ),
        "source_passage": (
            "Artificial intelligence (AI) is the simulation of human intelligence processes "
            "by machines, especially computer systems. Specific applications of AI include "
            "expert systems, natural language processing, speech recognition and machine vision."
        ),
    },
    # -----------------------------------------------------------------------
    # 2. Turing Test
    # -----------------------------------------------------------------------
    {
        "id": 2,
        "topic": "history",
        "question": "Who proposed the Turing Test and what does it measure?",
        "expected_answer": (
            "Alan Turing proposed the Turing Test in 1950 as a measure of a machine's "
            "ability to exhibit intelligent behaviour indistinguishable from that of a human."
        ),
        "source_passage": (
            "Alan Turing proposed the Turing test as a measure of machine intelligence in his "
            "1950 paper 'Computing Machinery and Intelligence'. The test checks whether a human "
            "evaluator can distinguish between a human and a machine on the basis of their "
            "written responses."
        ),
    },
    # -----------------------------------------------------------------------
    # 3. Founding of AI
    # -----------------------------------------------------------------------
    {
        "id": 3,
        "topic": "history",
        "question": "When was the term 'artificial intelligence' coined and by whom?",
        "expected_answer": (
            "The term 'artificial intelligence' was coined by John McCarthy in 1956 at the "
            "Dartmouth Conference, which is considered the founding event of AI as a field."
        ),
        "source_passage": (
            "The term 'artificial intelligence' was coined by John McCarthy, who organized the "
            "Dartmouth Conference in 1956. This conference is widely considered the founding "
            "event of artificial intelligence as a field of research."
        ),
    },
    # -----------------------------------------------------------------------
    # 4. Machine Learning definition
    # -----------------------------------------------------------------------
    {
        "id": 4,
        "topic": "machine_learning",
        "question": "What is machine learning and how does it differ from traditional programming?",
        "expected_answer": (
            "Machine learning is a subset of AI that enables systems to learn from data "
            "and improve their performance without being explicitly programmed. Unlike "
            "traditional programming, rules are inferred from data rather than hand-coded."
        ),
        "source_passage": (
            "Machine learning (ML) is a type of artificial intelligence that allows software "
            "applications to become more accurate at predicting outcomes without being explicitly "
            "programmed to do so. Machine learning algorithms use historical data as input to "
            "predict new output values."
        ),
    },
    # -----------------------------------------------------------------------
    # 5. Deep Learning
    # -----------------------------------------------------------------------
    {
        "id": 5,
        "topic": "deep_learning",
        "question": "What are neural networks and what role do they play in deep learning?",
        "expected_answer": (
            "Neural networks are computing systems inspired by the biological neural networks "
            "in animal brains. Deep learning uses many-layered neural networks to learn "
            "hierarchical representations of data, enabling breakthroughs in image and speech "
            "recognition."
        ),
        "source_passage": (
            "Deep learning is part of a broader family of machine learning methods based on "
            "artificial neural networks with representation learning. Learning can be "
            "supervised, semi-supervised or unsupervised. Deep learning architectures such as "
            "deep neural networks, recurrent neural networks, and convolutional neural networks "
            "have been applied to fields including computer vision, speech recognition, natural "
            "language processing, and audio recognition."
        ),
    },
    # -----------------------------------------------------------------------
    # 6. Natural Language Processing
    # -----------------------------------------------------------------------
    {
        "id": 6,
        "topic": "nlp",
        "question": "What is natural language processing used for?",
        "expected_answer": (
            "Natural language processing (NLP) is used to enable computers to understand, "
            "interpret, and generate human language for tasks such as translation, sentiment "
            "analysis, chatbots, and text summarization."
        ),
        "source_passage": (
            "Natural language processing (NLP) is a subfield of linguistics, computer science, "
            "and artificial intelligence concerned with the interactions between computers and "
            "human language, in particular how to program computers to process and analyze "
            "large amounts of natural language data. The goal is to enable computers to "
            "understand documents and the contextual nuances of the language within them."
        ),
    },
    # -----------------------------------------------------------------------
    # 7. AI winters
    # -----------------------------------------------------------------------
    {
        "id": 7,
        "topic": "history",
        "question": "What were the AI winters and why did they occur?",
        "expected_answer": (
            "AI winters were periods of reduced funding and interest in AI research, occurring "
            "in the 1970s and late 1980s, primarily because AI systems failed to meet overly "
            "optimistic expectations and practical applications were limited."
        ),
        "source_passage": (
            "The term 'AI winter' refers to a period of reduced funding and interest in "
            "artificial intelligence research. The first AI winter occurred in the 1970s when "
            "funding was cut after researchers failed to deliver on their ambitious promises. "
            "A second AI winter followed in the late 1980s and early 1990s when expert systems "
            "fell out of favour."
        ),
    },
    # -----------------------------------------------------------------------
    # 8. Expert Systems
    # -----------------------------------------------------------------------
    {
        "id": 8,
        "topic": "expert_systems",
        "question": "What is an expert system and what was DENDRAL?",
        "expected_answer": (
            "An expert system is an AI program that emulates the decision-making ability of a "
            "human expert using a knowledge base and an inference engine. DENDRAL was an early "
            "expert system developed at Stanford that identified chemical compounds."
        ),
        "source_passage": (
            "An expert system is a computer system that emulates the decision-making ability of "
            "a human expert. Expert systems are designed to solve complex problems by reasoning "
            "through bodies of knowledge, represented mainly as if-then rules. DENDRAL was one "
            "of the first expert systems, developed at Stanford University to identify unknown "
            "organic molecules by analyzing their mass spectra."
        ),
    },
    # -----------------------------------------------------------------------
    # 9. Computer Vision
    # -----------------------------------------------------------------------
    {
        "id": 9,
        "topic": "computer_vision",
        "question": "What is computer vision and what are its main applications?",
        "expected_answer": (
            "Computer vision is an AI field that trains computers to interpret and understand "
            "visual information from images and videos. Main applications include facial "
            "recognition, autonomous vehicles, medical imaging, and object detection."
        ),
        "source_passage": (
            "Computer vision is an interdisciplinary scientific field that deals with how "
            "computers can gain high-level understanding from digital images or videos. From "
            "the perspective of engineering, it seeks to understand and automate tasks that the "
            "human visual system can do. Applications include facial recognition, autonomous "
            "vehicles, medical image analysis, and augmented reality."
        ),
    },
    # -----------------------------------------------------------------------
    # 10. Robotics and AI
    # -----------------------------------------------------------------------
    {
        "id": 10,
        "topic": "robotics",
        "question": "How is artificial intelligence used in robotics?",
        "expected_answer": (
            "AI enables robots to perceive their environment through sensors, make decisions, "
            "plan actions, and learn from experience, allowing autonomous operation in "
            "manufacturing, surgery, exploration, and domestic settings."
        ),
        "source_passage": (
            "Robotics is a branch of computer science and engineering that involves the "
            "conception, design, manufacture, and operation of robots. AI is used in robotics "
            "to give robots the ability to perform tasks autonomously by perceiving their "
            "environment, reasoning about it, and acting upon it. AI-powered robots are used "
            "in manufacturing, surgery, space exploration, and household tasks."
        ),
    },
    # -----------------------------------------------------------------------
    # 11. AI safety
    # -----------------------------------------------------------------------
    {
        "id": 11,
        "topic": "ai_safety",
        "question": "What is AI safety and why is it considered important?",
        "expected_answer": (
            "AI safety is the research field focused on ensuring that AI systems behave as "
            "intended and do not cause unintended harm. It is important because advanced AI "
            "systems could have far-reaching societal impacts if misaligned with human values."
        ),
        "source_passage": (
            "AI safety is a research area that focuses on ensuring artificial intelligence "
            "systems behave in ways that are safe and beneficial for humans. As AI systems "
            "become more powerful, it becomes increasingly important to ensure they act in "
            "accordance with human values and intentions. Researchers study topics such as "
            "value alignment, robustness, and preventing harmful unintended consequences."
        ),
    },
    # -----------------------------------------------------------------------
    # 12. Narrow vs General AI
    # -----------------------------------------------------------------------
    {
        "id": 12,
        "topic": "definition",
        "question": "What is the difference between narrow AI and artificial general intelligence (AGI)?",
        "expected_answer": (
            "Narrow AI (or weak AI) is designed to perform a specific task, like playing chess "
            "or recognising faces. AGI refers to a hypothetical AI that can perform any "
            "intellectual task a human can, exhibiting general-purpose intelligence."
        ),
        "source_passage": (
            "Artificial intelligence can be classified as either narrow or general. Narrow AI "
            "(also called weak AI) is designed to perform a narrow task such as facial "
            "recognition or internet searches. Artificial general intelligence (AGI), also "
            "called strong AI, is a hypothetical AI that could perform any intellectual task "
            "that a human being can and would have a broad range of cognitive abilities."
        ),
    },
    # -----------------------------------------------------------------------
    # 13. Training data and bias
    # -----------------------------------------------------------------------
    {
        "id": 13,
        "topic": "ethics",
        "question": "How can biased training data affect AI systems?",
        "expected_answer": (
            "Biased training data causes AI models to replicate and amplify societal biases, "
            "leading to unfair outcomes in areas like hiring, lending, and criminal justice "
            "when the model makes predictions about under-represented groups."
        ),
        "source_passage": (
            "AI systems can inherit biases present in their training data. When historical "
            "data reflects societal inequalities or prejudices, models trained on that data "
            "can perpetuate and even amplify those biases. This can lead to discriminatory "
            "outcomes in applications such as hiring algorithms, credit scoring, and predictive "
            "policing, disproportionately affecting minority or under-represented groups."
        ),
    },
    # -----------------------------------------------------------------------
    # 14. Reinforcement Learning
    # -----------------------------------------------------------------------
    {
        "id": 14,
        "topic": "machine_learning",
        "question": "What is reinforcement learning and how does an agent learn in this paradigm?",
        "expected_answer": (
            "Reinforcement learning is a type of machine learning where an agent learns by "
            "interacting with an environment, receiving rewards for correct actions and "
            "penalties for incorrect ones, gradually discovering optimal strategies."
        ),
        "source_passage": (
            "Reinforcement learning (RL) is an area of machine learning concerned with how "
            "intelligent agents ought to take actions in an environment in order to maximise "
            "cumulative reward. The agent learns through trial and error, receiving positive "
            "reinforcement for good actions and negative reinforcement or penalties for bad "
            "actions, without being explicitly told which actions to take."
        ),
    },
    # -----------------------------------------------------------------------
    # 15. Transformers and large language models
    # -----------------------------------------------------------------------
    {
        "id": 15,
        "topic": "deep_learning",
        "question": "What is a transformer model and why did it revolutionise NLP?",
        "expected_answer": (
            "A transformer is a deep learning architecture introduced in 2017 that uses "
            "self-attention mechanisms to process sequential data in parallel. It revolutionised "
            "NLP by enabling the training of large language models like BERT and GPT that "
            "achieve state-of-the-art results across many tasks."
        ),
        "source_passage": (
            "The transformer architecture, introduced in the 2017 paper 'Attention Is All You "
            "Need', relies on self-attention mechanisms to model relationships between all "
            "positions in a sequence simultaneously, allowing much greater parallelism than "
            "recurrent neural networks. Transformers became the foundation for large language "
            "models such as BERT and the GPT series, achieving state-of-the-art performance "
            "on a wide range of natural language processing tasks."
        ),
    },
]


if __name__ == "__main__":
    print(f"Total QA pairs: {len(QA_PAIRS)}\n")
    for qa in QA_PAIRS:
        print(f"[{qa['id']:>2}] ({qa['topic']}) {qa['question']}")
