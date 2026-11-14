**Function calling** (also known as tool-calling) is a method by which models can reliably connect and interact with external tools or APIs.
**LLM** is a type of artificial intelligence (AI) program trained on massive amounts of text data to understand, summarize, translate, and generate human-like language.
**Tokenization** is the process of breaking raw text down into smaller pieces (words, sub-words, or characters) so a computer can map them into numerical IDs and vector embeddings.
**Neural network** is a computer system modeled loosely after the human brain that learns to recognize patterns and make decisions.
**BPE tokenizer** is a subword tokenization algorithm used by large language models (LLMs) like GPT to break raw text into smaller, manageable units (tokens) before processing.
**AI agent** is an autonomous software system that uses an artificial intelligence perceive a goal, plan steps, use external tools, and execute actions on its own.

LLM is two files: file of weights (tons of floats) and a short runner program.
Parameter weights file acts as a compressed representation of knowledge.
Runtime file executes the mathematical operations to process tokens.

---

How the prompt is beeing processed:

1. Tokenization
Text is broken into tokens, which can be words, parts of words, or characters.
Tokenizer splits input string into sub-word units using a lookup table which 
typically contain tens of thousands to over a hundred thousand unique tokens.
Tokenization quirks lead to specific failure modes in spelling, math and coding

Example mapping with BPE tokenization:
		Text input: "How do tokens look like?"
		Token chunks: ["How", " do", " tokens", " look", " like", "?"]
		Token IDs (Integers): [4486, 466, 16337, 1445, 588, 30]

The LLM never reads raw text strings like "How". It only receives a sequence of
integers: [4486, 466, 16337, 1445, 588, 30].

2. Vector Embedding Lookup
		Runtime program reads sequence [4486, 466, 16337, 1445, 588, 30] and 
		convert
		floating-point numbers) using weights file (parameters). 

Example embedded representation:
		Token ID 4486 ("How") -> [0.0312, -0.1421, 0.0891, ..., -0.0432]
		Token ID 466 (" do") -> [-0.1042, 0.2214, 0.4120, ..., 0.0811]
		Token ID 16337 (" tokens") -> [0.0142, 0.4109, 0.0031, ..., -0.1944]
		Token ID 1445 (" look") -> [0.1294, -0.3204, 0.1552, ..., 0.2103]
		Token ID 588 (" like") -> [-0.2105, 0.1198, -0.0784, ..., -0.0152]
		Token ID 30 ("?") -> [0.0012, -0.0041, 0.0023, ..., -0.0019]

3. Neural Network Processing
		Runtime script passes numerical vectors through consecutive neural 
		network layers. At each layer, matrix multiplication operations combine
		vector data across tokens to determine context and meaning.

4. Output Token Generation
		The final layer outputs a probability distribution over entire vocabulary 
		(32,000 to 128,000 possible token id's). The final vector goes through 
		a linear layer to match the vocabulary size. The raw output numbers 
		are called logits.

Example probability output for the next token:
		Token ID 2574 ("Tokens"): 82%
		Token ID 466 ("do"): 5%
		Token ID 682 ("They"): 3%

		The runtime picks an token ID based on the highest probability, decodes 
		integer 2574 back into text string "Tokens", displays it and appends 
		integer 2574 to the input sequence to generate the next token.

//////// GOTO DOWNLOADS

https://geeksforgeeks.org/artificial-intelligence/function-calling-in-llms
