LLM's run on a two-file architecture:

1. Weights file: which is massive binary file containing parameters in form of
floating numbers. It's lossy, compressed represenation of internet text.
2. Runtime program: executes matrix multiplications to calculate token 
probabilities

---

How the prompt is beeing processed:

1. Tokenization
Tokenizer splits input string into sub-word units using a lookup table.

Example mapping (BPE tokenization):
		Text input: "How do tokens look like?"
		Token chunks: ["How", " do", " tokens", " look", " like", "?"]
		Token IDs (Integers): [4486, 466, 16337, 1445, 588, 30]

The LLM never reads raw text strings like "How". It only receives a sequence of
integers: [4486, 466, 16337, 1445, 588, 30].

2. Vector Embedding Lookup
		Runtime program reads sequence [4486, 466, 16337, 1445, 588, 30] and 
		converts each integer into a high-dimensional vector (a long array of 
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
