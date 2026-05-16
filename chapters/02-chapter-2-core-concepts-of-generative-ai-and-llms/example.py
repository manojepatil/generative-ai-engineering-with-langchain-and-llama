import torch
import torch.nn as nn
import math

# This simplified code illustrates core concepts:
# 1. Transformer architecture (Attention Mechanism)
# 2. Sequential processing (though simplified, represents token-by-token)
# 3. Embedding (mapping input to a dense vector space)

# --- 1. Embedding Layer ---
# Represents converting input tokens (words) into numerical vectors.
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embedding.embedding_dim) # Scale by sqrt(d_model) for stability

# --- 2. Positional Encoding ---
# Adds information about the position of tokens in the sequence, crucial for non-recurrent models.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # Add batch dimension

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# --- 3. Simplified Self-Attention Mechanism (Core of Transformer) ---
# Allows each token to weigh the importance of other tokens in the sequence.
class SimpleSelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        # Scaled Dot-Product Attention: QK^T / sqrt(d_k)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.d_model)
        attention_weights = self.softmax(scores)
        output = torch.matmul(attention_weights, values)
        return output

# --- 4. Simulated LLM Forward Pass (combining components) ---
class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=10):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.attention = SimpleSelfAttention(d_model)
        # Output layer to get logits for next token prediction
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.attention(x)
        output_logits = self.output_layer(x)
        return output_logits

# --- Example Usage ---
if __name__ == "__main__":
    vocab_size = 10000 # Number of unique words/tokens
    d_model = 512      # Dimension of the model's internal representation
    max_seq_len = 5    # Max sequence length for our example

    # Simulate an input sequence (e.g., "hello world tokens")
    # Batch size 1, sequence length max_seq_len
    input_sequence = torch.randint(0, vocab_size, (1, max_seq_len))
    print(f"Input sequence (token IDs):\n{input_sequence}\n")

    model = SimpleLLM(vocab_size, d_model, max_len=max_seq_len)

    # Perform a forward pass
    output_logits = model(input_sequence)

    print(f"Output logits shape (batch, sequence_length, vocab_size):\n{output_logits.shape}\n")
    print(f"First token's output logits (probabilities for next word):\n{output_logits[0, 0, :10].detach().numpy()}...\n") # Display first 10 logits

    # To get the predicted next token for each position:
    predicted_tokens = torch.argmax(output_logits, dim=-1)
    print(f"Predicted next token IDs for each position in the sequence:\n{predicted_tokens}\n")
    print("This simple model demonstrates embedding, positional encoding, and self-attention,")
    print("which are foundational to modern LLMs like Transformers.")
