import esm
import torch
import torch.nn as nn

# Load pretrained ESM-2 model
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

# Sample protein sequence
data = [("protein1", "MKTAYIAKQRQISFVKSHFSRQDILDLWQ")]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

# Extract embeddings
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False)
    embeddings = results["representations"][model.num_layers]

# embeddings shape: (batch_size, seq_len, embedding_dim)
embedding_dim = embeddings.size(-1)

# Define a simple classifier model (for demo, random weights)
class SimpleSSClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes=3):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, num_classes)
    def forward(self, x):
        return self.linear(x)

classifier = SimpleSSClassifier(embedding_dim)
classifier.eval()  # Weights are random here, normally you'd train this!

# Run classifier on embeddings
with torch.no_grad():
    logits = classifier(embeddings)  # shape: (batch_size, seq_len, num_classes)
    predictions = torch.argmax(logits, dim=-1)  # predicted class per residue

# Map prediction indices to secondary structure labels
ss_labels = {0: 'H', 1: 'E', 2: 'C'}  # Helix, Sheet, Coil (example mapping)

# Print predicted secondary structure for your sequence
predicted_ss = [ss_labels[int(x)] for x in predictions[0]]
print("Sequence: ", batch_strs[0])
print("Predicted SS:", "".join(predicted_ss))
