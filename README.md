[README.md](https://github.com/user-attachments/files/29197727/README.md)
# OFEG Model: Opinion-oriented Feature Enhanced Graph Neural Network

## Introduction
This project implements an Opinion-oriented Feature Enhanced Graph Neural Network (OFEG) for Aspect-Based Sentiment Analysis (ABSA). The model leverages graph neural networks (GNNs) to capture syntactic dependencies within sentences and incorporates emotion-aware features from lexicons to enhance sentiment prediction.

## Model Architecture
The OFEG model consists of the following key components:

### 1. GCNLayer
A Graph Convolutional Network (GCN) layer that propagates information across the syntactic graph of a sentence. It takes node features (embeddings + fuzzy emotions) and an adjacency matrix as input, applies a linear transformation, and an activation function.

```python
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
    def forward(self, x, adj):
        h = torch.mm(adj, x)
        return F.relu(self.linear(h))
```

### 2. OFEG (Overall Model)
The main OFEG model integrates the GCN layer with multi-head attention and a final classification layer. It processes the node features and adjacency matrix, applies GCN, then uses multi-head attention to capture dependencies between the nodes' hidden states, and finally pools the attention output for sentiment classification.

```python
class OFEG(nn.Module):
    def __init__(self, hidden_size=768, gcn_hidden=256, num_classes=3):
        super().__init__()
        self.gcn = GCNLayer(hidden_size+6, gcn_hidden)
        self.attn = nn.MultiheadAttention(embed_dim=gcn_hidden, num_heads=4)
        self.fc = nn.Linear(gcn_hidden, num_classes)
    def forward(self, x, adj):
        h = self.gcn(x, adj)
        h = h.unsqueeze(0)
        attn_out, _ = self.attn(h, h, h)
        pooled = attn_out.mean(dim=1)
        return self.fc(pooled)
```

## Setup and Data Loading

### Dependencies
Ensure you have the necessary libraries installed:

```python
!pip install torch transformers spacy networkx scikit-learn pandas lxml
!python -m spacy download en_core_web_sm
```

### Data Loading and Preprocessing
The dataset (e.g., SemEval 2014 Laptop/Restaurant datasets) is loaded from XML files and preprocessed. This involves extracting sentences, aspects, and polarity labels, and converting polarity into numerical labels (0: positive, 1: negative, 2: neutral/conflict). Sentences are converted to lowercase.

```python
def load_semeval_dataset(path):
    # ... (function definition as in notebook)

def preprocess_dataset(df):
    # ... (function definition as in notebook)

laptop_train = preprocess_dataset(load_semeval_dataset("SemEval14/Laptop_Train.xml"))
restaurant_train = preprocess_dataset(load_semeval_dataset("SemEval14/Restaurants_Train.xml"))
laptop_test = preprocess_dataset(load_semeval_dataset("SemEval14/Laptop_Test.xml"))
restaurant_test = preprocess_dataset(load_semeval_dataset("SemEval14/Restaurants_Test.xml"))
```

### Tokenization and Embeddings
RoBERTa tokenizer and model are used to generate contextualized embeddings for sentences.

```python
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta = RobertaModel.from_pretrained("roberta-base")

def get_embeddings(sentence):
    # ... (function definition as in notebook)
```

### Syntactic Graph Construction
spaCy is used to parse sentences and build a syntactic graph, which represents word dependencies. This graph is then converted into an adjacency matrix.

```python
nlp = spacy.load("en_core_web_sm")

def build_syntactic_graph(sentence):
    # ... (function definition as in notebook)
```

### Emotion Lexicon and Node Features
An NRC emotion lexicon is used to generate fuzzy emotion vectors for words. These emotion vectors are concatenated with RoBERTa embeddings to form comprehensive node features for the GCN.

```python
lexicon = pd.read_csv("NRC_emotion_lexicon.csv")
# ... (emotion_index, emotion_dict, fuzzy_emotion_vector definitions as in notebook)

def build_node_features(sentence):
    # ... (function definition as in notebook)
```

### Dataset and DataLoader
Custom `SemEvalDataset` and `DataLoader` classes are defined to prepare the data for training and evaluation, creating batches of node features, adjacency matrices, and labels.

```python
from torch.utils.data import Dataset, DataLoader

class SemEvalDataset(Dataset):
    # ... (class definition as in notebook)

train_dataset = SemEvalDataset(laptop_train)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataset = SemEvalDataset(laptop_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
```

## Model Training
The model is trained using AdamW optimizer and Cross-Entropy Loss. The training loop iterates over epochs, calculating loss, performing backpropagation, and updating model weights.

```python
model = OFEG()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    # ... (training loop as in notebook)
```

## Model Evaluation
The trained model is evaluated on the test set using accuracy and F1-score (macro average).

```python
def evaluate_model(model, dataloader):
    # ... (function definition as in notebook)

acc, f1 = evaluate_model(model, test_loader)
print(f"Laptop Test - Accuracy: {acc:.2f}, F1: {f1:.2f}")
```

## Usage
To use this model:
1.  **Prepare your environment**: Install the required libraries and download the spaCy English model.
2.  **Download SemEval Datasets**: Place the `Laptop_Train.xml`, `Restaurants_Train.xml`, `Laptop_Test.xml`, `Restaurants_Test.xml` files in a `SemEval14` directory in your working environment.
3.  **Download NRC Emotion Lexicon**: Place the `NRC_emotion_lexicon.csv` file in your working environment.
4.  **Run the notebook**: Execute the cells sequentially to load data, define the model, train, and evaluate.
5.  **Adapt for new data**: Modify `load_semeval_dataset` and `preprocess_dataset` to load your specific ABSA dataset, or adapt `build_node_features` and `build_syntactic_graph` for custom input formats.

This `README.md` provides a comprehensive overview of the OFEG model, its implementation, and how to use it within a Colab environment.
