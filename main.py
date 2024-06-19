import time
import random
import unidecode
import string
import re
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import csv

# Setting seeds for reproducibility
RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Check if GPU is available and set the device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

# Hyperparameters
TEXT_PORTION_SIZE = 200
NUM_ITER = 4000  # Adjusted to 4000 iterations
LEARNING_RATE = 0.005
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
BATCH_SIZE = 32
NUM_LAYERS = 2
DROPOUT_RATE = 0.5

# Load dataset
with open('tinyshakespeare.txt', 'r') as f:
    textfile = f.read()

# Convert special characters and strip extra whitespaces
textfile = unidecode.unidecode(textfile)
textfile = re.sub(' +', ' ', textfile)
TEXT_LENGTH = len(textfile)
print(f'Number of characters in text: {TEXT_LENGTH}')

# Data augmentation: introduce noise into the text
def augment_text(text):
    text = list(text)
    if random.random() < 0.1:  # 10% chance to swap two characters
        idx1 = random.randint(0, len(text) - 2)
        idx2 = idx1 + 1
        text[idx1], text[idx2] = text[idx2], text[idx1]
    if random.random() < 0.1:  # 10% chance to delete a character
        del text[random.randint(0, len(text) - 1)]
    return ''.join(text)

# Function to get a random portion of text
def random_portion(textfile):
    start_index = random.randint(0, TEXT_LENGTH - TEXT_PORTION_SIZE)
    end_index = start_index + TEXT_PORTION_SIZE + 1
    text = textfile[start_index:end_index]
    return augment_text(text)

# Function to convert characters to tensor
def char_to_tensor(text):
    lst = [string.printable.index(c) for c in text]
    tensor = torch.tensor(lst).long()
    return tensor

# Function to draw a random sample of text with padding
def draw_random_sample(textfile, batch_size, seq_length):
    inputs = []
    targets = []
    for _ in range(batch_size):
        text_long = char_to_tensor(random_portion(textfile))
        inputs.append(text_long[:-1])
        targets.append(text_long[1:])
    
    # Pad sequences to the same length
    inputs = [torch.cat([seq, torch.full((seq_length - len(seq),), string.printable.index(' ')).long()]) if len(seq) < seq_length else seq for seq in inputs]
    targets = [torch.cat([seq, torch.full((seq_length - len(seq),), string.printable.index(' ')).long()]) if len(seq) < seq_length else seq for seq in targets]
    
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    return inputs, targets

# Define the RNN model with dropout for regularization
class RNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, num_layers=1, dropout_rate=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, character, hidden, cell_state):
        embedded = self.embed(character)
        embedded = self.dropout(embedded)
        output, (hidden, cell_state) = self.lstm(embedded, (hidden, cell_state))
        output = self.fc(output.contiguous().view(output.size(0) * output.size(1), -1))
        return output, hidden, cell_state
    
    def init_zero_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE))

# Initialize model, optimizer, and scheduler
model = RNN(len(string.printable), EMBEDDING_DIM, HIDDEN_DIM, len(string.printable), num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

# Function to evaluate the model
def evaluate(model, prime_str='A', predict_len=100, temperature=0.8):
    hidden, cell_state = model.init_zero_state(1)
    prime_input = char_to_tensor(prime_str).unsqueeze(0).to(DEVICE)
    predicted = prime_str

    for p in range(len(prime_str) - 1):
        inp = prime_input[:, p].unsqueeze(1)
        _, hidden, cell_state = model(inp, hidden, cell_state)
    
    inp = prime_input[:, -1].unsqueeze(1)
    
    for _ in range(predict_len):
        outputs, hidden, cell_state = model(inp, hidden, cell_state)
        output_dist = outputs.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        predicted_char = string.printable[top_i]
        predicted += predicted_char
        inp = char_to_tensor(predicted_char).unsqueeze(0).to(DEVICE)
    
    return predicted

# Additional performance metrics
def calculate_perplexity(loss):
    return torch.exp(loss).item()

def calculate_accuracy(predictions, targets):
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return correct / total

# Training loop with additional metrics and checkpointing
checkpoint_dir = 'model_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize CSV logging
with open('training_metrics.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Iteration', 'Loss', 'Perplexity', 'Accuracy'])

start_time = time.time()
loss_list = []
perplexity_list = []
accuracy_list = []

for iteration in range(NUM_ITER):
    model.train()
    hidden, cell_state = model.init_zero_state(BATCH_SIZE)
    optimizer.zero_grad()
    loss = 0
    total_accuracy = 0
    
    inputs, targets = draw_random_sample(textfile, BATCH_SIZE, TEXT_PORTION_SIZE)
    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
    
    for c in range(TEXT_PORTION_SIZE):
        outputs, hidden, cell_state = model(inputs[:, c].unsqueeze(1), hidden, cell_state)
        loss += nn.functional.cross_entropy(outputs, targets[:, c].view(-1))
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        accuracy = calculate_accuracy(predicted, targets[:, c].view(-1))
        total_accuracy += accuracy

    loss /= TEXT_PORTION_SIZE
    total_accuracy /= TEXT_PORTION_SIZE
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
    
    optimizer.step()
    scheduler.step()

    if iteration % 200 == 0 or iteration == NUM_ITER - 1:  # Ensure the last iteration is logged
        perplexity = calculate_perplexity(loss)
        print(f'Time elapsed: {(time.time() - start_time) / 60:.2f} min')
        print(f'Iteration {iteration} | Loss {loss.item():.2f} | Perplexity {perplexity:.2f} | Accuracy {total_accuracy:.2f}\n')
        print(evaluate(model, 'Th', 200), '\n')
        print('=' * 50)
        
        loss_list.append(loss.item())
        perplexity_list.append(perplexity)
        accuracy_list.append(total_accuracy)
        
        # Logging metrics to CSV
        with open('training_metrics.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([iteration, loss.item(), perplexity, total_accuracy])
        
        # Save model checkpoint
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'model_checkpoint_{iteration}.pth'))

# Final plots after training
plt.figure()
plt.plot(range(len(loss_list)), loss_list)
plt.ylim(bottom=0)  # Set y-axis bottom to 0 for loss
plt.ylabel('Loss')
plt.xlabel('Iteration x 200')
plt.title('Loss over Iterations')
plt.show()

plt.figure()
plt.plot(range(len(perplexity_list)), perplexity_list)
plt.ylabel('Perplexity')
plt.xlabel('Iteration x 200')
plt.title('Perplexity over Iterations')
plt.show()

plt.figure()
plt.plot(range(len(accuracy_list)), accuracy_list)
plt.ylabel('Accuracy')
plt.xlabel('Iteration x 200')
plt.title('Accuracy over Iterations')
plt.show()
