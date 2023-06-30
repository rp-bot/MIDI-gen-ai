import torch
from torch import nn
from torch.nn import functional as F


class CachedSelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead)

        # Cache for storing the key and value tensors
        self.key_cache = None
        self.value_cache = None

    def forward(self, x):
        # x: (S, N, E) where S is the source sequence length, N is the batch size, E is the feature number
        x = x.transpose(0, 1)  # Transpose x to (N, S, E)

        if self.key_cache is None:
            # This is the first step, no cache is available
            attn_output, _ = self.multihead_attn(x, x, x)  # (N, S, E)
            self.key_cache = self.multihead_attn.in_proj_k(
                x).transpose(0, 1)  # (S, N, E)
            self.value_cache = self.multihead_attn.in_proj_v(
                x).transpose(0, 1)  # (S, N, E)
        else:
            # Reuse the cached keys and values, and compute the new ones
            new_key = self.multihead_attn.in_proj_k(
                x).transpose(0, 1)  # (S, N, E)
            new_value = self.multihead_attn.in_proj_v(
                x).transpose(0, 1)  # (S, N, E)

            keys = torch.cat([self.key_cache, new_key], dim=0)  # (S, N, E)
            values = torch.cat(
                [self.value_cache, new_value], dim=0)  # (S, N, E)

            attn_output, _ = self.multihead_attn(x, keys, values)  # (N, S, E)

            # Update the cache
            self.key_cache = keys
            self.value_cache = values

        return attn_output.transpose(0, 1)  # Transpose back to (S, N, E)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()

        self.attn = CachedSelfAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (S, N, E) where S is the source sequence length, N is the batch size, E is the feature number
        attn_output = self.attn(x)
        x = self.norm1(x + attn_output)  # Add & Norm
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)  # Add & Norm
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        # assuming a maximum sequence length of 1000
        self.pos_encoder = nn.Embedding(1000, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (S, N) where S is the source sequence length, N is the batch size

        x = self.embedding(x)  # (S, N, E)

        pos = torch.arange(x.size(0), device=x.device).unsqueeze(1)  # (S, 1)
        x = x + self.pos_encoder(pos)

        for layer in self.layers:
            x = layer(x)  # (S, N, E)

        x = self.fc(x)  # (S, N, vocab_size)

        return F.log_softmax(x, dim=-1)


# define the model, loss function and optimizer
model = GPT(vocab_size=10000, d_model=512, nhead=8,
            num_layers=6, dim_feedforward=2048)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# number of epochs to train for
num_epochs = 10

# loop over the dataset multiple times
for epoch in range(num_epochs):
    running_loss = 0.0
    # train loop
    for i, batch in enumerate(train_data):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(batch[:-1])  # all tokens except the last
        # predict the next token in the sequence
        loss = criterion(outputs.view(-1, outputs.size(-1)),
                         batch[1:].view(-1))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    # validation loop
    with torch.no_grad():
        validation_loss = 0.0
        for i, batch in enumerate(val_data):
            outputs = model(batch[:-1])
            loss = criterion(
                outputs.view(-1, outputs.size(-1)), batch[1:].view(-1))
            validation_loss += loss.item()
        print('Epoch: %d, validation loss: %.3f' %
              (epoch + 1, validation_loss / len(val_data)))

print('Finished Training')
