import os
import numpy as np
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import tqdm
from sklearn.base import TransformerMixin
from collections import defaultdict
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.autograd import Variable


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2, emb_dropout=0.2):
        super(TCN, self).__init__()
        self.encoder = nn.Embedding(output_size, input_size)
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.decoder = nn.Linear(input_size, output_size)
        self.decoder.weight = self.encoder.weight
        self.drop = nn.Dropout(emb_dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # input has dimension (N, L_in), and emb has dimension (N, L_in, C_in)
        emb = self.drop(self.encoder(x))
        y = self.tcn(emb.transpose(1, 2))
        o = self.decoder(y.transpose(1, 2))
        return o.contiguous()

class Vocab:
    def __init__(self, wikitext_path, max_vocab_size=100000):
        self.wikitext_path = wikitext_path
        test_counts = self.process_tokens(os.path.join(wikitext_path, 'wiki.test.tokens'))
        train_counts = self.process_tokens(os.path.join(wikitext_path, 'wiki.train.tokens'))
        valid_counts = self.process_tokens(os.path.join(wikitext_path, 'wiki.valid.tokens'))
        counts = test_counts + train_counts + valid_counts
        self.vocab = [word for word, count in counts.most_common(max_vocab_size) if count > 1]
        self.vocab = ['<pad>', '<eos>'] + self.vocab

    def process_tokens(self, path):
        return Counter(open(path, 'r').read().lower().split())

EMBEDDING_DIM = 100
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000
UNKNOWN_TOKEN = '<unk>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'

class WordVectorTransformer(TransformerMixin):
    def __init__(
        self,
        embedding_dim=50,
        maxlen=1000,
        index2word=None,
    ):
        self.embedding_dim = embedding_dim
        self.index2word = index2word
        self.word2index = { word: index for index, word in enumerate(self.index2word) }
        self.maxlen = maxlen

    def _load_glove_vectors(self, path, embedding_dim):
        embeddings_index = defaultdict(lambda: np.zeros(embedding_dim))
        with open(path) as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        vocab = [PAD_TOKEN, UNKNOWN_TOKEN, EOS_TOKEN] + list(embeddings_index.keys())
        embedding_matrix = np.zeros((len(vocab), embedding_dim))

        for i, word in enumerate(vocab):
            word_embedding = embeddings_index[word]
            embedding_matrix[i] = word_embedding

        return vocab, embedding_layer, embedding_matrix

    def _embedding(self, word, word2index):
        if word in word2index:
            return word2index[word]
        return word2index[UNKNOWN_TOKEN]

    def _sequence_ids(self, tokenized):
        sequences = []
        for tokens in tokenized:
            ids = [self._embedding(token, self.word2index) for token in tokens]
            sequences.append(ids)

        return sequences

    def transform(self, tokenized):
        """
        Sequences is padded of size (batch, maxlen).
        """
        sequences = self._sequence_ids(tokenized)
        lengths = [len(ids) for ids in sequences]
        if self.maxlen:
            return pad_sequences(sequences, maxlen=self.maxlen)
        else:
            return pad_sequences(sequences, maxlen=max(lengths))

    def fit(self, X, y=None):
        return self

vocab = Vocab('data/wikitext-2')
index2word = vocab.vocab
word2index = {word: _id for _id, word in enumerate(index2word)}
lines = open(os.path.join('data/wikitext-2/wiki.train.tokens'), 'r').readlines()
x_texts = [line.lower() for line in lines if len(line) > 40]
wv = WordVectorTransformer(index2word=index2word, maxlen=None)

continuous_text = (' ' + EOS_TOKEN + ' ').join(x_texts).split(' ')
data = wv.transform([continuous_text])[0]

BPTT = 70
BATCH_SIZE=64
num_chunks = len(data) // BATCH_SIZE
data = data[:BATCH_SIZE * num_chunks]
chunks = data.reshape((BATCH_SIZE, num_chunks)).transpose()


tcn = TCN(50, len(wv.index2word), [100, 50])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(tcn.parameters(), lr=0.001)

epochs = 10
for e in range(epochs):
    for i in tqdm.tqdm(range(len(chunks))):
        optimizer.zero_grad()
        start = i * BPTT
        end = (i + 1) * BPTT
        # X is (timestep x batch_size)
        X = chunks[start:end].transpose()
        y = chunks[start+1:end+1].transpose()
        y = y.flatten()
        #y = to_categorical(y, num_classes=len(wv.index2word))

        output = tcn(Variable(torch.from_numpy(X).long()))
        output = output.view(output.shape[0] * output.shape[1], -1)
        loss = criterion(output, Variable(torch.from_numpy(y).long()))
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(loss)
