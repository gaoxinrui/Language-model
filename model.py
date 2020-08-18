import torch
import torch.nn as nn

class LMModel(nn.Module):
    # Language model is composed of three parts: a word embedding layer, a rnn network and a output layer. 
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding. 
    # The rnn network has input of each word embedding and output a hidden feature corresponding to each word embedding.
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary.
    def __init__(self, rnn_type, nvoc, ninput, nhid, nlayers, dropout, tie_weights=False):
        super(LMModel, self).__init__()
        self.drop = nn.Dropout(dropout)

        self.encoder = nn.Embedding(nvoc, ninput)
        # Construct you RNN model here. You can add additional parameters to the function.
        if rnn_type in ['RNN', 'LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninput, nhid, nlayers, dropout=dropout)
        else:
            raise ValueError("""'rnn_type' error, options are ['RNN', 'LSTM', 'GRU']""")
        self.decoder = nn.Linear(nhid, nvoc)

        if tie_weights:
            if nhid != ninput:  
                raise ValueError('When using the tied flag, hi_dim must be equal to em_dim')
            self.decoder.weight = self.encoder.weight
        
        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input, hidden):
        embeddings = self.drop(self.encoder(input))
        # With embeddings, you can get your output here. Output has the dimension of sequence_length * batch_size * number of classes
        output, hidden = self.rnn(embeddings, hidden)# output维度：
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        """初始化隐藏层参数，与batch_size相关"""
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':  # lstm：(h0, c0)
            return (weight.new(self.nlayers, bsz, self.nhid).zero_(),
                    (weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:  # gru 和 rnn：h0
            return weight.new(self.nlayers, bsz, self.nhid).zero_()
