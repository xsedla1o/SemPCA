import logging
import os
import sys

from sempca.CONSTANTS import SESSION, LOG_ROOT, device
from sempca.module.Attention import *
from sempca.module.CPUEmbedding import *
from sempca.module.Common import *
from sempca.utils import drop_input_independent


class AttGRUModel(nn.Module):
    # Dispose Loggers.
    AttGRULogger = logging.getLogger("AttGRU")
    AttGRULogger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
        )
    )

    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, "AttGRU.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
        )
    )

    AttGRULogger.addHandler(console_handler)
    AttGRULogger.addHandler(file_handler)
    AttGRULogger.info(
        "Construct logger for Attention-Based GRU succeeded, current working directory: %s, logs will be written in %s"
        % (os.getcwd(), LOG_ROOT)
    )

    def __init__(self, embedding, lstm_layers, lstm_hiddens, dropout):
        super(AttGRUModel, self).__init__()

        self.logger = AttGRUModel.AttGRULogger
        self.dropout = dropout
        self.logger.info("==== Model Parameters ====")
        vocab_size, word_dims = embedding.shape
        self.word_embed = CPUEmbedding(
            vocab_size, word_dims, padding_idx=vocab_size - 1
        )
        self.word_embed.weight.data.copy_(torch.from_numpy(embedding))
        self.word_embed.weight.requires_grad = False
        self.logger.info("Input Dimension: %d" % word_dims)
        self.logger.info("Hidden Size: %d" % lstm_hiddens)
        self.logger.info("Num Layers: %d" % lstm_layers)
        self.logger.info("Dropout %.3f" % dropout)
        self.rnn = nn.GRU(
            input_size=word_dims,
            hidden_size=lstm_hiddens,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.sent_dim = 2 * lstm_hiddens
        self.atten_guide = Parameter(torch.Tensor(self.sent_dim))
        self.atten_guide.data.normal_(0, 1)
        self.atten = LinearAttention(
            tensor_1_dim=self.sent_dim, tensor_2_dim=self.sent_dim
        )
        self.proj = NonLinear(self.sent_dim, 2)

    def reset_word_embed_weight(self, vocab, pretrained_embedding):
        vocab_size, word_dims = pretrained_embedding.shape
        self.word_embed = CPUEmbedding(
            vocab.vocab_size, word_dims, padding_idx=vocab.PAD
        )
        self.word_embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.word_embed.weight.requires_grad = False

    def forward(self, inputs):
        words, masks, word_len = inputs
        embed = self.word_embed(words)
        if self.training:
            embed = drop_input_independent(embed, self.dropout)
        if device.type == "cuda":
            embed = embed.cuda(device)
        batch_size = embed.size(0)
        atten_guide = torch.unsqueeze(self.atten_guide, dim=1).expand(-1, batch_size)
        atten_guide = atten_guide.transpose(1, 0)
        hiddens, state = self.rnn(embed)
        sent_probs = self.atten(atten_guide, hiddens, masks)
        batch_size, srclen, dim = hiddens.size()
        sent_probs = sent_probs.view(batch_size, srclen, -1)
        represents = hiddens * sent_probs
        represents = represents.sum(dim=1)
        outputs = self.proj(represents)
        return outputs  # , represents


class AttGRUModel_onehot(nn.Module):
    def __init__(self, embedding, lstm_layers, lstm_hiddens, dropout):
        super(AttGRUModel_onehot, self).__init__()
        # Dispose Loggers.
        AttGRULogger = logging.getLogger("AttGRU")
        AttGRULogger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
            )
        )

        file_handler = logging.FileHandler(os.path.join(LOG_ROOT, "AttGRU.log"))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"
            )
        )

        AttGRULogger.addHandler(console_handler)
        AttGRULogger.addHandler(file_handler)
        AttGRULogger.info(
            "Construct logger for Attention-Based GRU succeeded, current working directory: %s, logs will be written in %s"
            % (os.getcwd(), LOG_ROOT)
        )
        self.logger = AttGRULogger
        self.dropout = dropout
        self.logger.info("==== Model Parameters ====")
        self.logger.info("Input Dimension: %d" % 1)
        self.logger.info("Hidden Size: %d" % lstm_hiddens)
        self.logger.info("Num Layers: %d" % lstm_layers)
        self.logger.info("Dropout %.3f" % dropout)
        self.rnn = nn.GRU(
            input_size=1,
            hidden_size=lstm_hiddens,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.sent_dim = 2 * lstm_hiddens
        self.atten_guide = Parameter(torch.Tensor(self.sent_dim))
        self.atten_guide.data.normal_(0, 1)
        self.atten = LinearAttention(
            tensor_1_dim=self.sent_dim, tensor_2_dim=self.sent_dim
        )
        self.proj = NonLinear(self.sent_dim, 2)

    def reset_word_embed_weight(self, vocab, pretrained_embedding):
        vocab_size, word_dims = pretrained_embedding.shape
        self.word_embed = CPUEmbedding(
            vocab.vocab_size, word_dims, padding_idx=vocab.PAD
        )
        self.word_embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.word_embed.weight.requires_grad = False

    def forward(self, inputs):
        words, masks, word_len = inputs
        embed = torch.unsqueeze(words, dim=2).float()

        if self.training:
            embed = drop_input_independent(embed, self.dropout)
        batch_size = embed.size(0)
        atten_guide = torch.unsqueeze(self.atten_guide, dim=1).expand(-1, batch_size)
        atten_guide = atten_guide.transpose(1, 0)
        hiddens, state = self.rnn(embed)
        sent_probs = self.atten(atten_guide, hiddens, masks)
        batch_size, srclen, dim = hiddens.size()
        sent_probs = sent_probs.view(batch_size, srclen, -1)
        represents = hiddens * sent_probs
        represents = represents.sum(dim=1)
        outputs = self.proj(represents)
        return outputs  # , represents
