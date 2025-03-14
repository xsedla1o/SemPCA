import torch
from torch.autograd import Variable


class TInstWithLogits:
    def __init__(self, batch_size, slen, tag_size):
        self.src_ids = []
        self.src_words = Variable(
            torch.LongTensor(batch_size, slen).zero_(), requires_grad=False
        )
        self.src_masks = Variable(
            torch.Tensor(batch_size, slen).zero_(), requires_grad=False
        )
        self.tags = Variable(
            torch.FloatTensor(batch_size, tag_size).zero_(), requires_grad=False
        )
        self.g_truth = Variable(
            torch.LongTensor(batch_size).zero_(), requires_grad=False
        )
        self.word_len = Variable(
            torch.LongTensor(batch_size).zero_(), requires_grad=False
        )

    def to_cuda(self, device):
        self.src_words = self.src_words.cuda(device)
        self.src_masks = self.src_masks.cuda(device)
        self.tags = self.tags.cuda(device)
        self.g_truth = self.g_truth.cuda(device)
        self.word_len = self.word_len.cuda(device)

    @property
    def inputs(self):
        return self.src_words, self.src_masks, self.word_len

    @property
    def ids(self):
        return self.src_ids

    @property
    def targets(self):
        return self.tags

    @property
    def truth(self):
        return self.g_truth


class TensorInstance:
    def __init__(self, batch_size, slen):
        self.src_ids = []
        self.src_words = Variable(
            torch.LongTensor(batch_size, slen).zero_(), requires_grad=False
        )
        self.g_truth = Variable(
            torch.LongTensor(batch_size).zero_(), requires_grad=False
        )
        self.word_len = Variable(
            torch.LongTensor(batch_size).zero_(), requires_grad=False
        )
        self.mask = Variable(
            torch.LongTensor(batch_size, slen).zero_(), requires_grad=False
        )

    def to_cuda(self, device):
        self.src_words = self.src_words.cuda(device)
        self.g_truth = self.g_truth.cuda(device)
        self.word_len = self.word_len.cuda(device)
        self.mask = self.mask.cuda(device)

    @property
    def masks(self):
        return self.mask

    @property
    def inputs(self):
        return self.src_words, self.masks

    @property
    def ids(self):
        return self.src_ids

    @property
    def targets(self):
        return self.g_truth


class SequentialTensorInstance:
    def __init__(self, batch_size, slen):
        self.src_ids = []
        self.src_words = Variable(
            torch.FloatTensor(batch_size, slen, 1).zero_(), requires_grad=False
        )
        self.g_truth = Variable(
            torch.LongTensor(batch_size).zero_(), requires_grad=False
        )
        self.word_len = Variable(
            torch.LongTensor(batch_size).zero_(), requires_grad=False
        )
        self.mask = Variable(
            torch.LongTensor(batch_size, slen).zero_(), requires_grad=False
        )

    def to_cuda(self, device):
        self.src_words = self.src_words.cuda(device)
        self.g_truth = self.g_truth.cuda(device)
        self.word_len = self.word_len.cuda(device)
        self.mask = self.mask.cuda(device)

    @property
    def masks(self):
        return self.mask

    @property
    def inputs(self):
        return self.src_words, self.masks

    @property
    def ids(self):
        return self.src_ids

    @property
    def targets(self):
        return self.g_truth


class DualTensorInstance:
    def __init__(self, batch_size, slen, quantity_dim):
        self.src_ids = []
        self.sequential = Variable(
            torch.LongTensor(batch_size, slen).zero_(), requires_grad=False
        )
        self.quantity = Variable(
            torch.FloatTensor(batch_size, quantity_dim).zero_(), requires_grad=False
        )
        self.g_truth = Variable(
            torch.LongTensor(batch_size).zero_(), requires_grad=False
        )
        self.word_len = Variable(
            torch.LongTensor(batch_size).zero_(), requires_grad=False
        )
        self.mask = Variable(
            torch.LongTensor(batch_size, slen).zero_(), requires_grad=False
        )

    def to_cuda(self, device):
        self.sequential = self.sequential.cuda(device)
        self.quantity = self.quantity.cuda(device)
        self.g_truth = self.g_truth.cuda(device)
        self.word_len = self.word_len.cuda(device)
        self.mask = self.mask.cuda(device)

    @property
    def masks(self):
        return self.mask

    @property
    def inputs(self):
        return self.sequential, self.quantity, self.masks

    @property
    def ids(self):
        return self.src_ids

    @property
    def targets(self):
        return self.g_truth
