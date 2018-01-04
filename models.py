import math
import torch
import warnings

from helpers import compute_KL
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence


class Prior(object):
    def __init__(self, pi, log_sigma1, log_sigma2):
        self.pi_mixture = pi
        self.log_sigma1 = log_sigma1
        self.log_sigma2 = log_sigma2
        self.sigma1 = math.exp(log_sigma1)
        self.sigma2 = math.exp(log_sigma2)

        self.sigma_mix = math.sqrt(pi * math.pow(self.sigma1, 2) + (1.0 - pi) * math.pow(self.sigma2, 2))

    def lstm_init(self):
        """Returns parameters to use when initializing \theta in the LSTM"""
        rho_min_init = math.log(math.exp(self.sigma_mix / 4.0) - 1.0)
        rho_max_init = math.log(math.exp(self.sigma_mix / 2.0) - 1.0)
        return rho_min_init, rho_max_init

    def normal_init(self):
        """Returns parameters to use when initializing \theta in embedding/projection layer"""
        rho_min_init = math.log(math.exp(self.sigma_mix / 2.0) - 1.0)
        rho_max_init = math.log(math.exp(self.sigma_mix / 1.0) - 1.0)
        return rho_min_init, rho_max_init


def get_bbb_variable(shape, prior, init_scale, rho_min_init, rho_max_init):

    if rho_min_init is None or rho_max_init is None:
        rho_min_init = math.log(math.exp(prior.sigma_mix / 4.0) - 1.0)
        rho_max_init = math.log(math.exp(prior.sigma_mix / 2.0) - 1.0)

    mu = Parameter(torch.Tensor(*shape))
    rho = Parameter(torch.Tensor(*shape))

    # Initialize
    mu.data.uniform_(-init_scale, init_scale)
    rho.data.uniform_(rho_min_init, rho_max_init)

    return mu, rho


class BayesLSTM(Module):

    def __init__(self, input_size, hidden_size, prior, init_scale, name=None):

        super(BayesLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.prior = prior
        self.init_scale = init_scale
        self._forget_bias = 0.
        self.layer_name = name

        rho_min_init, rho_max_init = self.prior.lstm_init()

        # Layer 0
        mu0, rho0 = get_bbb_variable((self.input_size + self.hidden_size, 4 * self.hidden_size),
                                     self.prior,
                                     self.init_scale,
                                     rho_min_init,
                                     rho_max_init)

        bias0 = Parameter(torch.Tensor(4 * self.hidden_size))
        bias0.data.fill_(0.)

        self.bias_l0 = bias0
        self.mu_l0 = mu0
        self.rho_l0 = rho0

        # Layer 1
        mu1, rho1 = get_bbb_variable((self.input_size + self.hidden_size, 4 * self.hidden_size),
                                     self.prior,
                                     self.init_scale,
                                     rho_min_init,
                                     rho_max_init)

        bias1 = Parameter(torch.Tensor(4 * self.hidden_size))
        bias1.data.fill_(0.)

        self.bias_l1 = bias1
        self.mu_l1 = mu1
        self.rho_l1 = rho1

        self.kl = None

    def forward_layer(self, x, hidden, inference, layer_idx=0):

        # Get BBB params
        if layer_idx == 0:
            mean = self.mu_l0
            sigma = F.softplus(self.rho_l0) + 1E-5
            bias = self.bias_l0
        else:
            mean = self.mu_l1
            sigma = F.softplus(self.rho_l1) + 1E-5
            bias = self.bias_l1

        if inference:
            weights = mean
        else:
            # Sample weights
            # This way of creating the epsilon variable is faster than
            # from numpy or torch.randn or FloatTensor.normal_ when mean is already
            # on the GPU
            eps = Variable(mean.data.new(mean.size()).normal_(0., 1.))
            weights = mean + eps * sigma

        # Roll out hidden
        h, c = hidden

        # Store each hidden state in output
        output = []
        # Loop over time steps and obtain predictions
        for i in range(x.size(0)):

            concat = torch.cat([x[i, :, :], h], -1)
            concat = torch.matmul(concat, weights) + bias

            i, j, f, o = torch.split(concat, concat.size(1) // 4, dim=1)

            new_c = c * F.sigmoid(f + self._forget_bias) + F.sigmoid(i) * F.tanh(j)
            new_h = F.tanh(new_c) * F.sigmoid(o)

            h, c = new_h, new_c

            output.append(h)

        output = torch.stack(output, dim=0)

        # Compute KL divergence
        kl = compute_KL(weights.view(-1), mean.view(-1), sigma.view(-1),
                        self.prior)

        return output, (h, c), kl

    def forward(self, x, hidden=None, inference=False):
        """
        Args:
            x: A (seq_len, batch, input_size) tensor containing input
                features.
            hidden: A tuple (h, c), which contains the hidden
                and cell state, where the size of both states is
                (batch, hidden_size).

        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        if hidden is None:
            hidden0 = Variable(x.data.new(x.size(1), self.hidden_size).zero_())
            hidden0 = (hidden0, hidden0)

            hidden1 = Variable(x.data.new(x.size(1), self.hidden_size).zero_())
            hidden1 = (hidden1, hidden1)
        else:
            hidden0, hidden1 = hidden

        output, hidden0, kl0 = self.forward_layer(x, hidden0, inference, layer_idx=0)
        output, hidden1, kl1 = self.forward_layer(output, hidden1, inference, layer_idx=1)

        hidden = (hidden0, hidden1)

        self.kl = kl0 + kl1

        return output, hidden

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class BayesEmbedding(Module):

    def __init__(self, num_embeddings, embedding_dim, prior, init_scale):
        super(BayesEmbedding, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.prior = prior
        self.init_scale = init_scale
        self.max_norm = None
        self.norm_type = 2
        self.scale_grad_by_freq = False
        self.sparse = False
        self.padding_idx = -1

        emb_rho_min_init, emb_rho_max_init = prior.normal_init()
        mu, rho = get_bbb_variable([num_embeddings, embedding_dim],
                                   prior,
                                   init_scale,
                                   emb_rho_min_init,
                                   emb_rho_max_init)

        self.mu = mu
        self.rho = rho
        self.kl = None

    def forward(self, input, inference=False):

        # Sample weight
        mean = self.mu
        sigma = F.softplus(self.rho) + 1E-5

        if inference:
            weights = mean
        else:
            # This way of creating the epsilon variable is faster than
            # from numpy or torch.randn or FloatTensor.normal_ when mean is already
            # on the GPU
            eps = Variable(mean.data.new(mean.size()).normal_(0., 1.))
            weights = mean + eps * sigma

        # Compute KL divergence
        self.kl = compute_KL(weights.view(-1), mean.view(-1), sigma.view(-1),
                             self.prior)

        after_embed = self._backend.Embedding.apply(
            input, weights,
            self.padding_idx, self.max_norm, self.norm_type,
            self.scale_grad_by_freq, self.sparse
        )

        return after_embed

    def __repr__(self):
        s = '{name}({num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class BayesLinear(Module):

    def __init__(self, in_features, out_features, prior, init_scale):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior = prior
        self.init_scale = init_scale

        sft_rho_min_init, sft_rho_max_init = prior.normal_init()

        mu, rho = get_bbb_variable((out_features, in_features),
                                   self.prior,
                                   self.init_scale,
                                   sft_rho_min_init,
                                   sft_rho_max_init)

        bias = Parameter(torch.Tensor(out_features))
        bias.data.fill_(0.)

        self.mu = mu
        self.rho = rho
        self.bias = bias
        self.kl = None

    def forward(self, input, inference=False):

        # Sample weight
        mean = self.mu
        sigma = F.softplus(self.rho) + 1E-5

        if inference:
            weights = mean
        else:
            # Sample weights from normal distribution
            # This way of creating the epsilon variable is faster than
            # from numpy or torch.randn or FloatTensor.normal_ when mean is already
            # on the GPU
            eps = Variable(mean.data.new(mean.size()).normal_(0., 1.))
            weights = mean + eps * sigma

        logits = F.linear(input, weights, self.bias)

        # Compute KL divergence
        self.kl = compute_KL(weights.view(-1), mean.view(-1), sigma.view(-1),
                             self.prior)

        # print(self.kl)

        return logits

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
