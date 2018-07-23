import argparse
import math
import time
import reader
import torch.nn as nn
from tqdm import tqdm
import models
import torch


class PTBModel(nn.Module):

    def __init__(self, vocab_size, hidden_size, prior, init_scale):
        super(PTBModel, self).__init__()

        self.hidden_size = hidden_size
        self.inference = False

        # Layers
        self.encoder = models.BayesEmbedding(vocab_size, hidden_size, prior, init_scale)
        self.bayeslstm = models.BayesLSTM(hidden_size, hidden_size, prior, init_scale, name="lstm0")
        self.linear = models.BayesLinear(hidden_size, vocab_size, prior, init_scale)

        self.kl = None

    def forward(self, x, hidden):

        embedding = self.encoder(x, inference=self.inference)
        out, hidden = self.bayeslstm(embedding, hidden, inference=self.inference)
        logits = self.linear(out, inference=self.inference)

        self.kl = self.encoder.kl + self.linear.kl + self.bayeslstm.kl

        return logits, hidden


def run_epoch(model, criterion, train_data, corpus, lr, epoch):
    # Turn on training mode which enables dropout.
    model.train()
    total_likelihood_loss = 0
    total_kl_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = None

    for batch, i in enumerate(tqdm(range(0, train_data.size(0) - 1, args.num_steps))):
        data, targets = get_batch(train_data, i)
        model.zero_grad()

        output, hidden = model(data, hidden)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)

        likelihood_loss = criterion(output.view(-1, ntokens), targets) / args.batch_size
        # likelihood_loss.backward()
        total_likelihood_loss += likelihood_loss.data
        kl_loss = model.kl / (args.batch_size * train_data.size(0) // args.num_steps)
        loss = likelihood_loss + kl_loss
        loss.backward()
        total_kl_loss += kl_loss.data

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        if batch % args.log_interval == 0 and batch > 0:
            cur_likelihood_loss = total_likelihood_loss[0] / (args.log_interval * args.num_steps)
            cur_kl_loss = total_kl_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f} | KL {:8.2f}'.format(
                      epoch, batch, len(train_data) // args.num_steps, lr,
                      elapsed * 1000 / args.log_interval, cur_likelihood_loss, math.exp(cur_likelihood_loss), cur_kl_loss))
            total_likelihood_loss = 0
            total_kl_loss = 0
            start_time = time.time()


def repackage_hidden(h):

    hidden0, hidden1 = h

    hidden0 = (hidden0[0].detach(), hidden0[1].detach())
    hidden1 = (hidden1[0].detach(), hidden1[1].detach())

    h = hidden0, hidden1

    return h


def batchify(data, batch_size):
    # Work out how cleanly we can divide the dataset into batch_size parts.
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, evaluation=False):
    seq_len = min(args.num_steps, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def evaluate(model, corpus, criterion_eval, eval_batch_size, data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    model.inference = True
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = None
    for i in range(0, data_source.size(0) - 1, args.num_steps):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion_eval(output_flat, targets).data
        hidden = repackage_hidden(hidden)

    model.inference = False

    return total_loss[0] / len(data_source)


def run(args):
    if not args.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    corpus = reader.Corpus(args.data_path)

    eval_batch_size = 20
    train_data = batchify(corpus.train, args.batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)

    ntokens = len(corpus.dictionary)

    criterion = nn.CrossEntropyLoss(size_average=False)

    # Create model
    prior = models.Prior(args.prior_pi, args.log_sigma1, args.log_sigma2)
    model = PTBModel(ntokens, args.hidden_size, prior, args.init_scale)
    print(model)
    criterion_eval = nn.CrossEntropyLoss()

    if args.cuda:
        model.cuda()

    # Loop over epochs.
    lr = args.learning_rate

    for epoch in range(1, args.max_max_epoch + 1):
        epoch_start_time = time.time()
        run_epoch(model, criterion, train_data, corpus, lr, epoch)
        val_loss = evaluate(model, corpus, criterion_eval, eval_batch_size, val_data)
        print(val_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)

        # Learning rate annealing
        lr_decay = args.lr_decay ** max(epoch + 1 - args.max_epoch, 0.0)
        lr = args.learning_rate * lr_decay

    # Run on test data.
    test_loss = evaluate(model, corpus, criterion_eval, eval_batch_size, test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default='small',
                        choices=['small', 'medium', 'large', 'test'])
    parser.add_argument('--data_path', type=str, default='./data/penn')
    parser.add_argument('--save_path', type=str, default='./model/saved_new')
    parser.add_argument('--prior_pi', type=float, default=0.25)
    parser.add_argument('--log_sigma1', type=float, default=-1.0)
    parser.add_argument('--log_sigma2', type=float, default=-7.0)
    parser.add_argument('--inference_mode', type=str, default='mu', choices=['mu', 'sample'])
    parser.add_argument('--bbb_bias', action='store_true', help='Enable biases to be BBB variables')
    parser.add_argument('--var_mode', type=str, default="BBB",
                        choices=["BBB", "standard"], help='Enable biases to be BBB variables')
    parser.add_argument('--cuda', action="store_true")

    # Params from medium conf
    parser.add_argument('--init_scale', type=float, default=0.05)
    parser.add_argument('--learning_rate', type=float, default=1.)
    parser.add_argument('--max_grad_norm', type=float, default=5)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_steps', type=int, default=35)
    parser.add_argument('--hidden_size', type=int, default=650)
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--max_max_epoch', type=int, default=70)
    parser.add_argument('--keep_prob', type=float, default=1.0)
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--log_interval', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=1111)

    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    run(args)
