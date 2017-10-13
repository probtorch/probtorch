from random import random
import time
import datetime
import sys
sys.path.append('../')
import torch
from torch import nn
from torch.autograd import Variable
import probtorch

NUM_PIXELS = 784
NUM_HIDDEN = 256
NUM_DIGITS = 10
NUM_STYLE = 2

SAMPLE_SIZE = 4 # can set to None to revert to single-sample case
BATCH_SIZE = 128
NUM_EPOCHS = 10
LABEL_FRACTION = 0.5
LEARNING_RATE = 1e-3
BETA1 = 0.90

EPS = 1e-9

DATA_PATH = '../data'
WEIGHTS_PATH = '../weights'

class Encoder(nn.Module):
    def __init__(self, num_pixels=NUM_PIXELS, 
                       num_hidden=NUM_HIDDEN,
                       num_digits=NUM_DIGITS,
                       num_style=NUM_STYLE,
                       num_batch=BATCH_SIZE):
        super(self.__class__, self).__init__()
        self.enc_hidden = nn.Sequential( 
                            nn.Linear(num_pixels, num_hidden),
                            #nn.BatchNorm1d(num_hidden),
                            nn.ReLU())
        # encode digit
        self.log_weights = nn.Linear(num_hidden, num_digits)
        self.temp = Variable(torch.Tensor([0.66]))
        # encode style
        self.mean = nn.Linear(num_hidden + num_digits, num_style)
        self.log_std = nn.Linear(num_hidden + num_digits, num_style)

    def forward(self, images, labels=None, sample_size=None):
        q = probtorch.Trace()
        if not sample_size is None:
            expanded_size = [sample_size,] + list(images.size())
            images = images.expand(*expanded_size)
            if not labels is None:
                expanded_size = [sample_size,] + list(labels.size())
                labels = labels.expand(*expanded_size)
        hiddens = self.enc_hidden(images)
        digits = q.concrete(self.log_weights(hiddens),
                            self.temp,
                            log_pdf=False,
                            value=labels,
                            name='digits')
        styles_input = torch.cat([digits.float(), hiddens], -1)
        styles_mean = self.mean(styles_input)
        styles_std = torch.exp(self.log_std(styles_input))
        q.normal(styles_mean,
                 styles_std,
                 name='styles')
        return q

class Decoder(nn.Module):
    def __init__(self, num_pixels=NUM_PIXELS, 
                       num_hidden=NUM_HIDDEN,
                       num_digits=NUM_DIGITS,
                       num_style=NUM_STYLE,
                       num_batch=BATCH_SIZE):
        super(self.__class__, self).__init__()
        self.digit_log_weights = nn.Parameter(torch.zeros(num_digits))
        self.digit_temp = Variable(torch.Tensor([0.66]))
        self.style_mean = nn.Parameter(torch.zeros(num_style))
        self.style_log_std = Variable(torch.zeros(num_style))

        self.dec_hidden = nn.Sequential(
                            nn.Linear(num_style + num_digits, num_hidden),
                            #nn.BatchNorm1d(num_hidden),
                            nn.ReLU())
        self.dec_image = nn.Sequential(
                           nn.Linear(num_hidden, num_pixels),
                           nn.Sigmoid())
        #self.bce = nn.BCELoss(size_average=False)

    def forward(self, images, q, sample_size=None):
        p = probtorch.Trace()
        num_digits = len(self.digit_log_weights)
        batch_size = len(images)
        log_weights = self.digit_log_weights.expand(batch_size, num_digits)
        if not sample_size is None:
            log_weights = self.digit_log_weights.expand(sample_size, batch_size, num_digits)
            expanded_size = [sample_size,] + list(images.size())
            images = images.expand(*expanded_size)
        digits = p.concrete(log_weights,
                            self.digit_temp,
                            log_pdf=False,
                            value=q['digits'],
                            name='digits')
        styles = p.normal(self.style_mean,
                          torch.exp(self.style_log_std),
                          value=q['styles'],
                          name='styles')
        hiddens = self.dec_hidden(torch.cat([digits.float(), styles], -1))
        images = p.loss(#self.bce,
                        lambda x_hat, x: -(torch.log(x_hat + EPS) * x + torch.log(1 - x_hat + EPS) * (1-x)).sum(-1),
                        self.dec_image(hiddens),
                        images,
                        name='images')
        return p

# def unsup_loss(q, p):
#     log_like = probtorch.objectives.montecarlo.log_like(q, p)
#     kl = probtorch.objectives.montecarlo.kl(q, p)
#     return - (log_like - kl).mean()

# def kingma_loss(q, p):
#     log_like = probtorch.objectives.montecarlo.log_like(q, p)
#     kl = probtorch.objectives.montecarlo.kl(q, p)
#     log_w = probtorch.objectives.log_observed(q)
#     elbo_xy = log_like - kl + log_w
#     return - elbo_xy.mean() - 0.1 * log_w.mean()

# def jw_loss(q,p):
#     log_like = probtorch.objectives.montecarlo.log_like(q, p)
#     log_w = probtorch.objectives.log_observed(q)
#     #log_Z = probtorch.util.log_sum_exp(log_w)
#     w = torch.exp(log_w) 
#     wkl = probtorch.objectives.montecarlo.kl(q, p, weights=w)
#     return - (log_like - wkl).mean()

def elbo_loss(q, p, alpha=0.1):
    if SAMPLE_SIZE is None:
        return -probtorch.objectives.montecarlo.elbo(q, p, alpha, sample_dim=None, batch_dim=0)
    else:
        return -probtorch.objectives.montecarlo.elbo(q, p, alpha, sample_dim=0, batch_dim=1)

unsup_loss = elbo_loss
sup_loss = elbo_loss

def train(data, enc, dec, optimizer, 
          label_mask={}, label_fraction=LABEL_FRACTION):
    epoch_elbo = 0.0
    enc.train()
    dec.train()
    N = 0
    for b, (images, labels) in enumerate(data):
        if images.size()[0] == BATCH_SIZE:
            N += BATCH_SIZE
            images = Variable(images).view(-1, NUM_PIXELS)
            labels_onehot = torch.zeros(BATCH_SIZE, NUM_DIGITS)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_onehot = Variable(torch.clamp(labels_onehot, EPS, 1-EPS))
            optimizer.zero_grad()
            if b not in label_mask:
                label_mask[b] = (random() < label_fraction)
            if label_mask[b]:
                q = enc(images, labels_onehot, sample_size=SAMPLE_SIZE)
            else:
                q = enc(images, sample_size=SAMPLE_SIZE)
            p = dec(images, q, sample_size=SAMPLE_SIZE)
            if label_mask[b]:
                loss = sup_loss(q, p)
            else:
                loss = unsup_loss(q, p)
            loss.backward()
            optimizer.step()
            epoch_elbo -= loss.data.numpy()[0]
    return epoch_elbo / N, label_mask

def test(data, enc, dec):
    enc.eval()
    dec.eval()
    epoch_elbo = 0.0
    epoch_correct = 0
    N = 0
    for b, (images, labels) in enumerate(data):
        if images.size()[0] == BATCH_SIZE:
            N += BATCH_SIZE
            images = Variable(images).view(-1, NUM_PIXELS)
            q = enc(images, sample_size=SAMPLE_SIZE)
            p = dec(images, q, sample_size=SAMPLE_SIZE)
            epoch_elbo += unsup_loss(q, p).data.numpy()[0]
            _, y_pred = q['digits'].value.data.max(-1)
            epoch_correct += (labels == y_pred).sum()*1.0 / (SAMPLE_SIZE or 1.0)
    return epoch_elbo / N, epoch_correct / N

def git_revision():
    import subprocess
    result = subprocess.run(
                "git rev-parse --short HEAD".split(), 
                stdout=subprocess.PIPE)
    return result.stdout.strip()

if __name__ == '__main__':
    from torchvision import datasets, transforms
    import os 

    if not os.path.isdir(DATA_PATH):
        os.makedirs(DATA_PATH)

    train_data = torch.utils.data.DataLoader(
                    datasets.MNIST(DATA_PATH, 
                                   train=True, download=True,
                                   transform=transforms.ToTensor()),
                    batch_size=BATCH_SIZE, 
                    shuffle=True) 
    test_data = torch.utils.data.DataLoader(
                    datasets.MNIST(DATA_PATH, 
                                   train=False, download=True,
                                   transform=transforms.ToTensor()),
                    batch_size=BATCH_SIZE, 
                    shuffle=True) 
    enc = Encoder()
    dec = Decoder()
    optimizer =  torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),
                                  lr=LEARNING_RATE,
                                  betas=(BETA1, 0.999))
    mask = {}
    for e in range(NUM_EPOCHS):
        train_start = time.time()
        train_elbo, mask = train(train_data, enc, dec, 
                                 optimizer, mask, LABEL_FRACTION)
        train_end = time.time()
        test_start = time.time()
        test_elbo, test_accuracy = test(test_data, enc, dec)
        test_end = time.time()
        print('[Epoch %d] Train: ELBO %.4e (%ds) Test: ELBO %.4e, Accuracy %0.3f (%ds)' % (
                e, train_elbo, train_end - train_start, 
                test_elbo, test_accuracy, test_end - test_start))

    if not os.path.isdir(WEIGHTS_PATH):
        os.makedirs(WEIGHTS_PATH)
    hash = git_revision()
    id = datetime.datetime.now().strftime('%y%m%d-%H%M')

    torch.save(enc.state_dict(), 
               '%s/cmnist-%s-%s-enc.rar' % (WEIGHTS_PATH, id, hash))
    torch.save(dec.state_dict(), 
               '%s/cmnist-%s-%s-dec.rar' % (WEIGHTS_PATH, id, hash))

