import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """docstring for Encoder"""
    def __init__(self, args):
        super(Model, self).__init__()
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.args = args
        self.b_size = args.batch_size
        self.N = args.N
        self.K = args.K
        self.encoder = nn.Linear(self.embed_size, self.hidden_size, bias = False)
        self.relation_net = nn.Sequential(
                                 nn.Linear(2*self.hidden_size, 2*self.hidden_size, bias = False),
                                 nn.ReLU(),
                                 nn.Linear(2*self.hidden_size, 2*self.hidden_size, bias = False),
                                 nn.ReLU(),
                                 nn.Linear(2*self.hidden_size, 2*self.hidden_size, bias = False),
                                 nn.ReLU()
                                 )
        self.decoder = nn.Linear(self.hidden_size, 2*self.embed_size, bias = False)
        self._cuda = args.use_cuda
        self.dropout = nn.Dropout(p=args.dropout)
        self.inner_l_rate = nn.Parameter(torch.FloatTensor([args.inner_lr_init]))
        self.finetuning_lr = nn.Parameter(torch.FloatTensor([args.finetuning_lr_init]))

    def encode(self, inputs):
        # inputs -> [batch, N, K, embed_size]
        inputs = self.dropout(inputs)
        out = self.encoder(inputs)
        b_size, N, K , hidden_size = out.size()

        t1 = torch.repeat_interleave(out, K, dim = 2)
        t1 = torch.repeat_interleave(t1, N, dim = 1)
        t2 = out.repeat((1, N, K, 1))
        x = torch.cat((t1, t2), dim=-1)
        # x -> [batch, N*N, K*K, hidden_size]
        x = self.relation_net(x)
        x = x.view(b_size, N, N*K*K, -1)
        x = torch.mean(x, dim = 2)
        
        # b = a.reshape(b_size, N, N*K, K, -1)
        # c = b.reshape(b_size, N, N*K*K, -1)
        # print(a[0, :, :, 0])
        # c = t.view(b_size, N, N*K*K, -1)
        # print(c[0, :, :, 0])

        # TODO
        # do some operation       
        latents = self.sample(x, self.hidden_size)
        mean, var = x[:,:, :self.hidden_size], x[:,:, self.hidden_size:]
        kl_div = self.cal_kl_div(latents, mean, var)

        return latents, kl_div

    def cal_kl_div(self, latents, mean, var):
        if self._cuda:
            # print(torch.mean(self.cal_log_prob(latents, mean, var)))
            return torch.mean(self.cal_log_prob(latents, mean, var) - self.cal_log_prob(latents, torch.zeros(mean.size()).cuda(), torch.ones(var.size()).cuda()))
        else:
            return torch.mean(self.cal_log_prob(latents, mean, var) - self.cal_log_prob(latents, torch.zeros(mean.size()), torch.ones(var.size())))


    def cal_log_prob(self, x, mean, var):
        log_unnormalized = - 0.5 * ((x - mean)/ (var+1e-20))**2
        # print(torch.mean(log_unnormalized))
        log_normalization = torch.log(var+1e-20) + 0.5 * math.log(2*math.pi)
        return log_unnormalized - log_normalization


    def decode(self, latents):
        weights = self.decoder(latents)
        # print('decoder_output', weights.size())
        classifier_weights = self.sample(weights, self.embed_size)
        # print('classifier_weights', classifier_weights.size())
        return classifier_weights

    def sample(self, weights, size):
        eps = 1e-12
        # stddev = tf.exp(unnormalized_stddev)
        # stddev -= (1. - stddev_offset)
        mean, var = weights[:,:, :size], weights[:,:, size:]
        z = torch.normal(0.0, 1.0, mean.size()).cuda()
        # dist = torch.distributions.normal.Normal(mean, torch.exp(var)+eps)
        # dist -> [batch, num_class, size]
        return mean + var*z

    def predict(self, inputs, weights):
        # TODO
        # do dropout 
        # do einsum
        # print('weights', weights.size())
        b_size, N, K, embed_size = inputs.size()
        weights = weights.permute((0, 2, 1))
        # print('weights', weights.size())
        inputs = inputs.view(b_size, -1, embed_size)  
        # print('inputs', inputs.size())
        outputs = torch.bmm(inputs, weights)
        outputs = outputs.view(-1, outputs.size(-1))
        # print('outputs', outputs.size())
        outputs = F.log_softmax(outputs, dim = -1)
        return outputs

    def cal_target_loss(self, inputs, classifier_weights, target):
        outputs = self.predict(inputs, classifier_weights)
        # target -> [batch, num_classes]; pred -> [batch, num_classes]
        criterion = nn.NLLLoss()
        target = target.view(target.size(0), -1, target.size(-1))
        target = target.view(-1, target.size(-1)).squeeze()

        target_loss = criterion(outputs, target)
        # TODO
        # compute_acc
        pred = outputs.argmax(dim = -1)
        corr = (pred == target).sum()
        total = pred.fill_(1).sum()

        return target_loss, corr.float()/total.float(), pred

