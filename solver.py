import torch 
import torch.nn as nn
from model import Model
from data import Data_Utils
import os

class Solver():
    """docstring for Solver"""
    def __init__(self, args):
        # super(Solver, self).__init__()
        self.args = args
        self.data_utils = Data_Utils(args)
        self.model = Model(self.args)
        
        if args.use_cuda:
            self.model = self.model.cuda()
            
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def run_batch(self, batch, step, train=True):
        # do task-train (inner loop)
        latents, kl_div, encoder_penalty = self.meta_train_batch(batch['train']['input'], batch['train']['target'])
        # do inner fine-tuning & task-validate (outer loop)
        val_loss, val_acc = self.inner_finetuning(
                                    latents, 
                                    batch['train']['input'], 
                                    batch['train']['target'], 
                                    batch['val']['input'], 
                                    batch['val']['target'],
                                    self.args.verbose and step % self.args.print_every_step == 0 and train,
                                    step
                                    )
        orthogonality_penalty = self.orthogonality(list(self.model.decoder.parameters())[0])

        total_loss = val_loss + self.args.kl_weight * kl_div \
                   + self.args.encoder_penalty_weight * encoder_penalty + self.args.orthogonality_penalty_weight * orthogonality_penalty
                   # + self.args.l2_penalty_weight * l2_penalty 
        
        return total_loss, val_acc, kl_div, encoder_penalty, orthogonality_penalty


    def orthogonality(self, weight):
        w2 = torch.mm(weight, weight.transpose(0, 1))
        wn = torch.norm(weight, dim=1, keepdim=True) + 1e-20
        correlation_matrix = w2/ torch.mm(wn, wn.transpose(0, 1))
        assert correlation_matrix.size(0) == correlation_matrix.size(1)
        I = torch.eye(correlation_matrix.size(0)).cuda()
        return torch.mean((correlation_matrix-I)**2)


    def meta_train_batch(self, inputs, target):
        # TODO
        # inner_l_rate can be learned
        latents, kl_div = self.model.encode(inputs)
        latents_init = latents

        for i in range(self.args.inner_update_step):
            latents.retain_grad()
            classifier_weights = self.model.decode(latents)
            train_loss, _, _ = self.model.cal_target_loss(inputs, classifier_weights, target)
            train_loss.backward(retain_graph=True)

            latents = latents - self.model.inner_l_rate * latents.grad.data

        encoder_penalty = torch.mean((latents_init - latents) ** 2)
        return latents, kl_div, encoder_penalty

    def inner_finetuning(self, latents, inputs, target, val_input, val_target, verbose, step):
        finetuning_lr = self.args.finetuning_lr_init

        classifier_weights = self.model.decode(latents)
        classifier_weights.retain_grad()
        train_loss, train_acc, pred_train = self.model.cal_target_loss(inputs, classifier_weights, target)
        if verbose:
            print('Step: %d Training Loss: %4.4f Training Accuracy: %4.4f inner_lr: %4.4f finetuning_lr: %4.4f ' \
                   %(step, train_loss.item(), train_acc.item(), self.model.inner_l_rate, self.model.finetuning_lr))

        for j in range(self.args.finetuning_update_step):
            train_loss.backward(retain_graph=True)        
            classifier_weights = classifier_weights - self.model.finetuning_lr * classifier_weights.grad
            classifier_weights.retain_grad()
            train_loss, _, _ = self.model.cal_target_loss(inputs, classifier_weights, target)

        val_loss, val_accuracy, pred = self.model.cal_target_loss(val_input, classifier_weights, val_target)


        return val_loss, val_accuracy


    def train(self):
        
        optim = torch.optim.Adam(self.model.parameters(), lr=self.args.outer_lr, weight_decay=self.args.l2_penalty_weight)
        # optim = torch.optim.Adam(list(self.model.parameters()), lr=self.args.outer_lr, weight_decay=self.args.l2_penalty_weight)
        for step in range(self.args.num_steps):
            optim.zero_grad()
            # do training
            batch = self.data_utils.get_batch('train')
            val_loss, val_acc, kl_div, encoder_penalty, orthogonality_penalty = self.run_batch(batch, step)

            if self.args.verbose and step % self.args.print_every_step == 0:
                print('Step: %d Total Loss: %4.4f Valid Accuracy: %4.4f'%(step, val_loss.item(), val_acc.item()))
                print('KL: %4.4f encoder_penalty: %4.4f orthogonality_penalty: %4.4f'%(kl_div, encoder_penalty, orthogonality_penalty))

            val_loss.backward()
            nn.utils.clip_grad_value_(self.model.parameters(), self.args.clip_value)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_value)
            optim.step()

            if step % self.args.valid_every_step == 1:
                self.model.eval()
                val_losses = []
                val_accs = []
                for val_step in range(self.args.total_val_steps):
                    batch = self.data_utils.get_batch('val')
                    val_loss, val_acc, _, _, _ = self.run_batch(batch, step, False)
                    val_losses.append(val_loss.item())
                    val_accs.append(val_acc.item())

                #save checkpoint
                model_name = str(step//1000) + 'k_' + '%6.6f'%(sum(val_losses)/len(val_losses)) + '%2.4f_'%(sum(val_accs)/len(val_accs)) + 'model.pth'
                state = {'step': step, 'state_dict': self.model.state_dict()}
                if not os.path.exists(self.args.model_dir):
                    os.mkdir(self.args.model_dir)

                torch.save(state, os.path.join(self.args.model_dir, model_name))
                self.model.train()

                if self.args.verbose:
                    print()
                    print('=' * 50)
                    print('Meta Valid Loss: %4.4f Meta Valid Accuracy: %4.4f'%(sum(val_losses)/len(val_losses), sum(val_accs)/len(val_accs)))
                    print('=' * 50)
                    print()
                    print('Saving checkpoint %s...'%model_name)
                    print()

    def test(self):
        total_test_steps = self.args.total_test_instances// self.args.test_batch_size

        #load state dict
        state_dict = torch.load(self.args.load)['state_dict']
        self.model.load_state_dict(state_dict)

        self.model.eval()
        test_losses = []
        test_accs = []
        
        for test_step in range(total_test_steps):
            batch = self.data_utils.get_batch('test')
            test_loss, test_acc, _, _, _ = self.run_batch(batch, test_step, False)
            test_losses.append(test_loss.item())
            test_accs.append(test_acc.item())

        if self.args.verbose:
            print()
            print('=' * 50)
            print('Meta Test Loss: %4.4f Meta Test Accuracy: %4.4f'%(sum(test_losses)/len(test_losses), sum(test_accs)/len(test_accs)))
            print('=' * 50)
            print()