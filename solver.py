from comet_ml import Experiment
import torch 
import torch.nn as nn
from model import Model
from data import Data_Utils
import os
import yaml

class Solver():
    """docstring for Solver"""
    def __init__(self, args):
        config = yaml.load(open(args.config, 'r'), Loader=yaml.SafeLoader)
        if not args.disable_comet:
            self.comet_config = config['Comet']
        config = config[args.dataset]["%dshot"%args.K]

        self.data_utils = Data_Utils(args, config['data'])
        self.config = config['Solver']
        self.model = Model(
                           config['model'], 
                           args.N, 
                           args.K, 
                           not args.no_cuda
                           )
        
        
        if args.train:
            self.model_dir = os.path.join(args.model_dir, "%s_%dshot"%(args.dataset,args.K), args.exp_name)
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
        
        self._disable_comet = args.disable_comet
        self._print_every_step = args.print_every_step
        self._valid_every_step = args.valid_every_step
        self._verbose = args.verbose
        self._N = args.N
        self._K = args.K
        self._total_test_instances = args.total_test_instances
        self._total_val_steps = args.total_val_steps
        self._save_checkpoint = args.save_checkpoint
        self._load_model = args.load
        self._save_best = args.save_best 
        if self._save_best:
            self._best_acc = 0

        if not args.no_cuda:
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
                                    self._verbose and train,
                                    (not self._disable_comet) and train,
                                    step
                                    )
        orthogonality_penalty = self.orthogonality(list(self.model.decoder.parameters())[0])

        # calculate loss (l2 reg implemented with optimizer)
        total_loss = val_loss + self.config['kl_weight'] * kl_div \
                   + self.config['encoder_penalty_weight'] * encoder_penalty + self.config['orthogonality_penalty_weight'] * orthogonality_penalty
        
        return total_loss, val_acc, kl_div, encoder_penalty, orthogonality_penalty


    def orthogonality(self, weight):
        w2 = torch.mm(weight, weight.transpose(0, 1))
        wn = torch.norm(weight, dim=1, keepdim=True) + 1e-20
        correlation_matrix = w2/ torch.mm(wn, wn.transpose(0, 1))
        assert correlation_matrix.size(0) == correlation_matrix.size(1)
        I = torch.eye(correlation_matrix.size(0)).cuda()
        return torch.mean((correlation_matrix-I)**2)


    def meta_train_batch(self, inputs, target):
        latents, kl_div = self.model.encode(inputs)
        latents_init = latents

        for i in range(self.config['inner_update_step']):
            latents.retain_grad()
            classifier_weights = self.model.decode(latents)
            train_loss, _ = self.model.cal_target_loss(inputs, classifier_weights, target)
            train_loss.backward(retain_graph=True)

            latents = latents - self.model.inner_l_rate * latents.grad.data

        encoder_penalty = torch.mean((latents_init - latents) ** 2)
        return latents, kl_div, encoder_penalty

    def inner_finetuning(self, latents, inputs, target, val_input, val_target, verbose, logging, step):
        
        classifier_weights = self.model.decode(latents)
        classifier_weights.retain_grad()
        train_loss, train_acc = self.model.cal_target_loss(inputs, classifier_weights, target)
        
        # print info and logging
        if verbose and step % self._print_every_step == 0:
            print()
            print('(Meta-Train) [Step: %d/%d] Train Loss: %4.4f Train Accuracy: %4.4f Inner_Lr: %4.4f Finetuning_Lr: %4.4f ' \
                   %(step, self.config['total_steps'], train_loss.item(), train_acc.item(), self.model.inner_l_rate, self.model.finetuning_lr))
        
        if logging and step % self._print_every_step == 0:
            self.exp.log_metric('Training Accuracy', train_acc.item(), step=step)
            self.exp.log_metric('Training Loss', train_loss.item(), step=step)
            self.exp.log_metric('Inner Lr', float(self.model.inner_l_rate), step=step)
            self.exp.log_metric('Finetuning Lr', float(self.model.finetuning_lr), step=step)

        for j in range(self.config['finetuning_update_step']):
            train_loss.backward(retain_graph=True)        
            classifier_weights = classifier_weights - self.model.finetuning_lr * classifier_weights.grad
            classifier_weights.retain_grad()
            train_loss, _ = self.model.cal_target_loss(inputs, classifier_weights, target)

        val_loss, val_accuracy = self.model.cal_target_loss(val_input, classifier_weights, val_target)

        return val_loss, val_accuracy


    def train(self):
        if not self._disable_comet:
            # comet logging
            hyper_params = {
                "outer_lr": self.config['outer_lr'],
                "kl_weight": self.config['kl_weight'],
                "encoder_penalty_weight": self.config['encoder_penalty_weight'],
                "l2_penalty_weight": self.config['l2_penalty_weight'],
                "orthogonality_penalty_weight": self.config['orthogonality_penalty_weight'],
                "dropout": self.model.dropout,
                "embedding_size": self.model.embed_size,
                "hidden_size": self.model.hidden_size,
                "N": self._N,
                "K": self._K
            }

            self.exp = Experiment(
                                  project_name=self.comet_config['COMET_PROJECT_NAME'],
                                  workspace=self.comet_config['COMET_WORKSPACE'],
                                  auto_output_logging=None,
                                  auto_metric_logging=None,
                                  display_summary=False,
                                  )
            self.exp.log_parameters(hyper_params)
            self.exp.add_tags(['%d way'%self._N, '%d shot'%self._K, self.data_utils.dataset])


        # different optim for lrs and params (only l2 penalize on params)
        lr_list = ['inner_l_rate', 'finetuning_lr']
        params = [x[1] for x in list(filter(lambda kv: kv[0] not in lr_list, self.model.named_parameters()))]
        lr_params = [x[1] for x in list(filter(lambda kv: kv[0] in lr_list, self.model.named_parameters()))]
        optim = torch.optim.Adam(params, lr=self.config['outer_lr'], weight_decay=self.config['l2_penalty_weight'])
        optim_lr = torch.optim.Adam(lr_params, lr=self.config['outer_lr'])
        
        # update for (total_steps) steps
        for step in range(self.config['total_steps']):
            optim.zero_grad()
            optim_lr.zero_grad()
            # do training
            batch = self.data_utils.get_batch('train')
            val_loss, val_acc, kl_div, encoder_penalty, orthogonality_penalty = self.run_batch(batch, step)

            if self._verbose and step % self._print_every_step == 0:
                print('(Meta-Valid) [Step: %d/%d] Total Loss: %4.4f Valid Accuracy: %4.4f'%(step, self.config['total_steps'], val_loss.item(), val_acc.item()))
                print('(Meta-Valid) [Step: %d/%d] KL: %4.4f Encoder Penalty: %4.4f Orthogonality Penalty: %4.4f'%(step, self.config['total_steps'], kl_div, encoder_penalty, orthogonality_penalty))
                
            if not self._disable_comet and step % self._print_every_step == 0:
                self.exp.log_metric('Total Loss', val_loss.item(), step=step)
                self.exp.log_metric('Valid Accuracy', val_acc.item(), step=step)
                self.exp.log_metric('KL div', kl_div.detach().cpu().numpy(), step=step)
                self.exp.log_metric('Encoder Penalty', encoder_penalty.detach().cpu().numpy(), step=step)
                self.exp.log_metric('Orthogonality Penalty', orthogonality_penalty.detach().cpu().numpy(), step=step)

            val_loss.backward()
            nn.utils.clip_grad_value_(self.model.parameters(), self.config['clip_value'])
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_value'])
            optim.step()
            optim_lr.step()

            if step % self._valid_every_step == 1:
                self.model.eval()
                val_losses = []
                val_accs = []
                for val_step in range(self._total_val_steps):
                    batch = self.data_utils.get_batch('val')
                    val_loss, val_acc, _, _, _ = self.run_batch(batch, step, False)
                    val_losses.append(val_loss.item())
                    val_accs.append(val_acc.item())

                if self._save_checkpoint:
                    #save checkpoint                      
                    if not (self._save_best and sum(val_accs)/len(val_accs) < self._best_acc):
                        model_name = '%dk_%4.4f_%4.4f_model.pth' % (step//1000, sum(val_losses)/len(val_losses), sum(val_accs)/len(val_accs))
                        state = {'step': step, 'val_acc': sum(val_accs)/len(val_accs), 'state_dict': self.model.state_dict()}
                        if not os.path.exists(self.model_dir):
                            os.mkdir(self.model_dir)
                        torch.save(state, os.path.join(self.model_dir, model_name))
                
                self.model.train()

                if self._verbose:
                    print()
                    print('=' * 50)
                    print('Meta Valid Loss: %4.4f \nMeta Valid Accuracy: %4.4f'%(sum(val_losses)/len(val_losses), sum(val_accs)/len(val_accs)))
                    print('=' * 50)
                    print()
                    if self._save_checkpoint:
                        print('Saving checkpoint %s...'%model_name)
                        print()

                if not self._disable_comet:
                    self.exp.log_metric('Meta Valid Loss', sum(val_losses)/len(val_losses), step = step)
                    self.exp.log_metric('Meta Valid Accuracy', sum(val_accs)/len(val_accs), step = step)

    def test(self):
        total_test_steps = self._total_test_instances// self.data_utils.config['test_batch_size']

        #load state dict
        state_dict = torch.load(self._load_model)['state_dict']
        self.model.load_state_dict(state_dict)

        self.model.eval()
        test_losses = []
        test_accs = []
        
        for test_step in range(total_test_steps):
            batch = self.data_utils.get_batch('test')
            test_loss, test_acc, _, _, _ = self.run_batch(batch, test_step, False)
            test_losses.append(test_loss.item())
            test_accs.append(test_acc.item())

        if self._verbose:
            print()
            print('=' * 50)
            print('Meta Test Loss: %4.4f Meta Test Accuracy: %4.4f'%(sum(test_losses)/len(test_losses), sum(test_accs)/len(test_accs)))
            print('=' * 50)
            print()
