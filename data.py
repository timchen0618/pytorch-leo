# embedding_path = 'test_embeddings.pkl'
# pickle.load(open(embedding_path, 'rb'), encoding='latin1')
# keys -> image_names; labels -> image labels; embeddings -> image embeddings (instances, 640) <numpy.ndarray> 
import random
import torch.nn.functional as F
import os
import pickle
import numpy as np
import torch 
import itertools

class Data_Utils(object):
    """docstring for Data_Utils"""
    def __init__(self, args, data_config):
        super(Data_Utils, self).__init__()
        self.args = args
        self.no_cuda = args.no_cuda
        self.config = data_config
        self.dataset = args.dataset
        if args.train:
            self.metasplit = ['train', 'val']
        else:
            self.metasplit = ['test']
        self.verbose = args.verbose
        random.seed(self.args.seed)
        self.construct_data()
        

    def construct_data(self):
        # loading embeddings
        
        self.embedding_path = os.path.join(self.args.embedding_dir, self.args.dataset, self.args.pretraining_scheme)
        
        self.embeddings = {}
        for d in self.metasplit:
            if self.verbose:
                print('Loading data from ' + os.path.join(self.embedding_path, d+'_embeddings.pkl') + '...')
            self.embeddings[d] = pickle.load(open(os.path.join(self.embedding_path, d+'_embeddings.pkl'), 'rb'), encoding='latin1')
        
       
        # sort images by class
        self.image_by_class = {}
        self.embed_by_name = {}
        self.class_list = {}
        for d in self.metasplit:
            self.image_by_class[d] = {}
            self.embed_by_name[d] = {}
            self.class_list[d] = set()
            keys = self.embeddings[d]["keys"]
            for i, k in enumerate(keys):
                _, class_name, img_name = k.split('-')
                if(class_name not in self.image_by_class[d]):
                    self.image_by_class[d][class_name] = []
                self.image_by_class[d][class_name].append(img_name) 
                self.embed_by_name[d][img_name] = self.embeddings[d]["embeddings"][i]
                # construct class list
                self.class_list[d].add(class_name)
            
            self.class_list[d] = list(self.class_list[d])
            if self.verbose:
                print('Finish constructing ' + d + ' data, total %d classes.' %len(self.class_list[d]))
        


    def get_batch(self, metasplit):
        # train_data -> [batch, N, k, dim]
        # valid_data -> [batch, N, k, dim]
        if metasplit == 'train':
            b_size = self.config['batch_size']
        elif metasplit == 'val':
            b_size = self.config['val_batch_size']
        else:
            b_size = self.config['test_batch_size']

        K = self.args.K
        N = self.args.N
        val_steps = self.config['meta_val_steps']

        datasplit = ['train', 'val']
        batch = {}
        for d in datasplit:
            batch[d] = {'input':[], 'target':[], 'name':[]}

        for b in range(b_size):
            shuffled_classes = self.class_list[metasplit].copy()
            random.shuffle(shuffled_classes)

            shuffled_classes = shuffled_classes[:self.args.N]

            inp = {'train':[[] for i in range(N)], 'val':[[] for i in range(N)]}
            tgt = {'train':[[] for i in range(N)], 'val':[[] for i in range(N)]}

            for c, class_name in enumerate(shuffled_classes):
                images = np.random.choice(self.image_by_class[metasplit][class_name], K + val_steps)
                image_names = {'train':images[:K], 'val':images[K:]}

                for d in datasplit:
                    num_images = K if d == 'train' else val_steps
                    assert len(image_names[d]) == num_images
                    for i in range(num_images):
                        embed = self.embed_by_name[metasplit][image_names[d][i]]
                        inp[d][c].append(embed)
                        tgt[d][c].append(c)

            for d in datasplit:
                
                num_images = K if d == 'train' else val_steps

                assert(len(inp['train']) == N)
                assert(len(inp['val']) == N)

                permutations = list(itertools.permutations(range(self.args.N)))
                order = random.choice(permutations)
                inputs = [inp[d][i] for i in order]
                target = [tgt[d][i] for i in order]

                batch[d]['input'].append(np.asarray(inputs).reshape(N, num_images, -1))
                batch[d]['target'].append(np.asarray(target).reshape(N, num_images, -1))

            
        # convert to tensor
        for d in datasplit:
            num_images = K if d == 'train' else val_steps
            normalized_input = torch.from_numpy(np.array(batch[d]['input']))
            if self.no_cuda:
                batch[d]['input'] = F.normalize(normalized_input, dim = -1)
                batch[d]['target'] = torch.from_numpy(np.array(batch[d]['target']))
            else:
                batch[d]['input'] = F.normalize(normalized_input, dim = -1).cuda()
                batch[d]['target'] = torch.from_numpy(np.array(batch[d]['target'])).cuda()

            assert(batch[d]['input'].shape == (b_size, N, num_images, self.config['embedding_size']))
            assert(batch[d]['target'].shape == (b_size, N, num_images, 1))
        return batch
