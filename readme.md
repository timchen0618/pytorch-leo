# Pytorch-LEO: A Pytorch Implemtation of Meta-Learning with Latent Embedding Optimization(LEO)

## Running the code
### Prerequisites
*torch==1.4.0
*PyYAML==3.13

### Getting the data
We borrow the embedding from deepmind/leo repo
You can download the pretrained embeddings here
or 
`$ wget http://storage.googleapis.com/leo-embeddings/embeddings.zip`
`$ unzip embeddings.zip`

### Run Training 
`python3 main.py -train \ \n`
                `-verbose \ \n`
                `-N 5 \ \n`
                `-K 1 \ \n`
                `-embedding_dir ../embeddings/ \ \n`
                `-dataset miniImageNet \ \n`
                `-exp_name toy-example \ \n`
                `-save_checkpoint \n` 
where N, K means N-way K-shot training
exp_name help you keep track of your experiment 
for full arguments, see main.py

