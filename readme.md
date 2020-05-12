# Pytorch-LEO: A Pytorch Implemtation of Meta-Learning with Latent Embedding Optimization(LEO)

## Running the code
### Prerequisites
* torch==1.4.0
* PyYAML==3.13

### Getting the data
We borrow the embedding from the [deepmind/leo repo](https://github.com/deepmind/leo)  
You can download the pretrained embeddings [here](http://storage.googleapis.com/leo-embeddings/embeddings.zip)   
or   
```
$ wget http://storage.googleapis.com/leo-embeddings/embeddings.zip`
$ unzip embeddings.zip
```

### Run Training 
```
python3 main.py -train \ 
                -verbose \ 
                -N 5 \ 
                -K 1 \ 
                -embedding_dir ../embeddings/ \ 
                -dataset miniImageNet \ 
                -exp_name toy-example \ 
                -save_checkpoint
```                
where N, K means N-way K-shot training  
exp_name help you keep track of your experiment   

for full arguments, see main.py  

