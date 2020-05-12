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
where `-N`, `-K` means N-way K-shot training  
`-exp_name` help you keep track of your experiment   

for full arguments, see `main.py`  

### Run Testing
```
python3 main.py -test \
		-N 5 \
		-K 1 \
		-verbose \
    -load model_name.pth
```
The testing result will be printed on the console.

### Monitor Training
This projects comes with Comet.ml support. If you want to disable logging, just add `-disable_comet` as an argument.  
You will need to modify the `COMET_PROJECT_NAME` and `COMET_WORKSPACE` in `config.yml` to enable monitoring.

### Hyperparameters
