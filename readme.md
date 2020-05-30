# Pytorch-LEO: A Pytorch Implemtation of Meta-Learning with Latent Embedding Optimization(LEO)

## Running the code
### Prerequisites
* torch==1.4.0
* PyYAML==3.13

### Getting the data
We borrow the embedding from the [deepmind/leo repo](https://github.com/deepmind/leo)  
You can download the pretrained embeddings [here](http://storage.googleapis.com/leo-embeddings/embeddings.zip),   
or do   
```
$ wget http://storage.googleapis.com/leo-embeddings/embeddings.zip
$ unzip embeddings.zip
```

### Run Training 
```
python3 main.py -train \ 
                -verbose \ 
                -N 5 \ 
                -K 1 \ 
                -embedding_dir $(EMBEDDING_DIR) \ 
                -dataset miniImageNet \ 
                -exp_name toy-example \ 
                -save_checkpoint
```                
where
+ `-N`, `-K` means N-way K-shot training,  
+ `-exp_name` help you keep track of your experiment,     
+ `-save_checkpoint` to save model for later testing.

for full arguments, see `main.py`  

### Run Testing
```
python3 main.py -test \
		-N 5 \
		-K 1 \
		-verbose \
    		-load $(model_path) 
```
The testing result will be printed on the console.

### Monitor Training
This projects comes with [Comet.ml](https://www.comet.ml/site/) support. If you want to disable logging, just add `-disable_comet` as an argument.  
You will need to modify the `COMET_PROJECT_NAME` and `COMET_WORKSPACE` in `config.yml` to enable monitoring.

### Hyperparameters
You can modify the hyperparameters in `config.yml`, where detailed descriptions are also provided.
The hyperparameters that yield the best result in this code are as follow:
| Hyperparameters | miniImageNet 1-shot | miniImageNet 5-shot | tieredImageNet 1-shot | tieredImageNet 5-shot |
|:-------------|:-------------:|:-------------:|:-------------:|:-------------:| 
| `outer_lr` | 0.0005 | 0.0006 | 0.0006 | 0.0006 |
| `l2_penalty_weight` | 0.0001 | 8.5e-6 | 3.6e-10 | 3.6e-10 |
| `orthogonality_penalty_weight` | 303.0 | 0.00152 | 0.188 | 0.188 |
| `dropout` | 0.3 | 0.3 | 0.3 | 0.3 |
| `kl_weight` | 0 | 0.001 | 0.001 | 0.001 |
| `encoder_penalty_weight` | 1e-9 | 2.66e-7 | 5.7e-6 | 5.7e-6 |


## Result

| Implementation | miniImageNet 1-shot | miniImageNet 5-shot | tieredImageNet 1-shot | tieredImageNet 5-shot |
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:| 
| LEO Paper | 61.76 ± 0.08% | 77.59 ± 0.12% | 66.33 ± 0.05% | 81.44 ± 0.09% |
| this code | 59.46 ± 0.08% | 76.01 ± 0.09% | 66.62 ± 0.07% | 81.72 ± 0.09% |

*The result we obtained may not be comparable since the model is trained on both the training set and validation set in the paper. 
While our model is only trained on the training set and validated on the validation set.
Note: This project is licensed under the terms of the MIT license.
