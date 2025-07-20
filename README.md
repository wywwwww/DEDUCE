# Detect, Decide, Unlearn: A Transfer-Aware Framework for Continual Learning (DEDUCE)
# Package Requirements
- Pytorch 1.12.1
# Datasets
- CIFAR-100
- CIFAR-10
- TinyImageNet
- COER-50

NOTE: Datasets are automatically downloaded in data/.

- This can be changed by changing the base_path function in utils/conf.py or using the --base_path argument.
- The data/ folder should not be tracked by git and is created automatically if missing.
# Installation
- To execute the codes for running experiments, run the following.
```bash
pip install -r requirements.txt
```
- New models can be added to the models/ folder.
- New datasets can be added to the datasets/ folder.
# Examples
# Run a model
The following command will run the model derpp on the dataset seq-cifar100 with a buffer of 500 samples the some random hyperparameters for lr, alpha, and beta:
``` bash
python utils/main.py --model derpp --dataset seq-cifar100 --alpha 0.5 --beta 0.5 --lr 0.001 --buffer_size 500
```
To run the model with the best hyperparameters, use the --model_config=best argument:
``` bash
python utils/main.py --model derpp --load_best_args --dataset seq-cifar100 --alpha 0.5 --beta 0.5 --lr 0.001 --buffer_size 500
```
# Acknowledgements
Our implementation is based on https://github.com/aimagelab/mammoth
