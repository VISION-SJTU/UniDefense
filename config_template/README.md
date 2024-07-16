### Description of Model Configuration Keys

We use a yaml file to specify the hyperparameters of a model. All the training logs will be placed in `$PROJECT_ROOT/runs/$MODEL_NAME/$EXPERIMENT_ID`. An example are shown below.

```yaml
model:
  name: UDEB4           # Model name
  num_classes: 2
  drop_rate: 0.2
  extractor: efficientnet-b4
  extractor_weights: ckpt/adv-efficientnet-b4-44fb3a87.pth
config:
  distribute:
    backend: nccl
  find_unused: False    # Whether to find unused params
  warmup_step: 0
  lambda_triplet: 0.1   # Weight parameters for L_WT
  lambda_recons: 0.1    # Weight parameters for L_{Rec\_spat}
  lambda_freq: 1.0      # Weight parameters for L_{Rec\_freq}
  lambda_mask: 0.1      # Weight parameters for L_{Mask} and L_{Cons}
  lambda_fac: 0.1       # Weight parameters for L_{Orth}
  optimizer:
    name: adamw
    lr: 0.0001
    betas: [0.9, 0.999]
    weight_decay: 0.000005
    amsgrad: True
  scheduler:            # Optional
    name: StepLR
    step_size: 22500
    gamma: 0.5
  crop: nocrop
  resume: False
  resume_best: False
  id: FFppC40           # Unique experiment ID
  debug: False
data:
  train_batch_size: 10
  val_batch_size: 64
  test_batch_size: 96
  file: "./config/forgery/data_ffc40.yml"   # Dataset configuration path
  # file: "./config/forgery/data_ffc23.yml"
```