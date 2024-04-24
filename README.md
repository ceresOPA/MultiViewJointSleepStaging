## A Self-supervised Multiview Joint Pre-training Framework for Representation Learning in Sleep Staging

> PyTorch Implementation of the paper "A Self-supervised Multiview Joint Pre-training Framework for Representation Learning in Sleep Staging"

### 1. Self-supervised Training Stage

```
./contrastive_train.sh
```

Training the Time and Freq views, respectively.

### 2. MultiView Joint Training Stage

```
./cotrain.sh
```

### 3. Training for downstram task of Sleep Staging

```
./classifier_train.sh
```

### 4. Test and Evaluation

```
./classifier_test.sh
```