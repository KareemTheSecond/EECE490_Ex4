# CardClassification

Dataset: https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification

### What Card Dataset includes:
- train
- test
- valid
Please prepare the path names of the three downloaded folders above


Each folder has 53 subfolders for each type of card, Joker included.
- Ace of Spades
- Two of Spades
- Three of Spades
...

### Model Made

- The model uses CNN and then two normal feed-forward layers. When evaluated using the validation and test set, it returns around a 72% score.
Example output: 

Epoch [1/10], Loss: 3.8256
Epoch [2/10], Loss: 2.6423
Epoch [3/10], Loss: 1.9719
Epoch [4/10], Loss: 1.5069
Epoch [5/10], Loss: 1.1215
Epoch [6/10], Loss: 1.2263
Epoch [7/10], Loss: 0.8500
Epoch [8/10], Loss: 0.6060
Epoch [9/10], Loss: 0.4955
Epoch [10/10], Loss: 0.3929
Validation Accuracy: 72.45%
Test Accuracy: 73.96%


