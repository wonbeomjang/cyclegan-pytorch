# cyclegan-pytorch
This repository is a deep-running part of a program that changes the style of a photo. And I refer to the following paper.
<p>[A reference article] (https://arxiv.org/abs/1703.10593)</p>

# preparing dataset
We used the supplied cycleGAN dataset.You can download datasets from this link.(https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/)  

Run organize_sample_data.py to sort the downloaded files so they can be learned. As an example, the directory for the monet2photo dataset is changed to dataset \ monet, dataset \ photo.

## how to train
Once you have structured the dataset, you can train it by running the train in the Train class.
