# Session - 15 Assignment (In Progress)

## Requirement

- Train a pre-trained Resnet34 model on CIFAR100 with these hyperparameters:
more than 4+ GPUs
- 4 Epochs
- Use Spot instances
- Record the screen while training, clearly showing spot instance usage, 4 epoch logs, and inferences. 
- Upload the video to youtube (can be unlisted, but not private). 
- Share the link.


## Solution

A comparative study has been done to gauge the cost saving benefits of Spot instances.
The **youtube video** can be found **[here]**(https://youtu.be/wMIs3WGE_N8).


## Training Log with Spot instances
<pre>

epoch: 5 - loss: 0.8833218216896057
Saving the Checkpoint: /opt/ml/checkpoints/checkpoint.pth
Finished Training
Saving the model.
2022-02-21 17:56:40,920 sagemaker-training-toolkit INFO     Reporting training SUCCESS

2022-02-21 17:56:51 Uploading - Uploading generated training model
2022-02-21 17:56:51 Completed - Training job completed
Training seconds: 472
Billable seconds: 183
Managed Spot Training savings: 61.2%

</pre>

