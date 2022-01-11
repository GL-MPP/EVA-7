# Session - 13 Assignment

## Requirement

- To implement [blog](https://analyticsindiamag.com/hands-on-vision-transformers-with-pytorch/) on using ViT for Cats vs Dogs and train the ViT model for Cats vs Dogs.
- Train the model


## Approach

There are 2 parts to this exercise.

Part -1 Prepare the dataset for Cats and Dogs

Part -2 Build and train the ViT model on cats and dogs image dataset.


### Part -1
The dataset was downloaded from the [kaggle link](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data). The zip files were extracted and images were kept in separate Train and Test folders.
Following Transform rules were applied -
- transforms.Resize((224, 224))
- transforms.RandomResizedCrop(224)
- transforms.RandomHorizontalFlip()
- transforms.ToTensor()

### Part -2

The Transformer configurations are -
- dim=128
- seq_len=49+1  # 7x7 patches + 1 cls-token
- depth=12
- heads=8
- k=64

The ViT configurations are -
- dim=128
- image_size=224
- patch_size=32
- num_classes=2
- transformer= transformer with above configurations
- channels=3

## Result

Training logas as below - 

<pre>
100%
313/313 [02:38<00:00, 2.32it/s]
Epoch : 1 - loss : 0.6824 - acc: 0.5637 - val_loss : 0.6733 - val_acc: 0.5793

100%
313/313 [02:19<00:00, 2.59it/s]
Epoch : 2 - loss : 0.6770 - acc: 0.5714 - val_loss : 0.6672 - val_acc: 0.5920

100%
313/313 [02:19<00:00, 2.66it/s]
Epoch : 3 - loss : 0.6677 - acc: 0.5855 - val_loss : 0.6566 - val_acc: 0.6060

100%
313/313 [02:19<00:00, 2.63it/s]
Epoch : 4 - loss : 0.6530 - acc: 0.6061 - val_loss : 0.6582 - val_acc: 0.5930

100%
313/313 [02:19<00:00, 2.65it/s]
Epoch : 5 - loss : 0.6450 - acc: 0.6144 - val_loss : 0.6640 - val_acc: 0.5953

100%
313/313 [02:19<00:00, 2.63it/s]
Epoch : 6 - loss : 0.6422 - acc: 0.6162 - val_loss : 0.6232 - val_acc: 0.6513

100%
313/313 [02:19<00:00, 2.62it/s]
Epoch : 7 - loss : 0.6333 - acc: 0.6318 - val_loss : 0.6399 - val_acc: 0.6205

100%
313/313 [02:19<00:00, 2.56it/s]
Epoch : 8 - loss : 0.6239 - acc: 0.6399 - val_loss : 0.6219 - val_acc: 0.6430

100%
313/313 [02:19<00:00, 2.53it/s]
Epoch : 9 - loss : 0.6216 - acc: 0.6469 - val_loss : 0.6196 - val_acc: 0.6485

100%
313/313 [02:19<00:00, 2.57it/s]
Epoch : 10 - loss : 0.6191 - acc: 0.6491 - val_loss : 0.6168 - val_acc: 0.6551
</pre>
Highest train accuracy achieved is 64%.

Highest validation accuracy achieved is 65%.




# Notes on Classes used in ViT

### 1. PatchEmbedding 

This is basically a convolution (nn.Conv2d) which takes in the following parameters (with sample values) 
- image_size=224 (Image size 224X224)
- patch_size=16 (Patch size 16x16)
- num_channels=3 (Input image channels - RGB)
- embed_dim=768 (The dimension we decide on once the 16x16 patch is flattened)

The shape of the projected embedding is derived based on standard convolution -
nn.Conv2d(in_channel=num_channels, out_channel=embed_dim, kernel_size=patch_size, stride=patch_size)
or

patch_embedding = nn.Conv2D(3x768x16x16) followed by flattening will yeild ==> (14x14x768) = (196x768) or (-1,196,768)


### 2. Class Token is then added on top of the embeddings -
Shape of CLS Token = (-1, 1, 768)

### 3. Position embedding
Position embeddings(Learnable parameters) of shape (-1, 14*14+1,768) = (-1,197,768) is created.

### 4. Class ViTEmbeddings 
This class constructs (concats) the CLS token + patch embeddings and position to arrive at the final embedding and of shape (-1,197,768).

### 5. Class ViTConfig
This class stores the basic configuration values

{'attention_probs_dropout_prob': 0.0,
 'hidden_act': 'gelu',
 'hidden_dropout_prob': 0.0,
 'hidden_size': 768,
 'image_size': 224,
 'initializer_range': 0.02,
 'intermediate_size': 3072,
 'layer_norm_eps': 1e-12,
 'num_attention_heads': 12,
 'num_channels': 3,
 'num_hidden_layers': 12,
 'patch_size': 16}

### 6. Class ViTSelfAttention
This class basically takes in all the required configuration values and does the the following -
1. Generate the Key, Query and Value vectors.
2. Does necessary Matrix transpose operation to enable Matrix multiplication.
3. Generate the context layer (Softmax(Query x Key) x Value) and Attention probabilities (Softmax(Query x Key)) as output

### 7. class ViTSelfOutput
This is the Projection layer (Linear Layer) sitting on top of the Transformer mechanism.
### 8. Class ViTEncoder
This Classs is the inclusion of all the classes above to depict the overall architecture of Transformer (Encoder portion).