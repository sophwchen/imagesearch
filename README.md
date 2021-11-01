# Week3Capstone

Create a class that organizes all of the COCO data. It might store the following (William/Koa)
- All the image IDs
- All the caption IDs
- Various mappings between image/caption IDs, and associating caption-IDs with captions
    - image-ID -> [cap-ID-1, cap-ID-2, ...]
    - caption-ID -> image-ID
    - caption-ID -> caption (e.g. 24 -> "two dogs on the grass")

Ability to embed any caption / query text (using GloVe-200 embeddings weighted by IDFs of words across captions) (Clarice)
- An individual word not in the GloVe or IDF vocabulary should yield an embedding vector of just zeros.

Create a MyNN model for embedding image descriptors: 𝑑⃗ img→𝑤̂ img (Clarice/Sophia)

Extract sets of (caption-ID, image-ID, confusor-image-ID) triples (training and validation sets) (Ozan)

Write function to compute loss (margin ranking loss) and accuracy

Train the model (Ozan/William/Koa)
- get the caption embedding
- embed the “true” image
- embed the “confusor” image
- compute similarities (caption and good image, caption and bad image)
- compute loss and accuracy
- take optimization step

Write functionality for saving / loading trained model weights

Generate image feature vectors (descriptors) using ResNet18
- given N image-IDs be able to return an array of N descriptor vectors

Create image database by mapping image feature vectors to semantic embeddings with the trained model

Write function to query database with a caption-embedding and return the top-k images

Write function to display set of images given COCO image ids
