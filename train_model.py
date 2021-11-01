from coco_class import *
from resnet_stuff import *
from model import *

import pickle
import numpy as np
import mygrad as mg
import random

from mynn.optimizers.sgd import SGD
from mygrad.nnet.losses import margin_ranking_loss

from noggin import create_plot

def compute_cos_dist(v1, v2):
    """
    computes the cos distance between v1 and v2
    """
    return mg.einsum("ni,ni -> n", v1, v2)

def generate_confusor_image(
    glove, glove_shape, image_ID: int, image_caption: str, n: int,id_list, idf_dict) -> int:
    """
    params
    --------
    image_ID: int
        the ID of the image for which a confusor image is being generated
    image_caption: str
        a caption of the image for which a confusor image is being generated
    n: int
        the "tournament size": the number of randomly generated images from 
        which the "hardest" confusor image will be found
    returns
    --------
    confusor_ID: int
        the ID of the "hardest" confusor image
    """
    ids = [] 
    captions = [image_caption] 
    while len(ids) < n:
        id_ = random.choice(id_list)
        if id_ == image_ID:
            continue
        confusor_caption_id = random.choice(image_ids_to_caption_ids_dict[id_])
        confusor_caption = caption_ids_to_captions_dict[confusor_caption_id]
        captions.append(confusor_caption[0])
        ids.append(id_)
    embedded_captions = embed_idf_weighted_sum(captions, idf_dict, glove, glove_shape)
    original_caption = embedded_captions[0]
    confusor_captions = embedded_captions[1:]
    dists = np.array([
        compute_cos_dist(original_caption.reshape(1,-1), confusor_caption.reshape(1,-1))
        for confusor_caption in confusor_captions
    ])
    return ids[np.argmax(dists)]

def generate_train_triplets(resnet18_features, image_ids, n, all_image_ids, idf_dict, glove, glove_shape) -> tuple:
    """
    Parameters
    --------
    image_ID: int
        ID of the image for which a caption and confusor image will be selected
    n: int
        see generate_confusor_image()
    Returns
    --------
        `tuple(caption_embeddings, image_descriptors, confusor_descriptors)`
    """
    captions = []
    confusors = []

    for image_id in image_ids:
        image_caption_id = random.choice(image_ids_to_caption_ids_dict[image_id])
        image_caption = caption_ids_to_captions_dict[image_caption_id]
        captions.append(image_caption[0])
        
        confusors.append(generate_confusor_image(glove, glove_shape, image_id, image_caption[0], n, all_image_ids, idf_dict))
    
    caption_embeddings = np.array(embed_idf_weighted_sum(captions, idf_dict, glove, glove_shape))
    image_descriptors = np.array([resnet18_features[image_id] for image_id in image_ids])
    confusor_descriptors = np.array([resnet18_features[confusor] for confusor in confusors])
    return caption_embeddings, image_descriptors, confusor_descriptors

def generate_train_data(resnet18_features, idf_dict, n, glove, glove_shape):
    """
    Parameters
    ----------
    resnet18_features_ids
        Array of IDs from the resnet18 dictionary (pkl file).

    Returns
    -------
    `training_data`
        4/5 of entire dataset
    `validation_data`
        the other 1/5 of the dataset
    """
    resnet18_features_ids = list(resnet18_features.keys())
    all_image_IDs = np.array(resnet18_features_ids)
    idxs = np.arange(len(all_image_IDs))
    
    shuffled_image_ids = all_image_IDs[idxs]
    np.random.shuffle(idxs)
    # gets 4/5 of the data for training
    cutoff = int(len(shuffled_image_ids) * 0.8)
    training_data_ids = shuffled_image_ids[:cutoff]
    validation_data_ids = shuffled_image_ids[cutoff:]
    return generate_train_triplets(resnet18_features, training_data_ids, n, all_image_IDs, idf_dict, glove, glove_shape), generate_train_triplets(resnet18_features, validation_data_ids, n, all_image_IDs, idf_dict, glove, glove_shape)

# Change as needed
"""
file_path = "...Week3Capstone_Datasets/resnet18_features.pkl"
resnet18_features = get_resnet(file_path)

# All the image ids and descriptors in the resnet18 dataset
resnet18_features_ids = list(resnet18_features.keys())
resnet18_features_descriptors = list(resnet18_features.values())
image_to_captions = coco_dataset.image_ids_to_caption_ids()

n = 10 # tournament size for confusor id generation
# good_caption_ids, good_img_ids, bad_image_ids = list(zip(batch_triples))
training_data_triplet, validation_data_triplet = generate_train_data(resnet18_features_ids)

model = Model(512, 200)
num_epochs = 5
batch_size = 32
optimizer = SGD(model.parameters,learning_rate=1e-3,momentum=0.9)
margin = 0.25

# N should be consistent across the captions, images, and confusors
N = training_data_triplet[0].shape[0]

# setting up a noggin plot
plotter, fig, ax = create_plot(["loss", "accuracy"])

for epoch in range(num_epochs):
    # Indexing the tuple of the training data triplets
    # Training data triplets are not individual
    w_captions = training_data_triplet[0]
    d_imgs = training_data_triplet[1]
    d_confusors = training_data_triplet[2]
    
    for batch_cnt in range(0, N // batch_size):
        batch_indices = np.arange(batch_cnt * batch_size, ((batch_cnt + 1) * batch_size))
        
        # Creating batches of training data
        batch_w_captions = w_captions[batch_indices]
        batch_d_img = d_imgs[batch_indices]
        batch_d_confusor = d_confusors[batch_indices]
        
        # Forward pass
        w_img = model(batch_d_img)
        w_confusor = model(batch_d_confusor)
        
        # Compute error
        sim_match = compute_cos_dist(w_img, batch_w_captions)
        sim_confuse = compute_cos_dist(w_confusor, batch_w_captions)
        loss = margin_ranking_loss(sim_match, sim_confuse,1,margin)

        # Backprop + gradient step
        loss.backward()
        optimizer.step()

        # Accuracy calculation
        acc = sum([1 for sm, sc in (sim_match, sim_confuse) if sm > sc]) / batch_size

        plotter.set_train_batch({"loss": loss.item(), "accuracy": acc}, batch_size=batch_size)
        mg.turn_memory_guarding_off()
    
    #unpack validation data
    test_w_captions, test_d_imgs, test_d_confusors = validation_data_triplet
    N_test = validation_data_triplet[0].shape[0]

    for batch_cnt in range(0, N_test // batch_size):

        batch_indices = np.arange(batch_cnt * batch_size, ((batch_cnt + 1) * batch_size))
        
        # Creating batches of validation data
        batch_w_captions = test_w_captions[batch_indices]
        batch_d_img = test_d_imgs[batch_indices]
        batch_d_confusor = test_d_confusors[batch_indices]

        with mg.no_autodiff:
            test_w_img = model(batch_d_img)
            test_w_confusor = model(batch_d_confusor)

            test_sim_match = compute_cos_dist(test_w_img, batch_w_captions)
            test_sim_confuse = compute_cos_dist(test_w_confusor, batch_w_captions)
            test_acc = sum([1 for sm, sc in (test_sim_match, test_sim_confuse) if sm > sc]) / test_w_captions.shape[0]
    
        plotter.set_test_batch({"accuracy" : test_acc}, batch_size=batch_size)
    
    plotter.set_train_epoch()
"""
    # from cifar10
"""
    test_idxs = np.arange(len(x_test))
    
    for batch_cnt in range(0, len(x_test)//batch_size):
        batch_indices = test_idxs[batch_cnt*batch_size : (batch_cnt + 1)*batch_size]
        
        batch = x_test[batch_indices]
        truth = y_test[batch_indices]
        
        # We do not want to compute gradients here, so we use the
        # no_autodiff context manager to disable the ability to
        with mg.no_autodiff:
            # Get your model's predictions for this test-batch
            # and measure the test-accuracy for this test-batch
            # <COGINST>
            prediction = model(batch)
            test_accuracy = accuracy(prediction, truth)
            # </COGINST>
        
        # pass your test-accuracy here; we used the name `test_accuracy`
        plotter.set_test_batch({"accuracy" : test_accuracy}, batch_size=batch_size)
    plotter.set_test_epoch()
"""