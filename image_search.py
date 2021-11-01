import numpy as np
from resnet_stuff import get_resnet
from model import *
from train_model import compute_cos_dist

# import requests
# from PIL import Image

# Retrived from 
# https://keestalkstech.com/2020/05/plotting-a-grid-of-pil-images-in-jupyter
import io
from urllib.request import urlopen, Request
from urllib.parse import urlparse
from PIL import Image
from PIL.Image import Image as PilImage
import matplotlib.pyplot as plt
import textwrap, os

def urls_to_pils(urls: list) -> PilImage:
    images = []
    for url in urls:
        data = urlopen(Request(url)).read()
        bts = io.BytesIO(data)
        im = Image.open(bts)
        im.filename = urlparse(url).path
        images.append(im)
    return images

def display_images(urls: list, 
    columns=5, width=20, height=8, max_images=15, 
    label_wrap_length=50, label_font_size=8):
    
    images = urls_to_pils(urls)
    
    if not images:
        print("No images to display.")
        return 

    if len(images) > max_images:
        print(f"Showing {max_images} images of {len(images)}:")
        images=images[0:max_images]

    height = max(height, int(len(images)/columns) * height)
    plt.figure(figsize=(width, height))
    for i, image in enumerate(images):

        plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        plt.imshow(image)

        if hasattr(image, 'filename'):
            title=image.filename
            if title.endswith("/"): title = title[0:-1]
            title=os.path.basename(title)
            title=textwrap.wrap(title, label_wrap_length)
            title="\n".join(title)
            plt.title(title, fontsize=label_font_size); 

# update as needed
# file_path = r"file_path"
# resnet_features = get_resnet(file_path)

def get_caption_descriptors(model, resnet_features):
    """
    Returns the caption descriptors of all the images in the resnet dataset
    """
    image_descriptors = np.array(list(resnet_features.values()))
    caption_descriptors = model(image_descriptors)
    return caption_descriptors

"""
make sure to call this in the jupyter notebook
get_caption_descriptors(model, resnet_features)
"""

# def download_image(img_url: str) -> Image:
#     """Fetches an image from the web.

#     Parameters
#     ----------
#     img_url : string
#         The url of the image to fetch.

#     Returns
#     -------
#     PIL.Image
#         The image."""

#     response = requests.get(img_url)
#     return Image.open(io.BytesIO(response.content))

# path = "file path"
# glove = get_glove(path)
# glove_shape = glove["word"].shape

def search_image(
    query: str, idf_dict, glove, glove_shape, 
    caption_descriptors, resnet_features
) -> np.ndarray:
    """
    Returns 
    -------
    sorted_image_ids
        the sorted image ids from best to worst
    """
    # should return the weighted sum of the query as a vector
    # Change (1, 200) to (200, 1) array
    embedded_input = np.array(embed_idf_weighted_sum(
        [query], idf_dict, glove, glove_shape
    )).T
    
#     print(embedded_input)
    
    # (N, 200) @ (200, 1) = (N, 1)
    # How well does the query match up against all the caption descriptors?
    match_scores = (caption_descriptors @ embedded_input).flatten()
#     print(match_scores, match_scores.shape)
    
    # the indices would be useful later when retrieving the image ids
    # shape-(82612,)
    match_scores_indices = np.argsort(np.array(match_scores))
    
    # getting the sorted image ids from best match to worse match
    sorted_image_ids = np.array(list(resnet_features.keys()))[match_scores_indices]
#     print(sorted_image_ids)

    return sorted_image_ids

def display_top_images(
    image_ids_to_urls: dict, sorted_image_ids: np.ndarray, top_k: int
):
    """
    Displays top-k images in some kind of grid form
    """
    # get from image ids to urls dictionary, should be sorted
    urls = [
        image_ids_to_urls[sorted_image_ids[i]] for i in range(top_k)
    ]
    display_images(urls)

