from pathlib import Path
import json
import random

from collections import defaultdict

# load COCO metadata
filename = "captions_train2014.json"
with Path(filename).open() as f:
    coco_json_data = json.load(f)

"""82,783 images with at least 5 captions each"""
"""
Create a class that organizes all of the COCO data. It might store
the following
- All the image IDs
- All the caption IDs
- Various mappings between image/caption IDs, and associating 
caption-IDs with captions
    - image-ID -> [cap-ID-1, cap-ID-2, ...]
    - caption-ID -> image-ID
    - caption-ID -> caption (e.g. 24 -> "two dogs on the grass")
"""

class CocoData:
    def __init__(self, dataset):
        """
        Takes in the Coco data and assings it's images list and annotations list to two seperate variables
        """
        self.dataset = dataset
        self.images = dataset["images"]
        self.captions = dataset["annotations"]
    
    def image_ids_to_caption_ids(self) -> defaultdict:
        """
        Returns dictionary of image ids mapped to a list of caption ids
        """
        image_ids_to_caption_ids_dict = defaultdict(list)
        
        for caption in self.captions:
            image_ids_to_caption_ids_dict[caption["image_id"]].append(caption["id"])
            # puts all the caption ids in one defaultdict(list)
            
        # caption is a dictionary, self.captions is a list
        return image_ids_to_caption_ids_dict
    
    def caption_ids_to_image_ids(self) -> dict:
        """
        Returns dictionary of caption ids mapped to image ids
        """
        caption_ids_to_image_ids_dict = dict()
        
        for caption in self.captions:
            caption_ids_to_image_ids_dict[caption["id"]] = caption["image_id"]
            # puts all the caption ids in one defaultdict(list)
            
        return caption_ids_to_image_ids_dict
    
    def caption_ids_to_captions(self) -> defaultdict:
        """
        Returns dictionary of caption ids mapped to captions
        """
        caption_ids_to_captions_dict = defaultdict(list)
        
        for caption in self.captions:
            caption_ids_to_captions_dict[caption["id"]].append(caption["caption"])
            # puts all the captions in one defaultdict(list)
            
        return caption_ids_to_captions_dict

    """
    {'license': 5,
    'file_name': 'COCO_train2014_000000057870.jpg',
    'coco_url': 'http://images.cocodataset.org/train2014/COCO_train2014_000000057870.jpg',
    'height': 480, 'width': 640, 'date_captured': '2013-11-14 16:28:13',
    'flickr_url': 'http://farm4.staticflickr.com/3153/2970773875_164f0c0b83_z.jpg',
    'id': 57870}
    """
    
    def image_ids_to_urls(self) -> dict:
        """
        Returns dictionary of image ids mapped to their coco urls
        """
        image_ids_to_urls_dict = dict()
        
        for image in self.images:
            image_ids_to_urls_dict[image["id"]] = image["coco_url"]
            # puts all the image ids in one defaultdict(list)
            
        return image_ids_to_urls_dict
    

    # {'image_id': 373960, 'id': 73592, 'caption': 'A group of skiers skiing towards a large mountain'}

    def get_image(self, image_id_to_captions, image_ids_to_urls, image_id=None) -> tuple:
        """
        Get an image from the dataset based on an image id
        If one is not provided, get a random image

        Returns
        captions for one image, url
        """
        if image_id is None:
            # random image id
            image_id = random.randint(len(self.images))
        
        image_url = image_ids_to_urls[image_id]
        image_caption = image_id_to_captions[image_id]
        
        return image_caption, image_url

coco_dataset = CocoData(coco_json_data)

image_ids_to_caption_ids_dict = coco_dataset.image_ids_to_caption_ids()
caption_ids_to_image_ids_dict = coco_dataset.caption_ids_to_image_ids()
caption_ids_to_captions_dict = coco_dataset.caption_ids_to_captions()
image_ids_to_urls_dict = coco_dataset.image_ids_to_urls()

# print(image_ids_to_caption_ids_dict)
# print(caption_ids_to_image_ids_dict)
# print(list(caption_ids_to_captions_dict.items())[:100])
# print(image_ids_to_urls_dict)