import pickle

# print(type(resnet_18_features)) #dict mapping image ids to descriptors
# path = r"C:/Users/student/Documents/BWSI Stuff/Week3Capstone_Datasets/resnet18_features.pkl"

def get_resnet(filepath):
    with open(filepath, mode="rb") as f:
        resnet18_features = pickle.load(f)
    return resnet18_features