from datasets.cityscapes_halfres_features_dataset import CityscapesHalfresFeaturesDataset

dataset = CityscapesHalfresFeaturesDataset(train=False)
print(dataset[0])