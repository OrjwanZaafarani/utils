from load_transforms import *
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset
from collections import defaultdict
from tqdm import tqdm
import torch
from PIL import Image
import random

class DatasetClass(Dataset):
    def __init__(self, df , df_class_mapping_dict, centroids, transform = None):
        self.df = df
        self.transform = transform
        self.centroids = centroids
        self.df_class_mapping_dict = df_class_mapping_dict
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        #locally
#         img_path = self.df['imagePath'][index]
        # on DGX    
        img_path = self.df['imagePath'][index].replace('/home/dmmil/Desktop/nas','/app')
        
        img_label = self.df['class'][index]
        img_label_idx = torch.tensor(self.df_class_mapping_dict[img_label])
        resize_size = self.centroids[self.df['w_h_cluster'][index]]
        img_PIL = Image.open(img_path)
        if img_PIL.mode != "RGB":
            img_PIL = img_PIL.convert('RGB')          
        border = (self.df['xmin'][index],\
                  self.df['ymin'][index],\
                  self.df['xmax'][index],\
                  self.df['ymax'][index]) # left, top, right, bottom
        img_PIL = img_PIL.crop(border)
        if int(resize_size[0]) >= 500 or int(resize_size[1]) >=500:
            img_PIL = img_PIL.resize((int(resize_size[0]/2),int(resize_size[1]/2)))
        else:
            img_PIL = img_PIL.resize((int(resize_size[0]),int(resize_size[1])))
        
        img_PIL = self.transform(img_PIL)
        return img_PIL, img_label_idx


class BucketingClass(Sampler):

    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.buckets = self._get_buckets(dataset)
        self.num_examples = len(dataset)

    def __iter__(self):
        batch = []
        # Process buckets in random order
        dims = random.sample(list(self.buckets), len(self.buckets))
        for dim in dims:
            # Process images in buckets in random order
            bucket = self.buckets[dim]
            bucket = random.sample(bucket, len(bucket))
            for idx in bucket:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            # Yield half-full batch before moving to next bucket
            if len(batch) > 0:
                yield batch
                batch = []

    def __len__(self):
        return self.num_examples

    def _get_buckets(self, dataset):
        buckets = defaultdict(list)
        print('Building the dictionary of buckets:')
        for i in tqdm(range(len(dataset)), position=0, leave=True):
            img, _ = dataset[i]
            for im in img:
#                 dims = im.size
                dims = im.shape
                buckets[dims].append(i)
        return buckets


def initialize_datasets(train_df, val_df, train_dfs, val_dfs, df_class_mapping_dict, centroids, transforms):
    
    # Training set
    low_class_train_dataset1 = DatasetClass(train_dfs[0], df_class_mapping_dict, centroids[0], transform = transforms[0])
    low_class_train_dataset2 = DatasetClass(train_dfs[0], df_class_mapping_dict, centroids[0], transform = transforms[1])
    low_class_train_dataset3 = DatasetClass(train_dfs[0], df_class_mapping_dict,  centroids[0], transform = transforms[2])

    medium_class_train_dataset1 = DatasetClass(train_dfs[1], df_class_mapping_dict,  centroids[0], transform = transforms[3])

    all_class_train_dataset1 = DatasetClass(train_df, df_class_mapping_dict,  centroids[0], transform = transforms[4])
    all_class_train_dataset2 = DatasetClass(train_df, df_class_mapping_dict,  centroids[0], transform = transforms[5])
    all_class_train_dataset3 = DatasetClass(train_df, df_class_mapping_dict,  centroids[0], transform = transforms[6])




    # Validation set
    low_class_val_dataset1 = DatasetClass(val_dfs[0], df_class_mapping_dict,  centroids[1] ,transform = transforms[0])
    low_class_val_dataset2 = DatasetClass(val_dfs[0], df_class_mapping_dict,  centroids[1] ,transform = transforms[1])
    low_class_val_dataset3 = DatasetClass(val_dfs[0], df_class_mapping_dict,  centroids[1] ,transform = transforms[2])

    medium_class_val_dataset1 = DatasetClass(val_dfs[1], df_class_mapping_dict,  centroids[1] ,transform = transforms[3])

    all_class_val_dataset1 = DatasetClass(val_df, df_class_mapping_dict,  centroids[1] ,transform = transforms[4])
    all_class_val_dataset2 = DatasetClass(val_df, df_class_mapping_dict,  centroids[1] ,transform = transforms[5])
    all_class_val_dataset3 = DatasetClass(val_df, df_class_mapping_dict,  centroids[1] , transform = transforms[6])
    
    
    # Concatenate datasets
    df_train_concatenated_dataset = ConcatDataset([low_class_train_dataset1, low_class_train_dataset2, \
                                                   low_class_train_dataset3, medium_class_train_dataset1, \
                                                   all_class_train_dataset1, all_class_train_dataset2, \
                                                   all_class_train_dataset3])

    df_val_concatenated_dataset = ConcatDataset([low_class_val_dataset1, low_class_val_dataset2, low_class_val_dataset3,
                                                       medium_class_val_dataset1, all_class_val_dataset1, all_class_val_dataset2,
                                                       all_class_val_dataset3])
    
    return df_train_concatenated_dataset, df_val_concatenated_dataset


def prepare_dataloaders(df_class_mapping_dict, train_batch_size, val_batch_size, train_dataset, val_dataset, num_workers):
    
    bucketized_train_dataset = BucketingClass(train_dataset, train_batch_size)
    dataloader_train_dataset = torch.utils.data.DataLoader(train_dataset,\
                                                           num_workers=num_workers,\
                                                           batch_sampler=bucketized_train_dataset)
    
    bucketized_val_dataset = BucketingClass(val_dataset, val_batch_size)
    dataloader_val_dataset = torch.utils.data.DataLoader(val_dataset,\
                                                         num_workers=num_workers,\
                                                         batch_sampler=bucketized_val_dataset)
    
    dataloaders = {'train': dataloader_train_dataset, 'val': dataloader_val_dataset}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    
    return dataloaders

