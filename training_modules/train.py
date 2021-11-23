from training_script import *
from load_the_model import *
from preprocess_dataframe import *
from prepare_datasets_dataloaders import *
from clustering import *
from load_transforms import *
import torch
import pandas as pd
import argparse
import wandb



msg = "Training the logo recognition system"

# Initialize parser
parser = argparse.ArgumentParser(description = msg)

# Adding optional argument
parser.add_argument("-s", "--seed", type=int, default = 17, help = "seed")
parser.add_argument("-n", "--exp_num", type=int, help = "the experiment number")
parser.add_argument("-p", "--csv_path", type=str, help = "the path to the dataset csv file")
parser.add_argument("-tb", "--train_batch", type=int, default = 128, help = "the training batch size")
parser.add_argument("-vb", "--val_batch", type=int, default = 64, help = "the validation batch size")
parser.add_argument("-nw", "--num_workers", type=int, default = 8, help = "the number of workers for the dataloaders")
parser.add_argument("-m", "--model", type=str, default = 'tf_efficientnet_b8', help = "the efficient net model name")
parser.add_argument("-ts", "--train_split", type=float, default = 0.8, help = "the training split percentage")
parser.add_argument("-w", "--weights_path", type=str, help = "the path to the weights")
parser.add_argument("-of", "--out_feat", type=int, help = "the out features of the previous model")
parser.add_argument("-if", "--in_feat", type=int, help = "the in features of the previous model")
parser.add_argument("-lr", "--learning_rate", type=float, default = 0.001, help = "the learning rate")
parser.add_argument("-mo", "--momentum", type=float, default = 0.9, help = "the learning rate")
parser.add_argument("-sss", "--scheduler_step_size", type=int, default = 10, help = "the scheduler step size")
parser.add_argument("-sg", "--scheduler_gamma", type=float, default = 0.1, help = "the scheduler gamma")
parser.add_argument("-e", "--num_epochs", type=int, default = 50, help = "the number of epochs")
parser.add_argument("-c", "--n_clusters", type=int, default = 50, help = "the number of Kmeans clusters")

 
# Read arguments from command line
args = parser.parse_args()

wandb.init(project="LogoRecognition")
wandb.run.name = 'exp_'+str(args.exp_num)

config = wandb.config    

seed = args.seed
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
exp_num = args.exp_num
csv_path = args.csv_path
train_batch_size = args.train_batch
val_batch_size = args.val_batch
num_workers = args.num_workers
n_clusters = args.n_clusters
    
config.exp_num = args.exp_num
config.seed = args.seed
config.csv_path = args.csv_path         
config.train_batch_size = args.train_batch
config.val_batch_size = args.val_batch
config.num_workers = args.num_workers
config.model = args.model
config.train_split = args.train_split    
config.weights_path = args.weights_path
config.out_feat = args.out_feat
config.in_feat = args.in_feat
config.learning_rate = args.learning_rate
config.scheduler_step_size = args.scheduler_step_size
config.scheduler_gamma = args.scheduler_gamma
config.num_epochs = args.num_epochs
config.n_clusters = args.n_clusters
                     

df = extract_info_dataframe(csv_path)

X_train_df, X_val_df, y_train_df, y_val_df = split_train_val_indicies(df, train_size=args.train_split)

df_class_mapping_dict, train_DF, val_DF = prepare_train_val_dataframes(df, X_train_df, X_val_df)

train_DF, val_DF, df_train_classes_list, df_val_classes_list = find_low_medium_high_classes(df, train_DF, val_DF, X_train_df, X_val_df, y_train_df, y_val_df)

train_DF, val_DF, centroids_train, centroids_val, min_train_batch_size, min_val_batch_size = cluster(train_DF, val_DF, n_clusters)

train_dfs, val_dfs = create_low_medium_high_dfs(train_DF, val_DF, df_train_classes_list, df_val_classes_list)

centroids = [centroids_train, centroids_val]
train_dataset, val_dataset = initialize_datasets(train_DF, val_DF, train_dfs, val_dfs, \
                                                 df_class_mapping_dict, centroids, transforms)

dataloaders = prepare_dataloaders(df_class_mapping_dict, train_batch_size, val_batch_size, train_dataset, val_dataset, num_workers)

model = load_the_model(device, train_DF, out_features_of_prev_model = args.out_feat, model_name=args.model, weights_path = args.weights_path, in_features_of_prev_model=args.in_feat)

dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

criterion, optimizer_ft, exp_lr_scheduler = define_hyperparameters(device, model, lr=args.learning_rate, momentum=args.momentum, scheduler_step_size=args.scheduler_step_size, scheduler_gamma=args.scheduler_gamma)

wandb.init(project="exp_4")
torch.cuda.empty_cache()

train(exp_num, seed, model, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler, \
      device=device, num_epochs=args.num_epochs)

