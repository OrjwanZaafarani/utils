import torch
import timm
from torch import nn
import wandb

def load_the_model(device, train_df, out_features_of_prev_model, model_name='tf_efficientnet_b8', weights_path = None, in_features_of_prev_model=2816):
    
    print('out_features_of_prev_model', out_features_of_prev_model)
    print('in_features_of_prev_model',in_features_of_prev_model)
    
    model = timm.create_model(model_name, pretrained=True)
    
    unique_classes_length = len(train_df['class'].unique())
    print('out_features_of_current_model', unique_classes_length)
    
    config = wandb.config 
    config.in_features_of_prev_model = in_features_of_prev_model
    config.out_features_of_prev_model = out_features_of_prev_model
    config.out_features_of_current_model = unique_classes_length
    
    
    # model = model.to(device)
    if weights_path is None:
        model.classifier = nn.Sequential(
                nn.Linear(in_features_of_prev_model , 512),
                nn.Linear(512 , unique_classes_length)
            )
    else:
        model.classifier = nn.Sequential(
                nn.Linear(in_features_of_prev_model , 512),
                nn.Linear(512 , out_features_of_prev_model)
            )

    model = nn.DataParallel(model)

    # model.module.classifier = nn.Linear(in_features=in_features_of_prev_model ,out_features=out_features_of_prev_model)

    model.module.conv_stem.requires_grad = False
    model.module.act1.requires_grad = False
    model.module.blocks.requires_grad = False
    model.module.conv_head.requires_grad=False
    model.module.act2.require_grad=False
    model.module.global_pool.require_grad=True
    model.module.classifier._modules['0'].require_grad=True
    model.module.classifier._modules['1'].require_grad=False

    if weights_path is not None:
        weights = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(weights)
        
        model.module.classifier = nn.Sequential(
                nn.Linear(in_features_of_prev_model , 512),
                nn.Linear(512 , unique_classes_length)
            )
            
    # model = model.to(device)
        
    
    return model