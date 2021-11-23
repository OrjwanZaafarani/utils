import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def extract_info_dataframe(csv_path):

    # Import the csv file
    df = pd.read_csv(csv_path)
    
    # Print the length of dataframe
#     logger.log('The length of the dataframe ')
#     logger.log(str(len(df))
#     logger.log('\n)

    if csv_path.endswith('/LitW_DF.csv'):
        print('yes')
        df['xmin']=0
        df['xmax']=df['width']
        df['ymin']=0
        df['ymax']=df['height']
            
    return df.dropna().reset_index(drop=True)

def split_train_val_indicies(df, train_size=0.8):
    # Delete rows with classes of count 1
    counts = df['class'].value_counts()
    df = df[~df['class'].isin(counts[counts == 1].index)]
#     logger.log('Successfully deleted rows with classes of count 1 \n')
    
    X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(df['imagePath'], df['class'], 
                                                    train_size=train_size,
                                                    stratify=df['class'])
    return X_train_df, X_val_df, y_train_df, y_val_df


def prepare_train_val_dataframes(df, X_train_df, X_val_df):
    train_DF = df.iloc[X_train_df.index].reset_index()
    val_DF = df.iloc[X_val_df.index].reset_index()

    # Map classes to numbers
    df_class_mapping_dict = {}
    for i, label in enumerate(train_DF['class'].unique()):
        df_class_mapping_dict[label] = i
    return df_class_mapping_dict, train_DF, val_DF


def generate_weights(df):
    freq = df['class'].value_counts()
    label = df['class'].value_counts().index
    WRS_weights_DF = pd.DataFrame({'class': label, 'weight': freq})
    WRS_weights_DF = WRS_weights_DF.set_index('class')
    WRS_weights_DF['weight'] = 1/WRS_weights_DF['weight'] * (WRS_weights_DF['weight'][0])   
    WRS_weights_DF['weight'] = round(WRS_weights_DF['weight'] / WRS_weights_DF['weight'].mean(), 4)
    
    return WRS_weights_DF.to_dict()['weight']


def find_low_medium_high_classes(df, train_DF, val_DF, X_train_df, X_val_df, y_train_df, y_val_df):
      
    # Find weights
    # Training sets
    df_weights_train = generate_weights(df.iloc[X_train_df.index].reset_index())
    # Validation set
    df_weights_val = generate_weights(df.iloc[X_val_df.index].reset_index())
           
               
    df_train_q1 = np.percentile(list(df_weights_train.values()), 25)  # Q1
    df_train_q2 = np.percentile(list(df_weights_train.values()), 50)  # Q2
    df_train_q3 = np.percentile(list(df_weights_train.values()), 75)  # Q3

    df_val_q1 = np.percentile(list(df_weights_val.values()), 25)  # Q1
    df_val_q2 = np.percentile(list(df_weights_val.values()), 50)  # Q2
    df_val_q3 = np.percentile(list(df_weights_val.values()), 75)  # Q3
    
    
               
    # find low, medium, and high classes
    # Training set
    df_train_low_class = dict((k, v) for k, v in df_weights_train.items() if v < df_train_q2)
    df_train_medium_class = dict((k, v) for k, v in df_weights_train.items() if v >= df_train_q2 and v <= df_train_q3)
    df_train_high_class = dict((k, v) for k, v in df_weights_train.items() if v > df_train_q3)

    train_DF['bbox_width'] = train_DF['xmax'].astype(int) - train_DF['xmin'].astype(int)
    train_DF['bbox_height'] = train_DF['ymax'].astype(int) - train_DF['ymin'].astype(int)

    # Validation set
    df_val_low_class = dict((k, v) for k, v in df_weights_val.items() if v < df_val_q2)
    df_val_medium_class = dict((k, v) for k, v in df_weights_val.items() if v >= df_val_q2 and v <= df_val_q3)
    df_val_high_class = dict((k, v) for k, v in df_weights_val.items() if v > df_val_q3)

    val_DF['bbox_width'] = val_DF['xmax'].astype(int) - val_DF['xmin'].astype(int)    
    val_DF['bbox_height'] = val_DF['ymax'].astype(int) - val_DF['ymin'].astype(int)
    
    df_train_classes_list = [df_train_low_class, df_train_medium_class, df_train_high_class]
    df_val_classes_list = [df_val_low_class, df_val_medium_class, df_val_high_class]
               
    
    
    return train_DF, val_DF, df_train_classes_list, df_val_classes_list


def create_low_medium_high_dfs(train_DF, val_DF, df_train_classes_list, df_val_classes_list):
    # Training set
    df_train_low_class_indecies = []
    for k in list(df_train_classes_list[0].keys()):
        indecies = train_DF.index[train_DF['class'] == k].to_list()
        df_train_low_class_indecies.extend(indecies)

    df_train_medium_class_indecies = []
    for k in list(df_train_classes_list[1].keys()):
        indecies = train_DF.index[train_DF['class'] == k].to_list()
        df_train_medium_class_indecies.extend(indecies)

    df_train_high_class_indecies = []
    for k in list(df_train_classes_list[2].keys()):
        indecies = train_DF.index[train_DF['class'] == k].to_list()
        df_train_high_class_indecies.extend(indecies)


    # Validation set
    df_val_low_class_indecies = []
    for k in list(df_val_classes_list[0].keys()):
        indecies = val_DF.index[val_DF['class'] == k].to_list()
        df_val_low_class_indecies.extend(indecies)

    df_val_medium_class_indecies = []
    for k in list(df_val_classes_list[1].keys()):
        indecies = val_DF.index[val_DF['class'] == k].to_list()
        df_val_medium_class_indecies.extend(indecies)

    df_val_high_class_indecies = []
    for k in list(df_val_classes_list[2].keys()):
        indecies = val_DF.index[val_DF['class'] == k].to_list()
        df_val_high_class_indecies.extend(indecies)
               
    train_dfs = [train_DF.iloc[df_train_low_class_indecies].reset_index(), \
                 train_DF.iloc[df_train_medium_class_indecies].reset_index(), \
                 train_DF.iloc[df_train_high_class_indecies].reset_index()]
    val_dfs = [val_DF.iloc[df_val_low_class_indecies].reset_index(), \
               val_DF.iloc[df_val_medium_class_indecies].reset_index(), \
               val_DF.iloc[df_val_high_class_indecies].reset_index()]
    

    return train_dfs , val_dfs


def generate_low_medium_high_class_dataframes(csv_path):
    
    df = extract_info_dataframe(logger.log, csv_path)
    if df == None:
        return None
    else:
        X_train_df, X_val_df, y_train_df, y_val_df = split_train_val_indicies(df)

        indicies = find_low_medium_high_classes(df, X_train_df, X_val_df, y_train_df, y_val_df)

        df_train, df_val = split_dataframes(indicies[0], indicies[1], \
                                            indicies[2], indicies[3], \
                                            indicies[4], indicies[5])

        train_low_class_df = df_train.iloc[indicies[0]].reset_index()
        train_medium_class_df = df_train.iloc[indicies[1]].reset_index()
        train_dfs = [train_low_class_df, train_medium_class_df]

        val_low_class_df = df_val.iloc[indicies[3]].reset_index()
        val_medium_class_df = df_val.iloc[indicies[4]].reset_index()
        val_dfs = [val_low_class_df, val_medium_class_df]

        return train_dfs, val_dfs



