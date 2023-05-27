import splitfolders

def split_data (input_folder,output="data/random_images/split_images",train_ratio=.8, val_ratio=.2):
    # Dividim les dades en training i validation
    # amb una proporcio de 0.8 i 0.2 respectivament

    splitfolders.ratio(input_folder,
                       output=output,
                       seed=42,
                       ratio=(train_ratio, val_ratio),
                       group_prefix=None)