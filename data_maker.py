import pandas as pd
import numpy as np
from keras.preprocessing import image
import torch
import keras
import itertools



def celebA_loader(corr,data_length,n_clients):
    df = pd.read_csv('./celeb/anno.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    
    task_list = ["High_Cheekbones", "Mouth_Slightly_Open","Smiling","5_o_Clock_Shadow","Arched_Eyebrows","Attractive","Bags_Under_Eyes","Bald","Bangs","Big_Lips","Big_Nose","Black_Hair","Blond_Hair","Blurry","Brown_Hair","Bushy_Eyebrows","Chubby","Double_Chin","Eyeglasses","Goatee","Gray_Hair","Heavy_Makeup","Male","Mustache","Narrow_Eyes","No_Beard","Oval_Face","Pale_Skin","Pointy_Nose","Receding_Hairline","Rosy_Cheeks","Sideburns","Straight_Hair","Wavy_Hair","Wearing_Earrings","Wearing_Hat","Wearing_Lipstick","Wearing_Necklace","Wearing_Necktie","Young"]
    task_list = task_list[:n_clients]
    df = df[["q", "file_name"]+ task_list]
    t_df = df


    for i in range(n_clients):
        t_df = t_df.loc[(df[task_list[i]] == 0)]
    
    new_df = t_df.head(n=int(data_length*corr/2))
    
    t_df = df


    for i in range(n_clients):
        t_df = t_df.loc[(df[task_list[i]] == 1)]
    
    t_df = t_df.head(n=int(data_length*corr/2))
    new_df = new_df.append(t_df)

    




    lst = [list(i) for i in itertools.product([0, 1], repeat=n_clients)]
    lst = lst[1:-1]
    rest = len(lst)
    # print(rest)
    for per in lst:
        t_df = df
        for i in range(n_clients):
            t_df = t_df.loc[(df[task_list[i]] == per[i])]
        t_df = t_df.head(n=int((data_length - int(data_length*corr))/rest))
        new_df = new_df.append(t_df)
    
    # print(new_df)
    # exit()
    


        


    # t_df = df.loc[(df['High_Cheekbones'] == 0) & (df['Mouth_Slightly_Open'] == 0)]
    # new_df = t_df.head(n=int(data_length*corr*0.5))

    # t_df = df.loc[(df['High_Cheekbones'] == 1) & (df['Mouth_Slightly_Open'] == 1)]
    # t_df = t_df.head(n=int(data_length*corr*0.5))
    # new_df = new_df.append(t_df)


    # t_df = df.loc[(df['High_Cheekbones'] == 0) & (df['Mouth_Slightly_Open'] == 1)]
    # t_df = t_df.head(n=int((data_length - int(data_length*corr))*0.5))
    # new_df = new_df.append(t_df)

    # t_df = df.loc[(df['High_Cheekbones'] == 1) & (df['Mouth_Slightly_Open'] == 0)]
    # t_df = t_df.head(n=int((data_length - int(data_length*corr))*0.5))
    # new_df = new_df.append(t_df)

    # for i in range(2):
    #     for j in range(2):
    #         if(i==j):
    #             new_df = df.head(n=int(data_length*corr*0.5))
    #         else:
                # new_df = df.head(n=(data_length - int(data_length*corr)*0.5))
                



    df = new_df.sample(frac=1).reset_index(drop=True)
    train_image = []
    for i in range(data_length):
        img = keras.utils.load_img('./celeb/img_align_celeba/'+df['file_name'][i],target_size=(178,218,3))
        img = keras.utils.img_to_array(img)
        img = img/255
        train_image.append(img.T)
    X = np.array(train_image)

    df = df.head(n=data_length)
    y = np.array(df.drop(['q', 'file_name'],axis=1))
    y = y.astype('float32')
    # y = y[:,0]
    y = torch.from_numpy(y)
    y = y.long()
    # y = torch.reshape(y, (1,y.shape[0]))
    # y = torch.squeeze(y)
    X = torch.from_numpy(X)
    # torch.save(X, 'X_val_100.pt')
    # torch.save(y, 'Y_val_100.pt')

    return X, y

