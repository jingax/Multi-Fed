import pandas as pd
import numpy as np
# from keras.preprocessing import image
# import torch
# import keras
# import itertools
import torch
from PIL import Image
import torchvision.transforms as transforms


np.random.seed(0)

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


def get_covid(_type_,_size_):
    df = pd.DataFrame(columns = ['image', 'label'])
    # normal, covid, nemoina, opa
    no_images = [10192, 3616, 1345, 6012]
    if(_type_ != "train"):
        _size_ *= 1    
    
    act_no = [_size_]*4
    label = 0
    permu = np.random.permutation(no_images[label])
    # print(min(act_no[label], no_images[label]))
    i = 0
    while i < min(act_no[label], no_images[label]):
        df = df.append({'image' : f'./covid/COVID-19_Radiography_Dataset/Normal/images/Normal-{permu[i]+1}.png', 'label' : label}, ignore_index = True)
        i += 1
    label = 1
    # print(min(act_no[label], no_images[label]))
    i = 0
    permu = np.random.permutation(no_images[label])
    while i < min(act_no[label], no_images[label]):
        df = df.append({'image' : f'./covid/COVID-19_Radiography_Dataset/COVID/images/COVID-{permu[i]+1}.png', 'label' : label}, ignore_index = True)
        i += 1
    i = 0
    label = 2
    permu = np.random.permutation(no_images[label])
    # print(min(act_no[label], no_images[label]))
    while i < min(act_no[label], no_images[label]):
        df = df.append({'image' : f'./covid/COVID-19_Radiography_Dataset/Viral Pneumonia/images/Viral Pneumonia-{permu[i]+1}.png', 'label' : label}, ignore_index = True)
        i += 1
    
    
    # i = 0
    # label = 3
    # permu = np.random.permutation(no_images[label])
    # # print(min(act_no[label], no_images[label]))
    # while i < min(act_no[label], no_images[label]):
    #     df = df.append({'image' : f'./covid/COVID-19_Radiography_Dataset/Lung_Opacity/images/Lung_Opacity-{permu[i]+1}.png', 'label' : label}, ignore_index = True)
    #     i += 1
    # df = df.sample(frac=1).reset_index(drop=True)
    
    images = []
    labels = []
    GRAY = transforms.Grayscale()
    transform = transforms.Compose([transforms.PILToTensor()])
    for index, row in df.iterrows():
        image = Image.open(row['image'])
        image = image.resize((200,200))
        image = GRAY(image)
        img_tensor = transform(image).to(torch.float)
        if(img_tensor.shape[0]!=1):
            print(img_tensor.shape)
            continue
        images.append(img_tensor)
        labels.append(row['label'])
        # print(index, img_tensor.dtype,row['label'])
    labels = torch.tensor(labels,dtype=torch.long)
    images = torch.stack(images, dim=0)
    # images = np.array(images)
    # print(type(images), type(labels))
    # print(images.shape, labels.shape)
    return images, labels
