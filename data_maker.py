import pandas as pd
import numpy as np
from keras.preprocessing import image
from tqdm import tqdm
import torch
import keras

df = pd.read_csv('./celeb/anno.csv')
df = df.sample(frac=1).reset_index(drop=True)
data_length = 1200
corr = 1.0

df = df[["q", "file_name", "High_Cheekbones", "Mouth_Slightly_Open"]]

t_df = df.loc[(df['High_Cheekbones'] == 0) & (df['Mouth_Slightly_Open'] == 0)]
new_df = t_df.head(n=int(data_length*corr*0.5))

t_df = df.loc[(df['High_Cheekbones'] == 1) & (df['Mouth_Slightly_Open'] == 1)]
t_df = t_df.head(n=int(data_length*corr*0.5))
new_df = new_df.append(t_df)


t_df = df.loc[(df['High_Cheekbones'] == 0) & (df['Mouth_Slightly_Open'] == 1)]
t_df = t_df.head(n=int((data_length - int(data_length*corr))*0.5))
new_df = new_df.append(t_df)

t_df = df.loc[(df['High_Cheekbones'] == 1) & (df['Mouth_Slightly_Open'] == 0)]
t_df = t_df.head(n=int((data_length - int(data_length*corr))*0.5))
new_df = new_df.append(t_df)

# for i in range(2):
#     for j in range(2):
#         if(i==j):
#             new_df = df.head(n=int(data_length*corr*0.5))
#         else:
            # new_df = df.head(n=(data_length - int(data_length*corr)*0.5))
            



df = new_df.sample(frac=1).reset_index(drop=True)
# print(df)

# exit()
# df = df[["q", "file_name", "High_Cheekbones", "Mouth_Slightly_Open", "Attractive"]]
# print(df.head())
train_image = []
print("Loading Images...")
for i in tqdm(range(data_length)):
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
# y = torch.reshape(y, (1,y.shape[0]))
# y = torch.squeeze(y)
X = torch.from_numpy(X)
print(y.shape)
print(X.shape)
torch.save(X, 'X_val_100.pt')
torch.save(y, 'Y_val_100.pt')
