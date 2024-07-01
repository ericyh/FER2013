import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/eric/Documents/VS_CODE/FER2013/fer2013.csv")
df.columns = ["emotion", "pixels"]
emotions = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]

train, val, test = np.split(df.sample(frac = 1), [int(0.7*len(df)), int(0.85*len(df))])
image = train.iloc[9][1].split(" ")
image = np.array([float(i) for i in image]).reshape(48,48)
plt.imshow(image)
plt.show()