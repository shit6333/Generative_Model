import torch
import numpy as np
import matplotlib.pyplot as plt

# 顯示圖片函數
def showImg( img_tensor ):
    img_tensor = img_tensor.view( -1, 28, 28 )
    n = int( np.ceil( np.sqrt( img_tensor.shape[0] ) ) )
    
    plt.figure(figsize=(8,8))
    for index, image in enumerate( img_tensor ):
        plt.subplot( n, n, index+1)
        plt.imshow( image )
        plt.axis('off')