import trimesh as tri
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def show_image(img, title=""):
    
    plt.imshow(img, cmap='gray')
    plt.show()
    return


def save_image(img, file_name=""):
    plt.imsave(file_name, img, cmap='gray')
    return
