import trimesh as tri
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def show_image(img, title=""):
    # img /= np.max(img)
    fig = plt.figure()
    
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(img, cmap='gray')
    plt.colorbar()
    plt.show()
    
    return


def save_image(img, file_name=""):
    plt.imsave(file_name, img, cmap='gray')
    return
