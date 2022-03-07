import trimesh as tri
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

def show_image(img, title=""):
    # img /= np.max(img)
    fig = plt.figure()
    
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(img, cmap='gray')
    plt.colorbar()
    plt.show()
    
    return


