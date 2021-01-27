import imageio
import numpy as np
from PIL import Image
from skimage.transform import resize
import os

def size(img, size, mode=None):
    size = [size[0], size[1]]
    nsize = size
    if mode is not None:
        if mode in ["crop", "fill"]:
            r = (img.shape[0]/img.shape[1]) / (size[0]/size[1])
            nsize = (int(size[0]*r), size[1]) if r > 1 else (size[0], int(size[1]/r))
        elif mode == "fit":
            r = (img.shape[0]/img.shape[1]) / (size[0]/size[1])
            r = 1 / r
            nsize = (int(size[0]/r), size[1]) if r > 1 else (size[0], int(size[1]*r))
        elif mode == "mean":
            r = np.sqrt(size[0]**2 + size[1]**2) / np.sqrt(img.shape[0]**2 + img.shape[1]**2)
            nsize = np.array(img.shape[:2]) * r
    nsize = np.int32(nsize)
    img = resize(img, nsize, anti_aliasing=True, preserve_range=True)
    if mode == "crop":
        img = img[(nsize[0]-size[0])//2:size[0]+(nsize[0]-size[0])//2, (nsize[1]-size[1])//2:size[1]+(nsize[1]-size[1])//2]
    return img

def zoom(img, scale=4):
  img = np.repeat(img, scale, 0)
  img = np.repeat(img, scale, 1)
  return img

def norm(img):
    min_ = np.min(img)
    max_ = np.max(img)
    return (img-min_) / (max_-min_)

def scale(img, scale):
    return size(img, [img.shape[0]*scale, img.shape[1]*scale])

def load(uri):
    return imageio.imread(uri).astype("float32")/255

def tosave(img):
    if img.dtype != np.uint8:
        img = np.uint8(np.clip(img,0,1)*255)
    if len(img.shape) == 4:
        img = img[0]
    if len(img.shape) == 3:
        if img.shape[-1] == 1:
            img = img[...,0]
        elif img.shape[-1] > 4:
            img = img[...,:4]
    return img

def save(img, uri):
    img = tosave(img)
    dirname = os.path.dirname(uri)
    if len(dirname) > 0:
        os.makedirs(os.path.dirname(uri), exist_ok=True)
    imageio.imwrite(uri, img)
    
def show(img):
    img = tosave(img)
    try:
        display(Image.fromarray(img))
    except:
        pass

def clear():
    try:
        from IPython.display import clear_output
        clear_output()
    except:
        pass


class Video:
    def __init__(self, uri='_autoplay.mp4', fps=30.0):
        self.uri = uri
        dirname = os.path.dirname(uri)
        if len(dirname) > 0:
            os.makedirs(os.path.dirname(uri), exist_ok=True)
        self.writer = imageio.get_writer(uri, fps=fps)
    def add(self, img):
        img = tosave(img)
        self.writer.append_data(img)
    def close(self):
        self.writer.close()
    def __enter__(self):
        return self
    def __exit__(self, *kw):
        self.close()
        self.show()
    def show(self):
        try:
            if os.path.splitext(self.uri)[-1].lower() == ".gif":
                from IPython.display import Image
                display(Image(self.uri))
            else:
                from IPython.display import Video
                display(Video(self.uri, html_attributes="loop autoplay muted controls"))
        except:
            pass