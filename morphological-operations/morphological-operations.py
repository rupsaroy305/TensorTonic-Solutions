import numpy as np
def morphological_op(image, kernel, operation):
    image=np.asarray(image)
    kernel=np.asarray(kernel)
    kh,kw=kernel.shape
    ph,pw=kh//2,kw//2
    pad=np.pad(image,((ph,ph),(pw,pw)))
    h,w=image.shape
    out=np.zeros((h,w),dtype=int)
    for i in range(h):
        for j in range(w):
            win=pad[i:i+kh,j:j+kw]
            if operation=="erode":
                out[i,j]=int(np.all(win[kernel==1]==1))
            else:
                out[i,j]=int(np.any(win[kernel==1]==1))
    return out.tolist()