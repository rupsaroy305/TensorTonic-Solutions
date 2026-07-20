import numpy as np
import math

def roi_pool(feature_map, rois, output_size):
    f=np.asarray(feature_map)
    out=[]
    for x1,y1,x2,y2 in rois:
        h,w=y2-y1,x2-x1
        p=np.empty((output_size,output_size),f.dtype)
        for i in range(output_size):
            hs=y1+(i*h)//output_size;he=y1+((i+1)*h)//output_size
            if he==hs:he=hs+1
            for j in range(output_size):
                ws=x1+(j*w)//output_size;we=x1+((j+1)*w)//output_size
                if we==ws:we=ws+1
                p[i,j]=f[hs:he,ws:we].max()
        out.append(p.tolist())
    return out