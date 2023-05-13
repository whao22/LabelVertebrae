from scipy.interpolate import interp1d
import numpy as np
from qtpy.QtCore import QPointF

def getCenterline(mask):
    contour=[]
    for p in mask.points:
        p_np=np.array([p.x(),p.y()])
        contour.append(p_np)
    contour=np.array(contour)

    p_left,p_right=contour[:17],contour[17:][::-1]

    f_left=interp1d(p_left[:,1],p_left[:,0],kind="quadratic")
    f_right=interp1d(p_right[:,1],p_right[:,0],kind="quadratic")
    y_min=max(p_left[0,1],p_right[0,1])
    y_max=min(p_left[-1,1],p_right[-1,1])
    y_new=np.linspace(y_min,y_max,num=17,endpoint=True)
    x_new_left=f_left(y_new)
    x_new_right=f_right(y_new)
    x_new=(x_new_left+x_new_right)/2
    
    centerline=[]
    for i in range(len(y_new)):
        p=QPointF(x_new[i],y_new[i])
        centerline.append(p)

    return centerline