# from time import pthread_getcpuclockid
import numpy as np
import math

def cal_angles(lines):
    l=np.ones([4,2])

    for line in lines:
        v=np.array([line.points[1].x(),line.points[1].y()])-np.array([line.points[0].x(),line.points[0].y()])
        l[int(line.label[-1])-1]=v

    def m(v):
        return np.sqrt((v**2).sum())

    mt=math.acos(abs(np.dot(l[0],l[1])/(m(l[0])*m(l[1]))))/math.pi*180
    pt=math.acos(abs(np.dot(l[1],l[2])/(m(l[1])*m(l[2]))))/math.pi*180
    tl=math.acos(abs(np.dot(l[2],l[3])/(m(l[2])*m(l[3]))))/math.pi*180
    
    return [mt,pt,tl]