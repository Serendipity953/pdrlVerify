
import portion as P
import numpy as np
from numpy.linalg import *

def get_intersection(dim,bound_1,bound_2):
    intersection=[]
    for i in range(dim):
        interval_1=P.closed(bound_1[i],bound_1[i+dim])
        interval_2=P.closed(bound_2[i],bound_2[i+dim])
        intersection.append(interval_1&interval_2)
    return intersection
def get_intersection_volum(intersection):
    dim=len(intersection)
    volum=0
    mat=np.zeros((dim,dim))
    flag=0
    for i in range(dim):
        if intersection[i]==P.empty():
            volum = 0
            flag = 1
            #print("the", i, "th dimension dosen't have intersection")
            break
        elif intersection[i].upper-intersection[i].lower==0:
            volum = 0
            flag = 1
            #print("the", i, "th dimension Two sections are connected.")
            break
        else:
            mat[i][i]=abs(intersection[0].upper-intersection[0].lower)
    if flag==0:
        #print(mat)
        volum = det(mat)
    return volum

if __name__ == "__main__":
    b1=[0.0,0.0,0.0,1,1,1]
    b2=[0.5,0.5,0.5,1.5,1.5,1.5]
    inter=get_intersection(3,b1,b2)
    v=get_intersection_volum(inter)
    print(inter,v,type(P.to_data(inter[1])))