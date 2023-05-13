import numpy as np
import matplotlib.pyplot as plt
import time

class BSpline(object):
    EPSILON=1e-5
    def __init__(self,h,p,P=None,U=None,D=None): # D表示数据点，如果D不空表示用于近似数据点得到B-spline，如果D为空表示使用控制点生成B-spline
        self.h = h # 控制点个数=h+1
        self.p = p # 次数
        if D is not None:
            self.m = self.h+self.p+1
            self.D, self.n = D, len(D)-1
            self.gene_T(self.D)
            self.gene_U(self.T)
            self.NT={}
            self.NU={}
            self.gene_N(self.NT,self.T)
            self.gene_N(self.NU,self.U)
            self.approximation(self.D)
        else:
            self.P = P # 控制点
            self.U = U # 节点向量
            self.m = len(U)-1 # 节点向量个数=m+1
            self.NU = {}
            self.gene_N(self.NU,self.U)
        

    def isZero(self,x):
        return True if abs(x)<self.EPSILON else False


    def gene_T(self,D,method="centripetal",a=1/2):
        get_mod_len=lambda k : np.sqrt(np.sum((D[1:k+1]-D[0:k])**2,axis=1))**a
        L=get_mod_len(self.n).sum()

        self.T=np.zeros((self.n+1,))
        for k in range(1,self.n+1):
            if k==0:
                self.T[k]=np.array(0.)
            elif k==self.n:
                self.T[k]=np.array(1.)
            else:
                self.T[k]=get_mod_len(k).sum()/L


    def gene_U(self,T,method="average"):
        valid_num=self.h-self.p # 表示了节点向量中非0、1的节点的个数
        index=np.linspace(0,self.n-1,valid_num+self.p-1).astype(np.int)
        t=T[index]

        self.U=np.zeros((self.m+1,))
        for i in range(0,self.m+1):
            if i<=self.p:
                self.U[i]=np.array(0.)
            elif i>=self.m-self.p:
                self.U[i]=np.array(1.)
            else:
                j=i-self.p-1
                self.U[i]=t[j:j+self.p].mean()


    def gene_N(self,N,uq):
        for i in range(self.p+1):
            N[i]={}
            for j in range(self.h+self.p+1):
                N[i][j]={}

        # 将所有节点值预先计算并保存起来
        for u in uq:
            for i in range(self.h+1):
                self.N_BasicFunc(u,i,self.p,N)


    def N_BasicFunc_DeBoor(self,u,i,p,N):
        assert u<=1.+self.EPSILON and u>=-self.EPSILON , "u should locate in the span of [0,1]."
        
        if u.item() in N[p][i]:
            return N[p][i][u.item()]

        if p==0:
            if u<=self.U[i+1] and u>=self.U[i]:
                N[p][i][u.item()]=1
            else:
                N[p][i][u.item()]=0
        else:
            a,a_=u-self.U[i],self.U[i+p]-self.U[i]
            b,b_=self.U[i+p+1]-u,self.U[i+p+1]-self.U[i+1]
            A=0 if self.isZero(a_) else a/a_
            B=0 if self.isZero(b_) else b/b_
            N[p][i][u.item()]=A*self.N_BasicFunc_DeBoor(u,i,p-1,N)+B*self.N_BasicFunc_DeBoor(u,i+1,p-1,N)
        return N[p][i][u.item()]
    


    def N_BasicFunc(self,u,h,p,N):
        assert u<=1.+self.EPSILON and u>=-self.EPSILON , "u should locate in the span of [0,1]."
        
        if u.item() in N[p][h]:
            return N[p][h][u.item()]

        if u==self.U[0]:
            return self.N_BasicFunc(u+0.0001,h,p,N)
        if u==self.U[self.m]:
            return self.N_BasicFunc(u-0.0001,h,p,N)
        
        # 确定u落在节点区间[u_k,u_{k+1})
        k=-1
        for i in range(1,self.m-1):
            if self.U[i]<=u<self.U[i+1]:
                k=i
                break
        
        # 计算至多p+1个p次基函数值
        N[0][k][u.item()]=1
        for d in range(1,self.p+1):
            # 上
            if k-d>=0:
                N[d][k-d][u.item()]=(self.U[k+1]-u)/(self.U[k+1]-self.U[k-d+1])*N[d-1][k-d+1][u.item()]
            # 中
            for i in range(k-d+1,k):
                N[d][i][u.item()]=N[d-1][i][u.item()]*(u-self.U[i])/(self.U[i+d]-self.U[i])+N[d-1][i+1][u.item()]*(self.U[i+d+1]-u)/(self.U[i+d+1]-self.U[i+1])
            # 下
            N[d][k][u.item()]=N[d-1][k][u.item()]*(u-self.U[k])/(self.U[k+d]-self.U[k])
        # 计算除了至多p+1个p次基函数值之外p次基函数值
        for i in range(0,self.h+1):
            if k-p<=i<=k:
                continue
            else:
                N[self.p][i][u.item()]=0

        return N[p][h][u.item()]


    def approximation(self,D):
        # 生成Q矩阵
        Q=[]
        for i in range(1,self.h):
            SUM_NQ=np.array([0,0],dtype=np.float)
            for k in range(1,self.n):
                Q_k=D[k]-self.NT[self.p][0][self.T[k].item()]*D[0]-self.NT[self.p][self.h][self.T[k].item()]*D[self.n]
                SUM_NQ+=self.NT[self.p][i][self.T[k].item()]*Q_k
            Q.append(SUM_NQ)
        Q_=np.stack(Q,axis=0)

        # 构造N矩阵
        N=[]
        for k in range(1,self.n):
            N_k=[]
            for i in range(1,self.h):
                N_k.append(self.NT[self.p][i][self.T[k].item()])
            N.append(N_k)
        N_=np.array(N)
        
        # 计算控制点
        M=N_.T @ N_
        P=np.linalg.inv(M) @ Q_
        P_=np.concatenate([D[:1],P,D[-1:]],axis=0)
        self.P=P_
        return P_

    def C(self,u):
        ''' calculate the points when u '''
        pt=np.array([0.,0.])
        for i in range(0,self.h+1):
            pt+=self.P[i]*self.N_BasicFunc(u,i,self.p,self.NU)
            # pt+=self.P[i]*self.N_BasicFunc_DeBoor(u,i,self.p,self.NU)
        return pt

    def run(self,uq):
        re=[]
        for u in uq:
            pt=self.C(u)
            re.append(pt)
        re=np.stack(re,axis=0)
        return re


if __name__ =="__main__":

    xx=np.array([0.0551, 0.1001, 0.1474, 0.1937, 0.2362, 0.2781, 0.3226, 0.3758, 0.4313,
                     0.4901, 0.5502, 0.6084, 0.6666, 0.7257, 0.7945, 0.8663, 0.9390])
    yy=np.array([0.4749, 0.4552, 0.4303, 0.3903, 0.3440, 0.2877, 0.2379, 0.2033, 0.1855,
        0.1926, 0.2247, 0.2889, 0.3673, 0.4577, 0.5320, 0.5773, 0.5930])
    a=np.argsort(xx)
    xx=xx[a]
    yy=yy[a]
    data = np.stack([xx,yy],axis=0).T
    print(data.shape)
    bs=BSpline(h=11,p=3,D=data)





    # P=np.array([[ 28.2112, 121.5744],
    #     [ 38.9139, 118.9219],
    #     [ 63.0846, 114.8490],
    #     [100.4503, 101.6915],
    #     [135.9197,  77.3349],
    #     [174.8804,  54.1307],
    #     [223.3508,  44.8938],
    #     [275.6040,  51.6746],
    #     [325.7046,  80.6259],
    #     [384.3779, 131.4625],
    #     [442.3181, 152.0212],
    #     [480.7680, 151.8080]])
    # U = np.array([0.0000, 0.0000, 0.0000, 0.0000, 0.0661, 0.1503, 0.2363, 0.3267, 0.4248,
    #     0.5282, 0.6428, 0.7685, 1.0000, 1.0000, 1.0000, 1.0000])
    # bs=BSpline(h=11,p=3,P=P,U=U)


    uq=np.linspace(0,1,100)
    s=time.time()
    re=bs.run(uq)
    e=time.time()
    print(re)
    print(e-s)
    plt.figure(figsize=(8,4))
    # plt.plot(P[:,0],P[:,1],"b")
    plt.plot(re[:,0],re[:,1],"r")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()