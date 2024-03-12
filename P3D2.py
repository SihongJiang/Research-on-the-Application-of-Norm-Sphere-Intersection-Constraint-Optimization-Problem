# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:19:00 2022

@author: 29911
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 18 08:59:41 2022

@author: 29911
"""

# -*- coding: utf-8 -*-

'''
采用(P3)，数据集为模拟数据

'''
import numpy as np
import scipy
import math
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

#数据集

#读取
GEi = pd.read_csv(r'C:\Users\29911\Documents\gliomaData_GE.csv')
CGHi = pd.read_csv(r'C:\Users\29911\Documents\gliomaData_CGH.csv')
ylabeli = pd.read_csv(r'C:\Users\29911\Documents\gliomaData_y.csv')
weishu=[]
(p,q) = GEi.shape
weishu.append(q)
X1 = [[0 for i in range(q)] for j in range(p)]
for i in range(p):
    for j in range(q):
        X1[i][j] = GEi.iloc[i,j]
(p,q) = CGHi.shape
weishu.append(q)
X2 = [[0 for i in range(q)] for j in range(p)]
for i in range(p):
    for j in range(q):
        X2[i][j] = CGHi.iloc[i,j]
(p,q) = ylabeli.shape
weishu.append(q)
X3 = [[0 for i in range(q)] for j in range(p)]
for i in range(p):
    for j in range(q):
        X3[i][j] = ylabeli.iloc[i,j]
X1_scale = preprocessing.scale(X1);X2_scale = preprocessing.scale(X2);X3_scale = preprocessing.scale(X3)
X1_scale = X1_scale/math.sqrt(weishu[0]);X2_scale = X2_scale/math.sqrt(weishu[1]);X3_scale = X3_scale/math.sqrt(weishu[2])
X = [X1_scale,X2_scale,X3_scale]

J=3
C = np.array([[0,0,1],[0,0,1],[1,1,0]])
S = [10,8,1]

#函数g定义
def g(x):
    #g(x)=x
    #return x
    #g(x) = abs(x)
    return abs(x)
    #g(x) = x**2
    #return x**2

#目标函数定义
def fun(J,C,X,A):
    temp = 0
    n = X[0].shape[0]
    for j in range(J):
        for k in range(j+1,J):
            if C[j][k]!=0:
                temp = temp + g((1/n)*np.dot(np.dot(A[j].T,X[j].T),np.dot(X[k],A[k]))[0][0])
    temp = temp*2
    return temp
#目标函数在A[j]处梯度定义
def gfun(J,j,C,X,A):
    n = X[0].shape[0]
    p = A[j].shape[0]
    #g(x)=x
    '''
    temp = np.array([0 for i in range(p)])
    for k in range(J):
        if C[j][k]!=0:
            temp = temp + (1/n)*np.dot(np.dot(A[k].T,X[k].T),X[j])
    temp = temp.reshape((p,1))
    '''
    #g(x)=abs(x)
    temp = np.array([0.0 for i in range(p)])
    for k in range(J):
        if C[j][k]!=0:
            t = np.dot(np.dot(A[j].T,X[j].T),np.dot(X[k],A[k]))[0][0]
            if t>0:
                temp = temp + (1/n)*np.dot(np.dot(A[k].T,X[k].T),X[j])
            else:
                temp = temp - (1/n)*np.dot(np.dot(A[k].T,X[k].T),X[j])
    temp = temp.reshape((p,1))
    #g(x)=x**2
    '''
    temp = np.array([0 for i in range(p)])
    for k in range(J):
        if C[j][k]!=0:
            temp = temp + 2*(1/n)*np.dot(np.dot(A[k].T,X[k].T),X[j])
    temp = temp.reshape((p,1))
    '''
    return temp

def S1S2_QASB(v,n,z):
    v = [v[i][0] for i in range(n)]
    delta = 1e-9
    U = np.array([0.0 for i in range(10)]); x = [0.0 for i in range(n)]
    flag=0
    lambda_1=0; lambda_2=0; lambda_Q=0; lambda_S=0; Lambda=0
    s_1=0; s_2=0; s=0; s_Q=0; s_S=0; q_1=0; q_2=0; q=0; q_Q=0; q_S=0; temp=0; norm2x=0; v_max=0
    iter_step=0
    z2=z*z 
    temp1=0
    if z<0:
        print("\n z should be nonnegative!")
        x = np.array(x).reshape((n,1))
        return x
    s=0; q=0
    for i in range(n):
        s = s+abs(v[i])
        q = q+v[i]*v[i]
        if(abs(v[i])>v_max):
            v_max=abs(v[i])
    r=v_max
    i=n-1
    if(s*s-z2*q<0):
        Lambda=(s-z*math.sqrt(((i+1)*q-s*s)/(i+1-z*z)))/(i+1)
        flag=0
    else:
        V_i=0; rho_1=0; rho_2=0; rho_Q=0; rho_S=0
        s_1=0; s_2=0; s_Q=0; s_S=0; q_1=0; q_2=0; q_Q=0; q_S=0; s=0; q=0
        lambda_1=0; lambda_2=r
        i=0
        for i in range(n):
            if(abs(v[i]) >= lambda_2):
                rho_2 = rho_2+1
                s_2 = s_2+abs(v[i])
                q_2 = q_2+v[i]*v[i]
                rho_1 = rho_1+1
                s_1 = s_1+abs(v[i])
                q_1 = q_1+v[i]*v[i]
                s = s+abs(v[i])
                q = q+v[i]*v[i]
            if(abs(v[i]) >= lambda_1)and(abs(v[i]) < lambda_2):
                x[V_i] = abs(v[i])
                s_1 = s_1+x[V_i]
                q_1 = q_1+x[V_i]*x[V_i]
                rho_1 = rho_1+1
                V_i = V_i+1
        z2=z*z
        f_lambda_1=(rho_1-z2)*(rho_1*lambda_1-2*s_1)*lambda_1+s_1*s_1-z2*q_1
        f_lambda_2=(rho_2-z2)*(rho_2*lambda_2-2*s_2)*lambda_2+s_2*s_2-z2*q_2
        
        if (f_lambda_1==f_lambda_2):
            #print("f_lambda_1==f_lambda_2",n)
            x = np.array(x).reshape((n,1))
            return x
        
        V_i_b=0
        V_i_e=V_i-1
        while (flag==0):
            iter_step = iter_step + 1
            temp1=math.sqrt((rho_1*q_1-s_1*s_1)/(rho_1-z2))
            lambda_Q=(s_1-z*temp1)/rho_1
            lambda_S=lambda_1-f_lambda_1*(lambda_2-lambda_1)/(f_lambda_2-f_lambda_1)
            if (abs(lambda_Q-lambda_S) <= delta):
                Lambda=lambda_Q; flag=1
                break
            Lambda=(lambda_Q+lambda_S)/2;
            s_Q=s_S=s=0;q_Q=q_S=q=0;rho_Q=rho_S=rho=0
            i=V_i_b; j=V_i_e
            while (i <= j):
                while( (i <= V_i_e) and (x[i] <= Lambda) ):
                    if (x[i]> lambda_Q):
                        s_Q = s_Q+x[i]; q_Q = q_Q+x[i]*x[i]; rho_Q = rho_Q+1
                    i = i+1
                while( (j>=V_i_b) and (x[j] > Lambda) ):
                    if (x[j] > lambda_S):
                        s_S = s_S+x[j]; q_S = q_S+x[j]*x[j]; rho_S = rho_S+1
                    else:
                        s = s+x[j]; q = q+x[j]*x[j]; rho = rho+1
                    j = j-1
                if (i<j):
                    if (x[i] > lambda_S):
                        s_S = s_S+x[i];q_S = q_S+x[i]*x[i]; rho_S=rho_S+1
                    else:
                        s = s+x[i];q = q+x[i]*x[i]; rho = rho+1
                    if (x[j]> lambda_Q):
                        s_Q = s_Q+x[j]; q_Q = q_Q+x[j]*x[j]; rho_Q = rho_Q+1
                    temp=x[i]; x[i]=x[j];  x[j]=temp;
                    i = i+1; j = j-1
            s_S = s_S+s_2; q_S = q_S+q_2; rho_S = rho_S+rho_2
            s = s+s_S; q = q+q_S; rho = rho+rho_S
            s_Q = s_Q+s; q_Q = q_Q+q; rho_Q = rho_Q+rho
            f_lambda_S=(rho_S-z2)*(rho_S*lambda_S-2*s_S)*lambda_S+s_S*s_S-z2*q_S
            f_lambda=(rho-z2)*(rho*Lambda-2*s)*Lambda+s*s-z2*q
            f_lambda_Q=(rho_Q-z2)*(rho_Q*lambda_Q-2*s_Q)*lambda_Q+s_Q*s_Q-z2*q_Q
            if ( abs(f_lambda)< delta ):
                flag=1
                break
            if ( abs(f_lambda_Q)< delta):
                flag=2
                Lambda=lambda_Q
                break
            if (f_lambda <0):
                lambda_2=Lambda;  s_2=s; q_2=q; rho_2=rho
                f_lambda_2=f_lambda        
                lambda_1=lambda_Q; s_1=s_Q; q_1=q_Q; rho_1=rho_Q
                f_lambda_1=f_lambda_Q
                V_i_e=j;  i=V_i_b
                while (i <= j):
                    while( (i <= V_i_e) and (x[i] <= lambda_Q) ):
                        i = i+1
                    while( (j>=V_i_b) and (x[j] > lambda_Q) ):
                        j = j-1
                    if (i<j):                    
                        x[j]=x[i]
                        i=i+1;   j=j-1
                V_i_b=i; V_i=V_i_e-V_i_b+1
            else:
                lambda_1=Lambda;  s_1=s;  q_1=q; rho_1=rho
                f_lambda_1=f_lambda
                lambda_2=lambda_S; s_2=s_S; q_2=q_S; rho_2=rho_S
                f_lambda_2=f_lambda_S
                V_i_b=i;  j=V_i_e
                while (i <= j):
                    while( (i <= V_i_e) and (x[i] <= lambda_S) ):
                        i=i+1
                    while( (j>=V_i_b) and (x[j] > lambda_S) ):
                        j = j-1
                    if (i<j):
                        x[i]=x[j];
                        i=i+1;   j=j-1
                V_i_e=j; V_i=V_i_e-V_i_b+1
            U[iter_step]=V_i
    norm2x=0; i=0
    for i in range(n):
        if (v[i] > Lambda):
            x[i]=v[i]-Lambda
            norm2x = norm2x+x[i]*x[i]
        else:
            if (v[i]< -Lambda):
                x[i]=v[i]+Lambda
                norm2x = norm2x+x[i]*x[i]
            else:
                x[i]=0
    i=0
    for i in range(n):
        x[i]=x[i]/math.sqrt(norm2x)
    x = np.array(x).reshape((n,1))
    return x

#投影函数:x表示需要投影的点（p维列向量）,t表示投影区域的1范约束系数
def prox(x,t):
    #p表示列向量x的维数
    p = x.shape[0]
    out = np.array([0.0 for i in range(p)])
    out = out.reshape((p,1))
    #vmax表示向量(存储为p*1维数组矩阵)x的项取绝对值后最大的绝对值
    vmax = max(abs(x))[0]
    #I1表示绝对值大于等于绝对值最大值的指标个数
    I1 = sum(abs(x)>=vmax)[0]
    if(I1<=t**2):
        if(sum(abs(x))[0]>t*math.sqrt(np.dot(x.T, x)[0][0])):
            #x1为球的的投影点
            x1=S1S2_QASB(x, p, t)
            for i in range(p):
                if(x1[i][0]!=0):
                    out[i][0]=x1[i][0]
        else:
            for i in range(p):
                fenmu=math.sqrt(np.dot(x.T,x)[0][0])
                out[i][0] = x[i][0]/fenmu
    else:
        d = np.array([x[i][0] for i in range(p)]).reshape((p,1))
        sf = np.sign(d)
        if (I1!=1):
            cen = (t/I1)*np.array([1 for i in range(I1)]).reshape((I1,1))
            cbound = np.array([0.0 for i in range(I1)]).reshape((I1,1))
            cbound[0][0] = t
            h = cbound-cen
            ss = math.sqrt((1-math.sqrt(np.dot(cen.T,cen)[0][0]))/math.sqrt(np.dot(h.T,h)[0][0]))
            cqiu = cen +ss*h
        else:
            cen = (t/I1)*np.array([1 for i in range(I1)]).reshape((I1,1))
            cqiu = cen
        shz = cqiu
        wz = np.array([0.0 for i in range(p)]).reshape((p,1))
        k = 0
        for i in range(p):
            if (abs(d[i][0])>=vmax):
                #print(i,d[i])
                wz[i] = shz[k]
                k = k+1
        out = wz*sf
    return out

#梯度投影法求A[j]
#最大迭代次数
maxit_GBB = 1000
#停机准则
xtol_GBB = 1e-8#自变量
gftol_GBB = 1e-8#梯度
ftol_GBB = 1e-10#函数值
tol_GBB = [xtol_GBB,gftol_GBB,ftol_GBB]
#默认步长
tau_GBB = 200
#线搜索准则中下降量参数
rhol_GBB = 1e-4
#步长衰减率
eta_GBB = 0.2
#非单调线搜索准则参数
gamma_GBB = 0.85
#梯度投影法
#由于求最大值，目标函数为-fun，运算结束后再乘一次-1
#sj为aj的1-范数约束系数,J为数据矩阵X中块的个数,j为当前要求外权的索引,C,X,A同前,A_n为更新得到的新的外权矩阵
#tol_GBB为停机准则,tau_GBB为默认步长,eta_GBB为步长衰减率,gamma_GBB为非单调线搜索准则参数
def fminGBB(S,J,j,C,X,A,A_n,tol_GBB,tau_GBB,eta_GBB,gamma_GBB):
    #1-范约束的常数记为sj
    sj = S[j]
    #记录当前外权A[j]的维数p_j,用p表示.temp实际表示n,即样本点个数,但程序中无意义,用于占位.
    [temp,p] = X[j].shape
    #计算初始点处的函数值和梯度
    #用于计算函数值的A矩阵(数据类型为list),记为A_tempt:前j-1项来源于A_n,后来源于A.(来源于块坐标下降法)
    A_tempt = [0.0 for i in range(J)]
    for i in range(J):
        pi = A[i].shape[0]
        A_tempt[i] = np.array([A[i][k][0] for k in range(pi)]).reshape((pi,1))
    for i in range(J):#此处考虑j=0的特殊情形而分类讨论:前j-1为更新过的,后为未更新的.j=0时仅使用未更新的.
        if i<j:
            pi = A[i].shape[0]
            for k in range(pi):
                A_tempt[i][k][0] = A_n[i][k][0]
        else:
            break
    #待求解变量为A[j],记为x
    x = [0.0 for i in range(p)]
    for i in range(p):
        x[i] = A[j][i][0]
    x = np.array(x).reshape((p,1))
    #先将x投影到可行域
    w = prox(x,sj)#w为投影点
    x = w
    f = -fun(J,C,X,A_tempt)#此时的函数值
    gf = -1*gfun(J,j,C,X,A_tempt)#此时的梯度
    nrmG = math.sqrt(np.dot(gf.T, gf)[0][0]) #梯度的2范
    #线搜索参数
    Q = 1
    Cval = f
    tau = tau_GBB
    #迭代主循环
    #限制在最大迭代次数内
    for itr in range(1,maxit_GBB):
        #复制前一步的自变量为x_o，函数值为f_o和梯度为gf_o
        #print("itr",itr)
        x_o = x; f_o = f; gf_o = gf
        #非精确线搜索。初始化线搜索次数nls=1
        #满足先搜索准则或超过10次步长衰减时退出线搜索否则进行步长衰减
        nls = 0#线搜索次数number_of_linear_search
        x = x_o - tau*gf_o#沿着负梯度方向按照步长位移
        w = prox(x,sj)#w为投影点
        x = w
        for i in range(p):
            A_tempt[j][i][0] = x[i][0]
        f = -fun(J,C,X,A_tempt)#此时的函数值
        #print(j,itr,f_o,f,abs(f-f_o))
        gf = -1*gfun(J,j,C,X,A_tempt)#此时的梯度
        s = x - x_o
        #判断是否收敛(是否满足停机准则)
        deltafj = abs(f-f_o)
        #print('nls',nls,'itr',itr,'f_o',f_o,'f',f)
        deltaxj = math.sqrt(np.dot(s.T,s))
        nrmG = math.sqrt(np.dot(gf.T, gf)[0][0]) #梯度的2范
        if ((abs(deltafj) < tol_GBB[2] and abs(deltaxj)<tol_GBB[0]) or (nrmG<tol_GBB[1])):
            #print("converge!")
            return x
            break
        #计算BB步长：
        y = gf - gf_o
        sy = abs(np.dot(s.T,y)[0][0])#sy表示自变量差和梯度差的内积
        tau = tau_GBB
        if sy>0:#否则无法计算BB步长
            #第一种步长计算方法：
            tau = abs(np.dot(s.T,s)[0][0])/sy
            #第二种步长计算方法：
            #tau = sy/abs(np.dot(y.T,y)[0][0])
        #将步长限定在一定范围内(该范围需实验，此处选取[1e-20,1e20])
        else:
            tau=tau_GBB
        tau = max(min(tau,1e20),1e-20)
        #计算线搜索准则参数
        Q_o = Q; Q = gamma_GBB *Q_o+1
        Cval = (gamma_GBB*Q_o*Cval)/Q
        
#坐标下降法
#坐标下降法最大迭代次数
maxit_kzb = 100
#坐标下降法的停机准则
tol_kzb = 1e-16
#坐标下降法函数
def kzbSGCCA(S,J,C,X,A,maxit_kzb,tol_kzb):
    B=[0.0 for i in range(J)]
    for i in range(J):
        B[i] = A[i]
    for s in range(maxit_kzb):
        A_n = [0 for j in range(J)]
        for j in range(J):
            #print("kzbSGCCA",s,j)
            p = A[j].shape[0]
            x = fminGBB(S,J,j,C,X,B,A_n,tol_GBB,tau_GBB,eta_GBB,gamma_GBB)
            A_n[j]=np.array([x[k][0] for k in range(p)]).reshape((p,1))
        deltaf = fun(J,C,X,A_n)-fun(J,C,X,A)
        if abs(deltaf)<tol_kzb:
            #print(abs(deltaf))
            break
        for j in range(J):
            pj = A[j].shape[0]
            for k in range(pj):
                A[j][k][0] = A_n[j][k][0]
    f = fun(J,C,X,A)
    return [A,s,f]

#进行10000次仿真
funvalue=[]
Num = 50
Time = []
Results = []
psc1=[]; psc2=[]; psc3=[]#sensitivity
psp1=[]; psp2=[]; psp3=[]#specitivity
for i in range(Num):
    start = time.time()
    #s为迭代次数,deltaf为函数值差值,A为外权
    a1 = np.random.normal(1,1,(weishu[0],1)); a2 = np.random.normal(1,1,(weishu[1],1)); a3 = np.random.normal(1,1,(weishu[2],1))    
    '''
    a1 = [0 for k in range(200)]; a2 = [0 for k in range(500)]; a3 = [0 for k in range(700)]
    a1[0]=1; a2[0]=1; a3[0]=1   
    '''
    A_0 = [a1,a2,a3]
    A = [0,0,0]
    for k in range(3):
        A[k] = prox(A_0[k],S[k])
    #print(fun(J,C,X,A))
    B = [0 for k in range(J)]
    [B,s,f]=kzbSGCCA(S,J,C,X,A,maxit_kzb,tol_kzb)
    print(f,s)
    funvalue.append(f)
    end = time.time()
    Results.append(B)
    Time.append(end-start)
idealfunv = int(max(funvalue)*100)/100
suc=0
SucTime=[]
for i in range(Num):
    if (int(funvalue[i]*100)/100==idealfunv):
        suc = suc+1
        SucTime.append(Time[i])
       
psuc = suc/Num
print("percent of successful cases:",psuc)
T = 0; T= sum(SucTime)/suc
print("average time consumption:",T)
'''
Nonzero=[]
for i in range(Num):
    print("结果",i)
    nonzero = []
    s=0
    print("GE:")
    for j in range(X[0].shape[1]):
        if (Results[i][0][j][0]!=0):
            #print(j,Results[i][0][j][0])
            nonzero.append(j)
            s = s+1
    Nonzero.append(nonzero)
    print("非0元素个数：",s)
    s=0
    print("CGH:")
    for j in range(X[1].shape[1]):
        if (Results[i][1][j][0]!=0):
            #print(j,Results[i][1][j][0])
            s = s+1
    print("非0元素个数：",s)
    s=0
    print("ylabel:")
    for j in range(X[2].shape[1]):
        if (Results[i][2][j][0]!=0):
            #print(j,Results[i][2][j][0])
            s = s+1
    print("非0元素个数：",s)
'''



