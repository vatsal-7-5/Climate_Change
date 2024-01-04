import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# print(df)
t=37
time=36
l=["ID","trial","InvestmentAgainstMitigation","IncomeAvailable"]
l1=["Health","Life","Property"]
insincome=3545
uniqueincome=[0,488,1173,1661,1884,2372,3057,3545]
def datadictionary(filename):
    df = pd.read_csv(filename)
    y = df[l1].to_numpy()
    yt = []
    for i in y:
        yt.append(np.sum(i) / insincome)
    x = df[l].to_numpy()
    x = np.insert(x, 2, yt, axis=1)
    # print(np.shape(x))
    z=len(x)
    y=z//t
    xr=np.reshape(x,(y,t,5))
    d=np.unique(df["ID"])
    empd={}
    for i in range(y):
        perl=xr[i][1:]
        key=perl[0][0]
        d1=[]
        for j in range(t-1):
            ti=perl[j][1:]
            d1.append(ti)
        empd[key]=d1
    return [empd,d]

def nonlinregcoeff(x,y): # need some changes
    xlist=x
    z=np.matmul(np.transpose(xlist),xlist)
    z=np.linalg.inv(z)  #problem
    z=np.matmul(z,np.transpose(xlist))
    w=np.matmul(z,y)
    return w

def xreglis(x,p):
    z=len(x)
    final=[]
    for i in range(z-p):
        xl=[1]
        for j in range(p):
            xl.append(x[i+j])
        final.append(xl)
    return final
def yreglis(x,p):#x4 se leke x36 tak
    return x[p:]
def autoregressioncoeff(time_series,lagvalue):
    ylist = yreglis(time_series, lagvalue)
    xlist = xreglis(time_series, lagvalue)
    w = nonlinregcoeff(xlist, ylist)
    return w

def predfinaltilln(time_series,lagvalue,end=50):
    z=len(time_series)
    w = autoregressioncoeff(time_series, lagvalue)
    predmat = time_series
    for posn in range(end-z):
        dotmat = [1]
        for i in range(lagvalue):
            dotmat.append(predmat[-lagvalue + i])
        xpr = np.dot(w, dotmat)
        predmat.append(xpr)
    return predmat

def prednext(time_series,w,lagvalue=5):
    z=len(time_series)
    dotmat=[1]
    for i in range(lagvalue):
        dotmat.append(time_series[-lagvalue+i])
    xpr=np.dot(w,dotmat)
    return xpr

def incomesys(pred_income):
    x=pred_income*insincome
    nl=[]
    for i in uniqueincome:
        nl.append((x-i)**2)
    z=np.argsort(nl)
    return (uniqueincome[z[0]]/insincome)

def rmse(orig,pred,lagvalue):
    l=min(len(pred),len(orig))
    sum=0
    for i in range(lagvalue,l):
        d=pred[i]-orig[i]
        sum+=d**2
    sum/=(l-lagvalue)
    return sum**0.5

def incomelist(person_list):
    per=person_list
    xincome=[]
    for i in per:
        xincome.append(i[1])
    return xincome

def avgincomelist(full_list): #Dictionary
    length=36
    a=[0.0]*length
    c=0
    for i in full_list:
        c+=1
        z=full_list[i]
        for j in range(len(z)):
            a[j]+=z[j][1]
    for i in range(len(a)):
        a[i]/=c
    return a

def mitigationlist(person_list):
    per = person_list
    xmitigation = []
    for i in per:
        xmitigation.append(i[2])
    return xmitigation


def avgmitigationlist(full_list):  # Dictionary
    length = 36
    a = [0.0] * length
    c = 0
    for i in full_list:
        c += 1
        z = full_list[i]
        for j in range(len(z)):
            a[j] += z[j][2]
    for i in range(len(a)):
        a[i] /= c
    return a

def totinclist(person_list):
    per = person_list
    xtotinc = []
    for i in per:
        xtotinc.append(i[3])
    return xtotinc


def avgtotinclist(full_list):  # Dictionary
    length = 36
    a = [0.0] * length
    c = 0
    for i in full_list:
        c += 1
        z = full_list[i]
        for j in range(len(z)):
            a[j] += z[j][3]
    for i in range(len(a)):
        a[i] /= c
    return a
def probclimatechange(average_investment,mitigation_factor,k_value):
    c=average_investment**k_value
    c=c*mitigation_factor
    return 1-c

def probclimatechangelist(mitigation_list,total_incomelist,k_value,mitigation_factor=0.85):
    length=len(mitigation_list)
    ml=mitigation_list
    ti=total_incomelist
    newl=[]
    for i in range(length):
        num=np.sum(ml[0:i+1])
        den=np.sum(ti[:i+1])
        avg=num/den
        newl.append(probclimatechange(avg,mitigation_factor,k_value))
    return newl

def probclimatechangevalue(mitigation_list,total_incomelist,k_value,mitigation_factor=0.85):
    ml=mitigation_list
    ti=total_incomelist
    num=np.sum(ml)
    den=np.sum(ti)
    avg=num/den
    val=probclimatechange(avg,mitigation_factor,k_value)
    return val

def totalpred(p_loss,totallist):
    n=np.random.random()
    val=totallist[-1]
    if n<p_loss:
        return val*.875
    else:
        return val

def reduceins(ins,total):
    i=0
    ins=round(insincome*ins)
    while True:
        if uniqueincome[i]==ins:
            break
        i+=1
    while True:
        if uniqueincome[i]<=total:
            break
        i-=1
    return (uniqueincome[i]/insincome)

def allpredtillend(inclist,mitlist,totallist,k_value,end=100):
    i=inclist
    m=mitlist
    t=totallist
    wi = autoregressioncoeff(i, 5)
    wm = autoregressioncoeff(m, 5)
    while (len(inclist)<=end):
        pcl=probclimatechangevalue(m,t,k_value)
        ploss=0.3*pcl
        tpred=totalpred(ploss,t)
        mpr=prednext(m,wm,5)
        ipr=prednext(i,wi,5)
        ipr=incomesys(ipr)
        t.append(tpred)
        if mpr<0:
            mpr=0
        if ((mpr+ipr*insincome)<tpred):
            m.append(mpr)
            i.append(ipr)
        else: #need to be changed later
            if (ipr*insincome<tpred):
                mpr=tpred-ipr*insincome
                m.append(mpr)
                i.append(ipr)
            else:
                ipr=reduceins(ipr,tpred)
                mpr=tpred-ipr*insincome
                m.append(mpr)
                i.append(ipr)

    pclimlist=probclimatechangelist(m,t,k_value)
    newl=[i,m,t,pclimlist]
    return newl

def finallist(filename,k_value):
    empd,d=datadictionary(filename)
    x1income=incomelist(empd[d[0]])
    xavg=avgincomelist(empd)
    x1mit=mitigationlist(empd[d[0]])
    xmitavg=avgmitigationlist(empd)
    x1tot=totinclist(empd[d[0]])
    xtotavg=avgtotinclist(empd)
    time=len(x1income)
    p=6
    posn=2
    end=50
    predfinmitgavg50=predfinaltilln(xmitavg.copy(),5,end)
    pred1mit=predfinaltilln(x1mit.copy(),5,end)
    # print(mitigationlist(empd[d[0]]))
    # print(totinclist(empd[d[0]]))
    # print(probclimatechangelist(mitigationlist(empd[d[0]]),totinclist(empd[d[0]]),3))
    ha=allpredtillend(xavg,xmitavg,xtotavg,k_value)
    inclist,mitlist,totlist,climlist=ha[0],ha[1],ha[2],ha[3]
    return[inclist,mitlist,totlist,climlist]


files = ["cubicfeed.csv", "linearfeed.csv", "linearnofeed.csv"]
files2=["cubicnofeed.csv"]
k_val=[3,1,1]
mainl=[]
time=list(range(101))
for i in range(len(files)):
    mainl.append(finallist(files[i],k_val[i]))
    plt.plot(time,mainl[i][3])
plt.legend(["1","2","3"])
plt.show()
