import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pprint
from scipy.optimize import differential_evolution
from scipy.optimize import shgo
from scipy.optimize import dual_annealing
from multiprocessing import Pool
import multiprocessing as multi
from scipy.optimize import LinearConstraint
from scipy.optimize import Bounds
from time import sleep




kappa=np.array([0.,0.,0.])#回復者が自由に働けるように特定される割合
beta=0.134#感染力
gamma=1/18#感染期間の逆数
chi=35*365

t_max=365
dt=1
t=np.arange(0,t_max,dt)
interval=30

theta=0.75
alpha=2
eta=0.9
rho_sc=1
rho=np.array([[1,rho_sc,rho_sc],[rho_sc,1,rho_sc],[rho_sc,rho_sc,1]])

w=np.array([1.,1.,0.26])
Delta=np.array([32.5*365,10.0*365,2.5*365])
zeta=0.3
phi=np.array([0.1,0.1,0.1])

fatality_rate=np.array([0.001,0.01,0.06])
sigma=0.0076
iota=sigma*fatality_rate
delta_d_underbar=gamma*fatality_rate/iota

def H(I):
    return sigma*(np.dot(fatality_rate,I))

def lam(N):
    return 1/(sigma*np.dot(fatality_rate,N))

def delta_d(N,I):
    return delta_d_underbar*(1+lam(N)*H(I))

def M(S,I,R,L):
    m=np.array([0.,0.,0.])
    for j in range(3):
        m[j]=(np.dot(rho[j],(S+eta*I+(1-kappa)*R)*(1-theta*L)+kappa*R))**(alpha-2)
    return m

def mgsir_eq_v1(state,t,*Lp):
    
    L=Lp[0:(t_max//interval+1)*2]
    p=Lp[(t_max//interval+1)*2:(t_max//interval+1)*3]
    
    L=np.array(L).reshape(t_max//interval+1,-1)
    L=np.insert(L,0,L[:,0],axis=1)
    
    #print(L)
    
    
    
    state=state.reshape(5,3)
    
    S=state[0]
    I=state[1]
    R=state[2]
    D=state[3]
    V=state[4]
    
    N=S+I+R+D
    
    po=p[min(int(t/interval),int(t_max/interval))]
    py=(1-po)*N[0]/(N[0]+N[1])
    pm=(1-po)*N[1]/(N[0]+N[1])
    p=np.array([py,pm,po])
    
    dI=M(S,I,R,L[min(int(t/interval),int(t_max/interval))])*beta*(1-theta*L[min(int(t/interval),int(t_max/interval))])*S*np.dot(rho,eta*(1-theta*L[min(int(t/interval),int(t_max/interval))])*I)-gamma*I-p*I/(t_max*N)
    dS=-dI-gamma*I-p*S/(t_max*N)
    dD=delta_d(S+I+R+D,I)*iota*I
    dR=(gamma-delta_d(S+I+R+D,I))*iota*I+gamma*(I-iota*I)-p*R/(t_max*N)
    dV=p*(S+I+R)/(t_max*N)
    
    return np.vstack([dS,dI,dR,dD,dV]).flatten()

def death(result):
    return result[-1,9]+result[-1,10]+result[-1,11]

def psi(result,L):
    Psi=np.array([0.0,0.0,0.0])
    L=np.array(L).reshape(t_max//interval+1,-1)
    for t in range(t_max):
        S=result[t,0:3]
        I=result[t,3:6]
        R=result[t,6:9]
        D=result[t,9:12]
        Psi=Psi+(1-zeta)*w*S*L[min(int(t/interval),int(t_max/interval))]+(1-zeta)*w*I*(1-eta*(1-L[min(int(t/interval),int(t_max/interval))]))+(1-zeta)*w*(1-kappa)*R*L[min(int(t/interval),int(t_max/interval))]+w*Delta*iota*delta_d(S+I+R+D,I)*I
    return np.sum(Psi)

count=0
mincost=100000
minL=()
def economical_cost_v1(Lp):
    Lp=tuple(Lp)
    L=Lp[0:(t_max//interval+1)*2]
    L=np.array(L).reshape(t_max//interval+1,-1)
    L=np.insert(L,0,L[:,0],axis=1)
    
    p=Lp[(t_max//interval+1)*2:(t_max//interval+1)*3]
    
    result=odeint(mgsir_eq_v1,state_ini_v1,t,args=Lp)[::int(1/dt),:]
    Death=death(result)
    Psi=psi(result,L)
    global count
    global mincost
    global minL
    global minp
    count=count+1
    if Psi+chi*Death<mincost:
        mincost=Psi+chi*Death
        minL=L
        minp=p
        
    if count%100==0:
        print(Psi+chi*Death,mincost,minL,minp)
    return Psi+chi*Death

def my_map(f,xs):
    print("my map is called")
    sleep(0.00005)
    return map(f,xs)


N_ini=np.array([0.54,0.26,0.20])
per_ini=0.0001
S_ini=(1-per_ini)*N_ini
I_ini=per_ini*N_ini
R_ini=np.array([0.,0.,0.])
D_ini=np.array([0.,0.,0.])
V_ini=np.array([0.,0.,0.])
state_ini_v1=np.vstack([S_ini,I_ini,R_ini,D_ini,V_ini])
state_ini_v1=state_ini_v1.flatten()

bounds=Bounds([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
lc = LinearConstraint([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1],[73/30-0.001],[73/30+0.001])
difopt_v1=differential_evolution(economical_cost_v1,bounds,disp=True,updating="deferred",workers=my_map,constraints=(lc),maxiter=100000,popsize=100)
pprint.pprint(difopt_v1)
