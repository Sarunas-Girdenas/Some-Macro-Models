#Basic Real Business Cycle Model
from datetime import datetime # import this to calculate script execution time
startTime=datetime.now()
import numpy as np
import sympy as smp
from scipy import linalg as lng
import matplotlib.pyplot as plt



# Parmeter Values 
alpha = 0.36 ; # Capital Income Share 
beta = 0.99; # Subjective Discount rate 
delta = 0.025; # Capital Depreciation Rate 
Psi = 2.0 ; # Marginal Disutility of Labor 
Z = 1.0 ; # Mean of Aggregate Technology 
sigma_z = 0.007; # Standard Deviation of Technology 
rho = 0.95 ; # Autocorrelation of Technology Shocks 


# Variables
cc,lc,kc,zc,ca,la,ka,za = smp.symbols('cc,lc,kc,zc,ca,la,ka,za') 
xx = [ca,la,ka,za,cc,lc,kc,zc]; 
# Variables with "a" / "c" denote forward/current variables, respectively 
# 1) Control Variables 
# c : Consumpton 
# l : Hours-worked 
# 2) State Variables 
# k : Capital 
# z : Technology 

varname = np.array(['Consumption', 'Hours-worked', 'Capital', 'Technology'])

# Simulation Settings
# 1) Impulse-Response Analysis 
h = 24; # Horizons of Impulse-Response Function 
hit = 2; # Period in which an innovation occurs 
imp = varname[3]; # Variable hit by an innovation 
#SI = ([[ 0], [ 1]]) # Innovation Vector 
SI = np.matrix('0; 1')


# Calculation of the Steady State # 

 
KY = 1/((1-beta*(1-delta))/(alpha*beta*Z)); 
CY = 1-delta*KY; 
LS = 1/(1+Psi/((1-alpha)/CY)); 
KS = LS*(alpha*beta*Z/(1-(beta*(1-delta))))**(1/(1-alpha)); 
YS = Z*(KS**alpha)*(LS**(1-alpha)); 
IS = delta*KS; 
CS = YS - IS; 
 
s = [CS, LS, KS, Z]; 

print "============================================================"
print "Steady State Solution"
print "KY =", KY, "CS =", CS, "KS =", KS, 
print "LS =", LS, "YS =", YS, "IS =", IS
print "============================================================"

 
# Equilibrium Conditions  

#	 Intratemporal Optimality Condition 
intra = Psi/(1-lc) - (1-alpha)*zc*((kc/lc)**(alpha))*(1/cc); 
 
#	 Intertemporal Optimality Condition 
inter = beta*(alpha*za*(la/ka)**(1-alpha) + (1-delta))*(1/ca) - (1/cc); 
 
#	 Resource Constraint
resource = zc*(kc**alpha)*(lc**(1-alpha)) + (1-delta)*kc - cc - ka; 
 
#	 Dynamics of Technology Shock 
tech = za - zc**rho; 
 
#	 Optimal Conditions
eqcon = [intra, inter, resource, tech];

# 	Differentiation
X = smp.Matrix([eqcon]) #equations
Y = smp.Matrix([xx])    #variables
J = X.jacobian(Y)      #jacobian

#	 Evaluation at the Steady State 
cc = CS; 
ca = CS; 
kc = KS; 
ka = KS; 
lc = LS; 
la = LS; 
zc = Z; 
za = Z; 

SS = np.array([s,s,s,s])

coef = J.evalf(subs={'cc':CS, 'ca':CS, 'kc':KS, 'ka':KS, 'lc':LS, 'la':LS, 'zc':Z,'za':Z})

# 	Solving the Model

#Define intervals

coef_intr1=np.shape(coef)[1]/2

coef_intr2=np.shape(coef)[1]

#Coefficients of current variables
A1=np.array([coef[:,coef_intr1],coef[:,coef_intr1+1],coef[:,coef_intr1+2],coef[:,coef_intr1+3]])

A = A1.T*SS;

#Coefficients of forward looking variables

B1 = np.array([ -coef[:,0],-coef[:,1],-coef[:,2],-coef[:,3]])

B = B1.T*SS

# NOTE 
# A*Y(t) = B*Y(t+1) 
# Y(t) = inv(A)*B*Y(t+1) 

# Eigenvalue Decomposition
 
# Eigenvalue Decomposition  s 

[L,Q]=lng.eig(np.dot(np.linalg.inv(A),B))  # columns of Q : eigenvector
# diagonal matrix L : eigenvalues 
# Q*L*inv(Q) = inv(A)*B 
print "============================================================"
print "Eigenvalues"
print L
print "============================================================"


L11=np.sort(abs(L)) #sort eigenvalues in ascending order
idx=np.argsort(L) #get an index of sorting

Q=Q[:,idx] # reorder eigenvectors associated with their eigenvalues 
QI=lng.inv(Q) #Q inverse

# No of Eigenvalues Inside/Outside a Unit Circle 

# 1) No of Control Variables X(t) 

n=0
for i in range(0,len(L)):
	if abs(L[i]) < 1.000:
		n = n + 1

# 2) No of State Variables S(t)

m = len(L)-n
# Partition Matrices
QI11 = QI[0:n, 0:n] 
QI21 = QI[n:n+m, 0:n] 
QI12 = QI[0:n, n:m+n]
QI22 = QI[n:m+n, n:m+n]

# Extracting Stable Roots

L1 = L11[0:n] # Note that 'L11' is sorted eigenvalues in ascending order
L2 = L11[n:m+n]
VL = np.diag(L2)

# Solution to the system

# Solution to Control Variables
PX = -np.dot(lng.inv(QI11),QI12)

print '=========================================='
print ' Coefficient Matrix for Control Variables ' 
print PX
print '=========================================='

# Solution to State Variables

# Intermediary steps

PS1 = lng.inv(QI22+np.dot(QI21,PX))
PS2 = lng.inv(VL)
PS3 = QI22+np.dot(QI21,PX)
PS  = np.dot(np.dot(PS1,PS2),PS3)  
PS  = PS.real

print '=========================================='
print ' Coefficient Matrix for State Variables ' 
print PS
print '=========================================='

# Impulse-Response Function 

SC = SI # SI will initially be used as the current state vector
# SC will be revised recursively

S=np.zeros((h,m)) # Simulated state variables which will be revised as the simulation proceeds

for t in range(hit-1,h): # Shock occurs in s-th period; matrix S is revised from 6th row.
	SA = np.dot(PS,SC) # New state vector 
	S[t,:]=SA.T # Revise i-th row of state matrix
	SC = S[t,:].T # Use revised i-th row of state matrix as state vector in next period



SI = S[hit-1,:] # Simulated endogenous & exogenous state variables

X = (np.dot(PX,S.T)).T

Y = np.concatenate((X,S),axis=1)

ct = Y[:, 0]
lt = Y[:, 1]
kt = Y[:, 2]
zt = Y[:, 3]

yt = Y[:, 3] + alpha*Y[:, 2] + (1-alpha)*Y[:, 1] # Output 
it = (YS/IS)*yt - (CS/IS)*Y[:, 0] # Investment # Investment
wt = yt - Y[:,1]; # Marginal Product of Labor


plt.xlabel('Horizon')
plt.ylabel('% Deviation from Steady State')
plt.plot(yt, 'k-.o',label='Output')
plt.plot(ct, 'm-.o', label='Consumption')
plt.plot(it, 'r-.o', label='Investment')
plt.plot(lt, 'b-.o', label='Hours-worked')
plt.plot(wt, 'c-.o', label='Labor Productivity')
plt.legend(loc='upper right')
plt.title('Impulse Response to a Unitary Technology Shock')
plt.show()



print 'Computation time:', datetime.now()-startTime, 'seconds.'






















