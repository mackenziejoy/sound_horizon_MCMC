#Determination of the Cosmological Sound Horizon, rs, WITHOUT LCDM assumptions on the shape of H(z)
#Mackenzie Joy

#This code uses BAO data from the BOSS survey and SNe data from the PANSTARR survey to
    #calculate cosmological parameters. The COSMOSLIK Metropolis Hastings MCMC is used.
    
#To run the code in mpi, use '$cosmoslik -n 3 FILENAME.py' with 3=number of chains to run
#This saves a chain file that can be accessed in a kernal by 'chain=load_chain("FILENAME.chain")'

#refer to LBAO_SN_LCDM.py to get more detailed explanation as most of the machinery is the same.


from cosmoslik import *
import numpy as np
import scipy as sp
from scipy.integrate import quad
from scipy.interpolate import *

c=2.998e8 #speed of light (m/s)
rfid=147.78 #fiducial value of the sound horizon (Mpc)

H0r=73.52 #H0 value reported by Riess et al 2018
H0_err=1.62 #error

class LBAOSNnonpar(SlikPlugin):
    def __init__(self):
        super().__init__()
        
        # define sampled parameters
        self.alphaBAO  = param(start=.2998, scale=.1) #multiply by 100
        self.alphaSN = param(start=.355, scale=.1) #multipy by 10000 to get real alphaSN
        self.H0  = param(start=70, scale=.1, min=0) #initial guesses for value of H(z)
        self.Hz1  = param(start=77, scale=.1, min=0)
        self.Hz2  = param(start=95, scale=.1, min=0)
        self.Hz3  = param(start=109, scale=.1, min=0)
        self.Hz4  = param(start=146, scale=.1, min=0)
        
        # set the sampler
        self.sampler = samplers.metropolis_hastings(
            self,
            num_samples=4e5,
            print_level=2,
            cov_est="cov_LBAOSN_nonpar.txt", #save this file after first time running chains
            output_file="LBAOSNnonpar.chain",
        )   
        
    #compute the likelihood
    def __call__(self):
        return self.L(self.alphaBAO, self.alphaSN, self.H0, self.Hz1, self.Hz2, self.Hz3, self.Hz4)
        
    def L(self,alphaBAO,alphaSN,H0,Hz1,Hz2,Hz3,Hz4):
        zguess=[0,.2,.57,.8,1.3]
        Hzguess=[1,Hz1/H0,Hz2/H0,Hz3/H0,Hz4/H0]
        Hz_cs=sp.interpolate.CubicSpline(zguess,Hzguess)
        
        zdata=np.loadtxt("z_SN.txt")
        zdata=np.array(zdata)
        mbdata=np.loadtxt("mb_SN.txt")
        dmbdata=np.loadtxt("dmb_SN.txt")
        
        Csys=np.loadtxt("SNe_Csys.txt")
        Csys=np.array(Csys)
        shape=(40,40)
        Csys=Csys.reshape(shape)
        Cstat=np.zeros((40,40))
        k=0
        j=0
        while k<40 and j<40:
            Cstat[k,j]=(dmbdata[k])**2 #statistical covariance matrix for SN data
            k=k+1
            j=j+1
        C_SN=Cstat+Csys
        
        z=[0.38,0.51,0.61]
        DdHd=np.array([1512.39,81.2087,1975.22,90.9029,2306.68,98.9647])
        mbmodel=np.array([])
        def integrand(z,Hz_cs):
            return 1/Hz_cs(z)
        i=0
        while i<40:
            DL=quad(integrand,0,zdata[i],args=(Hz_cs))[0]*alphaSN*10000*(1+zdata[i])
            mb= -19 + 5*np.log10(DL) + 25
            mbmodel=np.append(mbmodel,mb)
            i=i+1
        Mbmodel=np.array(mbmodel)
        
        be=np.loadtxt("BAO_consensus_covtot_dM_Hz.txt")
        C_BAO=np.array(be)
        md=np.array([])
        def integrand(z,Hz_cs):
            return 1/Hz_cs(z)
        n=0
        while n<3:
            Dz=quad(integrand,0,z[n],args=(Hz_cs))
            md=np.append(md,Dz[0]*alphaBAO*rfid*100)
            md=np.append(md,Hz_cs(z[n])*(c/1000)*rfid**(-1)*(alphaBAO*100)**(-1))
            n=n+1
        Md=np.array(md)
        
        LR18=((H0r-H0)**2)/((H0_err)**2)
        
        return 0.5*(np.dot(Mbmodel-mbdata,np.dot(np.linalg.inv(C_SN),(Mbmodel-mbdata)))) + 0.5*LR18 + 0.5*np.dot(Md-DdHd,np.dot(np.linalg.inv(C_BAO),(Md-DdHd)))
    







