#Determination of the Cosmological Sound Horizon, rs, with LCDM assumptions on the shape of H(z)
#Mackenzie Joy

#This code uses BAO data from the BOSS survey and SNe data from the PANSTARR survey to
    #calculate cosmological parameters. The COSMOSLIK Metropolis Hastings MCMC is used.
    
#To run the code in mpi, use '$cosmoslik -n 3 FILENAME.py' with 3=number of chains to run
#This saves a chain file that can be accessed in a kernal by 'chain=load_chain("FILENAME.chain")'


from cosmoslik import *
import numpy as np
import scipy as sp
from scipy.integrate import quad
from scipy.interpolate import *

k=8.62*10**(-5) #Boltzmann constant (eV/K)
hp=4.136*10**(-15) #Planck constant (eV*s)
Tg=2.73 #CMB photon temperature (K)
c=2.998e8 #speed of light (m/s)
mn=0.02/(c**2) #neutrino mass (eV/c^2)
L=mn*c**2/(k*Tg*(4/11)**(1/3))
h=0.72 #hubble
p0=1.88*h**2*10**(-26)*1.783*10**36 #current critical energy density
wr=5*10**(-5) #radiation content
rfid=147.78 #fiducial value of sound horizon (Mpc)

H0r=73.52 #R18 value of Hubble constant
H0_err=1.62

class LBAOSN(SlikPlugin):
    def __init__(self):
        super().__init__()
        
        # define sampled parameters
        self.alphaBAO  = param(start=.2998, scale=.1) #multiply by 100 to get real alphaBAO
                                                      #alphaBAO=c/(H0*rs)
        self.wm  = param(start=.3, scale=.1)
        
        self.alphaSN = param(start=.365, scale=.1) #multipy by 10000 to get real alphaSN
                                                   #alphaSN=c/(H0*Lsn)
        self.H0 = param(start=70, scale=1)
        
        # set the sampler
        self.sampler = samplers.metropolis_hastings(
            self,
            num_samples=4e5,
            print_level=2,
            cov_est="cov_LBAOSN_LCDM.txt",
            output_file="LBAOSN.chain",
        )   
        
    #compute the (-0.5 log) likelihood
    def __call__(self):
        return self.L_BAOSN(self.alphaBAO, self.wm, self.alphaSN, self.H0)#, self.rs)
        
    def L_BAOSN(self,alphaBAO,wm,alphaSN,H0):
        def wn(z): #neutrino energy density
            def n_integrand(x,z):
                return x**2*np.sqrt((1/(1+z))**2+x**2)/(np.exp(x*L)+1)
            n=quad(n_integrand,0,np.inf,args=(z))
            return n[0]*((3*c**5*mn**4)*(1/(1+z))**(-4)*p0**(-1)*(np.pi)**(-2)*hp**(-3))
        zgrid=np.arange(.01,1,.01)
        wngrid=[]
        for i in range(len(zgrid)):
            wngrid=np.append(wngrid,wn(zgrid[i]))
        wn_cs=sp.interpolate.CubicSpline(zgrid,wngrid) #spline interpolation of wn for faster code
        z=[0.38,0.51,0.61]
        DzHdata=np.array([1512.39,81.2087,1975.22,90.9029,2306.68,98.9647]) #alternating DA and 
                                                                            #H(z) for BAO data
        be=np.loadtxt("BAO_consensus_covtot_dM_Hz.txt")
        C_BAO=np.array(be) #BAO covariance matrix
        
        md=np.array([])
        def integrand(x,wm,wr,wn_cs):
            return 1/(np.sqrt(wm*x+(1-wm)*x**4+wr+wn_cs((1/x)-1)))
        i=0
        while i<3:
            a=1/(1+z[i])
            Dz=quad(integrand,1/(1+z[i]),1,args=(wm,wr,wn_cs))[0]*alphaBAO*100*rfid #ang.diam.dist
            H=c*(alphaBAO*100)**(-1)*rfid**(-1)*np.sqrt(wm*a**(-3)+(1-wm)+wr*a**(-4)+wn_cs(z[i]))/1000
            md=np.append(md,Dz)
            md=np.append(md,H)
            i=i+1
        DzHmodel=np.array(md)
        
        bigz=1090.05 #from new data set
        Dbigz=14196.5109 #Dz for z=1090.05 in the data
        C_bigz=.00071*rfid/.0104096**2 #error
        Dbigzguess=quad(integrand,1/(1+bigz),1,args=(wm,wr,wn_cs))[0]*alphaBAO*100*rfid
        Lbigz=((Dbigzguess-Dbigz)**2)/(C_bigz**2) #log likelihood
        
        #call the SNe data
        zdata=np.loadtxt("z_SN.txt") 
        zdata=np.array(zdata)
        mbdata=np.loadtxt("mb_SN.txt")
        dmbdata=np.loadtxt("dmb_SN.txt")
        
        Csys=np.loadtxt("SNe_Csys.txt")
        Csys=np.array(Csys)
        shape=(40,40)
        Csys=Csys.reshape(shape) #systematic error covariance matrix
        Cstat=np.zeros((40,40))
        k=0
        j=0
        while k<40 and j<40:
            Cstat[k,j]=(dmbdata[k])**2 #statistical covariance matrix for SN data
            k=k+1
            j=j+1
        C_SN=Cstat+Csys #supernovae covariance matrix
        m=0
        mbmodel=np.array([])
        while m<40:
            DL=quad(integrand,1/(1+zdata[m]),1,args=(wm,wr,wn_cs))[0]*alphaSN*10000*(1+zdata[m])
            mb= -19 + 5*np.log10(DL) + 25
            mbmodel=np.append(mbmodel,mb)
            m=m+1
        Mbmodel=np.array(mbmodel)
        
        
        LR18=((H0r-H0)**2)/((H0_err)**2) #Riess et al 2018 value of H0 log likelihood
        
        return 0.5*(np.dot(DzHmodel-DzHdata,np.dot(np.linalg.inv(C_BAO),(DzHmodel-DzHdata)))) + 0.5*(np.dot(Mbmodel-mbdata,np.dot(np.linalg.inv(C_SN),(Mbmodel-mbdata)))) + 0.5*LR18 +0.5*Lbigz
    




