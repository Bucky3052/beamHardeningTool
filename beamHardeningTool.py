import os
import numpy as np
import matplotlib.pyplot as plt

def Intensity(I_0:float, μ:float, x:float):
    '''
    Function for 1-dimensional photon attenuation
    :param I_0: initial intensity
    :param μ: macroscopic cross-section [cm^-1]
    :param x: path through material [cm]
    :return: final intensity
    '''
    return I_0*np.exp(-μ*x)

def CrossSection(coeff:float, ρ:float):
    '''
    Function for correcting attenuation coefficient for target density
    :param coeff: mass attenuation coefficient [cm^2/g]
    :param ρ: density [g/cm^3]
    :return: macroscopic cross-section [cm^-1]
    '''
    return coeff*ρ

def HR(H:float, S:float):
    '''
    Function for determining hardness ratio; Sources with an HR less than 0 have soft spectra, while those with an HR greater than 0 have hard spectra
    :param H: relative number of detected hard x-ray photons
    :param S: relative number of detected soft x-ray photons
    :return: hardness ratio
    '''
    return (H-S)/(H+S)

class Material:
    def __init__(self, ρ:float, massAttenData=np.array(list(zip([0],[0])))):
        self.nominalDensity = ρ # [g/cc]
        self.attenuationData = massAttenData

    def grabAttenuationCoeffs(self, path):
        data = np.genfromtxt(path, delimiter=" ", skip_header=2)
        ergs = data[:,0] # [MeV]
        massAtten = data[:,1] # [cm^2/g]
        self.attenuationData = np.array(list(zip(ergs,massAtten)))

    def μ(self, E:float, ρ=None):
        if ρ == None:
            ρ = self.nominalDensity
        logEp = np.log10(self.attenuationData[:,0])
        logμp = np.log10(ρ*self.attenuationData[:,1])
        return np.power(10.0, np.interp(E, logEp, logμp))

class Node:
    def __init__(self, mat:Material, δ:float, ρ:float=None):
        self.material = mat
        self.thickness = δ # [cm]
        if ρ==None:
            ρ = mat.nominalDensity
        self.density = ρ # [g/cc]
    
    def matAtten(self, E:float, θ=0.0):
        δprime = self.thickness/np.cos(θ)
        μ = self.material.μ(E, self.density)
        return Intensity(1, μ, δprime)
    
class Beamline:
    def __init__(self, SOD:float, SDD:float, S:float=1.0):
        self.SOD = SOD
        self.SDD = SDD
        self.Source = S
        self.nodeList = []

    def defineSoure(self, I0, x0):
        self.Source = I0*(4*np.pi*x0**2)

    def addNode(self, Node:Node, x:float):
        SDD = self.SDD
        # SOD = self.SOD
        if x > SDD:
            x = SDD
        if x < 0:
            x = 0
        self.nodeList.append((Node,x))
        self.nodeList = sorted(self.nodeList, key = lambda x: x[1])
        
    def matAtten(self, E:float, θ=0.0):
        I = 1.0
        for item in self.nodeList:
            node = item[0]
            # pos = item[1]
            I = I*node.matAtten(E, θ)
        return I

    def geomAtten(self, θ=0.0):
        SDDprime = self.SDD/np.cos(θ)
        I = self.Source/(4*np.pi*(SDDprime**2))
        return I

    def totalAtten(self, E:float, θ=0.0):
        return self.matAtten(E, θ)*self.geomAtten(θ)

def initSpectrum(V:float, N:int=10):
    # logE = np.linspace(0,np.log(V),N)
    # E = np.exp(logE)
    E = np.linspace(0, V, N)
    I = (2/V)-((2*E)/(V**2))
    return E, I

CuData = np.genfromtxt(f'CrossSectionData{os.path.sep}Cu_XCOM.txt', delimiter=" ", skip_header=2)
InconelData = np.genfromtxt(f'CrossSectionData{os.path.sep}Inconel_XCOM.txt', delimiter=" ", skip_header=2)

CuEnergies = CuData[:,0] # [MeV]
CuCoeffs = CuData[:,1] # [cm^2/g]
CuDensity = 8.96 # [g/cc]
CuCoeffs = CrossSection(CuCoeffs,CuDensity) # [cm^-1]
CuMuData = np.array(list(zip(CuEnergies,CuCoeffs)))
#coeffsNoCoh = CuData[:,2]
IncEnergies = InconelData[:,0] # [MeV]
IncCoeffs = InconelData[:,1] # [cm^2/g]
IncDensity = 8.43 # [g/cc]
IncCoeffs = CrossSection(IncCoeffs,IncDensity) # [cm^-1]
IncMuData = np.array(list(zip(IncEnergies,IncCoeffs)))
#IncCoeffsNoCoh = InconelData[:,2]

# plt.plot(CuEnergies, CuCoeffs)
# plt.plot(IncEnergies, IncCoeffs)
# plt.yscale('log')
# plt.xscale('log')
# plt.show()

EArray, IArray = initSpectrum(9, 1000)
# plt.plot(E, I, '.')
# plt.yscale('log')
# plt.xscale('log')
# plt.show()

SOD = 100 # [cm]
SDD = 300 # [cm]
myWorld = Beamline(SOD, SDD, 1E9)
CuMat = Material(CuDensity, CuMuData)
CuThickness = 2.5 # [cm]
CuFilter = Node(CuMat, CuThickness, CuDensity)
myWorld.addNode(CuFilter, 0)
IncMat = Material(IncDensity, IncMuData)
IncThickness = 10 # [cm]
Object = Node(IncMat, IncThickness, IncDensity)
myWorld.addNode(Object, SOD)

v = 42.7 # [cm]
N = 100 # elements
x = np.linspace(-v/2, v/2, N)
y = np.linspace(-v/2, v/2, N)
xx, yy = np.meshgrid(x, y)
θ = np.arctan(np.sqrt(xx**2 + yy**2)/SDD)

IOut = np.zeros_like(IArray)
Hi = 0
Si = 0
H = 0
S = 0
thres = 2 # [MeV]
for i in range(len(EArray)):
    E = EArray[i]
    IOut[i] = IArray[i]*myWorld.matAtten(E)
    if E > thres:
        Hi += IArray[i]
        H += IOut[i]
    else:
        Si += IArray[i]
        S += IOut[i]

plt.plot(EArray, IOut)
# plt.yscale('log')
# plt.xscale('log')
# plt.show()
print('Initial HR:', HR(Hi,Si))
print('Final HR:', HR(H,S))
