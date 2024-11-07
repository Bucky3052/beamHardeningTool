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
        if massAttenData != np.array(list(zip([0],[0]))):
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
        return np.exp(-μ*δprime)
    
class Beamline:
    def __init__(self, SOD:float, SDD:float):
        self.SOD = SOD
        self.SDD = SDD

    def addNode(self, Node:Node, x:float):
        SDD = self.SDD
        SOD = self.SOD
        if x > SDD:
            x = SDD
        if x < 0:
            x = 0

CuData = np.genfromtxt(f'CrossSectionData{os.path.sep}Cu_XCOM.txt', delimiter=" ", skip_header=2)
InconelData = np.genfromtxt(f'CrossSectionData{os.path.sep}Inconel_XCOM.txt', delimiter=" ", skip_header=2)

CuEnergies = CuData[:,0] # [MeV]
CuCoeffs = CuData[:,1] # [cm^2/g]
CuDensity = 8.96 # [g/cc]
CuCoeffs = CuCoeffs*CuDensity # [cm^-1]
CuMuData = np.array(list(zip(CuEnergies,CuCoeffs)))
#coeffsNoCoh = CuData[:,2]
IncEnergies = InconelData[:,0] # [MeV]
IncCoeffs = InconelData[:,1] # [cm^2/g]
IncDensity = 8.43 # [g/cc]
IncCoeffs = IncCoeffs*IncDensity # [cm^-1]
#IncCoeffsNoCoh = InconelData[:,2]

plt.plot(CuEnergies, CuCoeffs)
plt.plot(IncEnergies, IncCoeffs)
plt.yscale('log')
plt.xscale('log')
plt.show()