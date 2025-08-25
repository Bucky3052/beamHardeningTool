import beamHardnessModule as bhm
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd
# from scipy.integrate import trapezoid

# Import Material Data Files
CuDensity = 8.96 # [g/cc]
CuMuData = bhm.xSectDataFromFile(f'CrossSectionData{os.path.sep}Cu_XCOM.txt')

# Initialize Energy Spectrum
V = 9 # MeV
# EArray, IArray = bhm.initSpectrum(V, 100)
empiricalData = pd.read_csv('SpectrumDataNDA/XraySpectrum.9MV.csv', skiprows=3)
EArray=np.array(empiricalData['UpperKeV'].tolist())/1000
IArray=np.array(empiricalData['FluxDistribution'].tolist())

# Set Up World
SOD = 100 # [cm]
SDD = 300 # [cm]
myWorld = bhm.Beamline(SOD, SDD, 1E9)

# Copper Filter Node
CuMat = bhm.Material(CuDensity, CuMuData)
CuThickness = 2.54 # [cm]
CuFilter = bhm.Node(CuMat, CuThickness, CuDensity)
myWorld.addNode(CuFilter, 0)

IOut = np.zeros_like(IArray)
Hi = 0
Si = 0
H = 0
S = 0
thres = 0.100 # [MeV]
for i in range(len(EArray)):
    E = EArray[i]
    IOut[i] = IArray[i]*myWorld.matAtten(E)
    if E > thres:
        Hi += IArray[i]
        H += IOut[i]
    else:
        Si += IArray[i]
        S += IOut[i]

plt.plot(EArray, IArray, 'b', label='Unfiltered Spectrum', alpha=0.5)
plt.plot(EArray, IOut, 'r', label='Filtered Spectrum')
# plt.plot(EArray, IOut*2.025, label='Filtered Spectrum')
plt.yscale('log')
# plt.xscale('log')
plt.xlim(0, V)
# plt.ylim(0)
plt.xlabel('Energy [MeV]')
plt.ylabel('Intensity (Normalized)')
plt.title('Filtration Provided by 1" Copper')
plt.legend()
plt.tight_layout()
plt.savefig('plots/spectra.png')
plt.show()
plt.close()

print('Initial HR:', bhm.HR(Hi,Si))
print('Final HR:', bhm.HR(H,S))