import beamHardnessModule as bhm
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# Import Material Data Files
CuDensity = 8.96 # [g/cc]
CuMuData = bhm.xSectDataFromFile(f'CrossSectionData{os.path.sep}Cu_XCOM.txt')
IncDensity = 8.43 # [g/cc]
IncMuData = bhm.xSectDataFromFile(f'CrossSectionData{os.path.sep}Inconel_XCOM.txt')

# Initialize Energy Spectrum
EArray, IArray = bhm.initSpectrum(9, 100)

# Set Up World
SOD = 100 # [cm]
SDD = 300 # [cm]
myWorld = bhm.Beamline(SOD, SDD, 1E9)

# Copper Filter Node (Initial)
CuMat = bhm.Material(CuDensity, CuMuData)
CuThickness = 2.54 # [cm]
CuFilter = bhm.Node(CuMat, CuThickness, CuDensity)
myWorld.addNode(CuFilter, 0)
CuFilterThicknesses = 2.54*np.array([0.25,0.5,1]) # [in -> cm]
CuTestThickns = [0.0, 2.54*0.04] # for filters not worth combining[in -> cm]
#2.54*np.array([0.0,0.04,0.25,0.5,1,1.25,1.5,1.75,1.79]) # [cm]
for count in range(1, len(CuFilterThicknesses)+1):
    for t in combinations(CuFilterThicknesses, count):
        CuTestThickns.append(float(sum(t)))
CuTestThickns = sorted(CuTestThickns)
print(CuTestThickns)

# Inconel Sample Object Node (Initial)
IncMat = bhm.Material(IncDensity, IncMuData)
IncThickness = 5 # [cm]
Object = bhm.Node(IncMat, IncThickness, IncDensity)
myWorld.addNode(Object, SOD)
IncTestPts = np.arange(0, IncThickness, 0.1) # [cm]

# Determine Initial HR
thres = 0.100 # [MeV]
Hi = 0
Si = 0
for i in range(len(EArray)):
    E = EArray[i]
    if E > thres:
        Hi += IArray[i]
    else:
        Si += IArray[i]
HRi = bhm.HR(Hi,Si)
print('Initial HR:', HRi)

# Determine HR Profile Along Sample for Each Cu Thickness
dHRdx = np.array([])
dIdx = np.array([])
for δ_Cu in CuTestThickns:
    CuFilter.setThickness(δ_Cu)
    HRArray = np.array([])
    attenArray = np.array([])
    for δ_Inc in IncTestPts:
        Object.setThickness(δ_Inc)
        IOut = np.zeros_like(IArray)
        H = 0
        S = 0
        for i in range(len(EArray)):
            E = EArray[i]
            IOut[i] = IArray[i]*myWorld.matAtten(E)
            if E > thres:
                H += IOut[i]
            else:
                S += IOut[i]
        HR = bhm.HR(H,S)
        HRArray = np.append(HRArray, HR)
        attenArray = np.append(attenArray, sum(IOut)/sum(IArray))
    dIdx = np.append(dIdx, -(attenArray[-1]-attenArray[0])/(IncTestPts[-1]-IncTestPts[0]))
    dHRdx = np.append(dHRdx, (HRArray[-1]-HRArray[0])/(IncTestPts[-1]-IncTestPts[0]))
    plt.plot(IncTestPts, HRArray, label=f"{δ_Cu:.3g} cm")
plt.plot(0, HRi, 'x')
plt.legend(title='Cu Filter Thickness')
plt.title('Beam Hardening Through Sample')
plt.xlabel('Depth through Sample [cm]')
plt.ylabel('Hardness Ratio')
# plt.xscale('log')
plt.show()
plt.close()

plt.plot(CuTestThickns, dHRdx)
plt.title('Hardness Ratio Variability Against Filter Thickness')
plt.xlabel('Cu Filter Thickness [cm]')
plt.ylabel(f'Average HR Rate-of-Change in sample [cm^-1]')
plt.show()
plt.close()