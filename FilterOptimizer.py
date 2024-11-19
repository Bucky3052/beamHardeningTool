import beamHardnessModule as bhm
import os
import numpy as np
import matplotlib.pyplot as plt

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
CuTestThickns = 2.54*np.array([0.04,0.25,0.5,1]) # [cm]

# Inconel Sample Object Node (Initial)
IncMat = bhm.Material(IncDensity, IncMuData)
IncThickness = 10 # [cm]
Object = bhm.Node(IncMat, IncThickness, IncDensity)
myWorld.addNode(Object, SOD)
IncTestPts = np.arange(0, IncThickness, 1) # [cm]

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
    dIdx = np.append(dIdx, -(attenArray[-1]-attenArray[0])/(IncTestPts[-1]-IncTestPts[0])/attenArray[0])
    dHRdx = np.append(dHRdx, (HRArray[-1]-HRArray[0])/(IncTestPts[-1]-IncTestPts[0]))
    plt.plot(IncTestPts, attenArray, label=f"{δ_Cu:.3g} cm")
plt.plot(0, HRi, 'x')
plt.legend(title='Cu Filter Thickness')
plt.title('Attenuation Nonlinearity through Sample due to Beam Hardening')
plt.xlabel('Depth through Sample [cm]')
plt.ylabel('Attenuation')
plt.xscale('log')
plt.show()
plt.close()

plt.plot(CuTestThickns, dIdx)
plt.title('Attenuation Variability Against Filter Thickness')
plt.xlabel('Cu Filter Thickness [cm]')
plt.ylabel(f'Average Attenuation Rate-of-Change in sample [Fraction of $I_0$]')
plt.show()
plt.close()