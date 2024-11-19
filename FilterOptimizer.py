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

# Inconel Sample Object Node (Initial)
IncMat = bhm.Material(IncDensity, IncMuData)
IncThickness = 10 # [cm]
Object = bhm.Node(IncMat, IncThickness, IncDensity)
myWorld.addNode(Object, SOD)

IOut = np.zeros_like(IArray)
Hi = 0
Si = 0
H = 0
S = 0
thres = 1 # [MeV]
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
plt.xlabel('Energy [MeV]')
plt.ylabel('Intensity (Normalized)')
plt.title('Beamline Spectrum at Detector Centerpoint')
# plt.yscale('log')
# plt.xscale('log')
plt.show()
plt.close()
print('Initial HR:', bhm.HR(Hi,Si))
print('Final HR:', bhm.HR(H,S))