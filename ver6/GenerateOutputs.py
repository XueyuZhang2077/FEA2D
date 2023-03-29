import os

import pandas as pd
import shutil
from FEM2D_ver6_Order1 import *
# from FEM2D_Cupy_Order1 import *



vertices = pd.read_excel("Input.xlsx", sheet_name="Vertices_Coords").to_numpy()
domains = pd.read_excel("Input.xlsx", sheet_name="Domain_VertID").to_numpy()[:, 1:]

info = pd.read_excel("Input.xlsx", sheet_name="Info").to_numpy()
domain_Cp = pd.read_excel("Input.xlsx", sheet_name="Domain_Cp").to_numpy()[:, 1:]
loadings = pd.read_excel("Input.xlsx", sheet_name="Load_Condition").to_numpy()
G = info[0, 0]
nu = info[0, 1]
Lame = 2 * G * nu / (1 - 2 * nu)

vertices_coord = vertices[:, :2]
mshSize = vertices[:, 2]
domain_vertiexID = []
for i in range(domains.shape[0]):
    temp = domains[i].tolist()

    temp = np.int_(np.delete(temp, np.isnan(temp)))
    domain_vertiexID.append(np.array(temp))

# generate mesh
mesh = GenerateMeshes2D(vertices_coord, mshSize, domain_vertiexID)
# mesh.Plot(PointIDs=False,CellIDs=False,Regions=False)

V = FunctionSpace(mesh)

# set Plastic Right Cauchy-Green tensor Cp at each domain
Cp = np.zeros((V.cell_coord.shape[0], 2, 2))
for i in range(domains.shape[0]):
    Cp[V.domain_cellID[i], 0, 0] = domain_Cp[i, 0]
    Cp[V.domain_cellID[i], 0, 1] = domain_Cp[i, 1]
    Cp[V.domain_cellID[i], 1, 0] = domain_Cp[i, 1]
    Cp[V.domain_cellID[i], 1, 1] = domain_Cp[i, 2]
V.set_Cp(Cp)


s = Cp[:, 1, 0]
Plas_work = np.sum(s * V.cellArea)  # 10**-6 J/mm


#define test functions


# Define PK2(F,Cp) and psi(F,Cp) of the St.Venant-Kirchhoff constitutive law.
def PK2_func(F, Cp):
    C =np.einsum("gji,gjk->gik",F,F)

    CpI = np.linalg.inv(Cp)
    E = (C-Cp)*0.5

    return np.einsum("gij,gkl,gkl->gij",CpI,CpI,E,optimize="greedy")*Lame+  np.einsum("gik,gjl,gkl->gij",CpI,CpI,E,optimize="greedy")*2*G

def psi(F, Cp):  # Energy per unit thickness    * 10^-3 J/m   or    J/mm
    C = np.einsum("gji,gjk->gik", F, F)
    CpI = np.linalg.inv(Cp)

    E = (C-Cp)*0.5

    CpI_ddot_E_exp2 = np.einsum("gij,gij->g", CpI, E)**2

    CpI_dot_ET = np.einsum("gij,gkj->gik",CpI,E)
    E_dot_CpIT = np.einsum("gij,gkj->gik",E,CpI)

    CpI_dot_ET_ddot_E_dot_CpIT = np.einsum("gij,gij->g",CpI_dot_ET,E_dot_CpIT)
    return (CpI_ddot_E_exp2*Lame/2+CpI_dot_ET_ddot_E_dot_CpIT*G)*np.sqrt(np.linalg.det(Cp))


# strain_energy_TotWork = []
for testID in range(loadings.shape[0]):
    if type(loadings[testID, 0]) == str:
        loading = eval(loadings[testID, 0])
    else:
        loading = loadings[testID, 0]

    V.name = str(loadings[testID, 1])


    y_Trial = TrialVectorFunction(V)
    y_Tests = TestVectorFunctions(V)
    # if testID > 0:
    #     y_Trial.baseValue = np.loadtxt("yTrial.txt")
    # y_Trial.baseValue=loadtxt("yTrial_"+ V.name +".txt")

    problem = EquilibriumProblem(y_Trial, y_Tests, psi, PK2_func)

    ################boundary
    if loading==None:
        d1 = np.linalg.norm(V.node_coord - np.array([0, 0]), axis=1)
        _id1 = np.where((d1 == min(d1)))[0]


        d2 = np.linalg.norm(V.node_coord - np.array([10, 0]), axis=1)
        _id2 = np.where((d2 == min(d2)))[0]

        problem.set_dirichlet(np.concatenate((_id1*2,
                                              _id1*2+1,
                                              _id2*2+1)),
                              np.array([0,0,0.0]))

    else:


        x2_max = np.max(V.node_coord[:, 1])
        Ids_top = np.where(V.node_coord[:, 1] == x2_max)[0]*2+1
        u_top=np.zeros(Ids_top.shape[0])+loading


        d1 = np.linalg.norm(V.node_coord - np.array([0, 0]), axis=1)
        _id1 = np.where((d1 == min(d1)))[0]
        x2_min = np.min(V.node_coord[:, 1])
        Ids_bot = np.append(_id1*2,np.where(V.node_coord[:, 1] == x2_min)[0]*2+1)
        u_bot=np.zeros(Ids_bot.shape[0])



        Ids=np.append(Ids_top,Ids_bot,axis=0)


        u=np.append(u_top,u_bot,axis=0)
        problem.set_dirichlet(Ids,u)
    #####################


    y, energy = problem.Solve_ByNewton(0.001, 0.001)
    np.savetxt("yTrial_" + V.name + ".txt", y)
    # np.savetxt("yTrial.txt", y.baseValue)

    # saveVTKdata
    u = y-V.node_coord
    F = V.Nabla(y)
    EnerDens = psi(F, Cp)
    PK2 = PK2_func(F, Cp)
    PK1 =np.einsum("gij,gjk->gik",F,PK2)*np.sqrt(np.linalg.det(Cp))[:,np.newaxis,np.newaxis]
    Sigma = np.einsum("gij,gkj->gik",PK1,F) / np.linalg.det(F)[:,np.newaxis,np.newaxis]

    V.SaveFuncs([u, F, Cp, PK1, PK2, Sigma, EnerDens],
                ["u", "F", "Cp", "PK1", "PK2", "Sigma", "EnergyDensity"])
#     if loading == None:
#         strain_energy_TotWork.append(["None", energy])
#     else:
#         strain_energy_TotWork.append([loading / (x2_max - x2_min), energy,energy+Plas_work])
# savetxt("\strain_energy_J_per_mm.txt", array(strain_energy_TotWork))

