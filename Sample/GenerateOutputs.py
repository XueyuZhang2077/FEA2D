import os

import pandas as pd
import shutil
from DIY_FEM.Solver import *
# from FEM2D_Cupy_Order1 import *



vertices = pd.read_excel("Input.xlsx", sheet_name="Vertices_Coords").to_numpy()
domains = pd.read_excel("Input.xlsx", sheet_name="Domain_VertID").to_numpy()[:,1:]

info = pd.read_excel("Input.xlsx", sheet_name="Elasticity").to_numpy()
domain_Cp = pd.read_excel("Input.xlsx", sheet_name="Domain_Cp").to_numpy()[:, 1:]
loadings = pd.read_excel("Input.xlsx", sheet_name="Bound_Condition").to_numpy()
G = info[0, 0]
nu = info[0, 1]
lmda= 2 * G * nu / (1 - 2 * nu)

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

V = FunctionSpace2D(mesh,name="Sample")

# set Plastic Right Cauchy-Green tensor Cp at each domain
Cp = np.zeros((V.cell_coord.shape[0], 2, 2))
for i in range(domains.shape[0]):
    Cp[V.domain_cellID[i], 0, 0] = domain_Cp[i, 0]
    Cp[V.domain_cellID[i], 0, 1] = domain_Cp[i, 1]
    Cp[V.domain_cellID[i], 1, 0] = domain_Cp[i, 1]
    Cp[V.domain_cellID[i], 1, 1] = domain_Cp[i, 2]


s = Cp[:, 1, 0]


# Define P-K1 stress PK1(F,Cp) and energy density psi(F,Cp) according to the St.Venant-Kirchhoff constitutive law.
def PK1_func(F, Cp):
    C = np.einsum("gji,gjk->gik", F, F)

    CpI = np.linalg.inv(Cp)
    E = (C - Cp) * 0.5

    CpI_ddot_E = np.einsum("gij,gij->g", CpI, E)
    CpI_dot_E = np.einsum("gij,gjk->gik", CpI, E)
    CpI_dot_E_dot_CpIT = np.einsum("gij,gkj->gik", CpI_dot_E, CpI)

    return F @ (CpI * CpI_ddot_E[:, np.newaxis, np.newaxis] * lmda + CpI_dot_E_dot_CpIT * 2 * G) * np.sqrt(
        np.linalg.det(Cp))[:, np.newaxis, np.newaxis]


def psi(F, Cp):  # Energy per unit thickness    * 10^-3 J/m   or    J/mm
    C = np.einsum("gji,gjk->gik", F, F)
    CpI = np.linalg.inv(Cp)

    E = (C - Cp) * 0.5

    CpI_ddot_E_exp2 = np.einsum("gij,gij->g", CpI, E) ** 2

    CpI_dot_ET = np.einsum("gij,gkj->gik", CpI, E)
    E_dot_CpIT = np.einsum("gij,gkj->gik", E, CpI)

    CpI_dot_ET_ddot_E_dot_CpIT = np.einsum("gij,gij->g", CpI_dot_ET, E_dot_CpIT)
    return (CpI_ddot_E_exp2 * lmda / 2 + CpI_dot_ET_ddot_E_dot_CpIT * G) * np.sqrt(np.linalg.det(Cp))



# strain_energy_TotWork = []
for testID in range(loadings.shape[0]):
    if type(loadings[testID, 0]) == str:
        loading = eval(loadings[testID, 0])
    else:
        loading = loadings[testID, 0]

    V.name = str(loadings[testID, 1])


    y_Trial = TrialVectorFunction(V)
    if testID > 0:
        y = np.loadtxt("yTrial.txt")
    # y=loadtxt("yTrial_"+ V.name +".txt")


    #########set dirichlet boundary condition#######
    x2_max = np.max(V.node_coord[:, 1])
    Ids_top = np.where(V.node_coord[:, 1] == x2_max)[0]
    u_top = np.zeros(Ids_top.shape[0])
    Ids_top = np.append(Ids_top*2,Ids_top*2+1)
    u_top=np.append(u_top,u_top+loading)

    d1 = np.linalg.norm(V.node_coord - np.array([0, 0]), axis=1)
    _id1 = np.where((d1 == min(d1)))[0]
    x2_min = np.min(V.node_coord[:, 1])
    Ids_bot=np.where(V.node_coord[:, 1] == x2_min)[0]
    # Ids_bot = np.append(_id1*2,np.where(V.node_coord[:, 1] == x2_min)[0]*2+1)
    u_bot=np.zeros(Ids_bot.shape[0])
    Ids_bot=np.append(Ids_bot*2,Ids_bot*2+1)
    u_bot=np.append(u_bot,u_bot)



    Ids=np.append(Ids_top,Ids_bot,axis=0)


    u_bound=np.append(u_top,u_bot,axis=0)
    y_bound=V.node_coord.ravel()[Ids]+u_bound

    #set test functions & trial functions
    y_Tests=TestVectorFunctions(V,DirichletIDs=Ids)
    y_Trial=TrialVectorFunction(V,DirichletVals=y_bound)

    #define problem
    prob=EquilibriumProblem_Isotropic(y_Trial,y_Tests,Cp,lmda,G)





    #######solve the PDE of the stress equilibrium Eq.#######
    y= prob.solve(0.001, 0.001)
    np.savetxt("yTrial_" + V.name + ".txt", y)
    np.savetxt("yTrial.txt", y)

    # saveVTKdata
    u = y-V.node_coord
    F = V.Nabla(y)
    C = np.einsum("gji,gjk->gik", F, F)
    E = (C - Cp) * 0.5
    CpI = np.linalg.inv(Cp)
    S = lmda * np.einsum("...ij,...kl,...kl->...ij", CpI, CpI, E) + 2 * G * np.einsum("...ik,...jl,...kl->...ij", CpI,
                                                                                      CpI, E)
    P=F@S

    Sigma = np.einsum("gij,gkj->gik",P,F) / np.linalg.det(F)[:,np.newaxis,np.newaxis]

    psi=1/2*np.einsum("...ij,...kj,...ki->...",S,E,np.linalg.inv(F))

    V.SaveFuncs([u, F, Cp, P, S, Sigma, psi],
                ["u", "F", "Cp", "PK1", "PK2", "Sigma", "EnergyDensity"])


