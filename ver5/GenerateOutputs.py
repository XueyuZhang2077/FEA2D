import os

import pandas as pd
from numpy import *
import shutil
from FEM2D_ver5_Order1 import *
L_arr=concatenate((array([0]),
                   arange(1,111,2),
                   array([110,
                          111])))
# for i in range(L_arr.shape[0]):
for i in arange(56,57,1,dtype=int):
    L=L_arr[i]




    vertices = pd.read_excel("Input_%d.xlsx"%i, sheet_name="Vertices_Coords").to_numpy()
    domains = pd.read_excel("Input_%d.xlsx"%i, sheet_name="Domain_VertID").to_numpy()[:, 1:]
    info = pd.read_excel("Input_%d.xlsx"%i, sheet_name="Info").to_numpy()
    domain_Cp = pd.read_excel("Input_%d.xlsx"%i, sheet_name="Domain_Cp").to_numpy()[:, 1:]
    loadings = pd.read_excel("Input_%d.xlsx"%i, sheet_name="Load_Condition").to_numpy()
    G = info[0, 0]
    nu = info[0, 1]
    mu = 2 * G * nu / (1 - 2 * nu)

    vertices_coord = vertices[:, :2]
    mshSize = vertices[:, 2]
    domain_vertiexID = []
    for i in range(domains.shape[0]):
        temp = domains[i].tolist()

        temp = int_(delete(temp, isnan(temp)))
        domain_vertiexID.append(array(temp))

    # generate mesh
    mesh = GenerateMeshes2D(vertices_coord, mshSize, domain_vertiexID)
    # mesh.Plot()

    V = FunctionSpace(mesh)

    # set Plastic Right Cauchy-Green tensor Cp at each domain
    Cp_ = zeros((V.cell_coord.shape[0], 2, 2))
    for i in range(domains.shape[0]):
        Cp_[V.domain_cellID[i], 0, 0] = domain_Cp[i, 0]
        Cp_[V.domain_cellID[i], 0, 1] = domain_Cp[i, 1]
        Cp_[V.domain_cellID[i], 1, 0] = domain_Cp[i, 1]
        Cp_[V.domain_cellID[i], 1, 1] = domain_Cp[i, 2]
    Cp = MatrixFunction(Cp_)
    s = Cp_[:, 1, 0]
    Plas_work = sum(s * V.cellArea)  # 10**-6 J/mm


    # Defines PK2(F,Cp) and psi(F,Cp) of the St.Venant-Kirchhoff constitutive law.
    def PK2_func(F: MatrixFunction, Cp: MatrixFunction):
        C = F.T().times(F)
        CpI = Cp.I()
        CpIT = CpI.T()
        E = (C.minus(Cp)).multiply(0.5)
        return CpI.multiply(CpI.dot(E).baseValue[:, newaxis, newaxis]).multiply(mu).plus(
            (CpI.times(E).times(CpIT)).multiply(2 * G))


    def psi(F: MatrixFunction, Cp: MatrixFunction):  # Energy per unit thickness    * 10^-3 J/m   or    J/mm
        C = F.T().times(F)
        CpI = Cp.I()
        Ee = (C.minus(Cp)).multiply(0.5)
        EeT = Ee.T()
        CpIdotEe_exp2 = CpI.dot(Ee).exp(2)
        CpIEeT = CpI.times(EeT)
        EeCpIT = CpIEeT.T()
        CpIEeTdotEeCpIT = CpIEeT.dot(EeCpIT)
        return CpIdotEe_exp2.multiply(mu / 2).plus(CpIEeTdotEeCpIT.multiply(G)).times(Cp.det().exp(0.5))


    # strain_energy_TotWork = []
    for testID in range(loadings.shape[0]):
        if type(loadings[testID, 0]) == str:
            loading = eval(loadings[testID, 0])
        else:
            loading = loadings[testID, 0]

        V.name = str(loadings[testID, 1])

        # Solve The problem
        y_Trial = TrialVectorFunction(V)
        # if testID > 0:
        #     y_Trial.baseValue = loadtxt("yTrial.txt")
        # # y_Trial.baseValue=loadtxt("yTrial_"+ V.name +".txt")

        y_Tests = TestFunctions(V)

        problem = EquilibriumProblem(y_Trial, y_Tests, psi, PK2_func, Cp)

        ################boundary
        d1 = linalg.norm(V.node_coord - array([0, 0]), axis=1)
        _id1 = where((d1 == min(d1)))[0]
        _id1= append(_id1*2,_id1*2+1)


        d2 = linalg.norm(V.node_coord - array([110, 0]), axis=1)
        _id2 = 2*where((d2 == min(d2)))[0]+1


        problem.set_dirichlet(append(_id1,_id2),array([0,0,0]))
        #####################


        y, energy = problem.Solve_ByNewton(0.001, 0.001)
        savetxt("yTrial_" + V.name + ".txt", y.baseValue)
        # savetxt("yTrial.txt", y.baseValue)

        # saveVTKdata
        u = y.minus(VectorFunction(V.node_coord))
        F = V.Nabla(y)
        EnerDens = psi(F, Cp)
        PK2 = PK2_func(F, Cp)
        PK1 = F.times(PK2)
        Sigma = (PK1.times(F.T())).times(F.det().exp(-1))

        V.SaveFuncs([u, F, Cp, PK1, PK2, Sigma, EnerDens],
                    ["u", "F", "Cp", "PK1", "PK2", "Sigma", "EnergyDensity"])
    #     if loading == None:
    #         strain_energy_TotWork.append(["None", energy])
    #     else:
    #         strain_energy_TotWork.append([loading / (x2_max - x2_min), energy,energy+Plas_work])
    # savetxt("\strain_energy_J_per_mm.txt", array(strain_energy_TotWork))

