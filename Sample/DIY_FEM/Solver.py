import numpy as np
from DIY_FEM.Functions3D import *
from DIY_FEM.Functions2D import *
import time
import scipy.sparse as sp







class TrialVectorFunction:
    def __init__(self,V:FunctionSpace3D or FunctionSpace2D,DirichletVals=np.array([])):
        self.V=V
        self.DirichletVals=DirichletVals
        self.node_trial=V.node_coord.copy()
    # if DirichletIds==None:
    #     trial=V.node_coord.copy()
    # else:
    #     trial = V.node_coord.copy().ravel()
    #     trial[DirichletIds]=DirichletValues
    #     trial=np.reshape(trial,V.node_coord.shape)


class TestVectorFunctions:
    def __init__(self,V:FunctionSpace3D or FunctionSpace2D,DirichletIDs=np.array([])):
        dim=V.dim
        self.V=V
        self.DirchletIDs=DirichletIDs
        if DirichletIDs.shape[0]>0:
            self.FreeIdsParaIds=np.delete(np.arange(0,V.node_coord.shape[0]*dim,1,dtype=int),DirichletIDs)

        XI=V.XI
        cell_nodeID=V.cell_nodeID

        deltaF=np.zeros((cell_nodeID.shape[0],(dim+1),dim,dim,dim),dtype=float)

        deltaF_RowIds=np.array([[K*dim+L for L in range(dim)] for K in range(dim)]  ) #temp
        deltaF_RowIds=np.array([[deltaF_RowIds for J in range(dim)] for I in range(dim+1)]) #temp
        deltaF_RowIds=np.array([i*dim**2+deltaF_RowIds for i in range(cell_nodeID.shape[0])])

        deltaF_ColIds=deltaF_RowIds.copy() #temp
        deltaF_ParaIds=np.zeros((XI.shape[0],dim+1,dim),dtype=int) #temp
        deltaF_TransCellIds=np.array([np.full((dim+1,dim),i,dtype=int) for i in range(cell_nodeID.shape[0])])




        # deltaF[:,0,0,0,0],deltaF[:,0,0,0,1],deltaF[:,0,0,0,2]=XI[:,0,0],XI[:,1,0],XI[:,2,0]
        # deltaF[:,0,1,1,0],deltaF[:,0,1,1,1],deltaF[:,0,1,1,2]=XI[:,0,0],XI[:,1,0],XI[:,2,0]
        # deltaF[:,0,2,2,0],deltaF[:,0,2,2,1],deltaF[:,0,2,2,2]=XI[:,0,0],XI[:,1,0],XI[:,2,0]

        for I in range(dim+1):
            for J in range(dim):
                deltaF_ParaIds[:,I,J]=cell_nodeID[:,I]*dim+J
                deltaF[:, I, J, J, :] = XI[:, 0:dim, I]
                for K in range(dim):
                    for L in range(dim):
                        deltaF_ColIds[:,I,J,K,L]=cell_nodeID[:,I]*dim+J

        self.deltaF = deltaF.reshape( (cell_nodeID.shape[0] * (dim + 1) * dim, dim, dim))
        self.deltaF_RowIds=deltaF_RowIds.reshape((cell_nodeID.shape[0] * (dim + 1) * dim, dim, dim))
        self.deltaF_ColIds=deltaF_ColIds.reshape((cell_nodeID.shape[0] * (dim + 1) * dim, dim, dim))
        deltaF_ParaIds=deltaF_ParaIds.ravel()
        self.deltaF_TransCellIds=deltaF_TransCellIds.ravel()
        if DirichletIDs.shape[0]>0:
            self.deltaF_FreeIds=np.arange(0,deltaF_ParaIds.shape[0],1,dtype=int)
            idss=[]
            for DiriID in np.sort(DirichletIDs)[::-1]:
                ids0,ids1,ids2=np.where(self.deltaF_ColIds>DiriID)
                idss = idss + np.where(deltaF_ParaIds == DiriID)[0].tolist()
                self.deltaF_ColIds[ids0,ids1,ids2] = self.deltaF_ColIds[ids0,ids1,ids2] - 1

            self.deltaF_FreeIds=np.delete(self.deltaF_FreeIds,np.array(idss))
            self.deltaF_ColIds=self.deltaF_ColIds[self.deltaF_FreeIds]
            self.deltaF=self.deltaF[self.deltaF_FreeIds]
            self.deltaF_RowIds=self.deltaF_RowIds[self.deltaF_FreeIds]
            self.deltaF_TransCellIds=self.deltaF_TransCellIds[self.deltaF_FreeIds]

        self.SparseShape=(cell_nodeID.shape[0]*dim**2,V.node_coord.shape[0]*dim-DirichletIDs.shape[0])
        self.deltaFravel_nonzeroids=self.deltaF.ravel()!=0



class EquilibriumProblem_Isotropic:
    def __init__(self,TrialFunc:TrialVectorFunction,TestFuncs:TestVectorFunctions,CpField,lmda,G):
        '''

        :param self:
        :param TrialFunc:
        :param TestFuncs:
        :param ElasticityMatrix: scipy.SparseMatrix or np.ndarray
                                  [[C0000,C0011,C0022,C0012,C0020,C0001],
                                   [C0100,C0111,C0122,C0112,C0120,C0101],
                                   [C0200,C0211,C0222,C0212,C0220,C0201],
                                   [C1100,C1111,C1122,C1112,C1120,C1101],
                                   [C1200,C1211,C1222,C1212,C1220,C1201],
                                   [C2200,C2211,C2222,C2212,C2220,C2201]]
        :param CpField:
        :param CpField: lmda=2*G*nu/(1-2*nu)
        :return:
        '''
        self.TrialFunc,self.TestFuncs,self.Cp,self.CpI,self.lmda,self.G=TrialFunc,TestFuncs,CpField,np.linalg.inv(CpField),lmda,G
        self._set_sparse()
        self._set_para()
    def _set_sparse(self):
        TestFuncs=self.TestFuncs
        if TestFuncs.V.dim==3:
            dV=TestFuncs.V.cellVol
        else:
            dV=TestFuncs.V.cellArea
        self.SparseTests = sp.coo_matrix(((TestFuncs.deltaF*( ( np.sqrt(np.linalg.det(self.Cp))*dV )[TestFuncs.deltaF_TransCellIds,np.newaxis,np.newaxis])).ravel()[TestFuncs.deltaFravel_nonzeroids],
                                     (TestFuncs.deltaF_RowIds.ravel()[TestFuncs.deltaFravel_nonzeroids], TestFuncs.deltaF_ColIds.ravel()[TestFuncs.deltaFravel_nonzeroids])),
                                    shape=TestFuncs.SparseShape)

    def _set_para(self):
        TrialFunc=self.TrialFunc
        TestFuncs=self.TestFuncs
        para=TrialFunc.node_trial.ravel()
        if TestFuncs.DirchletIDs.shape[0]>0:
            para[TestFuncs.DirchletIDs]=TrialFunc.DirichletVals
        self.para=para

    def _get_err(self,free):
        TrialFunc,TestFuncs,Cp,CpI,lmda,G=self.TestFuncs,self.TestFuncs,self.Cp,self.CpI,self.lmda,self.G
        if TestFuncs.DirchletIDs.shape[0]>0:
            self.para[TestFuncs.FreeIdsParaIds]=free
        else:
            self.para=free


        F=TestFuncs.V.Nabla(self.para.reshape(TestFuncs.V.node_coord.shape))
        E=(np.einsum("...ij,...ik->...jk",F,F)-Cp)/2
        S=lmda*np.einsum("...ij,...kl,...kl->...ij",CpI,CpI,E)+2*G*np.einsum("...ik,...jl,...kl->...ij",CpI,CpI,E)

        return self.SparseTests.T.dot((F@S).ravel()),F,S

    def _get_jacob(self,F,S):
        TrialFunc, TestFuncs, lmda,G= self.TestFuncs, self.TestFuncs, self.lmda,self.G
        F,S,CpI=F[TestFuncs.deltaF_TransCellIds],S[TestFuncs.deltaF_TransCellIds],self.CpI.copy()[TestFuncs.deltaF_TransCellIds]


        D_E=(np.einsum("...ij,...ik->...jk",TestFuncs.deltaF,F)+np.einsum("...ij,...ik->...jk",F,TestFuncs.deltaF))/2
        D_S=lmda*np.einsum("...ij,...kl,...kl->...ij",CpI,CpI,D_E)+2*G*np.einsum("...ik,...jl,...kl->...ij",CpI,CpI,D_E)

        D_P=TestFuncs.deltaF@S+F@D_S
        return self.SparseTests.T.dot(
            sp.coo_matrix( (D_P.ravel(),(TestFuncs.deltaF_RowIds.ravel(),TestFuncs.deltaF_ColIds.ravel()  )  ),
                           shape=TestFuncs.SparseShape)
        )

    def solve(self,tol=1e-3,rel_tol=1e-3):
        TestFuncs= self.TestFuncs
        start = time.time()
        if TestFuncs.DirchletIDs.shape[0]>0:
            free=self.para[TestFuncs.FreeIdsParaIds]
        else:
            free=self.para
        err,F,S=self._get_err(free)
        err_norm=np.linalg.norm(err)/np.sqrt(err.shape[0])
        print(err_norm)
        while err_norm>tol:
            jacob=self._get_jacob(F,S).toarray()
            free=free-np.linalg.inv(jacob).dot(err)
            err, F, S = self._get_err(free)
            err_norm = np.linalg.norm(err)/np.sqrt(err.shape[0])
            print(err_norm)
        end = time.time()
        print("cost:",end-start)
        return self.para.reshape(TestFuncs.V.node_coord.shape)


class EquilibriumProblem_Anisotropic:
    def __init__(self,TrialFunc:TrialVectorFunction,TestFuncs:TestVectorFunctions,CpField,ElasticityMatrix):
        '''

        :param self:
        :param TrialFunc:
        :param TestFuncs:
        :param ElasticityMatrix: scipy.SparseMatrix or np.ndarray
                                  [[C0000,C0011,C0022,C0012,C0020,C0001],
                                   [C0100,C0111,C0122,C0112,C0120,C0101],
                                   [C0200,C0211,C0222,C0212,C0220,C0201],
                                   [C1100,C1111,C1122,C1112,C1120,C1101],
                                   [C1200,C1211,C1222,C1212,C1220,C1201],
                                   [C2200,C2211,C2222,C2212,C2220,C2201]]
        :param CpField:
        :param ElasticityMatrix:
        :return:
        '''
        self.TrialFunc,self.TestFuncs,self.Cp,self.CpI,self.ElasMat=TrialFunc,TestFuncs,CpField,np.linalg.inv(CpField),ElasticityMatrix
        self._set_sparse()
        self._set_para()
    def _set_sparse(self):
        TestFuncs=self.TestFuncs
        if TestFuncs.V.dim==3:
            dV=TestFuncs.V.cellVol
        else:
            dV=TestFuncs.V.cellArea
        self.SparseTests = sp.coo_matrix(((TestFuncs.deltaF*( ( np.sqrt(np.linalg.det(self.Cp))*dV )[TestFuncs.deltaF_TransCellIds,np.newaxis,np.newaxis])).ravel()[TestFuncs.deltaFravel_nonzeroids],
                                     (TestFuncs.deltaF_RowIds.ravel()[TestFuncs.deltaFravel_nonzeroids], TestFuncs.deltaF_ColIds.ravel()[TestFuncs.deltaFravel_nonzeroids])),
                                    shape=TestFuncs.SparseShape)

    def _set_para(self):
        TrialFunc=self.TrialFunc
        TestFuncs=self.TestFuncs
        para=TrialFunc.node_trial.ravel()
        if TestFuncs.DirchletIDs.shape[0]>0:
            para[TestFuncs.DirchletIDs]=TrialFunc.DirichletVals
        self.para=para

    def _get_err(self,free):
        TrialFunc,TestFuncs,Cp,CpI,ElasMat=self.TestFuncs,self.TestFuncs,self.Cp,self.CpI,self.ElasMat
        if TestFuncs.DirchletIDs.shape[0]>0:
            self.para[TestFuncs.FreeIdsParaIds]=free
        else:
            self.para=free

        F=TestFuncs.V.Nabla(self.para.reshape(TestFuncs.V.node_coord.shape))
        E=(np.einsum("...ij,...ik->...jk",F,F)-Cp)/2
        E=E@CpI
        S = E.copy()
        if TestFuncs.V.dim==3:
            S_=ElasMat.dot(np.array([E[:,0,0],
                                     E[:,1,1],
                                     E[:,2,2],
                                     E[:,1,2]+E[:,2,1], #此时E_i^j未必对称
                                     E[:,2,0]+E[:,0,2],
                                     E[:,0,1]+E[:,1,0]]))
            S[:,0,0],S[:,1,1],S[:,2,2],S[:,1,2],S[:,2,0],S[:,0,1]=S_[0],S_[1],S_[2],S_[3],S_[4],S_[5]
            S[:,2,1],S[:,0,2],S[:,1,0]=S[:,1,2],S[:,2,0],S[:,0,1]
        else:
            S_=ElasMat.dot(np.array([E[:,0,0],
                                     E[:,1,1],
                                     E[:,0,1]*2]))
            S[:,0,0],S[:,1,1],S[:,0,1]=S_[0],S_[1],S_[2]
            S[:,1,0]=S[:,0,1]
        S = S@CpI


        return self.SparseTests.T.dot((F@S).ravel()),F,S

    def _get_jacob(self,F,S):
        TrialFunc, TestFuncs, ElasMat= self.TestFuncs, self.TestFuncs,self.ElasMat
        F,S,CpI=F[TestFuncs.deltaF_TransCellIds],S[TestFuncs.deltaF_TransCellIds],self.CpI.copy()[TestFuncs.deltaF_TransCellIds]
        D_S=S.copy()
        D_E=(np.einsum("...ij,...ik->...jk",TestFuncs.deltaF,F)+np.einsum("...ij,...ik->...jk",F,TestFuncs.deltaF))/2
        D_E = D_E@CpI

        if TestFuncs.V.dim==3:
            D_S_=ElasMat.dot(np.array([D_E[:,0,0],
                                       D_E[:,1,1],
                                       D_E[:,2,2],
                                       D_E[:,1,2]+D_E[:,2,1],
                                       D_E[:,2,0]+D_E[:,0,2],
                                       D_E[:,0,1]+D_E[:,1,0]]))
            D_S[:,0,0],D_S[:,1,1],D_S[:,2,2],D_S[:,1,2],D_S[:,2,0],D_S[:,0,1]=D_S_[0],D_S_[1],D_S_[2],D_S_[3],D_S_[4],D_S_[5]
            D_S[:,2,1],D_S[:,0,2],D_S[:,1,0]=D_S[:,1,2],D_S[:,2,0],D_S[:,0,1]
        else:
            D_S_=ElasMat.dot(np.array([D_E[:,0,0],
                                       D_E[:,1,1],
                                       D_E[:,0,1]*2]))
            D_S[:,0,0],D_S[:,1,1],D_S[:,0,1]=D_S_[0],D_S_[1],D_S_[2]
            D_S[:,1,0]=D_S[:,0,1]

        D_S = D_S@CpI
        D_P=TestFuncs.deltaF@S+F@D_S
        return self.SparseTests.T.dot(
            sp.coo_matrix( (D_P.ravel(),(TestFuncs.deltaF_RowIds.ravel(),TestFuncs.deltaF_ColIds.ravel()  )  ),
                           shape=TestFuncs.SparseShape)
        )

    def solve(self,tol=1e-3):
        TestFuncs= self.TestFuncs
        start = time.time()
        if TestFuncs.DirchletIDs.shape[0]>0:
            free=self.para[TestFuncs.FreeIdsParaIds]
        else:
            free=self.para
        err,F,S=self._get_err(free)
        err_norm=np.linalg.norm(err)/np.sqrt(err.shape[0])
        print(err_norm)
        while err_norm>tol:
            jacob=self._get_jacob(F,S).toarray()
            free=free-np.linalg.inv(jacob).dot(err)
            err, F, S = self._get_err(free)
            err_norm = np.linalg.norm(err)/np.sqrt(err.shape[0])
            print(err_norm)
        end = time.time()
        print("cost:",end-start)
        return self.para.reshape(TestFuncs.V.node_coord.shape)