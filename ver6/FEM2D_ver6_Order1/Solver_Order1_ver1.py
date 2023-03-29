from FEM2D_ver6_Order1.Functions2D_Order1 import *
import time
import scipy.sparse as sp





def jac_PK1_i(PK1, Cp,nodeI_gaussID, nodei, y_, dy, d, y_shape, Nabla_func, PK2_func):
    nodeI = int(nodei / 2)
    localI_F, localI_gaussIDs = Nabla_func(np.reshape(y_ + dy[nodei],y_shape),  nodeI), nodeI_gaussID[nodeI]

    return (localI_F @ PK2_func(localI_F, Cp[localI_gaussIDs]) - PK1[localI_gaussIDs]).ravel() / d


def TrialVectorFunction(V:FunctionSpace,dim=2):
    if dim==2:
        trial=V.node_coord.copy()
    return trial

class TestVectorFunctions:
    def __init__(self,V:FunctionSpace,dim=2):
        node_coord=V.node_coord
        self.TestFuncs= np.reshape(np.identity(node_coord.shape[0]*dim),(node_coord.shape[0]*dim,node_coord.shape[0],dim))
        self.V=V
        self._dim=dim

        self._TestSparse_flag=False

    def _GenerateTestSparse(self,YparaID_to_YravelID):
        if self._TestSparse_flag:
            pass
        else:
            V = self.V
            nodeI_cellID, cellNum, Nabla, Jp_dV, v_, dim = V.nodeI_cellID, V.cell_nodeID.shape[0], V._Nabla_N3_NodeI, (V.cellArea * V.Jp)[:,np.newaxis,np.newaxis], self.TestFuncs, self._dim

            print("testSparseMat preparation...")
            # self.dim is the degree of freedom of the trial func at each point, 2*self.dim is the degree of freedom of its Nabla

            temp_Row = np.array([])
            indices_Col = np.array([])  # testids
            Data = np.array([])
            for i in range(YparaID_to_YravelID.shape[0]):
                nodei = YparaID_to_YravelID[i]
                I = int(nodei / 2)
                cellids = nodeI_cellID[I]
                temp_Row = np.append(temp_Row, cellids)
                indices_Col = np.append(indices_Col, i * np.ones(2 * dim * cellids.shape[0], dtype="int32"))  # testids
                Data = np.append(Data, (Nabla(v_[nodei], I) * Jp_dV[cellids]).ravel())
            temp_Row = 2 * dim * temp_Row  # cellIds*2*self.dim
            indices_Row = np.array([temp_Row + i for i in range(2 * dim)]).T.ravel()

            self.sparse_jacPK1_i, self.sparse_jacPK1_j, self.sparse_jacPK1_shape = indices_Col, indices_Row, (
            YparaID_to_YravelID.shape[0], cellNum * 2 * dim)
            self.testSparse = sp.coo_matrix((Data, (indices_Row, indices_Col)),
                                            shape=(cellNum * 2 * dim, YparaID_to_YravelID.shape[
                                                0]))  # [i,0,j]   j=(4*C,4*C+1,4*C+2,4*C+3) of (F_00,F_01,F_10,F_11)@(C-th cell)          i-th testfunc
            print("testSparse completed")
            self._TestSparse_flag = True


class EquilibriumProblem:
    def __init__(self,TrialFunc:np.ndarray,TestFuncs:TestVectorFunctions,psi_func,PK2_func):
        self.y_initial_param=TrialFunc.ravel()
        self.y_shape=TrialFunc.shape

        self.dim=1
        for i in TrialFunc.shape[1:]:
            self.dim=self.dim*i

        V=TestFuncs.V
        self.V=V
        self.gauss_area=V.cellArea
        self.PK2_func=PK2_func
        self.psi_func=psi_func
        self.Cp=V.Cp

        self.y_para_len=self.y_initial_param.shape[0]
        self.TestFuncs=TestFuncs

        # self.nodeIJ_cellID=V.nodeIJ_cellID
        self.nodeI_gaussID=V.nodeI_cellID



        self.d=1e-7
        self.dy=np.identity(self.y_para_len)*self.d

        self.y_=TrialFunc.copy()
        self.YparaID_to_YravelID=np.arange(0,self.V.node_coord.shape[0]*self.dim,1,dtype="int32")


        self._other_constraint_flag=False
        self._dirichlet_flag=False


    def _GenerateTestSparse(self):
        V=self.V
        nodeI_cellID,cellNum,Nabla,Jp_dV, YparaID_to_YravelID,v_,dim= V.nodeI_cellID,V.cell_nodeID.shape[0],V._Nabla_N3_NodeI,(V.cellArea * V.Jp)[:, np.newaxis, np.newaxis],self.YparaID_to_YravelID,self.TestFuncs,self.dim

        print("testSparseMat preparation...")
        #self.dim is the degree of freedom of the trial func at each point, 2*self.dim is the degree of freedom of its Nabla

        temp_Row=np.array([])
        indices_Col=np.array([])#testids
        Data=np.array([])
        for i in range(YparaID_to_YravelID.shape[0]):
            nodei=YparaID_to_YravelID[i]
            I=int(nodei/2)
            cellids=nodeI_cellID[I]
            temp_Row=np.append(temp_Row,cellids)
            indices_Col=np.append(indices_Col,i*np.ones(2*dim*cellids.shape[0],dtype="int32")) #testids
            Data=np.append(Data,(Nabla(v_[nodei],I)*Jp_dV[cellids]).ravel())
        temp_Row=2*dim*temp_Row #cellIds*2*self.dim
        indices_Row = np.array([temp_Row+i for i in range(2*dim)]).T.ravel()


        self.sparse_jacPK1_i,self.sparse_jacPK1_j,self.sparse_jacPK1_shape=indices_Col,indices_Row,(YparaID_to_YravelID.shape[0],cellNum*2*dim)
        self.testSparse = sp.coo_matrix((Data, (indices_Row, indices_Col)),
                                  shape=(cellNum*2*dim,YparaID_to_YravelID.shape[0]))  # [i,0,j]   j=(4*C,4*C+1,4*C+2,4*C+3) of (F_00,F_01,F_10,F_11)@(C-th cell)          i-th testfunc
        print("testSparse completed")


    def set_dirichlet(self,coords_indices,u):
        if self._dirichlet_flag:
            raise Exception("dirichlet boundary condition already set")

        nodei=coords_indices

        y_=self.y_.ravel()
        y_[coords_indices]=self.V.node_coord.ravel()[coords_indices]+u
        self.y_=np.reshape(y_,self.y_shape)

        self.YparaID_to_YravelID=np.delete(self.YparaID_to_YravelID,coords_indices)

        self.y_initial_param=self.y_.ravel()[self.YparaID_to_YravelID]
        self.y_para_len = self.YparaID_to_YravelID.shape[0]

        self._dirichlet_flag=True




    def set_other_constraint(self,constraint_func):
        if self._other_constraint_flag:
            raise Exception("other_constraint already set")
        self.constraint_func=constraint_func
        constraint_initial=self.constraint_func(self.y_initial_param)
        self.constraints_Num=constraint_initial.shape[0]
        self._other_constraint_flag=True

    def _jac_other_constraints(self,y_para):
        constraint0=self.constraint_func(y_para)
        y_=self.y_.ravel()
        y_[self.YparaID_to_YravelID]=y_para
        pary_i_cons_j=np.array(
            [(self.constraint_func(y_+self.dy[self.YparaID_to_YravelID[i]])-constraint0)/self.d for i in range(self.y_para_len)  ]
        )
        return pary_i_cons_j,constraint0

    def _Jac_and_Hess_E(self,y_para,tol,rel_tol):
        y_shape,y_,d,dy,test_sparse,GArea,Cp =self.y_shape,self.y_.ravel(), self.d,self.dy,self.TestFuncs.testSparse,self.gauss_area, self.Cp
        sparse_jacPK1_i, sparse_jacPK1_j, sparse_jacPK1_shape = self.TestFuncs.sparse_jacPK1_i, self.TestFuncs.sparse_jacPK1_j, self.TestFuncs.sparse_jacPK1_shape
        PK2_func,V_Nabla_N3_J,V_Nabla=self.PK2_func,self.V._Nabla_N3_NodeI,self.V.Nabla
        nodeI_gaussID, YparaID_to_YravelID= self.nodeI_gaussID, self.YparaID_to_YravelID


        y_[YparaID_to_YravelID]=y_para

        F=V_Nabla(np.reshape(y_,self.y_shape))
        PK1=F@PK2_func(F,Cp)

        jac=sp.csr_matrix(PK1.ravel()).dot(test_sparse).toarray().ravel()
        err,flag=np.linalg.norm(jac)/np.sqrt(jac.shape[0]),True
        hess=sp.csc_matrix([[0]])

        if self.IterNum==0:
            rel_err_ener=1
        else:
            rel_err_ener=np.abs(err-self.err)/self.err
            if rel_err_ener < rel_tol:
                flag=False
        self.err=err

        if err>tol and flag:

            Data=np.array([])
            for nodei in YparaID_to_YravelID:
                Data=np.append(Data,jac_PK1_i(PK1, Cp, nodeI_gaussID, nodei, y_, dy, d, y_shape, V_Nabla_N3_J, PK2_func))

            hess=sp.coo_matrix(  (Data,(sparse_jacPK1_i,sparse_jacPK1_j)),shape= sparse_jacPK1_shape    ).dot(test_sparse).toarray()

        ener=np.sum(self.psi_func(F,Cp)*GArea)

        return jac,hess,ener,err,rel_err_ener
    def _Loss_Iterator(self,y_para_and_lmda,tol,rel_tol):
        self.IterNum = 0
        y_para, lmda = y_para_and_lmda[:self.y_para_len], y_para_and_lmda[self.y_para_len:]
        y_para, lmda = y_para_and_lmda[:self.y_para_len], y_para_and_lmda[self.y_para_len:]
        jac_E, hess_E, energy, err_ener, rel_err_ener = self._Jac_and_Hess_E(y_para, tol, rel_tol)


        pary_i_cons_j, cons_i = self._jac_other_constraints(y_para)
        jac = np.append(jac_E + (np.mat(pary_i_cons_j) * (np.mat(lmda).T)).A.ravel(), cons_i)
        err_tot = np.linalg.norm(jac) / np.sqrt(jac.shape[0])
        print(f"Iter{self.IterNum},", " energy=",   energy," 10^-6 J/mm", ", rel_err=NaN", ", err=", err_ener,
              ", err_tot=", err_tot, ", lmda=", lmda)
        while err_tot > tol and rel_err_ener > rel_tol:
            hess0_ = np.append(hess_E, pary_i_cons_j, axis=1)
            hess1_ = np.append(pary_i_cons_j.T, np.zeros((self.constraints_Num, self.constraints_Num)), axis=1)
            hess   = np.append(hess0_,
                                hess1_,
                                axis=0)

            y_para_and_lmda = y_para_and_lmda - jac.dot( np.linalg.inv(hess)).toarray().ravel()
            y_para, lmda = y_para_and_lmda[:self.y_para_len], y_para_and_lmda[self.y_para_len:]
            self.IterNum += 1
            jac_E, hess_E, energy, err_ener, rel_err_ener = self._Jac_and_Hess_E(y_para, tol, rel_tol)
            pary_i_cons_j, cons_i = self._jac_other_constraints(y_para)


            jac = np.append(jac_E + (np.mat(pary_i_cons_j) * (np.mat(lmda).T)).A.ravel(), cons_i)
            err_tot = np.linalg.norm(jac) / np.sqrt(jac.shape[0])
            print(f"Iter{self.IterNum},", " energy=", energy," 10^-6 J/mm", ", rel_err=", rel_err_ener, ", err=", err_ener,
                  ", err_tot=", err_tot, ", lmda=", lmda)
        return y_para,energy
    def _Ener_Iterator(self,y_para,tol,rel_tol):
        self.IterNum=0
        jac, hess, energy, err,rel_err_ener = self._Jac_and_Hess_E(y_para,tol,rel_tol)
        print(f"Iter{self.IterNum},", " energy=", energy," 10^-6 J/mm", ", rel_err=NaN", ", err=", err,
              )

        while err>tol and rel_err_ener>rel_tol:
            y_para=y_para-jac.dot(np.linalg.inv(hess))

            self.IterNum+=1
            jac, hess, energy, err,rel_err_ener = self._Jac_and_Hess_E(y_para,tol,rel_tol)
            print(f"Iter{self.IterNum},", " energy=", energy," 10^-6 J/mm", ", rel_err=", rel_err_ener, ", err=", err,
                  )
        return y_para,energy
    def Solve_ByNewton(self,tol=0.01,rel_tol=0.01):
        self.TestFuncs._GenerateTestSparse(self.YparaID_to_YravelID)

        start=time.time()
        print("\n\nstart iteration...")
        if self._other_constraint_flag:
            y_para,energy=self._Loss_Iterator(np.append(self.y_initial_param,np.zeros(self.constraints_Num)),tol,rel_tol)
        else:
            y_para,energy=self._Ener_Iterator(self.y_initial_param,tol,rel_tol)
        end=time.time()
        print("Iteration end...   time cost: ", end - start," s,  \n\n")
        if self._dirichlet_flag:
            y_ = self.y_.ravel()
            y_[self.YparaID_to_YravelID] = y_para
            self.y_ = np.reshape(y_, self.y_shape)
            y=self.y_
        else:
            y=np.reshape(y_para,self.y_shape)
        return y,energy