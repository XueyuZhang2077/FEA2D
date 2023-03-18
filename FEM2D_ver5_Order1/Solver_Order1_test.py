from FEM2D_ver5_Order1.Functions2D_Order1 import *
import time
class TrialScalarFunction(ScalarFunction):
    def __init__(self,V:FunctionSpace):
        super(TrialScalarFunction, self).__init__(V.node_coord)
        self.funcSpace=V
class TrialVectorFunction(VectorFunction):
    def __init__(self,V:FunctionSpace):
        super(TrialVectorFunction, self).__init__(V.node_coord)
        self.funcSpace=V
class TrialMatrixFunction(MatrixFunction):
    def __init__(self,V:FunctionSpace):
        super(TrialMatrixFunction, self).__init__(V.node_coord)
        self.funcSpace=V

def TestFunctions(V:FunctionSpace,dim=1)->MatrixFunction:
    TestFuncs=[]
    if dim==1:
        v_=identity(V.node_coord.shape[0]*2)*10
        for i in range(V.node_coord.shape[0]*2):
            v=VectorFunction(reshape(v_[i],V.node_coord.shape))
            TestFuncs.append(V.Nabla(v))
    return TestFuncs

class EquilibriumProblem:
    def __init__(self,TrialFunc:TrialVectorFunction,TestFuncs:list,psi_func,PK2_func,Cp):
        self.y_initial_param=TrialFunc.baseValue.ravel()
        self.y_shape=TrialFunc.baseValue.shape

        self.V=TrialFunc.funcSpace
        self.gauss_area=self.V.cellArea
        self.PK2_func=PK2_func
        self.Cp=Cp



        self.y_para_len=self.y_initial_param.shape[0]
        self.TestFuncs=TestFuncs

        self.psi_func=psi_func
        self.nodeIJ_cellID,self.nodeI_gaussID,self.nodeI_node0toI=self.V.nodeIJ_cellID,self.V.nodeI_cellID,self.V.nodeI_node0toI
        self.i_0toi=[]
        for i in range(len(self.nodeI_node0toI)):
            self.i_0toi.append(append(self.nodeI_node0toI[i] * 2,self.nodeI_node0toI[i]*2+1))
            self.i_0toi.append(append(self.nodeI_node0toI[i] * 2, self.nodeI_node0toI[i] * 2 + 1))

        self.ReturnFunc=self.V.ReturnFunc[TrialFunc.baseValue.ndim-1]

        self.d=1e-7
        self.dy=identity(self.y_para_len)*self.d

        self.y_=TrialFunc.baseValue.copy()
        self.YparaID_to_YravelID=arange(0,self.V.node_coord.shape[0]*2,1,dtype=int)
        self.YparaID_to_NodeID=int_(self.YparaID_to_YravelID/2)

        self._other_constraint_flag=False
        self._dirichlet_flag=False
        self._dirichlet_u0Independent_flag=False


    def set_dirichlet(self,coords_indices,u):
        if self._dirichlet_flag:
            raise Exception("dirichlet boundary condition already set")

        nodei=coords_indices

        y_=self.V.node_coord.ravel()
        y_[coords_indices]=y_[coords_indices]+u
        self.y_=reshape(y_,self.y_shape)

        self.YparaID_to_YravelID=delete(self.YparaID_to_YravelID,coords_indices)
        self.YparaID_to_NodeID=delete(self.YparaID_to_NodeID,coords_indices)
        self.y_initial_param=self.y_.ravel()[self.YparaID_to_YravelID]
        self.y_para_len = self.YparaID_to_YravelID.shape[0]

        self._dirichlet_flag=True


        self.where_independentNodei__equal_nodej=[]
        for i in range(len(self.i_0toi)):
            self.i_0toi[i]=int_(intersect1d(self.i_0toi[i],self.YparaID_to_YravelID))
            temp=[]
            for nodej in self.i_0toi[i]:
                temp.append(where(self.YparaID_to_YravelID==nodej)[0])
            self.where_independentNodei__equal_nodej.append(temp)











    def set_other_constraint(self,constraint_func):
        if self._other_constraint_flag:
            raise Exception("other_constraint already set")
        self.constraint_func=constraint_func
        constraint_initial=self.constraint_func(self.y_initial_param)
        self.constraints_Num=constraint_initial.shape[0]
        self._other_constraint_flag=True





    def _jac_other_constraints(self,y_para):
        constraint0=self.constraint_func(y_para)
        pary_i_cons_j=zeros((self.y_para_len,constraint0.shape[0]))
        y_=self.y_.ravel()
        y_[self.YparaID_to_YravelID]=y_para
        for i in range(self.y_para_len):

            pary_i_cons_j[i,:]=(self.constraint_func(y_+self.dy[self.YparaID_to_YravelID[i]])-constraint0)/self.d
        return pary_i_cons_j,constraint0

    def _Jac_and_Hess_E(self,y_para,tol,rel_tol):
        y_shape=self.y_shape
        PK2_func=self.PK2_func
        ReturnFunc=self.ReturnFunc
        GArea, YparaID_to_YravelID,YparaID_to_NodeID= ScalarFunction(self.gauss_area), self.YparaID_to_YravelID,self.YparaID_to_NodeID
        test_funcs, nodeI_gaussID= self.TestFuncs, self.nodeI_gaussID
        i_0toi, Cp = self.i_0toi, self.Cp

        d, dy, self_y_ = self.d, self.dy, self.y_
        IntersectN3_IJ_gaussID, V_Nabla_IntersectN3_IJ = self.V.IntersectN3_IJ_cellID, self.V._Nabla_IntersectN3_IJ
        where_independentNodei__equal_nodej=self.where_independentNodei__equal_nodej

        y_=self.y_.ravel()
        y_[YparaID_to_YravelID]=y_para
        self.y_=reshape(y_,self.y_shape)

        y=ReturnFunc(self.y_)
        F=self.V.Nabla(y)
        PK1=F.times(PK2_func(F,self.Cp))

        jac,hess=zeros(YparaID_to_YravelID.shape[0]),zeros((YparaID_to_YravelID.shape[0],YparaID_to_YravelID.shape[0]))
        for i in range(YparaID_to_YravelID.shape[0]):
            nodeI,nodei=YparaID_to_NodeID[i],YparaID_to_YravelID[i]
            localI_gaussids=nodeI_gaussID[nodeI]
            jac[i]=test_funcs[nodei].Get(localI_gaussids).dot(PK1.Get(localI_gaussids)).dot(GArea.Get(localI_gaussids))
        err,flag=linalg.norm(jac)/sqrt(jac.shape[0]),True
        if self.IterNum==0:
            rel_err_ener=1
        else:
            rel_err_ener=abs(err-self.err)/self.err
            if rel_err_ener < rel_tol:
                flag=False
        self.err=err

        if err>tol and flag:
            for i in range(YparaID_to_YravelID.shape[0]):
                nodeI,nodei=YparaID_to_NodeID[i],YparaID_to_YravelID[i]
                nodej_arr=i_0toi[nodei]

                jid=0
                for nodej in nodej_arr:
                    nodeJ=int(nodej/2)
                    localIJ_gaussids = IntersectN3_IJ_gaussID[nodeI][nodeJ]
                    localIJ_Cp  = Cp.Get(localIJ_gaussids)
                    localIJ_err = PK1.Get(localIJ_gaussids).dot(test_funcs[nodei].Get(localIJ_gaussids)).dot(GArea.Get(localIJ_gaussids))

                    y_p_dyj=ReturnFunc(reshape(y_+dy[nodej],y_shape))
                    localIJ_F=V_Nabla_IntersectN3_IJ(y_p_dyj,nodeI,nodeJ)
                    next_jac=localIJ_F.times(PK2_func(localIJ_F,localIJ_Cp)).dot(test_funcs[nodei].Get(localIJ_gaussids)).dot(GArea.Get(localIJ_gaussids))

                    j=where_independentNodei__equal_nodej[nodei][jid]
                    hess[i,j]=(next_jac - localIJ_err)/d
                    hess[j,i]=hess[i,j]
                    jid+=1

        ener=self.psi_func(F,Cp).dot(GArea)

        return jac,hess,ener,err,rel_err_ener
    def _Loss_Iterator(self,y_para_and_lmda,tol,rel_tol):
        self.IterNum = 0
        y_para, lmda = y_para_and_lmda[:self.y_para_len], y_para_and_lmda[self.y_para_len:]
        y_para, lmda = y_para_and_lmda[:self.y_para_len], y_para_and_lmda[self.y_para_len:]
        jac_E, hess_E, energy, err_ener, rel_err_ener = self._Jac_and_Hess_E(y_para, tol, rel_tol)
        pary_i_cons_j, cons_i = self._jac_other_constraints(y_para)
        jac = append(jac_E + (mat(pary_i_cons_j) * (mat(lmda).T)).A.ravel(), cons_i)
        err_tot = linalg.norm(jac) / sqrt(jac.shape[0])
        print(f"Iter{self.IterNum},", " energy=",   energy," J/mm", ", rel_err=NaN", ", err=", err_ener,
              ", err_tot=", err_tot, ", lmda=", lmda)
        while err_tot > tol and rel_err_ener > rel_tol:
            hess0_ = append(hess_E, pary_i_cons_j, axis=1)
            hess1_ = append(pary_i_cons_j.T, zeros((self.constraints_Num, self.constraints_Num)), axis=1)
            hess = append(hess0_,
                          hess1_,
                          axis=0)

            y_para_and_lmda = (mat(y_para_and_lmda).T - mat(hess).I * (mat(jac).T)).A.ravel()
            y_para, lmda = y_para_and_lmda[:self.y_para_len], y_para_and_lmda[self.y_para_len:]
            self.IterNum += 1
            jac_E, hess_E, energy, err_ener, rel_err_ener = self._Jac_and_Hess_E(y_para, tol, rel_tol)
            pary_i_cons_j, cons_i = self._jac_other_constraints(y_para)
            jac = append(jac_E + (mat(pary_i_cons_j) * (mat(lmda).T)).A.ravel(), cons_i)
            err_tot = linalg.norm(jac) / sqrt(jac.shape[0])
            print(f"Iter{self.IterNum},", " energy=", energy," J/mm", ", rel_err=", rel_err_ener, ", err=", err_ener,
                  ", err_tot=", err_tot, ", lmda=", lmda)
        return y_para,energy
    def _Ener_Iterator(self,y_para,tol,rel_tol):
        self.IterNum=0
        jac, hess, energy, err,rel_err_ener = self._Jac_and_Hess_E(y_para,tol,rel_tol)
        print(f"Iter{self.IterNum},", " energy=", energy," J/mm", ", rel_err=NaN", ", err=", err,
              )

        while err>tol and rel_err_ener>rel_tol:
            y_para=(mat(y_para).T-mat(hess).I*(mat(jac).T)).A.ravel()
            self.IterNum+=1
            jac, hess, energy, err,rel_err_ener = self._Jac_and_Hess_E(y_para,tol,rel_tol)
            print(f"Iter{self.IterNum},", " energy=", energy," J/mm", ", rel_err=", rel_err_ener, ", err=", err,
                  )
        return y_para,energy
    def Solve_ByNewton(self,tol=0.01,rel_tol=0.01):

        start=time.time()
        print("\n\nstart iteration...")
        if self._other_constraint_flag:
            y_para,energy=self._Loss_Iterator(append(self.y_initial_param,zeros(self.constraints_Num)),tol,rel_tol)
        else:
            y_para,energy=self._Ener_Iterator(self.y_initial_param,tol,rel_tol)
        end=time.time()
        print("Iteration end...   time cost: ", end - start," s,  \n\n")
        if self._dirichlet_flag:
            y_ = self.y_.ravel()
            y_[self.YparaID_to_YravelID] = y_para
            self.y_ = reshape(y_, self.y_shape)
            y=self.ReturnFunc(self.y_)
        else:
            y=self.ReturnFunc(reshape(y_para,self.y_shape))
        return y,energy