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

        self.ReturnFunc=self.V.ReturnFunc[TrialFunc.baseValue.ndim-1]

        self.independentNode_ids=arange(0,self.V.node_coord.shape[0],dtype=int)

        self.d=1e-7
        self.dy=identity(self.y_para_len)*self.d

        self.y_=TrialFunc.baseValue.copy()


        self._other_constraint_flag=False
        self._dirichlet_flag=False
        self._dirichlet_u0Free_flag=False
    def set_dirichlet(self,nodeIDs,u):
        if self._dirichlet_flag:
            raise Exception("dirichlet boundary condition already set")
        self.y_para_len=self.y_para_len-nodeIDs.shape[0]*2
        self.dy=identity(self.y_para_len)*self.d

        self.dirichletNode_ids=nodeIDs
        self.dirichletNode_u=u
        self.independentNode_ids=setdiff1d(self.independentNode_ids,nodeIDs)

        self.y_[self.dirichletNode_ids]=self.V.node_coord[self.dirichletNode_ids]+u

        self._dirichlet_flag=True

        self.y_initial_param=self.y_[self.independentNode_ids].ravel()
        self.where_independentNode_ids__equal__nodeJ=[]

        for i in range(len(self.nodeI_node0toI)):
            self.nodeI_node0toI[i]=int_(intersect1d(self.nodeI_node0toI[i],self.independentNode_ids))
            temp=[]
            for nodeJ in self.nodeI_node0toI[i]:
                temp.append(where(self.independentNode_ids == nodeJ)[0])
            self.where_independentNode_ids__equal__nodeJ.append(temp)






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
        for i in range(self.y_para_len):
            pary_i_cons_j[i,:]=(self.constraint_func(y_para+self.dy[i])-constraint0)/self.d
        return pary_i_cons_j,constraint0

    def _Jac_and_Hess_E(self,y_para,tol,rel_tol):
        y_shape=self.y_shape
        PK2_func=self.PK2_func
        ReturnFunc=self.ReturnFunc
        GArea, y_para_len = ScalarFunction(self.gauss_area), self.y_para_len
        test_funcs, nodeI_gaussID= self.TestFuncs, self.nodeI_gaussID
        nodeI_node0toI, Cp = self.nodeI_node0toI, self.Cp

        d, dy, self_y_ = self.d, self.dy, self.y_
        IntersectN3_IJ_gaussID, V_Nabla_IntersectN3_IJ = self.V.IntersectN3_IJ_cellID, self.V._Nabla_IntersectN3_IJ
        where_independentNode_ids__equal__nodeJ=self.where_independentNode_ids__equal__nodeJ


        if self._dirichlet_flag:
            independentNode_ids=self.independentNode_ids
            self.y_[independentNode_ids]=reshape(y_para,(independentNode_ids.shape[0],2))
            y=ReturnFunc(self.y_)
            F=self.V.Nabla(y)
            PK1=F.times(PK2_func(F,self.Cp))

            jac,hess=zeros(y_para_len),zeros((y_para_len,y_para_len))
            for i in range(y_para_len):
                nodeI,coord_id=independentNode_ids[int(i/2)],i%2
                localI_gaussids=nodeI_gaussID[nodeI]
                jac[i]=test_funcs[nodeI*2+coord_id].Get(localI_gaussids).dot(PK1.Get(localI_gaussids)).dot(GArea.Get(localI_gaussids))
            err,flag=linalg.norm(jac)/sqrt(jac.shape[0]),True
            if self.IterNum==0:
                rel_err_ener=1
            else:
                rel_err_ener=abs(err-self.err)/self.err
                if rel_err_ener < rel_tol:
                    flag=False
            self.err=err

            if err>tol and flag:
                for i in range(y_para_len):
                    nodeI,coord_id=independentNode_ids[int(i/2)],i%2
                    nodeJ_arr,jid=nodeI_node0toI[nodeI],0

                    for nodeJ in nodeJ_arr:

                        localIJ_gaussids=IntersectN3_IJ_gaussID[nodeI][nodeJ]
                        localIJ_Cp=Cp.Get(localIJ_gaussids)
                        localIJ_err=PK1.Get(localIJ_gaussids).dot(test_funcs[nodeI*2+coord_id].Get(localIJ_gaussids)).dot(GArea.Get(localIJ_gaussids))

                        j = where_independentNode_ids__equal__nodeJ[nodeI][jid]*2

                        y_= self_y_.copy()
                        y_[nodeJ,0]=y_[nodeJ,0]+d
                        y_p_dyj=ReturnFunc(y_)
                        localIJ_F=V_Nabla_IntersectN3_IJ(y_p_dyj,nodeI,nodeJ)
                        next_jac=localIJ_F.times(PK2_func(localIJ_F,localIJ_Cp)).dot(test_funcs[nodeI*2+coord_id].Get(localIJ_gaussids)).dot(GArea.Get(localIJ_gaussids))
                        hess[i,j]=(next_jac - localIJ_err)/d
                        hess[j,i]=hess[i,j]

                        y_=self_y_.copy()
                        y_[nodeJ,1]=y_[nodeJ,1]+d
                        y_p_dyj = ReturnFunc(y_)
                        localIJ_F = V_Nabla_IntersectN3_IJ(y_p_dyj, I=nodeI, J=nodeJ)
                        next_jac = localIJ_F.times(PK2_func(localIJ_F, localIJ_Cp)).dot(
                            test_funcs[nodeI*2+coord_id].Get(localIJ_gaussids)).dot(GArea.Get(localIJ_gaussids))
                        hess[i, j+1] = (next_jac - localIJ_err) / d
                        hess[j+1, i] = hess[i, j+1]

                        jid+=1
            ener=self.psi_func(F,Cp).dot(GArea)
        else:
            y = self.ReturnFunc(reshape(y_para,y_shape))
            F = self.V.Nabla(y)
            PK1=F.times(PK2_func(F,self.Cp))

            jac,hess=zeros(y_para_len),zeros((y_para_len,y_para_len))
            for i in range(y_para_len):
                nodeI = int(i / y_shape[1])
                localI_gaussids = nodeI_gaussID[nodeI]
                jac[i] = test_funcs[i].Get(localI_gaussids).dot(PK1.Get(localI_gaussids)).dot(
                    GArea.Get(localI_gaussids))
            err,flag = linalg.norm(jac) / sqrt(jac.shape[0]),True
            if self.IterNum == 0:
                rel_err_ener = 1
            else:
                rel_err_ener = abs(err - self.err) / self.err
                if rel_err_ener < rel_tol:
                    flag = False
            self.err = err
            if err > tol and flag:
                for i in range(y_para_len):
                    nodeI = int(i / 2)
                    nodeJ_arr = nodeI_node0toI[nodeI]
                    for nodeJ in nodeJ_arr:
                        localIJ_gaussids = IntersectN3_IJ_gaussID[nodeI][nodeJ]
                        localIJ_Cp = Cp.Get(localIJ_gaussids)
                        localIJ_err = PK1.Get(localIJ_gaussids).dot(test_funcs[i].Get(localIJ_gaussids)).dot(
                            GArea.Get(localIJ_gaussids))

                        j = 2 * nodeJ
                        y_p_dyj = ReturnFunc(reshape(y_para + dy[j], y_shape))
                        localIJ_F = V_Nabla_IntersectN3_IJ(y_p_dyj, I=nodeI, J=nodeJ)
                        next_jac = localIJ_F.times(PK2_func(localIJ_F, localIJ_Cp)).dot(
                            test_funcs[i].Get(localIJ_gaussids)).dot(GArea.Get(localIJ_gaussids))
                        hess[i, j] = (next_jac - localIJ_err) / d
                        hess[j, i] = hess[i, j]

                        j = 2 * nodeJ + 1
                        y_p_dyj = ReturnFunc(reshape(y_para + dy[j], y_shape))
                        localIJ_F = V_Nabla_IntersectN3_IJ(y_p_dyj, I=nodeI, J=nodeJ)
                        next_jac = localIJ_F.times(PK2_func(localIJ_F, localIJ_Cp)).dot(
                            test_funcs[i].Get(localIJ_gaussids)).dot(GArea.Get(localIJ_gaussids))
                        hess[i, j] = (next_jac - localIJ_err) / d
                        hess[j, i] = hess[i, j]
            ener = self.psi_func(F, Cp).dot(GArea)
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
            self.y_[self.independentNode_ids]=reshape(y_para,(self.independentNode_ids.shape[0],2))
            y=self.ReturnFunc(self.y_)
        else:
            y=self.ReturnFunc(reshape(y_para,self.y_shape))
        return y,energy