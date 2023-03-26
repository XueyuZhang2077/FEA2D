import meshio
from FEM2D_ver5_Order1.MeshManager import *


def CellArea(node_coord,cell_nodeID):
    cell_node_coord=node_coord[cell_nodeID]
    P1 = cell_node_coord[:, 0, :]
    P2 = cell_node_coord[:, 1, :]
    P3 = cell_node_coord[:, 2, :]

    cell_area = (abs(cross(P3 - P1, P2 - P1)) / 2)
    cell_gaussCoord = sum(cell_node_coord,axis=1)/3

    return cell_gaussCoord, cell_area







class FunctionSpace:
    def __init__(self,mesh:Mesh2D,name="unknownSpace")->None:
        print("Preparing FunctionSpace")
        self.node_coord=mesh.node_coord
        self.cell_nodeID=mesh.cell_nodeID
        self.domain_cellID=mesh.domain_cellID
        self.name=name
        
        #################Gauss(Cell)###############
        self.cell_coord,self.cellArea=CellArea(self.node_coord,self.cell_nodeID)
        self.cell_node_coord=self.node_coord[self.cell_nodeID]
        x0,x1,x2=self.cell_node_coord[:,0,:],self.cell_node_coord[:,1,:],self.cell_node_coord[:,2,:]
        self.X=MatrixFunction(append((x1-x0)[:,:,newaxis],(x2-x0)[:,:,newaxis],axis=2))
        self.XI=self.X.I()
        #######################################################
        print("           Local linearMapping completed.")
        #############preparedForselfDefinedIterator#############
        self.nodeI_cellID=[]
        self.nodeI_node0toI=[]
        for i in range(self.node_coord.shape[0]):
            self.nodeI_cellID.append(where(self.cell_nodeID==i)[0])
            temp=unique(self.cell_nodeID[self.nodeI_cellID[i]].ravel())
            self.nodeI_node0toI.append(temp[temp<i+1])


        self.nodeIJ_cellID=zeros((self.node_coord.shape[0],self.node_coord.shape[0])).tolist()
        self.IntersectN3_IJ_cellID=zeros((self.node_coord.shape[0],self.node_coord.shape[0])).tolist()
        for I in range(self.node_coord.shape[0]):
            for J in self.nodeI_node0toI[I]:
                self.nodeIJ_cellID[I][J]=unique(append(self.nodeI_cellID[I],self.nodeI_cellID[J]))
                self.IntersectN3_IJ_cellID[I][J]=intersect1d(self.nodeI_cellID[I],self.nodeI_cellID[J])
        print("           Node_cell_topology completed.")
        print(f"           NodeNum={self.node_coord.shape[0]}, CellNum={self.cell_nodeID.shape[0]}   ")

        ##############data_typeAfterNabla######################
        self.ReturnFunc=[ScalarFunction,VectorFunction,MatrixFunction]
        self.NablaFunc=[self._NablaScalar,self._NablaVec,self._NablaMatrix]
        self._inter_nablaIJ_basis=[self._Nabla_IntersectN3_IJ_Scalar,self._Nabla_IntersectN3_IJ_Vector]
        self._nablaI_basis=[self._Nabla_N3_NodeI_Scalar,self._Nabla_N3_NodeI_Vector]
    #     self._getGauss_2Order()
    # def _getGauss_2Order(self):
    #     cell_node_coord=self.node_coord[self.cell_nodeID]
    #     p1,p2,p3=cell_node_coord[:,0,:],cell_node_coord[:,1,:],cell_node_coord[:,2,:]
    #     g1=(4*p1+p2+p3)/6
    #     g2=(4*p2+p1+p3)/6
    #     g3=(4*p3+p1+p2)/6
    #     self.cell_g3_coord=concatenate((g1[:,newaxis,:],
    #                                     g2[:,newaxis,:],
    #                                     g3[:,newaxis,:]),axis=1)

    # def Node2G3(self,f):
    #     cell_node_f=f.baseValue[self.cell_nodeID]
    #     f1,f2,f3=cell_node_f[:,0],cell_node_f[:,1],cell_node_f[:,2]
    #     g1=(4*f1+f2+f3)/6
    #     g2=(4*f2+f3+f1)/6
    #     g3=(4*f3+f2+f1)/6
    #     return eval(f.msg+"Function(concatenate((g1[:,newaxis],g2[:,newaxis],g3[:,newaxis]),axis=1),g3base=True)")
        print("FunctionSpace preparation completed...")
    def Gauss2Node(self,f):
        NodeValue=[]
        if f.baseValue.ndim>2:
            Y=zeros((f.baseValue.shape[0],4))
            Y[:,0]=f.baseValue[:,0,0]
            Y[:,1]=f.baseValue[:,0,1]
            Y[:,2]=f.baseValue[:,1,0]
            Y[:,3]=f.baseValue[:,1,1]
            Y=mat(Y)
        elif f.baseValue.ndim==2:
            Y=mat(f.baseValue)
        else:
            Y=mat(f.baseValue).T
        for i in range(self.node_coord.shape[0]):
            cellids=self.nodeI_cellID[i]
            if cellids.shape[0]>2:
                cell_coord=self.cell_coord[cellids]
                X=mat(c_[ones((cell_coord.shape[0],1)),cell_coord])
                b=(X.T*X).I*X.T*(Y[cellids])
                x=mat(c_[array([[1]]),self.node_coord[i:i+1,:]])
                yPred=(x*b).A
            else:
                yPred=sum(Y.A[cellids],axis=0)/cellids.shape[0]

            if f.baseValue.ndim>2:
                NodeValue.append(reshape(yPred,(2,2)))
            elif f.baseValue.ndim==2:
                NodeValue.append(yPred.ravel())
            else:
                NodeValue.append(yPred.ravel()[0])
        return self.ReturnFunc[f.baseValue.ndim-1](array(NodeValue))
    def Node2Gauss(self,f):
        GaussValue=sum(f.baseValue[self.cell_nodeID],axis=1)/3
        return self.ReturnFunc[f.baseValue.ndim-1](GaussValue)
    def _Nabla_N3_NodeI(self,f,I):
        return self._nablaI_basis[f.baseValue.ndim-1](f,I)

    def _Nabla_IntersectN3_IJ(self,f,I,J):
        return self._inter_nablaIJ_basis[f.baseValue.ndim-1](f,I,J)

    def _Nabla_N3_NodeI_Vector(self,y,I):
        cellids=self.nodeI_cellID[I]
        cell_node_y=y.baseValue[self.cell_nodeID[cellids]]
        y0,y1,y2=cell_node_y[:,0],cell_node_y[:,1],cell_node_y[:,2]
        Y=MatrixFunction(append((y1-y0)[:,:,newaxis],(y2-y0)[:,:,newaxis],axis=2))
        return Y.times(self.XI.Get(cellids))

    def _Nabla_N3_NodeI_Scalar(self,y,I):
        cellids=self.nodeI_cellID[I]
        cell_nodey=y.baseValue[self.cell_nodeID[cellids]]
        y0,y1,y2=cell_nodey[:,0],cell_nodey[:,1],cell_nodey[:,2]
        Y=VectorFunction(append((y1-y0)[:,newaxis],(y2-y0)[:,newaxis],axis=1))
        return Y.times(self.XI.Get(cellids))

    def _Nabla_IntersectN3_IJ_Vector(self,y,I,J):
        cellids=self.IntersectN3_IJ_cellID[I][J]
        cell_node_y=y.baseValue[self.cell_nodeID[cellids]]
        y0,y1,y2=cell_node_y[:,0],cell_node_y[:,1],cell_node_y[:,2]
        Y=MatrixFunction(append((y1-y0)[:,:,newaxis],(y2-y0)[:,:,newaxis],axis=2))
        return Y.times(self.XI.Get(cellids))
    def _Nabla_IntersectN3_IJ_Scalar(self,y,I,J):
        cellids=self.IntersectN3_IJ_cellID[I][J]
        cell_nodey=y.baseValue[self.cell_nodeID[cellids]]
        y0,y1,y2=cell_nodey[:,0],cell_nodey[:,1],cell_nodey[:,2]
        Y=VectorFunction(append((y1-y0)[:,newaxis],(y2-y0)[:,newaxis],axis=1))
        return Y.times(self.XI.Get(cellids))
    def Nabla(self,f):
        return self.NablaFunc[f.baseValue.ndim-1](f)

    def _NablaMatrix(self,F):
        cell_node_F=F.baseValue[self.cell_nodeID]
        cell_node_Fi0, cell_node_Fi1= cell_node_F[:,:, :, 0], cell_node_F[:,:, :, 1]
        #Fi0
        y0,y1,y2=cell_node_Fi0[:,0],cell_node_Fi0[:,1],cell_node_Fi0[:,2]
        Y=MatrixFunction(append((y1-y0)[:,:,newaxis],(y2-y0)[:,:,newaxis],axis=2))
        cell_Fi0_parj=Y.times(self.XI)

        #Fi1
        y0,y1,y2=cell_node_Fi1[:,0],cell_node_Fi1[:,1],cell_node_Fi1[:,2]
        Y=MatrixFunction(append((y1-y0)[:,:,newaxis],(y2-y0)[:,:,newaxis],axis=2))
        cell_Fi1_parj=Y.times(self.XI)
        return cell_Fi0_parj,cell_Fi1_parj
    def _NablaVec(self,y):
        cell_node_y=y.baseValue[self.cell_nodeID]
        y0,y1,y2=cell_node_y[:,0],cell_node_y[:,1],cell_node_y[:,2]
        Y=MatrixFunction(append((y1-y0)[:,:,newaxis],(y2-y0)[:,:,newaxis],axis=2))
        return Y.times(self.XI)
    def _NablaScalar(self,y):
        cell_nodey=y.baseValue[self.cell_nodeID]
        y0,y1,y2=cell_nodey[:,0],cell_nodey[:,1],cell_nodey[:,2]
        Y=VectorFunction(append((y1-y0)[:,newaxis],(y2-y0)[:,newaxis],axis=1))
        return Y.times(self.XI)

    def SaveFuncs(self,funcs:list,funcNames:list):
        self.functions_dict_pointdata={}
        self.functions_dict_celldata={}
        points=self.node_coord
        cells={"triangle":self.cell_nodeID}
        for i in range(len(funcs)):
            func=funcs[i]
            if func.msg=="Scalar":
                if func.baseValue.shape[0]==self.cell_nodeID.shape[0]:#celldata
                    self.functions_dict_celldata[funcNames[i]]=[func.baseValue]
                    self.functions_dict_pointdata[funcNames[i]]=self.Gauss2Node(func).baseValue
                else:#pointdata
                    self.functions_dict_pointdata[funcNames[i]]=func.baseValue
            elif func.msg=="Vector":
                if func.baseValue.shape[0]==self.cell_nodeID.shape[0]:#celldata
                    self.functions_dict_celldata[funcNames[i]]=[func.baseValue]
                    self.functions_dict_pointdata[funcNames[i]] = self.Gauss2Node(func).baseValue
                else:#pointdata
                    self.functions_dict_pointdata[funcNames[i]]=func.baseValue
            else:#"Matrix"
                if func.baseValue.shape[0]==self.cell_nodeID.shape[0]:#celldata
                    raveledFunc=VectorFunction(append(func.baseValue[:,0],
                                                                       func.baseValue[:,1],
                                                                       axis=1))
                    self.functions_dict_celldata[funcNames[i]]=[raveledFunc.baseValue]
                    self.functions_dict_pointdata[funcNames[i]] = self.Gauss2Node(raveledFunc).baseValue
                else:#pointdata
                    self.functions_dict_pointdata[funcNames[i]]=append(func.baseValue[:,0],
                                                                       func.baseValue[:,1],
                                                                       axis=1)
        meshio.write_points_cells(self.name+".vtk",points,cells,self.functions_dict_pointdata,self.functions_dict_celldata)


class ScalarFunction:
    def __init__(self,baseValue):
        self.baseValue=baseValue
        self.msg="Scalar"
    def Get(self,baseID):
        return ScalarFunction(self.baseValue[baseID])
    def multiply(self,const):
        return ScalarFunction(self.baseValue*const)
    def exp(self,const):
        return ScalarFunction(self.baseValue**const)

    def dot(self,scalarfunc):
        return sum(self.baseValue*scalarfunc.baseValue)
    def plus(self,scalarFunc):
        return ScalarFunction(self.baseValue+scalarFunc.baseValue)
    def minus(self,scalarFunc):
        return ScalarFunction(self.baseValue-scalarFunc.baseValue)
    def times(self,M_or_v):
        return multiplyer["Scalar"][M_or_v.msg](self, M_or_v)



class VectorFunction:
    def __init__(self,baseValue):
        self.baseValue=baseValue
        self.msg="Vector"
    def Get(self,baseID):
        return VectorFunction(self.baseValue[baseID])
    def dot(self,v):
        return ScalarFunction(sum(self.baseValue*v.baseValue,axis=1))

    def minus(self,v):
        return VectorFunction(self.baseValue-v.baseValue)
    def plus(self,v):
        return VectorFunction(self.baseValue+v.baseValue)

    def times(self,M):
        x1, x2= self.baseValue[:, 0:1], self.baseValue[ :, 1:2]
        y_1, y_2= M.baseValue[ :, :, 0], M.baseValue[ :, :, 1]
        return VectorFunction(x1 * y_1 + x2 * y_2 )


class MatrixFunction:
    def __init__(self,baseValue):
        self.baseValue=baseValue
        self.msg="Matrix"
    def dot(self,M):
        return ScalarFunction(sum(sum(self.baseValue*M.baseValue,axis=2),axis=1))
    def Get(self,baseID):
        return MatrixFunction(self.baseValue[baseID])
    def det(self):
        cell_base_value = self.baseValue
        x11 = cell_base_value[ :, 0, 0]
        x12 = cell_base_value[ :, 0, 1]
        x21 = cell_base_value[ :, 1, 0]
        x22 = cell_base_value[ :, 1, 1]
        return ScalarFunction(x11*x22-x12*x21)
    def I(self):
        cell_base_value = self.baseValue
        x11 = cell_base_value[ :, 0:1, 0:1]
        x12 = cell_base_value[ :, 0:1, 1:2]
        x21 = cell_base_value[ :, 1:2, 0:1]
        x22 = cell_base_value[ :, 1:2, 1:2]
        J = x11 * x22 - x12 * x21
        _1 = append(x22, -x12, axis=2)
        _2 = append(-x21, x11, axis=2)
        return MatrixFunction(append(_1, _2, axis=1) / J)
    def T(self):

        cell_base_value =self.baseValue
        cell_base_col1 = cell_base_value[ :, 0, :, newaxis]
        cell_base_col2 = cell_base_value[ :, 1, :, newaxis]
        cell_base_value = append(cell_base_col1, cell_base_col2, axis=2)
        return MatrixFunction(cell_base_value)
    def plus(self,M):
        return MatrixFunction(self.baseValue+M.baseValue)
    def minus(self,M):
        return MatrixFunction(self.baseValue - M.baseValue)
    def times(self,M_or_v):
        return multiplyer["Matrix"][M_or_v.msg](self,M_or_v)
    def multiply(self,const):
        return MatrixFunction(self.baseValue*const)
    def exp(self,const):
        return MatrixFunction(self.baseValue**const)
    def QU_decomp(self):

        # Cp=self.T().times(self).baseValue
        # x11,x12,x22=Cp[:,0:1,0:1],Cp[:,0:1,1:2],Cp[:,1:2,1:2]
        # sqrt_=sqrt(x11**2 - 2*x11*x22 + 4*x12**2 + x22**2)
        # rho1=sqrt(1+(-2*x12/(sqrt_+x11-x22+1e-10))**2)
        # rho2=sqrt(1+(-2*x12/(-sqrt_+x11-x22+1e-10))**2)
        # U_11=4*x12**2*sqrt(sqrt_/2 + x11/2 + x22/2)/(rho2**2*(-sqrt_ + x11 - x22)**2) + 4*x12**2*sqrt(-sqrt_/2 + x11/2 + x22/2)/(rho1**2*(sqrt_ + x11 - x22)**2)
        # U_12=-2*x12*sqrt(sqrt_/2 + x11/2 + x22/2)/(rho2**2*(-sqrt_ + x11 - x22)) - 2*x12*sqrt(-sqrt_/2 + x11/2 + x22/2)/(rho1**2*(sqrt_ + x11 - x22))
        # U_21=U_12
        # U_22=sqrt(sqrt_/2 + x11/2 + x22/2)/rho2**2 + sqrt(-sqrt_/2 + x11/2 + x22/2)/rho1**2
        # U_1=append(U_11,U_12,axis=2)
        # U_2=append(U_21,U_22,axis=2)
        # U=MatrixFunction(append(U_1,U_2,axis=1))
        # Q=self.times(U.I())
        Q,U=[],[]
        for i in range(self.baseValue.shape[0]):
            F=self.baseValue[i]
            C=mat(F).T*mat(F)
            eigvals,eigvecs=linalg.eig(C)
            e1,e2=eigvecs[:,0],eigvecs[:,1]
            l1,l2=eigvals[0],eigvals[1]
            U.append(sqrt(l1)*e1*e1.T+sqrt(l2)*e2*e2.T)
            Q.append(F*U[i].I)
        Q=MatrixFunction(array(Q))
        U=MatrixFunction(array(U))
        return Q,U
    def trace(self):

        return ScalarFunction(self.baseValue[:,0,0]+self.baseValue[:,1,1])



#FuncMultiplier

def Mat_time_Mat(M1:MatrixFunction,M2:MatrixFunction)->MatrixFunction:

    x1, x2 = M1.baseValue[ :, :, 0:1], M1.baseValue[ :, :, 1:2]
    y1, y2 = M2.baseValue[ :, 0:1, :], M2.baseValue[ :, 1:2, :]
    return MatrixFunction(x1 * y1 + x2 * y2)

def Mat_time_Vec(M:MatrixFunction,v:VectorFunction)->VectorFunction:
    x_1, x_2 = M.baseValue[ :, :, 0], M.baseValue[ :, :, 1]
    y1, y2 = v.baseValue[ :, 0:1], v.baseValue[ :, 1:2]
    return VectorFunction(x_1 * y1 + x_2 * y2)
def Mat_time_Sca(M:MatrixFunction,s:ScalarFunction)->MatrixFunction:
    return MatrixFunction(M.baseValue*s.baseValue[:,newaxis,newaxis])

def Sca_time_Mat(s:ScalarFunction,M:MatrixFunction)->MatrixFunction:
    return MatrixFunction(M.baseValue*s.baseValue[:,newaxis,newaxis])
def Sca_time_Vec(s:ScalarFunction,v:VectorFunction)->VectorFunction:
    return VectorFunction(v.baseValue*s.baseValue[:,newaxis])
def Sca_time_Sca(s1:ScalarFunction,s2:ScalarFunction)->ScalarFunction:
    return ScalarFunction(s1.baseValue*s2.baseValue)


multiplyer={"Matrix":{"Matrix":Mat_time_Mat,"Vector":Mat_time_Vec,"Scalar":Mat_time_Sca},
            "Scalar":{"Matrix":Sca_time_Mat,"Vector":Sca_time_Vec,"Scalar":Sca_time_Sca}}



def Generate_S(V:FunctionSpace,domain_s)->MatrixFunction:
    cell_row_col=zeros((V.cell_nodeID.shape[0],2,2))
    for i in range(len(V.domain_cellID)):
        cell_row_col[V.domain_cellID[i],0,0]=1
        cell_row_col[V.domain_cellID[i],0,1]=0
        cell_row_col[V.domain_cellID[i],1,0]=domain_s[i]
        cell_row_col[V.domain_cellID[i],1,1]=1
    return MatrixFunction(cell_row_col)
def Generate_Q(V:FunctionSpace,domain_s)->MatrixFunction:
    cell_row_col=zeros((V.cell_nodeID.shape[0],2,2))
    for i in range(len(V.domain_cellID)):
        cell_row_col[V.domain_cellID[i],0,0]=(4-domain_s[i]**2)/(4+domain_s[i]**2)
        cell_row_col[V.domain_cellID[i],0,1]=4*domain_s[i]/(4+domain_s[i]**2)
        cell_row_col[V.domain_cellID[i],1,0]=-4*domain_s[i]/(4+domain_s[i]**2)
        cell_row_col[V.domain_cellID[i],1,1]=4-domain_s[i]**2/(4+domain_s[i]**2)
    return MatrixFunction(cell_row_col)


class MinimaProblem:
    def __init__(self,y_initial:VectorFunction,V:FunctionSpace,psi_func,Cp):
        self.y_initial_param=y_initial.baseValue.ravel()
        self.y_shape=y_initial.baseValue.shape
        self.V=V
        self.cellArea=V.cellArea
        self.psi=psi_func
        self.Cp=Cp
        self._set_consttraint_flag=False
    def Energy(self,y_para):
        y=VectorFunction(reshape(y_para,self.y_shape))
        F=self.V.Nabla(y)
        cellPsi=self.psi(F,self.Cp).baseValue
        return sum(cellPsi*self.cellArea)
    def set_constraint(self,constraint_func):
        self._set_consttraint_flag=True
        self.constraint_func=constraint_func
    def _callback(self,y_para):
        print("err:",self.Energy(y_para))

    def Solve(self,tol=0.001):
        if self._set_consttraint_flag:
            conditions={"type":"eq","fun":self.constraint_func}
            res=minimize(self.Energy,self.y_initial_param,method="SLSQP",constraints=conditions,tol=tol,callback=self._callback)
        else:
            res = minimize(self.Energy, self.y_initial_param, method="SLSQP", tol=tol,callback=self._callback)
        y_para=res.x
        y=VectorFunction(reshape(y_para,self.y_shape))
        print("Energy=", res.fun, ", y=", y.baseValue, "succeed?:", res.success, "Reason for failure:", res.message)
        return y





if __name__ == '__main__':
    G=21
    nu=0.26
    mu=2*G*nu/(1-2*nu)
    h0=0.2



    a=GenerateMeshes2D(array([[-0.6,-0.6],
                          [0.6,-0.6],
                          [0.6,-h0],
                          [-h0,0],
                          [0.6,0.2],
                          [0.6,0.6],
                          [-0.6,0.6]]),array([0.5,0.5,0.5,0.02,0.5,0.5,0.5]),)


    b=GenerateMesh2D(array([[-h0,0],
                          [0,0],
                          [0.6,0.15],
                          [0.6,0.2]]),array([0.02,0.02,0.5,0.5]),order=1)
    c=GenerateMesh2D(array([[-h0,0],
                          [0,0],
                          [0.6,-0.15],
                          [0.6,-h0]]),array([0.02,0.02,0.5,0.5]),order=1)
    d=GenerateMesh2D(array([
                          [0,0],
                          [0.6,-0.15],
                          [0.6,0.15]]),array([0.02,0.5,0.5]),order=1)





    a.Add([b,c,d])

    a.Plot(PointIDs=False,CellIDs=True)



    V=FunctionSpace(a)


    y_Trial=VectorFunction(V.node_coord)

    S=Generate_S(V,array([0,-0.5,0.5,0]))
    Cp=S.T().times(S)
    def psi(F:MatrixFunction,Cp:MatrixFunction):
        C=F.T().times(F)
        CpI=Cp.I()
        Ee=(C.minus(Cp)).multiply(0.5)
        EeT=Ee.T()

        CpIdotEe_exp2=CpI.dot(Ee).exp(2)
        CpIEeT=CpI.times(EeT)
        EeCpIT=CpIEeT.T()

        CpIEeTdotEeCpIT = CpIEeT.dot(EeCpIT)

        return CpIdotEe_exp2.multiply(mu/2).plus(CpIEeTdotEeCpIT.multiply(G))

    d1=linalg.norm(V.node_coord-array([0,0]),axis=1)
    idO=where((d1==min(d1)))[0][0]
    d2=linalg.norm(V.node_coord-array([-h0,0]),axis=1)
    idh0 = where((d2 == min(d2)))[0][0]
    def constrain(y_para):
        return array([y_para[idO*2]-0.0,
                      y_para[idO*2+1]-0.0,
                      y_para[idh0*2+1]-0.0])
    problem=MinimaProblem(y_Trial,V,psi,Cp)
    y=problem.Solve(tol=0.001)
    u=y.minus(VectorFunction(V.node_coord))

    F=V.NablaVec(y)

    QeQpUe=F.times(S.I())
    QeQp,Ue=QeQpUe.QU_decomp()
    print(QeQp.baseValue)
    phi=ScalarFunction(arccos((QeQp.trace().baseValue)/2)*sign(QeQp.baseValue[:,0,1]))
    phiInNode=V.Gauss2Node(phi)
    partial_phi=V.NablaScalar(phiInNode)

    V.SaveFuncs([u,F,phi,partial_phi],["u","F","phi","partial_phi"])

