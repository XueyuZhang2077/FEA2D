import meshio
from FEM2D_ver6_Order1.MeshManager import *


def CellArea(node_coord:np.ndarray,cell_nodeID:np.ndarray):
    cell_node_coord=node_coord[cell_nodeID]
    P1 = cell_node_coord[:, 0, :]
    P2 = cell_node_coord[:, 1, :]
    P3 = cell_node_coord[:, 2, :]

    cell_area = (np.abs(np.cross(P3 - P1, P2 - P1)) / 2)
    cell_gaussCoord = np.sum(cell_node_coord,axis=1)/3

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
        cell_node_coord=self.node_coord[self.cell_nodeID]
        x0,x1,x2=cell_node_coord[:,0,:],cell_node_coord[:,1,:],cell_node_coord[:,2,:]

        self.XI=np.linalg.inv(np.append((x1-x0)[:,:,np.newaxis],(x2-x0)[:,:,np.newaxis],axis=2))
        #######################################################
        print("           Local linearMapping completed.")
        #############preparedForselfDefinedIterator#############
        self.nodeI_cellID=[np.where(self.cell_nodeID==i)[0].astype("int32") for i in range(self.node_coord.shape[0])]
        print("           Node_cell_topology completed.")
        print(f"           NodeNum={self.node_coord.shape[0]}, CellNum={self.cell_nodeID.shape[0]}   ")


        ##############data_typeAfterNabla######################
        self.NablaFunc=[self._NablaScalar,self._NablaVec]
        self._nablaI_basis=[self._Nabla_N3_NodeI_Scalar,self._Nabla_N3_NodeI_Vector]



        print("FunctionSpace preparation completed...")
    def set_Cp(self,Cp):
        self.Jp=np.sqrt(np.linalg.det(Cp))
        self.Cp=Cp
    def Gauss2Node(self,f:np.ndarray):
        NodeValue=[]
        if f.ndim>2:
            Y=np.zeros((f.shape[0],4))
            Y[:,0]=f[:,0,0]
            Y[:,1]=f[:,0,1]
            Y[:,2]=f[:,1,0]
            Y[:,3]=f[:,1,1]
            Y=np.mat(Y)
        elif f.ndim==2:
            Y=np.mat(f)
        else:
            Y=np.mat(f).T
        for i in range(self.node_coord.shape[0]):
            cellids=self.nodeI_cellID[i]
            if cellids.shape[0]>2:
                cell_coord=self.cell_coord[cellids]
                X=np.mat(np.c_[np.ones((cell_coord.shape[0],1)),cell_coord])
                b=(X.T*X).I*X.T*(Y[cellids])
                x=np.mat(np.c_[np.array([[1]]),self.node_coord[i:i+1,:]])
                yPred=(x*b).A
            else:
                yPred=np.sum(Y.A[cellids],axis=0)/cellids.shape[0]

            if f.ndim>2:
                NodeValue.append(np.reshape(yPred,(2,2)))
            elif f.ndim==2:
                NodeValue.append(yPred.ravel())
            else:
                NodeValue.append(yPred.ravel()[0])
        return np.array(NodeValue)
    def Node2Gauss(self,f):
        GaussValue=np.sum(f[self.cell_nodeID],axis=1)/3
        return GaussValue
    def _Nabla_N3_NodeI(self,f,I):
        return self._nablaI_basis[f.ndim-1](f,I)

    def _Nabla_N3_NodeI_Vector(self,y,I):
        cellids=self.nodeI_cellID[I]
        cell_node_y=y[self.cell_nodeID[cellids]]
        y0,y1,y2=cell_node_y[:,0],cell_node_y[:,1],cell_node_y[:,2]
        Y=np.append((y1-y0)[:,:,np.newaxis],(y2-y0)[:,:,np.newaxis],axis=2)
        return np.einsum("...ij,...jk->...ik",Y,self.XI[cellids],optimize="greedy")

    def _Nabla_N3_NodeI_Scalar(self,y,I):
        cellids=self.nodeI_cellID[I]
        cell_nodey=y[self.cell_nodeID[cellids]]
        y0,y1,y2=cell_nodey[:,0],cell_nodey[:,1],cell_nodey[:,2]
        Y=np.append((y1-y0)[:,np.newaxis],(y2-y0)[:,np.newaxis],axis=1)
        return np.einsum("...i,...ij->...j",Y,self.XI[cellids],optimize="greedy")


    def Nabla(self,f):
        return self.NablaFunc[f.ndim-1](f)

    def _NablaVec(self,y):
        cell_node_y=y[self.cell_nodeID]
        y0,y1,y2=cell_node_y[:,0],cell_node_y[:,1],cell_node_y[:,2]
        Y=np.append((y1-y0)[:,:,np.newaxis],(y2-y0)[:,:,np.newaxis],axis=2)
        return np.einsum("...ij,...jk->...ik",Y,self.XI,optimize="greedy")
    def _NablaScalar(self,y):
        cell_nodey=y[self.cell_nodeID]
        y0,y1,y2=cell_nodey[:,0],cell_nodey[:,1],cell_nodey[:,2]
        Y=np.append((y1-y0)[:,np.newaxis],(y2-y0)[:,np.newaxis],axis=1)
        return np.einsum("...i,...ij->...j",Y,self.XI,optimize="greedy")

    def SaveFuncs(self,funcs:list,funcNames:list):
        self.functions_dict_pointdata={}
        self.functions_dict_celldata={}
        points=self.node_coord
        cells={"triangle":self.cell_nodeID}
        for i in range(len(funcs)):
            func=funcs[i]
            if func.ndim==1: #scalar func
                if func.shape[0]==self.cell_nodeID.shape[0]:#celldata
                    self.functions_dict_celldata[funcNames[i]]=[func]
                    self.functions_dict_pointdata[funcNames[i]]=self.Gauss2Node(func)
                else:#pointdata
                    self.functions_dict_pointdata[funcNames[i]]=func
            elif func.ndim==2: #vec func
                if func.shape[0]==self.cell_nodeID.shape[0]:#celldata
                    self.functions_dict_celldata[funcNames[i]]=[func]
                    self.functions_dict_pointdata[funcNames[i]] = self.Gauss2Node(func)
                else:#pointdata
                    self.functions_dict_pointdata[funcNames[i]]=func
            else:#"Matrix"
                if func.shape[0]==self.cell_nodeID.shape[0]:#celldata
                    raveledFunc=np.append(func[:,0],
                                                                       func[:,1],
                                                                       axis=1)
                    self.functions_dict_celldata[funcNames[i]]=[raveledFunc]
                    self.functions_dict_pointdata[funcNames[i]] = self.Gauss2Node(raveledFunc)
                else:#pointdata
                    self.functions_dict_pointdata[funcNames[i]]=append(func[:,0],
                                                                       func[:,1],
                                                                       axis=1)
        meshio.write_points_cells(self.name+".vtk",points,cells,self.functions_dict_pointdata,self.functions_dict_celldata)







if __name__ == '__main__':
    pass
