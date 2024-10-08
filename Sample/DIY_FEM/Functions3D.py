import meshio
from DIY_FEM.MeshManager3D import *


def CellVol(cell_node_coord):
    P1 = cell_node_coord[:, 0, :]
    P2 = cell_node_coord[:, 1, :]
    P3 = cell_node_coord[:, 2, :]
    P4 = cell_node_coord[:, 3, :]

    cellVol = np.abs(np.sum((P4-P1)*np.cross(P3-P1,P2-P1)/6,axis=1))
    cell_gaussCoord = np.sum(cell_node_coord,axis=1)/4

    return cell_gaussCoord, cellVol




class FunctionSpace3D:
    def __init__(self,mesh:Mesh3D,name="unknownSpace",PrepareXI=True,printInfo=True)->None:
        if printInfo:
            print("Preparing FunctionSpace")
        self.node_coord=mesh.node_coord
        self.cell_nodeID=mesh.cell_nodeID
        self.domain_cellID=mesh.domain_cellID
        self.name=name
        self.dim=3

        #################Gauss(Cell)###############

        cell_node_coord=self.node_coord[self.cell_nodeID]
        self.cell_node_coord=cell_node_coord
        self.cell_coord, self.cellVol = CellVol(cell_node_coord)
        x0,x1,x2,ones=cell_node_coord[:,:,0],cell_node_coord[:,:,1],cell_node_coord[:,:,2],np.ones(cell_node_coord[:,:,2].shape,dtype=float)
        if PrepareXI:
            self.XI=np.linalg.inv(np.concatenate((x0[:,:,np.newaxis],x1[:,:,np.newaxis],x2[:,:,np.newaxis],ones[:,:,np.newaxis]),axis=2))
            self.Na=np.einsum("ij,...jk->...ik",np.array([[1,0,0,0],
                                                          [0,1,0,0],
                                                          [0,0,1,0]]),self.XI)
        #######################################################
        if printInfo:
            print("           Local linearMapping completed.")
        #############preparedForselfDefinedIterator#############
        self.nodeI_cellID=[np.where(self.cell_nodeID==i)[0].astype("int32") for i in range(self.node_coord.shape[0])]
        if printInfo:
            print("           Node_cell_topology completed.")
            print(f"           NodeNum={self.node_coord.shape[0]}, CellNum={self.cell_nodeID.shape[0]}   ")


        ##############data_typeAfterNabla######################
        self.NablaFunc=[self._NablaScalar,self._NablaVec]

        if printInfo:
            print("FunctionSpace preparation completed...")




    def match_nodecoord2cellid(self,nodecoord):
        cell_node_coord=self.cell_node_coord


        xid_cellid=[]
        temp=cell_node_coord[:,:1,:].copy()
        for x in nodecoord:
            temp[:,0,:]=x

            vol_diff=( CellVol(np.concatenate( (cell_node_coord[:,0:1,:],cell_node_coord[:,1:2,:],cell_node_coord[:,2:3,:],temp) ,axis=1))[1]+
                       CellVol(np.concatenate( (cell_node_coord[:,0:1,:],cell_node_coord[:,1:2,:],cell_node_coord[:,3:4,:],temp) ,axis=1))[1]+
                       CellVol(np.concatenate( (cell_node_coord[:,0:1,:],cell_node_coord[:,2:3,:],cell_node_coord[:,3:4,:],temp) ,axis=1))[1]+
                       CellVol(np.concatenate( (cell_node_coord[:,1:2,:],cell_node_coord[:,2:3,:],cell_node_coord[:,3:4,:],temp) ,axis=1))[1])-self.cellVol

            xid_cellid.append( np.where(np.abs(vol_diff)<1e-7)[0][0]  )
        return np.array(xid_cellid)

    def map(self,xs,ys):
        cell_node_coord=self.node_coord[self.cell_nodeID]
        cell_node_newcoord=ys[self.cell_nodeID]
        xid_cellid=self.match_nodecoord2cellid(xs)

        cell_Fi_j=self.Nabla(ys)

        Y=[]
        for i in range(xs.shape[0]):
            x=xs[i]
            O=cell_node_coord[xid_cellid[i],0,:]
            new_O=cell_node_newcoord[xid_cellid[i],0,:]

            F=cell_Fi_j[xid_cellid[i]]

            y=new_O+F@(x-O)
            Y.append(y)
        return np.array(Y)

    def Gauss2Node(self,f:np.ndarray):
        NodeValue=[]
        if f.ndim==3:
            Y=np.zeros((f.shape[0],9))
            Y[:,0],Y[:,1],Y[:,2]=f[:,0,0],f[:,0,1],f[:,0,2]
            Y[:,3],Y[:,4],Y[:,5]=f[:,1,0],f[:,1,1],f[:,1,2]
            Y[:,6],Y[:,7],Y[:,8]=f[:,2,0],f[:,2,1],f[:,2,2]
            Y=np.mat(Y)

        elif f.ndim==2:
            Y=np.mat(f)
        elif f.ndim==1:
            Y=np.mat(f).T
        else:
            raise Exception("only support f.ndim=1,2,3")
        for i in range(self.node_coord.shape[0]):
            cellids=self.nodeI_cellID[i]
            cell_coord = self.cell_coord[cellids]
            O=np.sum(cell_coord,axis=0)/cellids.shape[0]
            v=(cell_coord-O)/(np.linalg.norm(cell_coord-O,axis=1)[:,np.newaxis])
            determinant=np.linalg.det((v.T)@v)

            if determinant>1e8:

                    X=np.mat(np.c_[np.ones((cell_coord.shape[0],1)),cell_coord])
                    b=(X.T*X).I*X.T*(Y[cellids])
                    x=np.mat(np.c_[np.array([[1]]),self.node_coord[i:i+1,:]])
                    yPred=(x*b).A
            else:
                yPred = np.sum(Y.A[cellids], axis=0) / cellids.shape[0]

            if f.ndim>2:
                NodeValue.append(np.reshape(yPred,(3,3)))
            elif f.ndim==2:
                NodeValue.append(yPred.ravel())
            else:
                NodeValue.append(yPred.ravel()[0])
        return np.array(NodeValue)
    def Node2Gauss(self,f):
        GaussValue=np.sum(f[self.cell_nodeID],axis=1)/4
        return GaussValue


    def Nabla(self,f):
        return self.NablaFunc[f.ndim-1](f)

    def _NablaVec(self,y):
        return np.einsum("...ij,...jk->...ki",self.Na,y[self.cell_nodeID],optimize="greedy")

    def _NablaScalar(self,y):
        return np.einsum("...ij,...j->...i",self.Na,y[self.cell_nodeID],optimize="greedy")

    def SaveFuncs(self,funcs:list,funcNames:list):
        self.functions_dict_pointdata={}
        self.functions_dict_celldata={}
        points=self.node_coord
        cells={"tetra":self.cell_nodeID}
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
                    raveledFunc=np.concatenate((func[:,0],
                                                                       func[:,1],
                                                                        func[:,2]),
                                                                       axis=1)
                    self.functions_dict_celldata[funcNames[i]]=[raveledFunc]
                    self.functions_dict_pointdata[funcNames[i]] = self.Gauss2Node(raveledFunc)
                else:#pointdata
                    self.functions_dict_pointdata[funcNames[i]]=np.concatenate((func[:,0],
                                                                       func[:,1],
                                                                       func[:,2]),
                                                                       axis=1)
        meshio.write_points_cells(self.name+".vtk",points,cells,self.functions_dict_pointdata,self.functions_dict_celldata)







if __name__ == '__main__':
    mesh=GenerateMeshes3D(np.array([[0,0.0,0],
                                    [1,0,0],
                                    [1,1,0],
                                    [0,1,0],
                                    [0,0,1]]),
                          np.array([1,1,1,1,1]),
                          [np.array([0,1,4]),
                           np.array([0,2,4]),
                           np.array([0,3,4]),
                           np.array([1,2,4]),
                           np.array([2,3,4]),
                           np.array([0,1,2]),
                           np.array([0,2,3])],
                          [np.array([0,1,3,5]),
                           np.array([1,2,4,6])])
    # mesh.Plot()
    V=FunctionSpace3D(mesh)
    x=V.node_coord
    y=V.node_coord+np.array([1,1,1])
    u=y-x
    F=V.Nabla(y[:,0])
    print(V.map(np.array([[0.4,0.4,0.3],
                          [0.3,0.3,0.2]]),y))
    V.SaveFuncs([u,F],["u","F"])