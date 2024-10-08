import meshio
import numpy as np
from DIY_FEM.MeshManager2D import *


def CellArea(node_coord:np.ndarray,cell_nodeID:np.ndarray):
    '''
    メッシュの内部の各セルの面積を計算するための関数
    :param node_coord: np.array([[nodeid=0　のノードの　基準配置での横座標，nodeid=0　のノードの　基準配置での縦座標],
                                 [nodeid=1　のノードの　基準配置での横座標，nodeid=1　のノードの　基準配置での縦座標],
                                 ...])
    :param cell_nodeID:np.array([[cellid=0　のセルの　頂点0　が対応するnodeid,　cellid=0　のセルの　頂点1　が対応するnodeid,　cellid=0　のセルの　頂点2　が対応するnodeid],
                                 [cellid=1　のセルの　頂点0　が対応するnodeid,　cellid=1　のセルの　頂点1　が対応するnodeid,　cellid=1　のセルの　頂点2　が対応するnodeid],
                                 ...])
    :return:np.array([cellid=0のセルの面積,cellid=1のセルの面積,...])
    '''


    cell_node_coord=node_coord[cell_nodeID]
    '''
    "node_coord[cell_nodeID]"のステップで，以下の
    ようなものが得られる．     　　　
    np.array([ [[cellid=0　のセルの　頂点0　基準配置での横座標,　cellid=0　のセルの　頂点0　基準配置での縦座標],
                [cellid=0　のセルの　頂点1　基準配置での横座標,　cellid=0　のセルの　頂点1　基準配置での縦座標],
                [cellid=0　のセルの　頂点2　基準配置での横座標,　cellid=0　のセルの　頂点2　基準配置での縦座標]],
                   
               [[cellid=1　のセルの　頂点0　基準配置での横座標,　cellid=1　のセルの　頂点0　基準配置での縦座標],
                [cellid=1　のセルの　頂点1　基準配置での横座標,　cellid=1　のセルの　頂点1　基準配置での縦座標],
                [cellid=1　のセルの　頂点2　基準配置での横座標,　cellid=1　のセルの　頂点2　基準配置での縦座標]], 
               
               [[cellid=2　のセルの　頂点0　基準配置での横座標,　cellid=2　のセルの　頂点0　基準配置での縦座標],
                [cellid=2　のセルの　頂点1　基準配置での横座標,　cellid=2　のセルの　頂点1　基準配置での縦座標],
                [cellid=2　のセルの　頂点2　基準配置での横座標,　cellid=2　のセルの　頂点2　基準配置での縦座標]], 
                
                ...
                
               ])
    '''


    P1 = cell_node_coord[:, 0, :]
    P2 = cell_node_coord[:, 1, :]
    P3 = cell_node_coord[:, 2, :]
    '''
    P1=np.array([[cellid=0 のセルの頂点0　基準配置での横座標, cellid=0 のセルの頂点0　基準配置での縦座標],
                 [cellid=1 のセルの頂点0　基準配置での横座標, cellid=1 のセルの頂点0　基準配置での縦座標],
                 ...])
    P2=np.array([[cellid=0 のセルの頂点1　基準配置での横座標, cellid=0 のセルの頂点1　基準配置での縦座標],
                 [cellid=1 のセルの頂点1　基準配置での横座標, cellid=1 のセルの頂点1　基準配置での縦座標],
                 ...])
    P3=np.array([[cellid=0 のセルの頂点2　基準配置での横座標, cellid=0 のセルの頂点2　基準配置での縦座標],
                 [cellid=1 のセルの頂点2　基準配置での横座標, cellid=1 のセルの頂点2　基準配置での縦座標],
                 ...])           
    
    '''



    cell_area = (np.abs(np.cross(P3 - P1, P2 - P1)) / 2) #各セルの面積を計算する
    cell_gaussCoord = np.sum(cell_node_coord,axis=1)/3   #各gaussian積分点の座標(一次三角形要素の場合では，中心の座標となる)を計算する

    return cell_gaussCoord, cell_area


def QU_decompose(F):
    '''
    極分解するための関数
    :param F: 変形勾配（の場）
        np.array([ [[id=0　の点での　F_00,　id=0　の点での　F_01],
                    [id=0  の点での　F_10,　id=0　の点での　F_11]],

                   [[id=1　の点での　F_00,　id=1　の点での　F_01],
                    [id=1  の点での　F_10,　id=1　の点での　F_11]],

                   ...
               ])

    :return:
    '''
    C=np.einsum("...ji,...jk->...ik",F,F) #各点において, C_ik=F_ji F_jkを計算する（）
    lmda,u=np.linalg.eig(C)#各点での右コーシーグリーンテンソルのeigen val.とeigen vec.を計算する

    #後はバタチャリヤの本の極分解の手法に沿ったアルゴリズムです．
    mu=np.sqrt(lmda)
    mu_u=np.einsum("...i,...ji->...ji",mu,u)
    U=np.einsum("...ji,...ki->...jk",mu_u,u)
    Q=np.einsum("...ij,...jk->...ik",F,np.linalg.inv(U))
    return Q,U

class FunctionSpace2D:
    def __init__(self,mesh:Mesh2D,name="unknownSpace",PrepareXI=True,PrepareSurfIDs=True):
        '''
        関数空間 {y(x)}=V　のy(x)を
        y(x)=y(x;node0におけるy値)+y(x;node1におけるy値)+y(x;node2におけるy値)+...
        のように分解して得られる関数からなる近似用の関数空間V^h

        :param mesh: object mesh, MeshManagerのファイルに定義されている,
        :param name: .vtkファイルに保存するときのファイル名 -> {name}.vtk
        :param PrepareXI:
                (cell... vert1 y0 は　...番目のセル　の　一番目の頂点　における現配置の横座標，cell... vert1 y0 は　...番目のセル　の　一番目の頂点　における現配置の縦座標を表す)
                [[cell... vert0 y0, cell... vert0 y1],       [[ 1, cell... vert0 x0, cell... vert0 x1],          [[cell... c_0 , cell... c_1],
                 [cell... vert1 y0, cell... vert1 y1],  =     [ 1, cell... vert1 x0, cell... vert1 x1],   \dot    [cell... F_00, cell... F_10],
                 [cell... vert2 y0, cell... vert2 y1]]        [ 1, cell... vert2 x0, cell... vert2 x1]]           [cell... F_01, cell... F_11]]
                を
                  cell0_Y=  cell0_X  \dot   cell0_cF
                  cell1_Y=  cell1_X  \dot   cell1_cF
                cell..._Y=cell..._X  \dot cell..._cF
                で記述すると，
                cell..._cFは  cell..._cF = cell..._X^inverse \dot cell..._Y
                によって算出できます．

                そして   PrepareXI=True　の場合では，
                事前に   cell..._X^inverse　を属性にして，これからのcell..._cFの計算において，    cell..._X^inverseに関する重複な計算を避けることができ，計算コストを改善できる．

        '''
        print("Preparing FunctionSpace")
        self.node_coord=mesh.node_coord
        self.cell_nodeID=mesh.cell_nodeID
        self.domain_cellID=mesh.domain_cellID
        self.name=name
        self.dim=2
        '''
        domain_cellID=[domain0に属するたくさんのセルのindices(例,2,3,5,6),
                       domain1に属するたくさんのセルのindices(例,1,4,7,8,9,10),
                       ....]
        domainの意味：例えばdomain0をキンク変形した領域，domain1を未変形部とすることができます．
                                
        '''


        #################Gauss(Cell)###############

        self.cell_coord,self.cellArea=CellArea(self.node_coord,self.cell_nodeID)#cell_coordは，各セルのgaussian積分点の座標を表し，cellAreaは各セルの面積を表す．
        cell_node_coord=self.node_coord[self.cell_nodeID]#CellArea()関数のcell_node_coordに関する説明に参照
        x0,x1,ones=cell_node_coord[:,:,0],cell_node_coord[:,:,1],np.ones(cell_node_coord[:,:,1].shape,dtype=float)
        if PrepareXI:
            self.XI=np.linalg.inv(np.concatenate((x0[:,:,np.newaxis],x1[:,:,np.newaxis],ones[:,:,np.newaxis]),axis=2))
            self.Na=np.einsum("ij,...jk->...ik",np.array([[1,0,0],
                                                          [0,1,0]]),self.XI)

        ############FreeSurfaces###############################
        if PrepareSurfIDs:
            cell_surf_nodeID=np.concatenate((self.cell_nodeID[:,[0,1]][:,np.newaxis],
                                            self.cell_nodeID[:,[0,2]][:,np.newaxis],
                                            self.cell_nodeID[:,[1,2]][:,np.newaxis]),axis=1)
            cell_surfSingleID=np.concatenate((self.cell_nodeID[:,[2]],
                                              self.cell_nodeID[:,[1]],
                                              self.cell_nodeID[:,[0]]),axis=1)

            cell_surf_nodeID=cell_surf_nodeID.reshape((cell_surf_nodeID.shape[0]*cell_surf_nodeID.shape[1],cell_surf_nodeID.shape[2]))
            cell_surfSingleID=cell_surfSingleID.ravel()

            cell_surf_nodeID=np.sort(cell_surf_nodeID,axis=1)
            unique,indices,counts=np.unique(cell_surf_nodeID,axis=0,return_index=True,return_counts=True)

            cell_surfSingleID=cell_surfSingleID[indices]
            surfID=(counts==1)
            self.surf_vtxNID=unique[surfID]
            self.surf_outNID=cell_surfSingleID[surfID]

            surf_node_coord=self.node_coord[self.surf_vtxNID]
            surf_outnodecoord=self.node_coord[self.surf_outNID]
            self.surf_coord=np.sum(surf_node_coord,axis=1)/2
            self.surf_dA=np.einsum("ij,...j->...i",np.array([[0,-1],
                                                             [1,0]]),surf_node_coord[:,0]-surf_node_coord[:,1])
            self.surf_dA=self.surf_dA*np.sign( np.sum(self.surf_dA*(surf_node_coord[:,0]-surf_outnodecoord)   ,axis=1) )[:,np.newaxis]
            self.surf_cellID=np.int_(indices[counts==1]/3)
        #######################################################



        #######################################################
        print("           Local linearMapping completed.")
        #############preparedForselfDefinedIterator#############
        self.nodeI_cellID=[np.where(self.cell_nodeID==i)[0].astype("int32") for i in range(self.node_coord.shape[0])]
        '''
        nodeI_cellID=[node0を共通点とするセルのindices(例,0,1,2,3),
                      node1を共通点とするセルのindices(例,2,3,4,5,6)]
        '''
        print("           Node_cell_topology completed.")
        print(f"           NodeNum={self.node_coord.shape[0]}, CellNum={self.cell_nodeID.shape[0]}   ")


        ##############data_typeAfterNabla######################
        self.NablaFunc=[self._NablaScalar,self._NablaVec]
        '''
        _NablaScalar: [node0 scalar, node1 scalar,...] -> [cell0 scalarの勾配, cell1 scalarの勾配,...]
        _NablaVec: 例[node0 y, node1 y,...] -> [cell0 変形勾配, cell1 変形勾配, ...] 
        '''

        print("FunctionSpace preparation completed...")




    def match_nodecoord2cellid(self,nodecoord):
        '''
        ある点の基準配置での座標が与えられた時，
        この点が属するセルのindexを返す関数
        :param nodecoord: [点0の基準配置での座標，点1の基準配置での座標,...]
        :return:
        nodeI_cellID=[node0を共通点とするセルのindices(例,0,1,2,3),
                      node1を共通点とするセルのindices(例,2,3,4,5,6)]
        '''
        cell_node_coord=self.node_coord[self.cell_nodeID]
        eps=np.array([[0,1],
                      [-1,0]])

        xid_cellid=[]
        for x in nodecoord:
            cell_edgevec_coord=cell_node_coord-x
            area_diff=(np.abs(np.einsum("ij,...i,...j->...",eps,cell_edgevec_coord[:,0,:],cell_edgevec_coord[:,1,:]))/2+
                       np.abs(np.einsum("ij,...i,...j->...",eps,cell_edgevec_coord[:,1,:],cell_edgevec_coord[:,2,:]))/2+
                       np.abs(np.einsum("ij,...i,...j->...",eps,cell_edgevec_coord[:,2,:],cell_edgevec_coord[:,0,:]))/2)-self.cellArea
            xid_cellid.append( np.where(np.abs(area_diff)<1e-7)[0][0]  )
            '''
            ...番目のセルは,その三つの頂点p1,p2,p3はx点と一緒に，xp1p2,xp2p3,xp3p1という三つの三角形を作ることができるます．
            こういう三つの三角形の面積の和が...番目のセルの面積に等しいであることは，
            点ｘが...番目のセルに属していることとは等価である．
            '''
        return np.array(xid_cellid)

    def map(self,xs,ys):
        '''
        メッシュの各nodeにおけるyの値が与えられた時，
        与えられた点xにおけるyの値をinterpolateするための関数
        (点iはnode iではないです．node iはmesh自分自身の属性です．点iは，使用者に指定された点で，この点における情報を知りたいという気持ちは使用者が持っています．)

        :param xs: np.array([[点0 x0, 点0 x1],
                             [点1 x0, 点1 x1],
                             ...])
        :param ys:　np.array([[node0 y0, node0 y1],
                              [node1 y0, node1 y1],
                              ...])
        :return:    np.array([[点0 y0, 点0 y1],
                              [点1 y0, 点1 y1],
                              ...])
        '''
        cell_node_coord=self.node_coord[self.cell_nodeID]
        cell_node_newcoord=ys[self.cell_nodeID]
        xid_cellid=self.match_nodecoord2cellid(xs)#点xが属するセルのidを記録する

        cell_Fi_j=self.Nabla(ys)

        Y=[]

        for i in range(xs.shape[0]):
            x=xs[i]
            O=cell_node_coord[xid_cellid[i],0,:]#点xが属するセルのある頂点の基準配置での座標をOとする
            new_O=cell_node_newcoord[xid_cellid[i],0,:]#この頂点の現配置での座標をnew_Oとする
            F=cell_Fi_j[xid_cellid[i]]#このセルでの変形勾配
            y=new_O+F@(x-O)#現配置での座標を計算する
            Y.append(y)
        return np.array(Y)

    def FitFromIrregularPointData(self,point_coords,pointData,fit_radius=5):
        '''
        各点での座標と各点での関数値[y(点0),y(点1),y(点2),...]が与えられたが，
        セルに関する情報(cellID_nodeID)がまだ定義されていない場合，
        本メッシュを使ってフィッティングを行って，[y(node0),y(node1),y(node2),...]のデータの列に変換する
        :param point_coords: [x(点0),x(点1),x(点2),...]
        :param pointData: [y(点0),y(点1),y(点2),...]
        :param fit_radius: y(node_i)の値を得るために，node_iとの距離<fit_radiusのpointらにおける関数値を使ってフィッティングを行います．
        :return:
        '''
        data=[]
        for x in self.node_coord:
            dx=point_coords-x
            ids=np.where(np.linalg.norm(dx,axis=1)<fit_radius)[0]
            if ids.shape[0]>2:
                '''node_iとの距離<fit_radiusのpointらにおける関数値に対して，
                線形回帰を行って，node_inにおける関数値をinterpolateする
                '''
                X=np.c_[np.ones(ids.shape[0]),dx[ids]]
                XTX=X.T@X
                if np.linalg.det(XTX)>1e-8:
                    y=pointData[ids]
                    k=np.linalg.inv(X.T@X)@X.T@y
                    data+=[k[0]]
                else:
                    data+=[np.nan]#もしフィッティングの品質が悪いだったら，フィッティングの結果をnp.nanで記述する
            else:
                data+=[np.nan]
        return np.array(data)

    def Gauss2Node(self,f:np.ndarray):
        '''
        Gaussian積分点での関数値から，
        nodeでの関数値を内挿する
        :param f:　[f(gauss0), f(gauss1), f(gauss2), ...] (一次三角形要素の場合では，gaussian積分点のindexはセルのindexと同じです)
        :return:
        '''
        NodeValue=[]
        if f.ndim>2:
            #[cellのid,変形勾配の行id,変形勾配の列id]のように，.ndim=3
            Y=np.zeros((f.shape[0],4))
            Y[:,0]=f[:,0,0]
            Y[:,1]=f[:,0,1]
            Y[:,2]=f[:,1,0]
            Y[:,3]=f[:,1,1]
            Y=np.mat(Y) #cellNum * 4の形となるように変換する
        elif f.ndim==2:
            # [cellのid,ベクトルの成分id]のように，.ndim=2
            Y=np.mat(f)
        else:
            # [cellのidにおけるscalar値]のように，.ndim=1
            Y=np.mat(f).T
        for i in range(self.node_coord.shape[0]):
            cellids=self.nodeI_cellID[i]
            #各nodeの周りのgaussian積分点のindices(このnodeを共通点とするセルらのindicesと同じ)
            if cellids.shape[0]>2:
                #周りのセル数>=3の場合，線形回帰法を利用して周りのgaussian積分点における関数値から，nodeでの関数値を内挿することが可能である．
                cell_coord=self.cell_coord[cellids]
                X=np.mat(np.c_[np.ones((cell_coord.shape[0],1)),cell_coord])

                b=(X.T*X).I*X.T*(Y[cellids])
                x=np.mat(np.c_[np.array([[1]]),self.node_coord[i:i+1,:]])
                yPred=(x*b).A
            else:
                # 周りのセル数<3の場合，簡単に周りのgaussian積分点の関数の和を周りのセルの数で割ることで nodeにおける関数値を推測する
                yPred=np.sum(Y.A[cellids],axis=0)/cellids.shape[0]

            if f.ndim>2:
                # [cellのid,変形勾配の行id,変形勾配の列id]のように，.ndim=3
                NodeValue.append(np.reshape(yPred,(2,2)))#shape=(4,)のyPredを，shape=( 2 , 2 ) の形となるように変換する
            elif f.ndim==2:
                # [cellのid,ベクトルの成分id]のように，.ndim=2
                NodeValue.append(yPred.ravel())
            else:
                # [cellのidにおけるscalar値]のように，.ndim=1
                NodeValue.append(yPred.ravel()[0])
        return np.array(NodeValue)
    def Node2Gauss(self,f):
        '''
        ノードでの関数値から，
        Gaussian積分点での関数値を内挿する
        :param f: [f(node0), f(node1), f(node2), ...]
        :return: [f(gauss0), f(gauss1), f(gauss2), ...]
        '''
        GaussValue=np.sum(f[self.cell_nodeID],axis=1)/3
        '''
        f[self.cell_nodeID]=
    np.array([ [[cellid=0　のセルの　頂点0　f値,　cellid=0　のセルの　頂点0　f値],
                [cellid=0　のセルの　頂点1　f値,　cellid=0　のセルの　頂点1　f値],
                [cellid=0　のセルの　頂点2　f値,　cellid=0　のセルの　頂点2　f値]],
                   
               [[cellid=1　のセルの　頂点0　f値,　cellid=1　のセルの　頂点0　f値],
                [cellid=1　のセルの　頂点1　f値,　cellid=1　のセルの　頂点1　f値],
                [cellid=1　のセルの　頂点2　f値,　cellid=1　のセルの　頂点2　f値]], 
               
               [[cellid=2　のセルの　頂点0　f値,　cellid=2　のセルの　頂点0　f値],
                [cellid=2　のセルの　頂点1　f値,　cellid=2　のセルの　頂点1　f値],
                [cellid=2　のセルの　頂点2　f値,　cellid=2　のセルの　頂点2　f値]], 
        
        np.sum(f[self.cell_nodeID],axis=1)は，セルごとに，その内部の頂点012のf値を合算して3で割ることを意味する．
        '''

        return GaussValue

    def Nabla(self,f):
        '''
        すべての点における勾配を算出する
        :param f: [f(node0), f(node1), f(node2), ....] 各ノードにおける関数値
        :return:  [Nabla f @ cell0, Nabla f @ cell1, ...]
        self.NablaFunc=[self._NablaScalar,self._NablaVec]であるため，
        fはベクトルばn.dimが1である場合，scalar場として認識されself.NablaFunc[0]で指定されるscalar fieldの勾配を計算するための関数を使う．
        fはベクトルばn.dimが2である場合，vector場として認識されself.NablaFunc[1]で指定されるvector fieldの勾配を計算するための関数を使う．
        '''
        return self.NablaFunc[f.ndim-1](f)

    def _NablaVec(self,y):
        return np.einsum("...ij,...jk->...ki",self.Na,y[self.cell_nodeID],optimize="greedy")

    def _NablaScalar(self,y):
        return np.einsum("...ij,...j->...i",self.Na,y[self.cell_nodeID],optimize="greedy")

    def PlotSurf(self):
        for nodes in self.node_coord[self.surf_vtxNID]:
            plt.plot(nodes[:,0],nodes[:,1])

        plt.quiver(self.surf_coord[:,0],self.surf_coord[:,1],self.surf_dA[:,0],self.surf_dA[:,1])
        plt.show()
    def SaveFuncs(self,funcs:list,funcNames:list):
        '''
        各テンソル場とメッシュを {self.name}.vtkのファイルに保存する
        :param funcs: [テンソル場0,テンソル場1,...]
        テンソル場i=np.array([テンソル場iの値 @ node0, テンソル場iの値 @ node1,...])
        or
        テンソル場i=np.array([テンソル場iの値 @ cell0, テンソル場iの値 @ cell1,...])
        :param funcNames: [テンソル場0の名前,テンソル場1の名前,...]
        :return:
        '''
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
                    self.functions_dict_pointdata[funcNames[i]]=np.append(func[:,0],
                                                                          func[:,1],
                                                                           axis=1)
        meshio.write_points_cells(self.name+".vtk",points,cells,self.functions_dict_pointdata,self.functions_dict_celldata)





if __name__ == '__main__':
    msh=GenerateMeshes2D(np.array([[0.0,0.0],
                                   [1.0,0.0],
                                   [1.0,1.0],
                                   [0.0,1.0]]),np.array([0.1,0.1,0.1,0.1]),[np.array([0,1,2,3])])
    V=FunctionSpace2D(msh)
    V.PlotSurf()

