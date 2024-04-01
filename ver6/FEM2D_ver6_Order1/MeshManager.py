import numpy as np
import matplotlib.pyplot as plt
import pygmsh





def Geom(geom: pygmsh.occ.Geometry, vertex_coord: np.ndarray, lcar: np.ndarray, domain_vertexID: list):
    '''
    Meshの生成用のobejec 「geom」に，すべての点（未変形部と変形した領域に分割するための点）の座標と，これら点の付近においてどれぐらい大きいなmeshを使ってdomainを細かく分割するのに関する情報，および各domainの頂点のindexの情報をつけるための関数．
    :param geom: Meshの生成用のobejec
    :param vertex_coord: np.array([点0の座標,点1の座標，点2の座標,...])
    :param lcar:   lcarは点付近のmeshsizeを表す． 例：np.array([0.2,0.1,0.1,...])は，点0付近ではメッシュ化に0.2サイズの要素が使われていることと，点1付近ではメッシュ化に0.1サイズの要素が使われていることを意味している．
    :param domain_vertexID: [domain0の頂点が対応する点のindices,
                             domain1の頂点が対応する点のindices,
                             ...]
    :return: None (obj geomの属性はこの関数内部において処理されたので, geomを新たに返す必要はありません)
    '''
    len_vertex_coord = len(vertex_coord)
    len_domain_vertexID=len(domain_vertexID)
    pL = []
    '''「vertex_coord」のたくさんの点の座標をgeomの属性に導入する'''
    for i in range(len_vertex_coord):
        pL.append(geom.add_point(vertex_coord[i], lcar[i]))

    line_vertexID=[]
    '''domainを分割するために使用される線分の両端の点のindicesを導入して，
    line_vertexID=np.array([[line0始点が対応するvertexのindex,line0終点が対応するvertexのindex],
                            [line1始点が対応するvertexのindex,line1終点が対応するvertexのindex],
                            ...])'''
    for i in range(len_domain_vertexID):
        region=domain_vertexID[i]
        for j in range(1,len(region)):
            line_vertexID.append([region[j-1],region[j]])
        line_vertexID.append([region[len(region) - 1], region[0]])
    line_vertexID= np.unique(np.sort(np.array(line_vertexID) ,axis=1) ,axis=0)

    '''線をつなげて，線のループ(loop of Lines, 略称:lL)を作る'''
    lL=[]
    for line in line_vertexID:
        lL.append(geom.add_line(pL[line[0]],pL[line[1]]))

    '''線のループに囲まれた領域から，surfaces(ss)が得られる'''
    ss = []
    for i in range(len_domain_vertexID):
        region = domain_vertexID[i]

        loop = []
        for j in range(1, len(region)):
            edge_v0,edge_v1=region[j-1],region[j]
            v0_inlineI=np.where(line_vertexID==edge_v0)[0]
            v1_inlineI=np.where(line_vertexID==edge_v1)[0]
            lineID=np.intersect1d(v0_inlineI,v1_inlineI)[0]
            if line_vertexID[lineID,0]==edge_v0:
                loop.append(lL[lineID])
            else:
                loop.append(-lL[lineID])

        edge_v0, edge_v1 = region[len(region) - 1], region[0]
        v0_inlineI = np.where(line_vertexID == edge_v0)[0]
        v1_inlineI = np.where(line_vertexID == edge_v1)[0]
        lineID = np.intersect1d(v0_inlineI, v1_inlineI)[0]
        if line_vertexID[lineID, 0] == edge_v0:
            loop.append(lL[lineID])
        else:
            loop.append(-lL[lineID])
        ll = geom.add_curve_loop(loop)
        '''lLをgeomの属性にし，返したobjをさらに，add_plane_surface()という関数を利用して，surfaceを表すobj.を作成する．作成されたsurfaceをlist　ss に入れる'''
        ss.append(geom.add_plane_surface(ll))

        '''surface i に'domain_i' という名前をつける'''
        geom.add_physical(ss[i],f"domain_{i}")
        # geom.boolean_union(ss)

    return


class Mesh2D:
    def __init__(self, node_coord:np.ndarray, cell_nodeID:np.ndarray, domain_cellID:list,order=1):
        '''
        Mesh object
        :param node_coord: np.array([[nodeid=0　のノードの　基準配置での横座標，nodeid=0　のノードの　基準配置での縦座標],
                                     [nodeid=1　のノードの　基準配置での横座標，nodeid=1　のノードの　基準配置での縦座標],
                                     ...])
        :param cell_nodeID:  np.array([[cellid=0　のセルの　頂点0　が対応するnodeid,　cellid=0　のセルの　頂点1　が対応するnodeid,　cellid=0　のセルの　頂点2　が対応するnodeid],
                                       [cellid=1　のセルの　頂点0　が対応するnodeid,　cellid=1　のセルの　頂点1　が対応するnodeid,　cellid=1　のセルの　頂点2　が対応するnodeid],
                                       ...])
        :param domain_cellID:  [domain0に属するセルらのindices,
                                domain1に属するセルらのindices,
                                ...]
        :param order=1: セル内部の近似用関数が一次関数である
        '''
        self.order=order
        self.node_coord = node_coord[:,:2] #2D有限要素法のコードであるため，[:,:2]を使って，xyの座標のみを考慮し，z方向の座標を無視する．
        self.cell_nodeID = cell_nodeID
        self.domain_cellID = domain_cellID

        cell_node_coord = self.node_coord[self.cell_nodeID]
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
        #各セルのgaussian積分点の座標を計算する
        self.cell_coord = (P1 + P2 + P3) / 3
        #各セルの面積を計算する
        self.cellArea = (np.abs(np.cross(P3 - P1, P2 - P1)) / 2)

    def savetxt(self, mesh_name: str):
        '''
        Meshの情報（node_coord,cell_nodeIDと,domain_cellID）を保存する．
        :param mesh_name: 保存するファイルの名前
        :return:
        '''
        np.savetxt(mesh_name + "__node_coord.txt", self.node_coord)
        np.savetxt(mesh_name + "__cell_nodeID.txt", self.cell_nodeID)
        np.savetxt(mesh_name + "__domainNum.txt", np.array([len(self.domain_cellID)]))
        for i in range(len(self.domain_cellID)):
            np.savetxt(mesh_name + f"__domain{i}_cell.txt", self.domain_cellID[i])

    def ChangeToOrder2(self):
        '''
        Meshの基底関数を区分的な二次関数に変換するために，三つの頂点を持つ三角形要素を，六つの頂点を持つ三角形要素へ変換する必要があります．
        :return: None
        '''
        if self.order == 2:
            raise Exception("order==2")
        else:
            cell_nodeID_coord = self.node_coord[self.cell_nodeID]
            cell_nodeID = self.cell_nodeID.tolist()
            for i in range(len(self.cell_nodeID)):

                node1_coord, node2_coord, node3_coord = cell_nodeID_coord[i, 0], cell_nodeID_coord[i, 1], \
                                                        cell_nodeID_coord[i, 2]
                #セルごとに，新しいnodeの座標を生成する
                node4_coord = (node1_coord + node2_coord) / 2
                node5_coord = (node2_coord + node3_coord) / 2
                node6_coord = (node3_coord + node1_coord) / 2
                new_node_coord = [node4_coord, node5_coord, node6_coord]

                for j in range(3):
                    new_coord = new_node_coord[j]
                    #新たに生成されたnodeと本来そもそも存在するnodeとの距離を計算する
                    #距離が0の点は，重複な点として認識され，node_coordに追加されない．
                    d = np.linalg.norm(self.node_coord - new_coord, axis=1)
                    if any(d < 0.0000001):
                        cell_nodeID[i].append(np.where(d < 0.0000001)[0][0])
                    else:
                        # 重複していない新しいnodeのみを,node_coordnに追加する
                        self.node_coord = np.append(self.node_coord, np.array([new_coord]), axis=0)
                        cell_nodeID[i].append(len(self.node_coord) - 1)
            self.cell_nodeID = np.array(cell_nodeID)
            self.order = 2
    def Plot(self, Points=True, PointIDs=True, Cells=True, CellIDs=True):
        '''
        Meshの様子を大まかに見るための関数
        :param Points: True->nodeをプロットする　　False->nodeをプロットしない
        :param PointIDs: True->nodeのindexをプロットする False ->nodeのindexをプロットしない
        :param Cells:    True->cellをプロットする　False->cellをプロットしない
        :param CellIDs:  True->cellのindexをプロットする  False->cellのindexをプロットしない
        :return:
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set(aspect=1)
        x, y = self.node_coord[:, 0], self.node_coord[:, 1]
        cell_nodeID_num = len(self.cell_nodeID.T)
        if Points:
            ax.scatter(x, y, c='tab:blue', alpha=0.33)
        if PointIDs:
            i = 0
            for point in self.node_coord:
                ax.text(point[0], point[1], f"p{i}")
                i += 1
        if Cells:
            for cell in self.cell_nodeID:
                points = self.node_coord[cell[:3]]

                px = points[:, 0].ravel()
                py = points[:, 1].ravel()
                px = np.append(px, px[0])
                py = np.append(py, py[0])
                ax.plot(px, py, lw=0.66, color="tab:blue")
        if CellIDs:
            i = 0
            for p in self.cell_coord:
                ax.text(p[0], p[1], f"c{i}")
                i += 1
        plt.legend()
        plt.show()

def GenerateMeshes2D(vertex_coord: np.ndarray, vertex_mshSize: np.ndarray, domain_vertexID: list) :
    print("Generating Meshes")
    with pygmsh.occ.Geometry() as geom:
        Geom(geom, vertex_coord, vertex_mshSize, domain_vertexID)
        msh = geom.generate_mesh()


    mesh=Mesh2D(msh.points,msh.cells_dict["triangle"].astype("int32"),domain_cellID=[ msh.cell_sets[f"domain_{i}"][1]   for i in range(len(domain_vertexID))  ])

    print("Meshes completed")
    return mesh


def GenerateRegularMesh2D(h_range,v_range,h_num,v_num):
    '''
    規格化した長方形二次元メッシュを作成する
    :param h_range: horizontal range 横方向に沿った幅
    :param v_range: vertical range   縦方向に沿った幅
    :param h_num:   horizontal cell num 横方向に沿ったcellの数
    :param v_num: 　vertical cell num縦方向に沿ったcellの数
    :return:
    '''
    print("Generating Meshes")
    x=np.linspace(h_range[0],h_range[1],h_num)
    y=np.linspace(v_range[0],v_range[1],v_num)

    x,y=np.meshgrid(x,y)
    node_coord=np.append(x.ravel()[:,np.newaxis],y.ravel()[:,np.newaxis],axis=1)

    cellID_nodeID=[]
    for j in range(v_num-1):
        for i in range(h_num-1):
            cellID_nodeID.append([i+j*h_num,i+1+j*h_num,i+h_num+j*h_num])
            cellID_nodeID.append([i+1+j*h_num,i+h_num+j*h_num,i+h_num+1+j*h_num])
    cellID_nodeID=np.array(cellID_nodeID)

    domain_cellID=[ np.arange(0,cellID_nodeID.shape[0],1,dtype=int)   ]
    mesh=Mesh2D(node_coord,cellID_nodeID,domain_cellID=domain_cellID)
    print("Meshes completed")
    return mesh







if __name__ == '__main__':
    mesh=GenerateMeshes2D(np.array([[0.0,0.0],
                                    [1.0,0.0],
                                    [0.0,1.0],
                                    [1.0,1.0]]),
                          np.array([0.2,0.2,0.2,0.2]),
                          [np.array([0,1,2]),
                           np.array([1,2,3])])
    mesh.Plot()