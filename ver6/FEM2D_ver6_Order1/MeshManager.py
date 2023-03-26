import numpy as np
import matplotlib.pyplot as plt
import pygmsh





def Geom(geom: pygmsh.occ.Geometry, vertices: np.ndarray, lcar: np.ndarray, regions: list):
    len_vertices = len(vertices)
    len_regions=len(regions)
    pL = []
    for i in range(len_vertices):
        pL.append(geom.add_point(vertices[i], lcar[i]))

    line_vertexID=[]
    for i in range(len_regions):
        region=regions[i]
        for j in range(1,len(region)):
            line_vertexID.append([region[j-1],region[j]])
        line_vertexID.append([region[len(region) - 1], region[0]])
    line_vertexID= np.unique(np.sort(np.array(line_vertexID) ,axis=1) ,axis=0)

    lL=[]
    for line in line_vertexID:
        lL.append(geom.add_line(pL[line[0]],pL[line[1]]))

    ss = []
    for i in range(len_regions):
        region = regions[i]

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
        ss.append(geom.add_plane_surface(ll))
        # geom.boolean_union(ss)

    return


class Mesh2D:
    def __init__(self, node_coord:np.ndarray, cell_nodeID:np.ndarray, domain_cellID:list,order=1):
        self.order=order
        self.node_coord = node_coord[:,:2]
        self.cell_nodeID = cell_nodeID
        self.domain_cellID = domain_cellID

        cell_node_coord=self.node_coord[self.cell_nodeID]
        P1 = cell_node_coord[:, 0, :]
        P2 = cell_node_coord[:, 1, :]
        P3 = cell_node_coord[:, 2, :]
        self.cell_coord = (P1 + P2 + P3) / 3
        self.cellArea = (np.abs(np.cross(P3 - P1, P2 - P1)) / 2)
        #############

    def savetxt(self, mesh_name: str):
        np.savetxt(mesh_name + "__node_coord.txt", self.node_coord)
        np.savetxt(mesh_name + "__cell_nodeID.txt", self.cell_nodeID)
        np.savetxt(mesh_name + "__domainNum.txt", np.array([len(self.domain_cellID)]))
        for i in range(len(self.domain_cellID)):
            np.savetxt(mesh_name + f"__domain{i}_cell.txt", self.domain_cellID[i])

    def ChangeToOrder2(self):
        if self.order == 2:
            raise Exception("order==2")
        else:
            cell_nodeID_coord = self.node_coord[self.cell_nodeID]
            cell_nodeID = self.cell_nodeID.tolist()
            for i in range(len(self.cell_nodeID)):

                node1_coord, node2_coord, node3_coord = cell_nodeID_coord[i, 0], cell_nodeID_coord[i, 1], \
                                                        cell_nodeID_coord[i, 2]

                node4_coord = (node1_coord + node2_coord) / 2
                node5_coord = (node2_coord + node3_coord) / 2
                node6_coord = (node3_coord + node1_coord) / 2
                new_node_coord = [node4_coord, node5_coord, node6_coord]

                for j in range(3):
                    new_coord = new_node_coord[j]
                    d = np.linalg.norm(self.node_coord - new_coord, axis=1)
                    if any(d < 0.0000001):
                        cell_nodeID[i].append(np.where(d < 0.0000001)[0][0])
                    else:
                        self.node_coord = np.append(self.node_coord, np.array([new_coord]), axis=0)
                        cell_nodeID[i].append(len(self.node_coord) - 1)
            self.cell_nodeID = np.array(cell_nodeID)
            self.order = 2
    def Plot(self, Points=True, PointIDs=True, Cells=True, CellIDs=True, Regions=False):
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
        if Cells:  # plot一下cell的轮廓线而不plot其面色， 面色用于标记region
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
        if Regions:
            i = 0
            for region in self._regions:
                vertices = self._vertices[region]
                ax.fill(vertices[:, 0], vertices[:, 1], alpha=0.33, label='region:' + f'{i}')
                i += 1
        plt.legend()
        plt.show()
def LargeMeshCellID_InAsubMesh(cell_node_coordFromSubMesh,cell_coordFromLargeMesh):
    P1, P2, P3 = cell_node_coordFromSubMesh[:, 0], cell_node_coordFromSubMesh[:, 1], cell_node_coordFromSubMesh[:, 2]
    cellArea_SubMesh = (np.abs(np.cross(P3 - P1, P2 - P1)) / 2)

    res=np.array([])
    C=cell_coordFromLargeMesh
    for i in range(cell_node_coordFromSubMesh.shape[0]):
        P1,P2,P3=cell_node_coordFromSubMesh[i, 0],cell_node_coordFromSubMesh[i, 1],cell_node_coordFromSubMesh[i, 2]
        Area_CPiPj=np.abs(np.cross(C - P1, C - P2)) / 2+np.abs(np.cross(C - P2, C - P3)) / 2+np.abs(np.cross(C - P3, C - P1)) / 2

        err=Area_CPiPj-cellArea_SubMesh[i]

        res=np.append(res,np.where(np.abs(err)<1e-7)[0])

    return np.int_(np.unique(np.array(res)))

def Get_domainCell(cell_coord_LargeMesh,vertex_coord,domain_vertexID):
    domain_cellID=[]
    for i in range(len(domain_vertexID)):
        print(f"          Assigning cells indices to domain {i}...")
        with pygmsh.occ.Geometry() as geom:
            # Define the vertices of the polygon
            points=vertex_coord[domain_vertexID[i]]
            # Create the polygon
            polygon = geom.add_polygon(points, 100)
            # Generate the mesh
            mesh = geom.generate_mesh()
            cell_node_coord_SubMesh=mesh.points[:,:2][mesh.cells_dict["triangle"]]
            domain_cellID.append(LargeMeshCellID_InAsubMesh(cell_node_coord_SubMesh,cell_coord_LargeMesh))
    return domain_cellID
def GenerateMeshes2D(vertex_coord: np.ndarray, vertex_mshSize: np.ndarray, domain_vertexID: list) :
    print("Generating Meshes")
    with pygmsh.occ.Geometry() as geom:
        Geom(geom, vertex_coord, vertex_mshSize, domain_vertexID)
        msh = geom.generate_mesh()
    mesh=Mesh2D(msh.points,msh.cells_dict["triangle"].astype("int32"),domain_cellID=None)
    mesh.domain_cellID=Get_domainCell(mesh.cell_coord, vertex_coord, domain_vertexID)
    print("Meshes completed")
    return mesh











