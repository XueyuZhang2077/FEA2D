import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pygmsh





def Geom(geom: pygmsh.occ.Geometry, vertices_coords: np.ndarray, lcar: np.ndarray, surfaceID_verticesID:list,domainID_surfaceID: list):
    len_vertices = len(vertices_coords)
    len_surfaces=len(surfaceID_verticesID)
    len_domains=len(domainID_surfaceID)
    pL = []
    for i in range(len_vertices):
        pL.append(geom.add_point(vertices_coords[i], lcar[i]))

    line_vertexID=[]
    for i in range(len_surfaces):
        verticesID=surfaceID_verticesID[i]
        for j in range(1,len(verticesID)):
            line_vertexID.append([verticesID[j-1],verticesID[j]])
        line_vertexID.append([verticesID[len(verticesID) - 1], verticesID[0]])
    line_vertexID= np.unique(np.sort(np.array(line_vertexID) ,axis=1) ,axis=0)

    lL=[]
    for verticesID in line_vertexID:
        lL.append(geom.add_line(pL[verticesID[0]],pL[verticesID[1]]))



    ss = []
    for i in range(len_surfaces):
        verticesID = surfaceID_verticesID[i]

        loop = []
        for j in range(1, len(verticesID)):
            edge_v0,edge_v1=verticesID[j-1],verticesID[j]
            v0_inlineI=np.where(line_vertexID==edge_v0)[0]
            v1_inlineI=np.where(line_vertexID==edge_v1)[0]
            lineID=np.intersect1d(v0_inlineI,v1_inlineI)[0]
            if line_vertexID[lineID,0]==edge_v0:
                loop.append(lL[lineID])
            else:
                loop.append(-lL[lineID])

        edge_v0, edge_v1 = verticesID[len(verticesID) - 1], verticesID[0]
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

    ss=np.array(ss)
    vv=[]
    i=0
    for surfacesID in domainID_surfaceID:
        vL=geom.add_surface_loop(ss[surfacesID])
        vv.append(geom.add_volume(vL))
        geom.add_physical(vv[i],f"domain_{i}")
        i+=1

    return


class Mesh3D:
    def __init__(self, node_coord:np.ndarray, cell_nodeID:np.ndarray, domain_cellID:list,order=1):
        self.order=order
        self.node_coord = node_coord
        self.cell_nodeID = cell_nodeID
        self.domain_cellID = domain_cellID

        cell_node_coord=self.node_coord[self.cell_nodeID]
        P1 = cell_node_coord[:, 0, :]
        P2 = cell_node_coord[:, 1, :]
        P3 = cell_node_coord[:, 2, :]
        P4 = cell_node_coord[:, 3, :]
        self.cell_coord = (P1 + P2 + P3 + P4) / 4
        self.cellVol = np.abs(np.sum((P4-P1)*np.cross(P3-P1,P2-P1)/6,axis=1))
        #############
    def savetxt(self, mesh_name: str):
        np.savetxt(mesh_name + "__node_coord.txt", self.node_coord)
        np.savetxt(mesh_name + "__cell_nodeID.txt", self.cell_nodeID)
        np.savetxt(mesh_name + "__domainNum.txt", np.array([len(self.domain_cellID)]))
        for i in range(len(self.domain_cellID)):
            np.savetxt(mesh_name + f"__domain{i}_cell.txt", self.domain_cellID[i])


    def Plot(self, Points=True, PointIDs=True, Cells=True, CellIDs=True):
        fig = plt.figure()
        ax = fig.add_subplot(111,projection="3d")
        ax.set_box_aspect((1,1,1))
        x, y, z = self.node_coord[:, 0], self.node_coord[:, 1], self.node_coord[:,2]
        cell_nodeID_num = len(self.cell_nodeID.T)
        if Points:
            ax.scatter(x, y,z, c='tab:blue', alpha=0.33)
        if PointIDs:
            i = 0
            for point in self.node_coord:
                ax.text(point[0], point[1],point[2], f"p{i}")
                i += 1


        if Cells:  # plot一下cell的轮廓线而不plot其面色， 面色用于标记region
            for cell in self.cell_nodeID:
                points = self.node_coord[cell[:4]]

                px = points[:, 0].ravel()
                py = points[:, 1].ravel()
                pz = points[:, 2].ravel()
                px = np.append(px, px[0])
                py = np.append(py, py[0])
                pz = np.append(pz, pz[0])


                p1 = points[0]
                p2 = points[1]
                p3 = points[2]
                p4 = points[3]


                ax.plot([p2[0],p3[0],p4[0],p2[0]], [p2[1],p3[1],p4[1],p2[1]],[p2[2],p3[2],p4[2],p2[2]], lw=0.66, color="tab:blue")
                ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],lw=0.66,color="tab:blue")
                ax.plot([p1[0],p3[0]],[p1[1],p3[1]],[p1[2],p3[2]],lw=0.66,color="tab:blue")
                ax.plot([p1[0],p4[0]],[p1[1],p4[1]],[p1[2],p4[2]],lw=0.66,color="tab:blue")


        if CellIDs:
            i = 0
            for p in self.cell_coord:
                ax.text(p[0], p[1], p[2], f"c{i}")
                i += 1

        plt.legend()
        plt.show()


def LargeMeshCellID_InAsubMesh(cell_node_coordFromSubMesh,cell_coordFromLargeMesh,cellVol_SubMesh=None):
    P1, P2, P3, P4 = cell_node_coordFromSubMesh[:, 0], cell_node_coordFromSubMesh[:, 1], cell_node_coordFromSubMesh[:, 2], cell_node_coordFromSubMesh[:,3]
    cellVol_SubMesh = np.abs(np.sum((P4-P1)*np.cross(P3-P1,P2-P1)/6,axis=1))

    res=np.array([])
    C=cell_coordFromLargeMesh
    for i in range(cell_node_coordFromSubMesh.shape[0]):
        P1,P2,P3,P4=cell_node_coordFromSubMesh[i, 0],cell_node_coordFromSubMesh[i, 1],cell_node_coordFromSubMesh[i, 2], cell_node_coordFromSubMesh[i,3]
        Area_CPiPj= np.abs(np.sum((P1-C)*np.cross(P2-C,P3-C)/6,axis=1))+np.abs(np.sum((P1-C)*np.cross(P2-C,P4-C)/6,axis=1))+np.abs(np.sum((P1-C)*np.cross(P3-C,P4-C)/6,axis=1))+np.abs(np.sum((P2-C)*np.cross(P3-C,P4-C)/6,axis=1))

        err=Area_CPiPj-cellVol_SubMesh[i]

        res=np.append(res,np.where(np.abs(err)<1e-7)[0])

    return np.int_(np.unique(np.array(res)))

def Get_domainCell(cell_coord_LargeMesh,vertex_coord,surface_vertexID,domain_surfaceID):


    domain_cellID=[]
    for i in range(len(domain_surfaceID)):
        print(f"          Assigning cells indices to domain {i}...")
        with pygmsh.occ.Geometry() as geom:
            surfacesID=domain_surfaceID[i]

            Localsurf_vertexID=[surface_vertexID[surfaceID] for surfaceID in surfacesID ]
            Localdomain_surfID=[np.arange(0,surfacesID.shape[0],1,dtype=int)]


            Geom(geom,vertex_coord,666*np.ones(vertex_coord.shape[0]),Localsurf_vertexID,Localdomain_surfID)
            # Generate the mesh
            mesh = geom.generate_mesh()
            cell_node_coord_SubMesh=mesh.points[mesh.cells_dict["tetra"]]
            domain_cellID.append(LargeMeshCellID_InAsubMesh(cell_node_coord_SubMesh,cell_coord_LargeMesh))
    return domain_cellID
def GenerateMeshes3D(vertex_coord: np.ndarray, vertex_mshSize: np.ndarray,surface_vertexID:list, domain_surfaceID: list) :
    print("Generating Meshes")
    with pygmsh.occ.Geometry() as geom:
        Geom(geom, vertex_coord, vertex_mshSize,surface_vertexID,domain_surfaceID)
        msh = geom.generate_mesh()

    mesh=Mesh3D(msh.points,msh.cells_dict["tetra"].astype("int32"),domain_cellID=[msh.cell_sets[f"domain_{i}"][2]  for i in range(len(domain_surfaceID))])


    print("Meshes completed")
    return mesh


def GenerateRegularMesh2D(x_range,y_range,z_range,x_num,y_num,z_num):
    print("Generating Meshes")
    x=np.linspace(x_range[0],x_range[1],x_num)
    y=np.linspace(y_range[0],y_range[1],y_num)
    z=np.linspace(z_range[0],z_range[1],z_num)
    dx,dy,dz=x[1]-x[0],y[1]-y[0],z[1]-z[0]
    x0,y0,z0=x[0],y[0],z[0]

    x,y,z=np.meshgrid(x,y,z)
    node_coord=np.concatenate((x.ravel()[:,np.newaxis],y.ravel()[:,np.newaxis],z.ravel()[:,np.newaxis]),axis=1)

    cellID_nodeID=[]
    for k in range(z_num-1):
        for j in range(y_num-1):
            for i in range(x_num-1):
                #eightvertices
                origin=np.array([x0+i*dx,y0+j*dy,z0+k*dy])
                vertices_coords=np.array([[0.0,0.0,0.0],
                                          [dx,0.0,0.0],
                                          [0.0,dy,0.0],
                                          [0.0,0.0,dz],
                                          [dx,dy,0.0],
                                          [dx,0.0,dz],
                                          [0.0,dy,dz],
                                          [dx,dy,dz]])+origin
                xyz_ids=np.array([ np.where(np.linalg.norm(node_coord-xyz,axis=1)==0  )[0][0]       for xyz in vertices_coords])


                cellID_nodeID=cellID_nodeID+xyz_ids[ [[0,1,2,3],
                                                      [1,2,3,5],
                                                      [2,3,5,6],
                                                      [1,4,5,2],
                                                      [2,4,5,6],
                                                      [4,6,5,7]] ].tolist()
    cellID_nodeID=np.array(cellID_nodeID)

    domain_cellID=[ np.arange(0,cellID_nodeID.shape[0],1,dtype=int)   ]
    mesh=Mesh3D(node_coord,cellID_nodeID,domain_cellID=domain_cellID)
    print("Meshes completed")
    return mesh






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

    # mesh=GenerateRegularMesh2D([-1,1],[-1,1],[-1,1],3,3,3)
    print(mesh.domain_cellID)
    mesh.Plot()




