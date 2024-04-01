from FEM2D_ver7_Order1.Functions2D_Order1 import *
import time
import scipy.sparse as sp





def jac_PK1_i(PK1, Cp,nodeI_gaussID, nodei, y_, dy, d, y_shape, Nabla_func, PK1_func):
    '''
    i番目のnodeにおけるy(x)の関数値(nodei=2Iの場合->横座標の関数値,nodei=2I+1の場合->縦座標の間数値)に関するPK1の偏微分
    (yが変わると，Nabla y(x)が変わる，結果として, PK1(Nabla y(x))も変わります)

    PK1(節点におけるyの解) \cdot \delta F   =  0 (体積力や面力を考ゼロにする場合)                (\delta F はテスト関数の変形勾配を表す．Galerkinに基づく本programは，メッシュが決定される次第，\delta Fも決められる．)
    上式の「節点におけるyの解」を得るために，
    まず「節点におけるyのtrial」を上式に代入する．
    そうすると,errが得られる：
    err(節点におけるyのtrial) = PK1(節点におけるyのtrial) \cdot \delta F        (\delta F はテスト関数の変形勾配を表す．Galerkinに基づく本programは，メッシュが決定される次第，\delta Fも決められる．)
    そしてニュートン法では，
    節点におけるyの次回のtrial = 節点におけるyのtrial-[ errが節点におけるyに対する偏微分 ].inverse \cdot err(節点におけるyのtrial)
    の式を利用して近似解を探す．

    上式において一番大事なのは　errのyに関するjacobiである　[errが節点におけるyに対する偏微分] を算出ことである．
    err(節点におけるyのtrial)の式を　[errが節点におけるyに対する偏微分] (err)に代入すると，
    [errが節点におけるyに対する偏微分] = [PK1が節点におけるyに対する偏微分] \cdot \delta F         (\delta F はテスト関数の変形勾配を表す．Galerkinに基づく本programは，メッシュが決定される次第，\delta Fも決められる．)

    したがって，[errが節点におけるyに対する偏微分]の算出には，[PK1が節点におけるyに対する偏微分]が不可欠である．
    本関数が計算しているのは，この[PK1が節点におけるyに対する偏微分]である．

    :param PK1: 第一Piola Kirchhoffテンソルの場 np.array([cell0におけるPK1, cell1におけるPK1,...])
    :param Cp: eigen右コーシーグリーンテンソル（キンクの場合，C^p）
    :param nodeI_gaussID: I番目のノードの周りの第一近傍のセル(gaussian積分点)のindex
    :param nodei: i=2*Iの時：I番目の横座標に対するPK1の偏微分を計算する，i=2*I+1の時: I番目の縦座標に対するPK1の偏微分を計算する
    :param y_: np.array([y(node0),y(node1),...])
    :param dy: np.array([[d,0,0,0,....],    <- 0番目のnodeの横座標に沿ったdy
                         [0,d,0,0,....],    <- 0番目のnodeの縦座標に沿ったdy
                         [0,0,d,0,....],    <- 1番目のnodeの横座標に沿ったdy
                         ...)
    :param d:   数値微分用の係数　partial_x y(x) =[y(x+d) - y(x)]/d
    :param y_shape: np.array([y(node0),y(node1),...])のshape, (ベクトル関数の場合，shape=(nodeNum,2))
    :param Nabla_func: I番目のノードの第一近傍のセルにおける変形勾配を計算するための関数
    :param PK1_func:　PK1関数，入力は変形勾配とeigen右コーシーグリーンテンソル（キンクの場合，C^p）,PK1(F,Cp)の具体的な形式は後で定義する
    :return:  [PK1が節点におけるyに対する偏微分]
    '''
    nodeI = int(nodei / y_shape[1])
    localI_F, localI_gaussIDs = Nabla_func(np.reshape(y_ + dy[nodei],y_shape),  nodeI), nodeI_gaussID[nodeI]

    return ( PK1_func(localI_F, Cp[localI_gaussIDs]) - PK1[localI_gaussIDs]).ravel() / d


def TrialVectorFunction(V:FunctionSpace,dim=2):
    '''
    nodeの基準配置での座標を最初のtrial np.array([node0における関数値,node1における関数値,...])とする．
    :param V: 定義されたメッシュ上の関数空間
    :param dim=2:二次元y(x)からなる関数空間
    :return: trial
    '''
    if dim==2:
        trial=V.node_coord.copy()
    elif dim==3:
        trial=np.zeros((V.node_coord.shape[0],3),dtype=float)
        trial[:,[0,1]]=V.node_coord.copy()
    return trial

class TestVectorFunctions:
    def __init__(self,V:FunctionSpace,dim=2):
        '''
        Testfuncsを定義する
        Testfuncsの間数値行列は：
        np.array([[node0におけるtestfunc0の横座標，node0におけるtestfunc0の縦座標,node1におけるtestfunc0の横座標,...],
                  [node0におけるtestfunc1の横座標, node0におけるtestfunc1の縦座標,node1におけるtestfunc1の横座標,...],
                  ...])
          =
        np.array([[1,0,0,...],
                  [0,1,0,...],
                  [0,0,1,...],
                  ...])

        :param V: 関数空間，関数空間の基底関数をここでのtestfuncsとする
        :param dim:
        '''
        node_coord=V.node_coord
        self.TestFuncs= np.reshape(np.identity(node_coord.shape[0]*dim),(node_coord.shape[0]*dim,node_coord.shape[0],dim))
        self.V=V
        self._dim=dim

        self._TestSparse_flag=False

    def _GenerateTestSparse(self,YparaID_to_YravelID):
        '''
        Testfuncsの変形勾配:
        np.array([[node0 testfunc0の変形勾配の00成分, node0 testfunc0の変形勾配の01成分, node0 testfunc0の変形購買の10成分, node0 testfunc0の変形購買の11成分, node1 testfunc0の変形勾配の00成分, node1 testfunc0の変形勾配の01成分, node1 testfunc0の変形購買の10成分, node1 testfunc0の変形購買の11成分,...],
                  [node0 testfunc1の変形勾配の00成分, node0 testfunc1の変形勾配の01成分, node0 testfunc1の変形購買の10成分, node0 testfunc1の変形購買の11成分, node1 testfunc1の変形勾配の00成分, node1 testfunc1の変形勾配の01成分, node1 testfunc1の変形購買の10成分, node1 testfunc1の変形購買の11成分,...],
                  ...])
        を生成する

        :param YparaID_to_YravelID:
        np.array([独立変数0が対応するi index,
                  独立変数1が対応するi index,
                  独立変数2が対応するi index,
                  ...                    ])    i=2*Iの時,この独立変数がI番目のノードの横座標を表し，
        　　　　　　　　　　　　　　　　　　　　　　 i=2*I+1の時，この独立変数がI番目のノードの縦座標を表す．

        独立の意味は：dirichlet境界条件が設定されていない
        '''
        if self._TestSparse_flag:
            pass
        else:
            V = self.V
            nodeI_cellID, cellNum, Nabla, dV, v_, dim = V.nodeI_cellID, V.cell_nodeID.shape[0], V._Nabla_N3_NodeI, (V.cellArea )[:,np.newaxis,np.newaxis], self.TestFuncs, self._dim

            print("testSparseMat preparation...")
            # self.dim is the degree of freedom of the trial func at each point, 2*self.dim is the degree of freedom of its Nabla

            temp_Row = np.array([])
            indices_Col = np.array([])  # testids
            Data = np.array([])
            for i in range(YparaID_to_YravelID.shape[0]):
                nodei = YparaID_to_YravelID[i]
                I = int(nodei / self._dim)
                cellids = nodeI_cellID[I]
                temp_Row = np.append(temp_Row, cellids)
                indices_Col = np.append(indices_Col, i * np.ones(2 * dim * cellids.shape[0], dtype="int32"))  # testids
                Data = np.append(Data, (Nabla(v_[nodei], I) * dV[cellids]).ravel())
            temp_Row = 2 * dim * temp_Row  # cellIds*2*self.dim
            indices_Row = np.array([temp_Row + i for i in range(2 * dim)]).T.ravel()

            self.sparse_jacPK1_i, self.sparse_jacPK1_j, self.sparse_jacPK1_shape = indices_Row, indices_Col, (
             cellNum * 2 * dim,YparaID_to_YravelID.shape[0])
            self.testSparse = sp.coo_matrix((Data, (indices_Col, indices_Row)),
                                            shape=( YparaID_to_YravelID.shape[0],cellNum * 2 * dim))  # [i,0,j]   j=(4*C,4*C+1,4*C+2,4*C+3) of (F_00,F_01,F_10,F_11)@(C-th cell)          i-th testfunc
            self.testSparse.eliminate_zeros()
            self.TestFuncs=None  # release RAM
            print("testSparse completed")
            self._TestSparse_flag = True


class EquilibriumProblem:
    def __init__(self,TrialFunc:np.ndarray,TestFuncs:TestVectorFunctions,psi_func,PK1_func):
        '''
        Equilibriumの式：
        PK1(節点におけるyの解) \cdot \delta F   =  0 (体積力や面力をゼロにする場合)                (\delta F はテスト関数の変形勾配を表す．Galerkinに基づく本programは，メッシュが決定される次第，\delta Fも決められる．)
        上式の「節点におけるyの解」を得るために，
        まず「節点におけるyのtrial」を上式に代入する．
        そうすると,errが得られる：
        err(節点におけるyのtrial) = PK1(節点におけるyのtrial) \cdot \delta F        (\delta F はテスト関数の変形勾配を表す．Galerkinに基づく本programは，メッシュが決定される次第，\delta Fも決められる．)
        そしてニュートン法では，
        節点におけるyの次回のtrial = 節点におけるyのtrial-[ errが節点におけるyに対する偏微分 ].inverse \cdot err(節点におけるyのtrial)
        の式を利用して近似解を探す．

        上式において一番大事なのは　errのyに関するjacobiである　[errが節点におけるyに対する偏微分] を算出ことである．
        err(節点におけるyのtrial)の式を　[errが節点におけるyに対する偏微分] (err)に代入すると，
        [errが節点におけるyに対する偏微分] = [PK1が節点におけるyに対する偏微分] \cdot \delta F         (\delta F はテスト関数の変形勾配を表す．Galerkinに基づく本programは，メッシュが決定される次第，\delta Fも決められる．)

        [PK1が節点におけるyに対する偏微分]は関数「jac_PK1_i()」によって算出される．

        :param TrialFunc:  Trial関数，ニュートン法によって更新される関数
        :param TestFuncs: 　test関数s, Galerkinを使う本プログラムでは，testfuncs=basisfuncs
        :param psi_func:   弾性エネルギー密度関数，入力は変形勾配とeigen右コーシーグリーンテンソル
        :param PK1_func: 　第一Piola kirchhof関数，入力は変形勾配とeigen右コーシーグリーンテンソル
        '''
        self.y_initial_param=TrialFunc.ravel() #初回のtrial関数
        self.y_shape=TrialFunc.shape

        self.dim=1
        for i in TrialFunc.shape[1:]:
            self.dim=self.dim*i

        '''
        以下の手順で，Testfuncs,Trialfuncと関数空間V^hに関する情報をこのsolverの属性に追加する．
        '''

        V=TestFuncs.V
        self.V=V
        self.gauss_area=V.cellArea
        self.PK1_func=PK1_func
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



        self._dirichlet_flag=False





    def set_dirichlet(self,coords_indices,u):
        '''
        dirichlet条件を設定する
        :param coords_indices: dirichlet条件を設定した点と座標のindex　i=2*I+0: I番目nodeの横座標， i=2*I+1: I番目nodeの縦座標
        :param u: dirichlet条件を設定した座標の変化量(現配置での座標-基準配置での座標)（定数とする）
        '''
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



    def _Err_and_Jacob_and_psi(self,y_para,tol,rel_tol):
        '''

        :param y_para: dirichlet条件を設定していない点の座標　
        :param tol: err<tolとなった時，「収束した」と判別する．
        :param rel_tol: rel_err<rel_tolとなった時，「収束した」と判別する．
        :return:
        '''
        '''
        solverの属性を下のように引用する
        （y_shape=self.y_shapeを設定した場合，これからself.y_shapeを利用したいy_shapeを呼ぶだけでいいです．
        　self.y_shapeを読んでしまうと，ドットという操作の時間がカウントされるので，コストが上昇されます．
        Loopを使用しない場合では，self.を使ってもいいですが，
        今の_Err_and_Jacob_and_psiという関数はあるLoopに属されているので，self.の操作を回避するほうがいいです．）
        '''
        y_shape,y_,d,dy,test_sparse,GArea,Cp =self.y_shape,self.y_.ravel(), self.d,self.dy,self.TestFuncs.testSparse,self.gauss_area, self.Cp
        sparse_jacPK1_i, sparse_jacPK1_j, sparse_jacPK1_shape = self.TestFuncs.sparse_jacPK1_i, self.TestFuncs.sparse_jacPK1_j, self.TestFuncs.sparse_jacPK1_shape
        PK1_func,V_Nabla_N3_J,V_Nabla=self.PK1_func,self.V._Nabla_N3_NodeI,self.V.Nabla
        nodeI_gaussID, YparaID_to_YravelID= self.nodeI_gaussID, self.YparaID_to_YravelID

        #ravelした現配置での座標の独立変数である部分に,y_paraの値を与える
        y_[YparaID_to_YravelID]=y_para

        #変形勾配の場と第一PK応力の場を計算する
        F=V_Nabla(np.reshape(y_,self.y_shape))
        PK1=PK1_func(F,Cp)

        #残差力の場を計算する
        Err=test_sparse.dot(PK1.ravel())
        err,flag=np.linalg.norm(Err)/np.sqrt(Err.shape[0]),True
        #残差力のjacobiを定義する（まだ具体な値を入れていない）
        jacobi=sp.csc_matrix([[0.0]])

        if self.IterNum==0:
            #初回の更新において,ErrのL2normが前回のそれに対する相対的な変化量を１にする
            rel_err_ener=1
        else:
            rel_err_ener=np.abs(err-self.err)/self.err
            if rel_err_ener < rel_tol:
                flag=False
        self.err=err

        if err>tol and flag:
            #ErrのL2normがtolより大きい場合，Newton法の更新用のjacobiを計算する
            Data=np.array([])
            for nodei in YparaID_to_YravelID:
                Data=np.append(Data,jac_PK1_i(PK1, Cp, nodeI_gaussID, nodei, y_, dy, d, y_shape, V_Nabla_N3_J, PK1_func))
                #Dataはndim=1のnp.arrayであり，Dataのi番目の成分が対応するjacobiの行と列のindexに関する情報はすでにsparse_jacPK1_iとsparse_jacPK1_jに含まれている．
            jacobi=test_sparse.dot(sp.coo_matrix(  (Data,(sparse_jacPK1_i,sparse_jacPK1_j)),shape= sparse_jacPK1_shape    )).toarray()

        ener=np.sum(self.psi_func(F,Cp)*GArea)

        return Err,jacobi,ener,err,rel_err_ener

    def _Ener_Iterator(self,y_para,tol,rel_tol):
        '''
        newtonほうを使って近似解y_paraを見出す
        :param y_para: trialfuncというオブジェクトが持つたくさんのノードにおける現配置での座標のうち，独立な座標値からなるベクトル
        :param tol: 収束判定の基準値1
        :param rel_tol: 収束判定の基準値2
        :return: 近似解であるy_paraとその弾性エネルギー
        '''
        self.IterNum=0
        #Errと,y_paraの更新用のjacobiを計算する
        Err, jacobi, energy, err,rel_err_ener = self._Err_and_Jacob_and_psi(y_para,tol,rel_tol)
        print(f"Iter{self.IterNum},", " energy=", energy," 10^-6 J/mm", ", rel_err=NaN", ", err=", err,
              )

        while err>tol and rel_err_ener>rel_tol:
            #下記のnewton法の式で，y_paraを更新する
            y_para=y_para-np.linalg.inv(jacobi).dot(Err).ravel()

            self.IterNum+=1
            # 更新後のy_paraが対応するErrと,更新用のjacobiを計算する
            Err, jacobi, energy, err,rel_err_ener = self._Err_and_Jacob_and_psi(y_para,tol,rel_tol)
            print(f"Iter{self.IterNum},", " energy=", energy," 10^-6 J/mm", ", rel_err=", rel_err_ener, ", err=", err,
                  )
        return y_para,energy
    def Solve_ByNewton(self,tol=0.01,rel_tol=0.01):
        '''
        newton法で現配置での座標の近似解を見出す
        :param tol: 収束判定の基準値1
        :param rel_tol: 収束判定の基準値2
        '''
        #定数である（galerkin法の場合）test関数を生じる
        self.TestFuncs._GenerateTestSparse(self.YparaID_to_YravelID)

        start=time.time()
        print("\n\nstart iteration...")
        #Newton法の更新式を使って，y_paraの近似解を見出す
        y_para,energy=self._Ener_Iterator(self.y_initial_param,tol,rel_tol)
        end=time.time()
        print("Iteration end...   time cost: ", end - start," s,  \n\n")

        '''
        y_paraの値を，現配置での座標np.array([[node0の横座標,node0の縦座標],
                                           [node1の横座標,node1の縦座標],
                                           ...])に代入する
        '''
        if self._dirichlet_flag:
            y_ = self.y_.ravel()
            y_[self.YparaID_to_YravelID] = y_para
            self.y_ = np.reshape(y_, self.y_shape)
            y=self.y_
        else:
            y=np.reshape(y_para,self.y_shape)
        return y,energy


