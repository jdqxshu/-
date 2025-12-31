# ==============================================================================
# analysis_only_vis.py
# 批量特征编码模型分析脚本 (Batch Encoding Model Analysis)
# 功能：PCA降维 -> FIR时序对齐 -> 多被试岭回归 -> 结果汇总与可视化
# ==============================================================================

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

# 可视化库依赖检查
try:
    from brainspace.plotting.surface_plotting import plot_hemispheres
    from brainspace.mesh.mesh_io import read_surface
    from neuromaps.datasets import fetch_atlas
    VIZ_AVAILABLE = True
except ImportError:
    print("[Warning] brainspace 未安装，可视化模块将被禁用。")
    VIZ_AVAILABLE = False

try:
    from utils import fit_encoding_cv, concat_feature, extract_hemi_data_from_files
except ImportError:
    raise ImportError("错误：找不到 utils.py，请检查路径。")

# ================= 全局配置 =================
PATH_FMRI = '21styear_all_subs_rois.npy'  # 预处理后的 fMRI ROI 数据
INPUT_DIR = Path('./imagebind_all_layers_features')                     # 特征文件输入目录
DIR_RES = Path('./results_final')         # 结果输出目录

# 分析超参数
N_TRS = 2249             # 目标时间点数量
PCA_DIM = 250            # PCA 保留主成分数
FIR_WINDOW = 4           # FIR 窗口大小 (4 TRs)
FIR_OFFSET = 7           # FIR 延迟 (Hemodynamic Delay)
ALPHAS = [1e4, 1e5, 1e6] # 岭回归正则化系数

DIR_RES.mkdir(exist_ok=True)

# ================= 分析核心类 =================

class Analyzer:
    def __init__(self):
        print(f"[System] Loading fMRI data: {PATH_FMRI} ...")
        if not Path(PATH_FMRI).exists():
            raise FileNotFoundError(f"找不到文件: {PATH_FMRI}")
            
        self.fmris = np.load(PATH_FMRI, allow_pickle=True).item()
        self.subjects = list(self.fmris.keys())
        print(f"[System] Data loaded. Subjects: {len(self.subjects)}")
        
        if VIZ_AVAILABLE:
            self._init_viz()

    def _init_viz(self):
        """初始化 Brainspace 绘图所需的 Surface 和 Atlas 数据"""
        print("[System] Initializing visualization resources...")
        try:
            # 加载 fsaverage-41k 表面模版
            self.fslr = fetch_atlas('fsaverage', '41k', data_dir=str(DIR_RES))
            self.surf_lh = read_surface(str(self.fslr['inflated'].L))
            self.surf_rh = read_surface(str(self.fslr['inflated'].R))
            
            # 加载 Glasser MMP 图谱用于 ROI 映射
            tpl_files = list(Path('.').glob('*MMP*gii'))
            if not tpl_files:
                print("[Warning] 未找到 MMP 图谱文件 (*.gii)，跳过绘图初始化。")
                self.viz_ready = False
                return
                
            self.wb_rois = extract_hemi_data_from_files(tpl_files, is_label=True, return_list=False).astype(int)
            self.viz_ready = True
        except Exception as e:
            print(f"[Error] Visualization init failed: {e}")
            self.viz_ready = False

    def run_file(self, file_path):
        """
        对单个特征文件执行完整的编码模型流水线
        Pipeline: Load -> Z-Score -> PCA -> FIR -> RidgeCV -> Plot
        """
        file_name = file_path.name
        label = file_path.stem 
        
        print(f"\n[Analysis] Processing: {file_name}")
        
        # 1. 加载特征
        try:
            features = np.load(file_path)
        except Exception as e:
            print(f"[Error] Load failed: {e}")
            return None

        # 2. 维度完整性检查
        if features.shape[0] != N_TRS:
            print(f"[Skip] Dimension mismatch: {features.shape[0]} != {N_TRS}")
            return None
            
        # 3. PCA 降维
        # 先进行标准化 (Z-score normalization)，这对 PCA 至关重要
        scaler = StandardScaler()
        X = scaler.fit_transform(features)
        
        n_comp = min(PCA_DIM, X.shape[1])
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X)
        
        # 4. FIR 时序建模 (Finite Impulse Response)
        # 将特征矩阵转换为包含时间延迟的 Toeplitz 矩阵，以模拟 HRF 延迟
        # Shape: (N_TRS, PCA_DIM) -> (N_TRS, PCA_DIM * FIR_WINDOW)
        X_fir = concat_feature(X_pca, window=FIR_WINDOW, offset=FIR_OFFSET)
        X_fir = X_fir.reshape(X_fir.shape[0], -1)
        
        # 5. 多被试交叉验证回归
        group_scores = []  # 存储每个被试的平均预测准确率 (Pearson r)
        group_maps = []    # 存储每个被试的全脑预测图 (用于计算 Group Mean)
        
        kf = KFold(n_splits=5, shuffle=False)
        
        for sub in tqdm(self.subjects, desc="Fitting", leave=False):
            y = self.fmris[sub]
            
            # 确保 X 和 y 长度一致 (处理 FIR 可能造成的边缘截断)
            min_len = min(len(X_fir), len(y))
            
            # 核心回归函数 (Ridge Regression with Inner CV for Alpha selection)
            _, corr_map = fit_encoding_cv(
                X=X_fir[:min_len], 
                y=y[:min_len],
                cv_splitter=kf, 
                alphas=ALPHAS,
                excluded_start=10, 
                excluded_end=10
            )
            
            group_scores.append(np.nanmean(corr_map))
            group_maps.append(corr_map)
            
        mean_r = np.mean(group_scores)
        std_r = np.std(group_scores)
        
        print(f"   -> Result: Mean R = {mean_r:.5f} (+/- {std_r:.5f})")
        
        # 6. 生成并保存可视化结果
        if VIZ_AVAILABLE and getattr(self, 'viz_ready', False):
            # 计算 Group Level 平均脑图
            avg_map = np.mean(np.vstack(group_maps), axis=0)
            self._plot_brain(avg_map, label, mean_r)
            
        return {"Name": file_name, "Mean_R": mean_r, "Std_R": std_r}

    def _plot_brain(self, roi_map, label, score):
        """绘制皮层投影图并保存为 PNG"""
        # 将 ROI 级数据映射回 Vertex 级数据
        surf_data = np.zeros(self.wb_rois.shape, dtype=np.float32)
        unique_rois = np.unique(self.wb_rois[self.wb_rois != 0])
        
        for roi in unique_rois:
            # 注意 ROI 索引偏移 (Atlas通常从1开始，Array从0开始)
            if roi-1 < len(roi_map):
                surf_data[self.wb_rois == roi] = roi_map[roi-1]
        
        # 掩码处理：隐藏 Medial Wall (0) 和负相关值
        surf_data[self.wb_rois == 0] = np.nan
        surf_data[surf_data < 0] = np.nan 
        
        try:
            save_name = DIR_RES / f"{label}_brain.png"
            plot_hemispheres(
                self.surf_lh, self.surf_rh, array_name=surf_data,
                nan_color=(0.8,0.8,0.8,1), size=(1000, 300),
                cmap='coolwarm', zoom=1.2, screenshot=True,
                filename=str(save_name),
                interactive=False, embed_nb=False,
                label_text={'left': [f"{label}"], 'right': [f'Mean R={score:.3f}']}
            )
        except Exception as e:
            print(f"[Error] Plotting failed: {e}")

# ================= 主程序入口 =================

def main():
    print(f"[Debug] Searching for features in: {INPUT_DIR.absolute()}")
    try:
        analyzer = Analyzer()
    except Exception as e:
        print(f"[Fatal] Initialization failed: {e}")
        return

    # 1. 文件扫描与过滤
    # 过滤规则：必须包含 'win' (表示带窗口特征) 且不含 'rois' (排除标签数据)
    all_files = sorted(list(INPUT_DIR.glob("*.npy")))
    feature_files = [
        f for f in all_files 
    ]

    print(f"\n[System] Found {len(feature_files)} files. Starting batch analysis...\n")
    
    leaderboard = []

    # 2. 批量执行
    for f_path in feature_files:
        res = analyzer.run_file(f_path)
        if res:
            leaderboard.append(res)

    # 3. 结果汇总
    if leaderboard:
        print("\n" + "="*60)
        print("Final Leaderboard (Sorted by Mean R)")
        print("="*60)
        
        leaderboard.sort(key=lambda x: x['Mean_R'], reverse=True)
        
        df_res = pd.DataFrame(leaderboard)
        print(f"{'Rank':<4} | {'File Name':<40} | {'Mean R':<10} | {'Std Dev'}")
        print("-" * 75)
        for i, row in df_res.iterrows():
            print(f"#{i+1:<3} | {row['Name']:<40} | {row['Mean_R']:.4f}     | {row['Std_R']:.4f}")
        
        csv_path = DIR_RES / "final_leaderboard.csv"
        df_res.to_csv(csv_path, index=False)
        print(f"\n[System] Leaderboard saved to: {csv_path}")
        print(f"[System] Brain maps saved to: {DIR_RES}")
    else:
        print("[System] No valid results generated.")

if __name__ == "__main__":
    main()