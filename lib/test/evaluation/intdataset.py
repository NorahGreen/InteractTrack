import os
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import re
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# 场景类别名（顺序固定）
SCENE_NAMES = [
    "日常活动",               # 0
    "体育活动（运动分析）",   # 1
    "无人机跟踪",             # 2
    "监控场景",               # 3
    "动物检测",               # 4
    "其他",                   # 5
]


class INTDataset(BaseDataset):
    """
    INT merged dataset
    """
    def __init__(self):
        super().__init__()
        self.base_path = Path(self.env_settings.int_path)
        self.sequence_list: List[str] = self._get_sequence_list()

    def __len__(self):
        return len(self.sequence_list)

    def get_sequence_list(self):
        seqs = []
        for s in self.sequence_list:
            seq = self._construct_sequence(s)
            if seq is not None:
                seqs.append(seq)
        return SequenceList(seqs)

    # ---------- Public helpers (for analysis) ----------

    def get_sequence_scene_matrix(self) -> Tuple[List[str], np.ndarray]:
        rel_list: List[str] = []
        rows: List[List[int]] = []
        for rel in self.sequence_list:
            vec = self._load_scene_vec(self.base_path / rel / "scene.txt")
            rel_list.append(rel)
            rows.append(vec.tolist())
        return rel_list, np.asarray(rows, dtype=np.int64)

    def get_scene_counts(self) -> dict:
        _, mat = self.get_sequence_scene_matrix()
        counts = mat.sum(axis=0).tolist()
        return {SCENE_NAMES[i]: int(counts[i]) for i in range(6)}

    # ---------- Internal helpers ----------

    def _get_sequence_list(self) -> List[str]:
        """
        使用 os.walk 动态寻找包含 groundtruth.txt 和 imgs 文件夹的目录。
        """
        seqs: List[str] = []
        for root, dirs, files in os.walk(self.base_path):
            if 'groundtruth.txt' in files and 'imgs' in dirs:
                rel = os.path.relpath(root, self.base_path).replace('\\', '/')
                seqs.append(rel)
                
        seqs = sorted(list(dict.fromkeys(seqs)))
        if len(seqs) == 0:
            print(f"[WARN] INTDataset: no sequences found under {self.base_path}")
        else:
            print(f"[INFO] INTDataset: found {len(seqs)} sequences under {self.base_path}")
        return seqs

    def _get_sequence_list_filter_MOT(self) -> List[str]:
        seqs: List[str] = []
        bad_seqdir = re.compile(r'_c\d{3}(?:-\d+)?$', re.IGNORECASE)

        for root, dirs, files in os.walk(self.base_path):
            seq_name = os.path.basename(root)
            if bad_seqdir.search(seq_name):   
                continue
            if 'groundtruth.txt' in files and 'imgs' in dirs:
                rel = os.path.relpath(root, self.base_path).replace('\\', '/')
                seqs.append(rel)

        seqs = sorted(list(dict.fromkeys(seqs)))
        if len(seqs) == 0:
            print(f"[WARN] INTDataset: no sequences found under {self.base_path}")
        else:
            print(f"[INFO] INTDataset: found {len(seqs)} sequences under {self.base_path}")
        return seqs

    def _construct_sequence(self, sequence_rel: str) -> Optional[Sequence]:
        seq_dir = (self.base_path / sequence_rel).resolve()
        img_dir = seq_dir / "imgs"

        frames_list = [
            str(p) for p in sorted(img_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ]
        n_img = len(frames_list)
        if n_img == 0:
            print(f"[WARN] {sequence_rel}: no images under imgs/")
            return None

        gt_path = seq_dir / "groundtruth.txt"
        gt_raw, vis_raw = self._load_gt_and_visibility(gt_path)

        vis_mask = vis_raw.copy()
        fo = seq_dir / "full_occlusion.txt"
        oov = seq_dir / "out_of_view.txt"
        if fo.is_file() and oov.is_file():
            try:
                fo_arr = self._load_mask_file(fo, len(gt_raw))
                oov_arr = self._load_mask_file(oov, len(gt_raw))
                vis_mask = np.logical_and(vis_mask, np.logical_and(fo_arr == 0, oov_arr == 0))
            except Exception:
                pass

        n_gt = int(gt_raw.shape[0])
        if n_img != n_gt:
            n = min(n_img, n_gt)
            print(f"[WARN] {sequence_rel}: frames={n_img}, gt={n_gt} -> truncate to {n}")
            frames_list = frames_list[:n]
            gt_raw = gt_raw[:n]
            vis_mask = vis_mask[:n]

        if not vis_mask[0]:
            idxs = np.flatnonzero(vis_mask)
            if idxs.size > 0:
                start = int(idxs[0])
                if start > 0:
                    frames_list = frames_list[start:]
                    gt_raw = gt_raw[start:]
                    vis_mask = vis_mask[start:]
            else:
                return None

        scene_vec = self._load_scene_vec(seq_dir / "scene.txt")

        # ================== 核心修改点 ==================
        # 提取真实扁平化的文件夹名作为序列名
        seq_name = Path(sequence_rel).name
        # ================================================

        seq = Sequence(seq_name, frames_list, 'int', gt_raw.reshape(-1, 4),
                       target_visible=vis_mask.astype(np.bool_))

        seq.scene_vec = scene_vec                    
        seq.scene_names = list(SCENE_NAMES)          
        seq.scene_tags = [SCENE_NAMES[i] for i, v in enumerate(scene_vec.tolist()) if v == 1]

        meta = getattr(seq, "meta", {})
        meta.update({
            "scene_vec": scene_vec,
            "scene_names": list(SCENE_NAMES),
            "scene_tags": seq.scene_tags,
        })
        seq.meta = meta

        return seq

    @staticmethod
    def _load_mask_file(path: Path, n_expect: int) -> np.ndarray:
        xs = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for ln in f:
                s = ln.strip().split(',')[0].split()[0] if ln.strip() else '0'
                try:
                    xs.append(int(float(s)))
                except Exception:
                    xs.append(0)
        arr = np.asarray(xs, dtype=np.int64)
        if arr.ndim != 1 or arr.size < n_expect:
            raise ValueError("mask length mismatch")
        return arr[:n_expect]

    @staticmethod
    def _load_gt_and_visibility(gt_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        rows, vis = [], []
        with open(gt_path, 'r', encoding='utf-8', errors='ignore') as f:
            for ln in f:
                s = ln.strip()
                if not s:
                    rows.append([0.0, 0.0, 0.0, 0.0]); vis.append(False); continue
                
                s = s.replace('\t', ' ').replace(',', ' ')
                toks = [t for t in s.split() if t]
                
                if len(toks) == 1:
                    try:
                        v = float(toks[0])
                        if np.isnan(v) or v == 0.0:
                            rows.append([0.0, 0.0, 0.0, 0.0]); vis.append(False); continue
                    except Exception:
                        rows.append([0.0, 0.0, 0.0, 0.0]); vis.append(False); continue

                if len(toks) < 4:
                    rows.append([0.0, 0.0, 0.0, 0.0]); vis.append(False); continue
                try:
                    x, y, w, h = map(float, toks[:4])
                except Exception:
                    rows.append([0.0, 0.0, 0.0, 0.0]); vis.append(False); continue

                if (np.isnan(x) or np.isnan(y) or np.isnan(w) or np.isnan(h) or
                    x < 0 or y < 0 or w <= 0 or h <= 0):
                    rows.append([0.0, 0.0, 0.0, 0.0]); vis.append(False); continue

                rows.append([x, y, w, h]); vis.append(True)

        gt = np.asarray(rows, dtype=np.float64)
        vm = np.asarray(vis, dtype=np.bool_)
        return gt, vm

    @staticmethod
    def _load_scene_vec(scene_path: Path) -> np.ndarray:
        vec = np.zeros(6, dtype=np.int64)
        try:
            if not scene_path.is_file():
                return vec
            line = scene_path.read_text(encoding='utf-8', errors='ignore').strip().splitlines()[0]
            line = line.replace('\t', ' ').replace(',', ' ')
            toks = [t for t in line.split() if t]
            
            vals: List[int] = []
            for t in toks[:6]:
                try:
                    v = int(float(t))
                    v = 1 if v >= 1 else 0
                except Exception:
                    v = 0
                vals.append(v)
            while len(vals) < 6:
                vals.append(0)
            vec = np.asarray(vals[:6], dtype=np.int64)
        except Exception:
            pass
        return vec