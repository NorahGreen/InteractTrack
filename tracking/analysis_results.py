import os
import csv
import torch
import numpy as np
import _init_paths
import matplotlib.pyplot as plt

from lib.test.evaluation import get_dataset, trackerlist
from lib.test.evaluation.environment import env_settings
from lib.test.evaluation.tracker import Tracker
from lib.test.analysis.plot_results import (
    check_and_load_precomputed_results,
    merge_multiple_runs,
    get_tracker_display_name,
    get_plot_draw_styles
)

# ========================================================================
# 0. Monkey Patch: Bypass PyTracking model initialization to avoid multiprocessing deadlocks
# ========================================================================
def _patched_tracker_init(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None, result_only=False):
    self.name = name
    self.parameter_name = parameter_name
    self.dataset_name = dataset_name
    self.run_id = run_id
    self.display_name = name if display_name is None else display_name
    self.env_settings = env_settings()
    self.results_dir = '{}/{}/{}'.format(self.env_settings.results_path, self.name, self.parameter_name)

Tracker.__init__ = _patched_tracker_init

# ========================================================================
# 1. Plotting Configurations & Constants
# ========================================================================
FONT_STACK = ['Inter', 'Arial', 'Helvetica Neue', 'DejaVu Sans', 'Liberation Sans']
TITLE_WEIGHT = 'semibold'
LABEL_WEIGHT = 'normal'
LEGEND_WEIGHT = 'bold'
AXES_LINEWIDTH = 1.5
TICK_LENGTH = 6

plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': FONT_STACK, 'font.size': 15,
    'axes.titlesize': 20, 'axes.labelsize': 16,
    'axes.titleweight': TITLE_WEIGHT, 'axes.labelweight': LABEL_WEIGHT,
    'axes.linewidth': AXES_LINEWIDTH,
    'xtick.labelsize': 20, 'ytick.labelsize': 20,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': TICK_LENGTH, 'ytick.major.size': TICK_LENGTH,
    'legend.fontsize': 14, 'legend.title_fontsize': 16,
    'figure.dpi': 150, 'savefig.dpi': 300,
})

SCENE_NAMES = ["Daily Activities", "Sports Analysis", "UAV Tracking",
               "Surveillance", "Wildlife Monitor", "Other"]

METRIC_CONFIG = {
    'success': {'array_field': 'ave_success_rate_plot_overlap', 'threshold_field': 'threshold_set_overlap', 'xlabel': 'Overlap threshold', 'ylabel': 'Success rate', 'xlim': (0.0, 1.0), 'ylim': (0.0, 1.0), 'title': 'Success plots of OPE', 'score_type': 'auc', 'normalize_x': True},
    'prec': {'array_field': 'ave_success_rate_plot_center', 'threshold_field': 'threshold_set_center', 'xlabel': 'Location error threshold', 'ylabel': 'Distance precision', 'xlim': (0.0, 1.0), 'ylim': (0.0, 1.0), 'title': 'Precision plots of OPE', 'score_type': 'index', 'score_index': 20, 'normalize_x': True},
    'norm_prec': {'array_field': 'ave_success_rate_plot_center_norm', 'threshold_field': 'threshold_set_center_norm', 'xlabel': 'Location error threshold', 'ylabel': 'Distance precision', 'xlim': (0.0, 1.0), 'ylim': (0.0, 1.0), 'title': 'Normalized Precision plots', 'score_type': 'index', 'score_index': 20, 'normalize_x': True}
}

AX_TICKS = np.round(np.arange(0.0, 1.01, 0.1), 2)
DEFAULT_Y_MAX = 0.85
PRECISION_Y_MAX = 0.82

# ========================================================================
# 2. Custom Metrics Evaluation (Interact, Resp, Perc)
# ========================================================================
def calculate_iou(boxA, boxB):
    if boxA is None or len(boxA) < 4 or boxB is None or len(boxB) < 4: return 0.0
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]), min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    unionArea = float(boxA[2] * boxA[3] + boxB[2] * boxB[3] - interArea)
    return interArea / unionArea if unionArea > 0 else 0.0

def find_file_recursively(root_path, target_filename):
    for dirpath, _, filenames in os.walk(root_path):
        if target_filename in filenames:
            return os.path.join(dirpath, target_filename)
    return None

def parse_box_line(line):
    try:
        parts = line.strip().replace(',', ' ').split()
        if not parts: return None
        box = [float(val) for val in parts]
        if len(box) >= 4 and all(v == 0 for v in box[:4]): return None
        if len(box) >= 4: return box[:4]
    except: pass
    return None

def evaluate_custom_metrics(dataset_path, predictions_path):
    segment_avg_ious = []
    total_switches, correct_switch_events = 0, 0
    total_perc_frames, successful_perc_events, total_iou_perc = 0, 0, 0.0

    valid_sequences = []
    for root, dirs, files in os.walk(dataset_path):
        if 'description.txt' in files and 'groundtruth.txt' in files and 'switch.txt' in files:
            seq_name = os.path.basename(root)
            valid_sequences.append((seq_name, root))
    valid_sequences.sort(key=lambda x: x[0])

    for seq_name, seq_path in valid_sequences:
        pred_file = find_file_recursively(predictions_path, f"{seq_name}.txt")
        if pred_file is None: continue

        with open(os.path.join(seq_path, 'description.txt'), 'r') as f: orig_desc = f.readlines()
        with open(os.path.join(seq_path, 'groundtruth.txt'), 'r') as f: orig_gt = f.readlines()
        with open(os.path.join(seq_path, 'switch.txt'), 'r') as f: orig_sw = f.readlines()
        with open(pred_file, 'r') as f: orig_pred = f.readlines()

        max_len = max(len(orig_desc), len(orig_gt), len(orig_pred), len(orig_sw))
        min_len = min(len(orig_desc), len(orig_gt), len(orig_pred))

        desc_pad = orig_desc + [''] * (max_len - len(orig_desc))
        gt_pad = orig_gt + [''] * (max_len - len(orig_gt))
        sw_pad = orig_sw + [''] * (max_len - len(orig_sw))
        pred_pad = orig_pred + [''] * (max_len - len(orig_pred))

        # A. Interactiveness
        starts = [i for i, line in enumerate(desc_pad) if line.strip()]
        for i in range(len(starts)):
            s = starts[i]
            e = starts[i+1] if i+1 < len(starts) else max_len
            if s >= e: continue
            ious = []
            for idx in range(s, e):
                gt = parse_box_line(gt_pad[idx])
                pd = parse_box_line(pred_pad[idx])
                if gt and pd: ious.append(calculate_iou(gt, pd))
            if ious: segment_avg_ious.append(np.mean(ious))

        # B. Responsiveness
        for i in range(max_len):
            if desc_pad[i].strip():
                total_switches += 1
                gt_box = parse_box_line(gt_pad[i])
                pred_box = parse_box_line(pred_pad[i])
                switch_box = parse_box_line(sw_pad[i]) if i < len(orig_sw) else None
                
                if gt_box is not None and pred_box is not None:
                    iou_gt = calculate_iou(gt_box, pred_box)
                    if switch_box is not None:
                        iou_sw = calculate_iou(switch_box, pred_box)
                        if iou_gt > iou_sw and iou_gt > 0.5: correct_switch_events += 1
                    else:
                        if iou_gt > 0.5: correct_switch_events += 1

        # C. Perception
        for i in range(min_len):
            if orig_desc[i].strip():
                gt_box = parse_box_line(orig_gt[i])
                if gt_box is not None:
                    total_perc_frames += 1
                    pred_box = parse_box_line(orig_pred[i])
                    curr_pred = pred_box if pred_box is not None else [0,0,0,0]
                    iou_gt = calculate_iou(gt_box, curr_pred)
                    total_iou_perc += iou_gt
                    if iou_gt > 0.5: successful_perc_events += 1

    interact = (np.mean(segment_avg_ious) * 100) if segment_avg_ious else 0.0
    resp = (correct_switch_events / total_switches * 100) if total_switches > 0 else 0.0
    perc_acc = (successful_perc_events / total_perc_frames * 100) if total_perc_frames > 0 else 0.0
    perc_prec = (total_iou_perc / total_perc_frames * 100) if total_perc_frames > 0 else 0.0

    return interact, resp, perc_acc, perc_prec

def _extract_tracking_scores(eval_data):
    mask = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)
    T = len(eval_data['trackers'])
    out = {}
    ov = torch.tensor(eval_data['ave_success_rate_plot_overlap'])       
    out['AUC'] = (ov[mask,:,:].mean(0) * 100.0).mean(-1) if mask.any() else torch.zeros(T)
    ctr = torch.tensor(eval_data['ave_success_rate_plot_center'])       
    out['Prec'] = (ctr[mask,:,:].mean(0) * 100.0)[:, 20] if mask.any() else torch.zeros(T)
    nctr = torch.tensor(eval_data['ave_success_rate_plot_center_norm']) 
    out['NPrec'] = (nctr[mask,:,:].mean(0) * 100.0)[:, 20] if mask.any() else torch.zeros(T)
    return out

# ========================================================================
# 3. Scene Extraction & Printing (Bypassing Cache)
# ========================================================================
def _force_read_scene_matrix(dataset_root, seq_names):
    """Bypass PyTracking cache and scan scene.txt directly from disk."""
    print("[INFO] Scanning scene.txt from disk (bypassing cache)...")
    seq_to_scenepath = {}
    for root, dirs, files in os.walk(dataset_root):
        if 'groundtruth.txt' in files:
            sname = os.path.basename(root)
            seq_to_scenepath[sname] = os.path.join(root, 'scene.txt')

    mat = []
    found_count = 0
    for sname in seq_names:
        vec = np.zeros(6, dtype=np.int64)
        scene_file = seq_to_scenepath.get(sname, None)
        
        if scene_file and os.path.exists(scene_file):
            try:
                with open(scene_file, 'r') as f:
                    line = f.readline().strip().replace(',', ' ').replace('\t', ' ')
                    toks = [t for t in line.split() if t]
                    for i, t in enumerate(toks[:6]):
                        vec[i] = 1 if int(float(t)) >= 1 else 0
                found_count += 1
            except Exception:
                pass
        mat.append(vec)
        
    print(f"[INFO] Successfully loaded scene labels for {found_count} / {len(seq_names)} sequences.\n")
    return np.stack(mat, axis=0) if len(mat) > 0 else np.zeros((0,6), dtype=np.int64)

def _score_from_eval_data(eval_data, mask_bool, want=('success','prec','norm_prec')):
    mask = torch.tensor(mask_bool, dtype=torch.bool)
    T = len(eval_data['trackers'])
    out = {}

    if 'success' in want:
        ov = torch.tensor(eval_data['ave_success_rate_plot_overlap'])       
        out['AUC'] = (ov[mask,:,:].mean(0) * 100.0).mean(-1) if mask.any() else torch.zeros(T)
    if 'prec' in want:
        ctr = torch.tensor(eval_data['ave_success_rate_plot_center'])       
        out['Precision'] = (ctr[mask,:,:].mean(0) * 100.0)[:, 20] if mask.any() else torch.zeros(T)
    if 'norm_prec' in want:
        nctr = torch.tensor(eval_data['ave_success_rate_plot_center_norm']) 
        out['Norm Precision'] = (nctr[mask,:,:].mean(0) * 100.0)[:, 20] if mask.any() else torch.zeros(T)

    out['num_seqs'] = int(mask.sum().item())
    return out

def _emit_scene_table(scene_name, tracker_disp_names, scores):
    print(f"=== Scene: {scene_name} | Seqs used: {scores['num_seqs']} ===")
    header = f"{'Tracker':<20} | {'AUC':>7} | {'Precision@20':>12} | {'NormPrec@0.2':>13}"
    print(header)
    for i, name in enumerate(tracker_disp_names):
        auc  = scores['AUC'][i].item() if 'AUC' in scores else 0.0
        pre  = scores['Precision'][i].item() if 'Precision' in scores else 0.0
        npre = scores['Norm Precision'][i].item() if 'Norm Precision' in scores else 0.0
        print(f"{name:<20} | {auc:7.2f} | {pre:12.2f} | {npre:13.2f}")
    print()

def print_results_by_scene(trackers, dataset, dataset_root, report_name, merge_results=False, plot_types=('success','prec','norm_prec')):
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, force_evaluation=False)
    if merge_results:
        eval_data = merge_multiple_runs(eval_data)

    valid = torch.tensor(eval_data['valid_sequence']).bool()
    tracker_disp_names = [get_tracker_display_name(trk) for trk in eval_data['trackers']]
    seq_names = eval_data.get('sequences', [])
    
    scene_mat = _force_read_scene_matrix(dataset_root, seq_names)

    for sid, sname in enumerate(SCENE_NAMES):
        scene_mask = (scene_mat[:, sid] == 1)
        mask = np.logical_and(scene_mask, valid.numpy())
        scores = _score_from_eval_data(eval_data, mask, want=plot_types)
        _emit_scene_table(sname, tracker_disp_names, scores)

    all_scores = _score_from_eval_data(eval_data, valid.numpy(), want=plot_types)
    _emit_scene_table("Overall (valid only)", tracker_disp_names, all_scores)

# ========================================================================
# 4. Plotting Sub-functions
# ========================================================================
def _annotate_rank_box(ax, tracker_names, scores, plot_styles, topk=24, fontsize=14):
    order = torch.argsort(scores, descending=True)
    if order.numel() == 0: return
    k = min(topk, order.numel())
    inset = ax.inset_axes([0.54, 0.02, 0.45, 0.95])
    inset.set_xticks([]); inset.set_yticks([]); inset.set_xlim(0, 1); inset.set_ylim(0, 1)
    inset.patch.set_facecolor((0.88, 0.94, 1.0, 0.82)); inset.patch.set_edgecolor('#5f8ab2'); inset.patch.set_linewidth(1.3)
    spacing = 1.0 / max(k + 1, 2)
    for i, idx in enumerate(order[:k]):
        y = 1.0 - (i + 1) * spacing
        style = plot_styles[idx % len(plot_styles)]
        inset.plot([0.05, 0.18], [y, y], color=style['color'], linestyle=style['line_style'], linewidth=2.0)
        inset.text(0.20, y, f"[{scores[idx].item():.3f}] {tracker_names[idx]}", ha='left', va='center', fontsize=fontsize, fontweight=LEGEND_WEIGHT, color='#1b1b1b')

def _metric_scores_from_curve(curves, metric_cfg):
    if metric_cfg['score_type'] == 'auc': return curves.mean(-1)
    if metric_cfg['score_type'] == 'index':
        idx = max(0, min(int(metric_cfg.get('score_index', 0)), curves.shape[-1] - 1))
        return curves[:, idx]
    raise ValueError(f"Unknown score_type: {metric_cfg['score_type']}")

def _prepare_metric_tensors(eval_data, metrics):
    metric_tensors, thresholds = {}, {}
    for key in metrics:
        cfg = METRIC_CONFIG[key]
        metric_tensors[key] = torch.tensor(eval_data[cfg['array_field']])
        thresholds[key] = torch.tensor(eval_data[cfg['threshold_field']])
    return metric_tensors, thresholds

def plot_scene_curves(trackers, dataset, dataset_root, report_name, merge_results=False, plot_types=('success', 'prec'), rank_box_topk=27):
    metrics = [m for m in plot_types if m in METRIC_CONFIG]
    if len(metrics) == 0: return

    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, force_evaluation=False)
    if merge_results: eval_data = merge_multiple_runs(eval_data)

    valid = torch.tensor(eval_data['valid_sequence']).bool()
    seq_names = eval_data.get('sequences', [])
    scene_mat = _force_read_scene_matrix(dataset_root, seq_names)
    
    if scene_mat.size == 0 or scene_mat.shape[0] != valid.numel(): return

    tracker_disp_names = [get_tracker_display_name(trk) for trk in eval_data['trackers']]
    metric_tensors, thresholds = _prepare_metric_tensors(eval_data, metrics)
    plot_styles = get_plot_draw_styles()
    scene_tensor = torch.tensor(scene_mat, dtype=torch.bool)

    settings = env_settings()
    out_dir = os.path.join(settings.result_plot_path, report_name)
    os.makedirs(out_dir, exist_ok=True)

    for metric_key in metrics:
        cfg = METRIC_CONFIG[metric_key]
        metric_tensor = metric_tensors[metric_key]
        thr = thresholds[metric_key]
        thr_max = float(thr.max().item()) if thr.numel() else 1.0
        if thr_max <= 0: thr_max = 1.0
        thr_norm = (thr / thr_max).tolist() if cfg.get('normalize_x', False) else thr.tolist()

        for sid, scene_name in enumerate(SCENE_NAMES):
            scene_mask = scene_tensor[:, sid]
            mask = scene_mask & valid
            if int(mask.sum().item()) == 0: continue

            fig, ax = plt.subplots(figsize=(8.0, 8.8))
            curves = metric_tensor[mask, :, :].mean(0) 
            scores = _metric_scores_from_curve(curves, cfg)
            draw_order = torch.argsort(scores, descending=False)

            for tid in draw_order:
                style = plot_styles[tid % len(plot_styles)]
                ax.plot(thr_norm, curves[tid].tolist(), color=style['color'], linestyle=style['line_style'], linewidth=2.2)

            ax.set_xlim(cfg['xlim'])
            ylim = (cfg['ylim'][0], min(PRECISION_Y_MAX if metric_key == 'prec' else DEFAULT_Y_MAX, cfg['ylim'][1]))
            ax.set_ylim(ylim)
            ax.set_xticks(AX_TICKS)
            ax.set_yticks([tick for tick in AX_TICKS if tick <= ylim[1] + 1e-6])
            ax.tick_params(labelsize=18, colors='#111111', length=TICK_LENGTH, width=1.2)
            ax.set_xlabel(cfg['xlabel'], fontsize=24, fontweight=LABEL_WEIGHT)
            ax.set_ylabel(cfg['ylabel'], fontsize=24, fontweight=LABEL_WEIGHT)
            ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.8, color='#b7c5d3')
            for spine in ax.spines.values():
                spine.set_linewidth(AXES_LINEWIDTH)
                spine.set_color('#1f1f1f')
            ax.set_facecolor('#fdfefe')
            ax.set_title(f"{cfg['title']} - {scene_name}", fontsize=24, fontweight=TITLE_WEIGHT, pad=12)

            _annotate_rank_box(ax, tracker_disp_names, scores, plot_styles, topk=rank_box_topk, fontsize=14)

            fig.tight_layout()
            safe_scene = scene_name.lower().replace(' ', '_').replace('/', '-')
            fig.savefig(os.path.join(out_dir, f"{metric_key}_{safe_scene}.pdf"), dpi=300, bbox_inches='tight')
            plt.close(fig)

# ========================================================================
# 5. Main Execution
# ========================================================================
if __name__ == '__main__':
    dataset_name = 'int'
    
    # 自动从环境配置中读取 dataset 和 results 目录
    settings = env_settings()
    dataset_root = os.path.join(settings.int_path, 'INT') if not settings.int_path.endswith('INT') else settings.int_path
    all_results_root = settings.results_path

    trackers = []
    # --- Register trackers ---
    trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300', dataset_name=dataset_name, run_ids=None, display_name='OSTrack', result_only=True))
    trackers.extend(trackerlist(name='ROMTrack', parameter_name='large_384_stage2', dataset_name=dataset_name, run_ids=None, display_name='ROMTrack', result_only=True))
    trackers.extend(trackerlist(name='artrack_seq', parameter_name='artrack_seq_large_384_full', dataset_name=dataset_name, run_ids=None, display_name='ARTrack', result_only=True))
    trackers.extend(trackerlist(name='citetrack', parameter_name='vitb_384_mae_ce_32x4_ep300', dataset_name=dataset_name, run_ids=None, display_name='CiteTrack', result_only=True))
    trackers.extend(trackerlist(name='dutrack', parameter_name='dutrack_384_full', dataset_name=dataset_name, run_ids=None, display_name='DUTrack', result_only=True))
    trackers.extend(trackerlist(name='droptrack', parameter_name='vitb_384_mae_ce_32x4_ep300', dataset_name=dataset_name, run_ids=None, display_name='DropTrack', result_only=True))
    trackers.extend(trackerlist(name='grm', parameter_name='vitl_320_ep300', dataset_name=dataset_name, run_ids=None, display_name='GRM', result_only=True))
    trackers.extend(trackerlist(name='jointnlt', parameter_name='swin_b_ep300', dataset_name=dataset_name, run_ids=None, display_name='JointNLT', result_only=True))
    trackers.extend(trackerlist(name='mcitrack', parameter_name='mcitrack_l384', dataset_name=dataset_name, run_ids=None, display_name='MCITrack', result_only=True))
    trackers.extend(trackerlist(name='mixformer_vit_online', parameter_name='baseline_large', dataset_name=dataset_name, run_ids=None, display_name='MixFormer', result_only=True))
    trackers.extend(trackerlist(name='odtrack', parameter_name='baseline_large_300', dataset_name=dataset_name, run_ids=None, display_name='ODTrack', result_only=True))
    trackers.extend(trackerlist(name='sa2va', parameter_name='sa2va1b', dataset_name=dataset_name, run_ids=None, display_name='Sa2VA', result_only=True))
    trackers.extend(trackerlist(name='seqtrack', parameter_name='seqtrack_l384', dataset_name=dataset_name, run_ids=None, display_name='SeqTrack', result_only=True))
    trackers.extend(trackerlist(name='simtrack', parameter_name='baseline', dataset_name=dataset_name, run_ids=None, display_name='SimTrack', result_only=True))
    trackers.extend(trackerlist(name='stark_st', parameter_name='baseline_R101', dataset_name=dataset_name, run_ids=None, display_name='STARK', result_only=True))
    trackers.extend(trackerlist(name='sutrack', parameter_name='sutrack_l384', dataset_name=dataset_name, run_ids=None, display_name='SUTrack', result_only=True))
    trackers.extend(trackerlist(name='tamos', parameter_name='tamos_swin_base', dataset_name=dataset_name, run_ids=None, display_name='TaMOs', result_only=True))
    trackers.extend(trackerlist(name='tomp', parameter_name='tomp101', dataset_name=dataset_name, run_ids=None, display_name='ToMP', result_only=True))
    trackers.extend(trackerlist(name='videolisa', parameter_name='videolisa3b', dataset_name=dataset_name, run_ids=None, display_name='VideoLiSA', result_only=True))
    trackers.extend(trackerlist(name='samurai', parameter_name='samurai_large', dataset_name=dataset_name, run_ids=None, display_name='SAMURAI', result_only=True))
    trackers.extend(trackerlist(name='vlsam2', parameter_name='vlsam2_large', dataset_name=dataset_name, run_ids=None, display_name='VL-SAM2', result_only=True))
    trackers.extend(trackerlist(name='dam4sam', parameter_name='sam21pp_l', dataset_name=dataset_name, run_ids=None, display_name='DAM4SAM', result_only=True))
    trackers.extend(trackerlist(name='sam2', parameter_name='sam2', dataset_name=dataset_name, run_ids=None, display_name='SAM2', result_only=True))
    trackers.extend(trackerlist(name='him2sam', parameter_name='him2sam', dataset_name=dataset_name, run_ids=None, display_name='HiM2SAM', result_only=True))
    trackers.extend(trackerlist(name='ours', parameter_name='ours', dataset_name=dataset_name, run_ids=None, display_name='Ours', result_only=True))

    dataset = get_dataset(dataset_name)

    # ========================== PHASE 1: Table 2 ==========================
    print("\n" + "="*120)
    print("  PHASE 1: Core Metrics Summary (Table 2 - Interact, Resp, Perc, Tracking) ")
    print("="*120)
    print("[INFO] Fetching cached tracking data (please wait if multiprocessing is compiling results)...")
    eval_data = check_and_load_precomputed_results(trackers, dataset, dataset_name, force_evaluation=False)
    eval_data = merge_multiple_runs(eval_data)
    tracking_scores = _extract_tracking_scores(eval_data)

    unified_results = []
    for i, trk in enumerate(trackers):
        disp_name = trk.display_name
        t_auc = tracking_scores['AUC'][i].item()
        t_prec = tracking_scores['Prec'][i].item()
        t_nprec = tracking_scores['NPrec'][i].item()
        
        pred_path = os.path.join(all_results_root, trk.name, trk.parameter_name)
        interact, resp, p_acc, p_prec = evaluate_custom_metrics(dataset_root, pred_path)
        
        unified_results.append({
            'Algorithm': disp_name, 'Interact': interact, 'Resp': resp,
            'Perc_Acc': p_acc, 'Perc_Prec': p_prec,
            'Track_AUC': t_auc, 'Track_Prec': t_prec, 'Track_NPrec': t_nprec
        })

    print(f"{'Algorithm':<20} | {'Interactiveness':>15} | {'Responsiveness':>14} | {'Perc(Acc)':>10} | {'Perc(Prec)':>10} | {'Track(AUC)':>10} | {'Track(Prec)':>11} | {'Track(NPrec)':>12}")
    print("-" * 120)
    for r in unified_results:
        print(f"{r['Algorithm']:<20} | {r['Interact']:>15.2f} | {r['Resp']:>14.2f} | {r['Perc_Acc']:>10.2f} | {r['Perc_Prec']:>10.2f} | {r['Track_AUC']:>10.2f} | {r['Track_Prec']:>11.2f} | {r['Track_NPrec']:>12.2f}")
    print("=" * 120)


    # ========================== PHASE 2: Per-Scene Breakdown ==========================
    print("\n\n" + "="*80)
    print("  PHASE 2: Per-Scene Tracking Breakdown (AUC, Prec, NPrec) ")
    print("="*80)
    print_results_by_scene(trackers, dataset, dataset_root, dataset_name, merge_results=True, plot_types=('success','prec','norm_prec'))


    # ========================== PHASE 3: Plotting curves ==========================
    # print("\n\n" + "="*80)
    # print("  PHASE 3: Plotting and Saving PDF Curves ")
    # print("="*80)
    # plot_scene_curves(trackers, dataset, dataset_root, dataset_name, merge_results=True, plot_types=('success', 'prec'))
    print("\n[SUCCESS] Evaluation perfectly finished! All plots are saved in your output directory.")