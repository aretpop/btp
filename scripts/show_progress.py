"""
scripts/show_progress.py
Print a clean training dashboard from MLflow.
Usage: python scripts/show_progress.py
"""
import mlflow, datetime, sys
import numpy as np

CLASSES = [
    'Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion',
    'Emphysema','Fibrosis','Hernia','Infiltration','Mass',
    'Nodule','Pleural_Thickening','Pneumonia','Pneumothorax'
]

mlflow.set_tracking_uri('sqlite:///mlflow.db')
client = mlflow.tracking.MlflowClient()

exp = client.get_experiment_by_name('ChestXRay14_Phase3_Regional')
if not exp:
    print("No experiment found yet. Training hasn't started logging.")
    sys.exit(0)

runs = client.search_runs(exp.experiment_id,
                          order_by=['start_time DESC'], max_results=10)
active   = [r for r in runs if r.info.status == 'RUNNING']
finished = [r for r in runs if r.info.status == 'FINISHED']

W = 62
print("=" * W)
print(f"  TRAINING DASHBOARD   {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * W)
print(f"  Active runs  : {len(active)}")
print(f"  Finished runs: {len(finished)}")
print()

for r in active + finished:
    status_tag = "RUNNING" if r.info.status == "RUNNING" else "DONE   "
    print(f"[{status_tag}]  {r.info.run_name}")

    auroc_hist = client.get_metric_history(r.info.run_id, 'val_macro_auroc')
    loss_hist  = client.get_metric_history(r.info.run_id, 'loss_total')
    n_epochs   = len(auroc_hist)

    print(f"  Epochs completed : {n_epochs} / 20")

    if auroc_hist:
        valid_aurocs = [m.value for m in auroc_hist if not np.isnan(m.value)]
        last_auroc   = auroc_hist[-1].value
        best_auroc   = max(valid_aurocs) if valid_aurocs else float('nan')
        last_loss    = loss_hist[-1].value if loss_hist else float('nan')

        print(f"  Last epoch AUROC : {last_auroc:.4f}  (epoch {auroc_hist[-1].step})")
        print(f"  Best AUROC       : {best_auroc:.4f}")
        print(f"  Last train loss  : {last_loss:.4f}")

        # Per-class breakdown from last epoch
        per_cls = []
        for cls in CLASSES:
            h = client.get_metric_history(r.info.run_id, f'auroc_{cls}')
            if h:
                per_cls.append((cls, h[-1].value))

        if per_cls:
            print("  Per-class AUROC (last epoch):")
            for cls, auc in sorted(per_cls, key=lambda x: -x[1]):
                bar_len = int(auc * 20) if not np.isnan(auc) else 0
                bar = "#" * bar_len
                print(f"    {cls:<22s}: {auc:.4f}  |{bar:<20s}|")
    else:
        print("  No epochs logged yet — epoch 1 still in progress...")

    print()

print("-" * W)
print("Re-run this script any time to refresh.")
