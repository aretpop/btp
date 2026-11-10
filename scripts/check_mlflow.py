import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
client = mlflow.tracking.MlflowClient()
runs = mlflow.search_runs(experiment_names=['ChestXRay14_Phase2_Attention'], order_by=['start_time DESC'])
print('=== Phase 2 Attention Runs ===')
for _, run in runs.iterrows():
    hist = client.get_metric_history(run.run_id, 'val_macro_auroc')
    best = max([m.value for m in hist]) if len(hist)>0 else 0
    print(f"{run.get('tags.mlflow.runName', 'unknown')}: Best Val AUROC = {best:.4f}")
