from __future__ import annotations
import numpy as np, torch
from .model import DirectLSTM

def fit_predict(train, pred, candidate, cfg, seed=42):
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(int(cfg["n_jobs"]))
    device=torch.device("cpu"); model=DirectLSTM(candidate["hidden_size"],candidate["num_layers"],candidate["dropout"]).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=candidate["learning_rate"],weight_decay=cfg["training"]["weight_decay"]); bs=int(cfg["training"]["batch_size"]); x=torch.tensor(train["sequences"],device=device); c=torch.tensor(train["contexts"],device=device); y=torch.tensor(np.nan_to_num(train["targets"],nan=0),device=device); m=torch.tensor(train["loss_mask"],device=device); best=None; stale=0
    model.train()
    for _ in range(int(candidate["epochs"])):
        order=torch.randperm(len(x)); losses=[]
        for ids in order.split(bs):
            opt.zero_grad(); z=model(x[ids],c[ids]); loss=((z-y[ids])**2*m[ids]).sum()/m[ids].sum().clamp_min(1); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.detach()))
        cur=float(np.mean(losses))
        if best is None or cur<best[0]-1e-7: best=(cur,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}); stale=0
        else: stale+=1
        if stale>=int(cfg["training"]["patience"]): break
    if best: model.load_state_dict(best[1])
    model.eval();
    with torch.no_grad(): out=model(torch.tensor(pred["sequences"],device=device),torch.tensor(pred["contexts"],device=device)).cpu().numpy()
    daylight=pred["target_daylight"]
    out=np.maximum(out,0.0); out[~daylight]=0.0
    return out,model
