"""LSTM v1.1 training with a real inner temporal validation window."""
from __future__ import annotations
import copy
import numpy as np
import torch
from .model import DirectLSTM

def _tensors(data, device):
    return (
        torch.tensor(data["sequences"], dtype=torch.float32, device=device),
        torch.tensor(data["contexts"], dtype=torch.float32, device=device),
        torch.tensor(np.nan_to_num(data["targets"], nan=0.0), dtype=torch.float32, device=device),
        torch.tensor(data["loss_mask"], dtype=torch.float32, device=device),
    )

def _masked_mse(pred, target, mask):
    return ((pred-target).square()*mask).sum()/mask.sum().clamp_min(1.0)

def _predict(model, data, device):
    model.eval()
    with torch.no_grad():
        x,c,_,_=_tensors(data,device); result=model(x,c).cpu().numpy()
    result=np.maximum(result,0.0); result[~data["daylight"]]=0.0
    return result

def fit_early_stopping(train, inner_validation, predict_data, candidate, cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(int(cfg["n_jobs"])); device=torch.device("cpu")
    model=DirectLSTM(candidate["hidden_size"],candidate["num_layers"],candidate["dropout"]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=candidate["learning_rate"],weight_decay=cfg["training"]["weight_decay"])
    x,c,y,m=_tensors(train,device); vx,vc,vy,vm=_tensors(inner_validation,device); batch=int(cfg["training"]["batch_size"])
    best_loss=float("inf"); best_epoch=0; best_state=None; stale=0; history=[]
    for epoch in range(1,int(cfg["training"]["max_epochs"])+1):
        model.train(); batch_losses=[]
        generator=torch.Generator().manual_seed(seed+epoch); order=torch.randperm(len(x),generator=generator)
        for ids in order.split(batch):
            optimizer.zero_grad(); output=model(x[ids],c[ids]); loss=_masked_mse(output,y[ids],m[ids]); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg["training"]["gradient_clip_norm"])); optimizer.step(); batch_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad(): validation_loss=float(_masked_mse(model(vx,vc),vy,vm).cpu())
        training_loss=float(np.mean(batch_losses)); history.append({"epoch":epoch,"training_loss":training_loss,"inner_validation_loss":validation_loss})
        if validation_loss < best_loss-float(cfg["training"]["minimum_delta"]): best_loss=validation_loss; best_epoch=epoch; best_state=copy.deepcopy(model.state_dict()); stale=0
        else: stale+=1
        if stale>=int(cfg["training"]["patience"]): break
    if best_state is None: raise RuntimeError("No valid inner-validation epoch")
    model.load_state_dict(best_state)
    return _predict(model,predict_data,device),model,history,best_epoch,best_loss

def fit_fixed_epochs(train, predict_data, candidate, cfg, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(int(cfg["n_jobs"])); device=torch.device("cpu")
    model=DirectLSTM(candidate["hidden_size"],candidate["num_layers"],candidate["dropout"]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=candidate["learning_rate"],weight_decay=cfg["training"]["weight_decay"])
    x,c,y,m=_tensors(train,device); batch=int(cfg["training"]["batch_size"]); history=[]
    for epoch in range(1,int(epochs)+1):
        model.train(); losses=[]; generator=torch.Generator().manual_seed(seed+epoch); order=torch.randperm(len(x),generator=generator)
        for ids in order.split(batch):
            optimizer.zero_grad(); output=model(x[ids],c[ids]); loss=_masked_mse(output,y[ids],m[ids]); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg["training"]["gradient_clip_norm"])); optimizer.step(); losses.append(float(loss.detach()))
        history.append({"epoch":epoch,"training_loss":float(np.mean(losses)),"inner_validation_loss":np.nan})
    return _predict(model,predict_data,device),model,history
