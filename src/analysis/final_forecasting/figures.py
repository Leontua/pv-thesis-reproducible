"""Chapter-ready SVG figures."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["persistence", "sarima", "rf", "lstm"]
LABELS = {"persistence": "Persistence", "sarima": "SARIMA", "rf": "Random Forest", "lstm": "LSTM v1.1"}
COLORS = {"persistence": "#8c8c8c", "sarima": "#4c78a8", "rf": "#59a14f", "lstm": "#e15759"}


def _save(fig, path: Path):
    fig.tight_layout(); fig.savefig(path, format="svg", bbox_inches="tight"); plt.close(fig); return path


def create_all(folder: Path, site_macro: pd.DataFrame, by_site: pd.DataFrame, by_horizon: pd.DataFrame, by_season: pd.DataFrame, bootstrap: pd.DataFrame):
    folder.mkdir(parents=True, exist_ok=True); paths=[]
    data=site_macro.set_index("model").loc[MODELS]
    fig,ax=plt.subplots(figsize=(7.5,4.5)); bars=ax.bar([LABELS[m] for m in MODELS],data.nRMSE_percent,[0.7]*4,color=[COLORS[m] for m in MODELS]); ax.bar_label(bars,fmt="%.2f%%",padding=3); ax.set_ylabel("Site-macro nRMSE (%)"); ax.set_ylim(0,data.nRMSE_percent.max()*1.2); ax.grid(axis="y",alpha=.2); paths.append(_save(fig,folder/"01_site_macro_nrmse.svg"))

    pivot=by_site.pivot(index="site_id",columns="model",values="nRMSE_percent").loc[:,MODELS]
    x=np.arange(len(pivot)); width=.2; fig,ax=plt.subplots(figsize=(12,5.5))
    for i,m in enumerate(MODELS): ax.bar(x+(i-1.5)*width,pivot[m],width,label=LABELS[m],color=COLORS[m])
    ax.set_xticks(x,pivot.index,rotation=40,ha="right"); ax.set_ylabel("nRMSE (%)"); ax.legend(ncol=4,frameon=False); ax.grid(axis="y",alpha=.2); paths.append(_save(fig,folder/"02_nrmse_by_site.svg"))

    fig,ax=plt.subplots(figsize=(9,4.8))
    for m in MODELS:
        part=by_horizon[by_horizon.model==m]; ax.plot(part.horizon,part.nRMSE_percent,marker="o",ms=3,label=LABELS[m],color=COLORS[m])
    ax.set_xlabel("Forecast horizon (hours ahead)"); ax.set_ylabel("Site-macro nRMSE (%)"); ax.set_xticks([1,4,8,12,16,20,24]); ax.legend(ncol=2,frameon=False); ax.grid(alpha=.2); paths.append(_save(fig,folder/"03_nrmse_by_horizon.svg"))

    order=[s for s in ["DJF","MAM","JJA","SON"] if s in set(by_season.season)]; pivot=by_season.pivot(index="season",columns="model",values="nRMSE_percent").loc[order,MODELS]; x=np.arange(len(order)); fig,ax=plt.subplots(figsize=(8,4.8))
    for i,m in enumerate(MODELS): ax.bar(x+(i-1.5)*width,pivot[m],width,label=LABELS[m],color=COLORS[m])
    ax.set_xticks(x,order); ax.set_ylabel("Site-macro nRMSE (%)"); ax.legend(ncol=4,frameon=False); ax.grid(axis="y",alpha=.2); paths.append(_save(fig,folder/"04_nrmse_by_season.svg"))

    macro=bootstrap[bootstrap.scope=="site_macro"].copy(); labels=[f"{LABELS[a]} − {LABELS[b]}" for a,b in zip(macro.model_a,macro.model_b)]; y=np.arange(len(macro)); xerr=np.vstack([macro.delta_nRMSE_pp_a_minus_b-macro.ci95_low,macro.ci95_high-macro.delta_nRMSE_pp_a_minus_b]); fig,ax=plt.subplots(figsize=(8,4.3)); ax.errorbar(macro.delta_nRMSE_pp_a_minus_b,y,xerr=xerr,fmt="o",capsize=5,color="#333333"); ax.axvline(0,color="#9c0006",ls="--"); ax.set_yticks(y,labels); ax.set_xlabel("Paired ΔnRMSE (percentage points); negative favors first model"); ax.grid(axis="x",alpha=.2); paths.append(_save(fig,folder/"05_bootstrap_site_macro_ci.svg"))

    ranks=by_site.pivot(index="site_id",columns="model",values="nRMSE_percent").loc[:,MODELS].rank(axis=1,method="min"); fig,ax=plt.subplots(figsize=(7.5,5.5)); im=ax.imshow(ranks.to_numpy(),aspect="auto",cmap="YlGn_r",vmin=1,vmax=4); ax.set_xticks(np.arange(4),[LABELS[m] for m in MODELS],rotation=20,ha="right"); ax.set_yticks(np.arange(len(ranks)),ranks.index)
    for i in range(len(ranks)):
        for j in range(4): ax.text(j,i,str(int(ranks.iloc[i,j])),ha="center",va="center")
    ax.set_title("Model rank by physical site (1 = best)"); fig.colorbar(im,ax=ax,label="Rank"); paths.append(_save(fig,folder/"06_model_rank_by_site.svg")); return paths
