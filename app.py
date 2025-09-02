# Import necessary libraries
from functions import functions
import streamlit as st 
import torch
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import math
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.monte_carlo import qExpectedImprovement, qProbabilityOfImprovement
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
import warnings
from botorch.optim.initializers import BadInitialCandidatesWarning
from botorch.models.utils.assorted import InputDataWarning
warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=InputDataWarning)
warnings.filterwarnings("ignore", category=UserWarning)  

# Interactive sidebar
st.sidebar.header("Settings")
func_name = st.sidebar.selectbox("Select function", ['Sphere', 'Branin', 'McCormick', 'Rosenbrock'])
acq_name = st.sidebar.selectbox("Acquisition Function", ['EI', 'PI', 'UCB', 'LCB'])
experiments = st.sidebar.slider("Number of BO iterations", 2, 30, 5)

# Load function metadata
with open("functions.json", "r") as f:
    func_data = json.load(f)
name = func_name # change this!
for f in func_data["functions"]:
        if f["name"] == name:
            func_info = f
x1_min, x1_max = func_info["domain"]["x1"]
x2_min, x2_max = func_info["domain"]["x2"]
func = getattr(functions, func_info["name"].lower())
optima = func_info["optima"]["value"]
equation = func_info["equation"]
st.markdown(f"<h2 style='text-align: center;'>{name} Function</h2>", unsafe_allow_html=True)
st.latex(equation)
st.markdown(f"<p style='text-align: center;'><b>Global Maximum:</b> {optima}</p>", unsafe_allow_html=True)

x1 = np.linspace(x1_min, x1_max, 30)
x2 = np.linspace(x2_min, x2_max, 30)
X1, X2 = np.meshgrid(x1, x2)
Z = func(np.column_stack([X1.ravel(), X2.ravel()])).reshape(X1.shape)
# Initialize BO
np.random.seed(123)
initial_points = 1
mc_samples = 256
perf_list = []
col1, col2, col3 = st.columns(3)
surf_plot = col1.empty()
bo_plot = col2.empty()
perf_plot = col3.empty()

fig_surf = plt.figure(figsize=(4,3))
ax = fig_surf.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none')
ax.set_title(f"Surface Plot of {name} Function", fontsize=10)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.text2D(1.1, 0.5, "f(x1,x2)", transform=ax.transAxes, rotation=270,
          va='center', ha='left')

# --- Initialize BO ---
np.random.seed(123)
initial_points = 1
mc_samples = 256
perf_list = []
bounds_tensor = torch.tensor([[x1_min, x2_min], [x1_max, x2_max]], dtype=torch.float)
point1 = x1_min + (x1_max - x1_min) * torch.rand(initial_points)
point2 = x2_min + (x2_max - x2_min) * torch.rand(initial_points)
X_train = torch.stack([point1, point2], dim=1)
Y_train = torch.tensor(func(X_train.numpy())).unsqueeze(-1)

st.markdown("Bayesian Optimization Progress")
col1, col2, col3 = st.columns(3)
surf_placeholder = col1.empty()
bo_placeholder   = col2.empty()
perf_placeholder = col3.empty()
surf_placeholder.pyplot(fig_surf)

fig_cont = plt.figure(figsize=(4,3))
plt.contourf(X1, X2, Z, levels=30, cmap='viridis')
plt.colorbar(label='f(x1,x2)')
plt.scatter(X_train[:,0], X_train[:,1], color='red', edgecolor='black', s=80)
plt.scatter(0,0, color='yellow', marker='*', s=150, edgecolor='black')  # global optimum
plt.title("Initial Points")
plt.xlabel("x1")
plt.ylabel("x2")
bo_placeholder.pyplot(fig_cont)

# BO loop
for i in range(initial_points, experiments):
    # Fit GP
    gp = SingleTaskGP(X_train, Y_train)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    # Acquisition function
    if acq_name == "EI":
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        acq = qExpectedImprovement(model=gp, best_f=Y_train.max().item(), sampler=sampler)
    elif acq_name == "PI":
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        acq = qProbabilityOfImprovement(model=gp, best_f=Y_train.max().item(), sampler=sampler)
    elif acq_name == "UCB":
        acq = UpperConfidenceBound(model=gp, beta=0.1)
    elif acq_name == "LCB":
        acq = UpperConfidenceBound(model=gp, beta=-0.1)
    # Optimize acquisition
    X_next, _ = optimize_acqf(acq, bounds=bounds_tensor, q=1, num_restarts=3, raw_samples=64)
    Y_next = torch.tensor(func(X_next.numpy())).unsqueeze(-1)
    # Update data
    X_train = torch.cat([X_train, X_next], dim=0)
    Y_train = torch.cat([Y_train, Y_next], dim=0)
    perf = float(Y_next) / float(optima)
    perf_list.append(perf)
    fig_cont = plt.figure(figsize=(4,3))
    plt.contourf(X1, X2, Z, levels=30, cmap='viridis')
    plt.colorbar(label='f(x1,x2)')
    all_pts = X_train.numpy()
    plt.scatter(all_pts[:,0], all_pts[:,1], color='white', edgecolor='black', s=80)
    plt.scatter(0,0, color='yellow', marker='*', s=150, edgecolor='black')
    for idx, (xval, yval) in enumerate(all_pts):
        plt.text(xval, yval, str(idx+1), color='black', fontsize=8, ha='center', va='bottom')
    plt.title("BO Campaign")
    plt.xlabel("x1")
    plt.ylabel("x2")
    bo_placeholder.pyplot(fig_cont)
    fig_perf = plt.figure(figsize=(4,3))
    plt.plot(range(1, len(perf_list)+1), perf_list, marker='o', label='BO Performance')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Optimum')
    plt.xlabel("Iteration")
    plt.ylabel("Performance")
    plt.title("BO Performance")
    plt.legend()
    perf_placeholder.pyplot(fig_perf)
    # Stop early if optimum reached
    if float(Y_next) >= optima:
        st.success(f"Optima Reached at experiment {i+1}!")
        break
