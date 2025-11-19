import os
import glob 
import pickle
import numpy as np
from copy import deepcopy
from omegaconf import OmegaConf
from argparse import ArgumentParser

import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 18})


argparse = ArgumentParser()
argparse.add_argument('--run_name', type=str, required=True)
argparse.add_argument('--trials', type=int, required=True)
argparse.add_argument('--results_dir', type=str, default='./res')
argparse.add_argument('--outdir', type=str, default='./res/plots')
argparse.add_argument('--single_run', action='store_true')
argparse.add_argument('--eval_only', action='store_true')


SENTINEL = -float('inf') # do not set to nan
VERBOSE = False 
VERBOSE = True
ERROR_METRICS = ['eval_accuracy']
GT_METRICS = ['eval_actual', 'eval_predicted']

# fn that loads and plots metrics eval_rewards, train_rewards, val_loss, etc.

def try_load_pkl(path):
    try:
        return pickle.load(open(path, "rb"))
    except Exception as e:
        print(f"error loading {path}: {e}")
        return []

def mean_ci95(array, dim=0):
    if array.size == 0: return np.zeros(1,), np.zeros(1,), 0
    n_filtered = np.count_nonzero(array != SENTINEL, axis=dim)
    mean = np.mean(array, axis=dim, where=array != SENTINEL)
    std = np.std(array, axis=dim, where=array != SENTINEL)
    if len(mean.shape) == 1: mean = mean.reshape(-1, 1)
    if len(std.shape) == 1: std = std.reshape(-1, 1)
    return mean, 1.96 * std / np.sqrt(n_filtered), n_filtered.min()

def dim_sum(array, dim=0):
    return np.sum(array, axis=0, where=(array != SENTINEL), keepdims=True)

def dim_max(array, dim=0):
    return np.max(array, axis=0, where=(array != SENTINEL), keepdims=True)

def list_shape(ll):
    # compute shape as extent of each dimension in nested lists
    return max(list(map(list_shape,ll)), default=[]) + [len(ll)] if isinstance(ll,list) or isinstance(ll,np.ndarray) else [1]

def to_matrix(lists):
    try:
        return np.array(lists)
    except:
        shape = list_shape(lists)
        shape.reverse()

        dim = 0 
        lists_filled = deepcopy(lists)
        stak = [lists_filled]
        while len(stak) > 0:
            # process dimension in stack
            for _ in range(len(stak)):
                d = stak.pop(0)
                if (shape[dim] - len(d)) > 0:
                    # fill missing scalars or vectors according to size of next dim
                    d.extend((shape[dim] - len(d))*([SENTINEL] if shape[dim+1] == 1 else [[SENTINEL]])) 
                # recurse inner dimensions
                stak.extend([inner for inner in d if isinstance(inner, list)])
            dim += 1

        return np.array(lists_filled)


if __name__ == '__main__':
    args = argparse.parse_args()

    out_path = os.path.join(args.outdir, args.run_name.replace(',', '_'))
    os.makedirs(out_path, exist_ok=True)

    loaded_results_count = 0
    unloaded_results_dirs = list()
    for run_name in args.run_name.split(','):
        results_path = os.path.join(args.results_dir, run_name)
        unloaded_results_dirs.extend([results_path])

    METRICS = ERROR_METRICS + GT_METRICS
    intervals = dict()
    results = dict()
    results.update({ metric: dict() for metric in METRICS })
    for env_dir in unloaded_results_dirs:
        # load metrics grouped by subdirs in worker dir 
        env_name = os.path.basename(env_dir).capitalize()
        for method_cfg in glob.glob(os.path.join(env_dir, "**/config.yaml"), recursive=True): 
            method_dir = os.path.dirname(method_cfg)
            method = os.path.basename(method_dir)
            if not args.single_run:
                method = f"{method}"

            method_env = (method, env_name)
            intervals[method_env] = OmegaConf.load(method_cfg).test_interval
            print(f"loading results for {method_env[0]} in {method_env[1]}")
            for metric in METRICS:
                # stack method trials  (for eval runs, test metrics are stored in last_trial+1.pkl)
                results[metric][method_env] = list()
                min_trial, max_trial = (args.trials, args.trials+1) if args.eval_only else (0,args.trials) 
                for t in range(min_trial, max_trial):
                    results[metric][method_env].append(try_load_pkl(os.path.join(method_dir, f"t-{t}_{metric}.pkl")))
            # count as loaded if all trials are loaded
            loaded_results_count += int(not any([None in results[m][method_env] for m in METRICS]))

    if loaded_results_count == 0:
        print("no results loaded")
        exit(-1) 

    print(f"saving plots to {out_path}")
    styles = ['solid', 'dashdot', 'dashed', 'dotted']
    for env_dir in unloaded_results_dirs:
        # plot and save each metric
        env_name = os.path.basename(env_dir).capitalize()
        print(f"saving results for {env_name}")

        for metric in METRICS:
            env_results = [(k, to_matrix(results[metric][k])) for k in results[metric].keys() if k[1] == env_name]
            max_vec_size = np.max([Y.shape[1] if len(Y.shape) > 1 else 1 for _, Y in env_results]) # trials x steps x dims
            assert max_vec_size > 0, f"missing some results for {metric}"

            if max_vec_size == 1: ax = np.array([ax])
            for (method_env, res) in sorted(env_results, key=lambda x: x[0][0]):
                method_name, env_name = method_env
                method_name = method_name.replace('_', ' ')

                interval = intervals[method_env] if metric in ERROR_METRICS else 1
                Y, ci, n_samples = mean_ci95(res)
                X = np.linspace(0, Y.shape[0] * interval, Y.shape[0])
                X_intervals = np.arange(0, Y.shape[0] * interval, intervals[method_env])
                if VERBOSE:
                    method_name += f" (n={n_samples})"

                plt.grid(color='grey', axis='y', linestyle='--', linewidth=0.7, alpha=0.4)
                plt.vlines(x=X_intervals, ymin=np.min(Y[:,0]), ymax=np.max(Y[:,0]), color='grey', linestyle='--', alpha=0.2)
                plt.plot(X, Y[:,0], label=f"{method_name}", linewidth=3)
                plt.fill_between(X, Y[:,0] + ci[:,0], Y[:,0] - ci[:,0], alpha=0.25)

            #plt.title(f"{env_name} {metric_name}")
            metric_name = metric.split('_')[1].replace('_', ' ').capitalize()
            plt.xlabel("Samples")
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1+0.1*len(env_results)), fancybox=True, ncol=1)
            plt.ylabel(metric_name)
            [spv.set_visible(False) for spv in plt.gca().spines.values()]
            plt.gcf().tight_layout(pad=0.2)
            plt.savefig(os.path.join(out_path, f"{env_name}_{metric}.png"), bbox_inches='tight')
            plt.close()

        if not GT_METRICS[0] in results:
            continue 

        env_results = [(k, to_matrix(results[GT_METRICS[0]][k])) for k in results[GT_METRICS[0]].keys() if k[1] == env_name]
        max_vec_size = max([Y.shape[2] if len(Y.shape) > 2 else 1 for _, Y in env_results]) # trials x steps x dims

        fig, ax = plt.subplots(max_vec_size, len(env_results), figsize=(5*len(env_results),5*max_vec_size))
        if max_vec_size == 1: ax = np.array(ax).reshape(1, -1)
        if len(ax.shape) == 1: ax = np.array(ax).reshape(-1, 1)
        for i, metric in enumerate(sorted(GT_METRICS)):
            for j, method_env in enumerate(sorted([k for k in results[metric].keys() if k[1] == env_name], key=lambda x: x[0])):
                method_name, env_name = method_env
                res = to_matrix(results[metric][method_env])

                Y, ci, n_samples = mean_ci95(res)
                X = np.linspace(0, Y.shape[0], Y.shape[0])
                X_intervals = np.arange(0, Y.shape[0], intervals[method_env])

                for dim in range(Y.shape[1]):
                    ax[dim, j].vlines(x=X_intervals, ymin=np.min(Y[:,dim]), ymax=np.max(Y[:,dim]), color='grey', linestyle='--', alpha=0.2)
                    ax[dim, j].plot(X, Y[:,dim], label=metric.lower(), linewidth=3, linestyle=styles[i])
                    ax[dim, j].fill_between(X, Y[:,dim] + ci[:,dim], Y[:,dim] - ci[:,dim], alpha=0.25)
                    # label cols 
                    ax[dim, j].set_xlabel("Samples")
                    ax[dim, j].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                ax[0, j].set_title(method_name.replace('_', ' '))
                ax[0, j].legend(loc='upper center', bbox_to_anchor=(0.75, 1), fancybox=True, ncol=1)
        # label rows 
        [ax[dim, 0].set_ylabel(f"cue[{dim}]") for dim in range(max_vec_size)]
        [spv.set_visible(False) for iax in ax.ravel() for spv in iax.spines.values()]
        plt.gcf().tight_layout(pad=0.2)
        plt.savefig(os.path.join(out_path, f"{env_name}_gt_metrics.png"), bbox_inches='tight')
        plt.close()

