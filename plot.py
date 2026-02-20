import os
import glob 
import numpy as np
from omegaconf import OmegaConf
from argparse import ArgumentParser
from common.extra_plot import try_load_pkl, mean_ci95, to_matrix

import matplotlib.pyplot as plt
FONT_SIZE = 28
plt.rcParams.update({"font.size": FONT_SIZE//2})
plt.rcParams['xtick.labelsize'] = FONT_SIZE
plt.rcParams['ytick.labelsize'] = FONT_SIZE
plt.style.use('seaborn-v0_8-notebook')


argparse = ArgumentParser()
argparse.add_argument('--results_dir', type=str, default='res', help="path to output directory")
argparse.add_argument('--run_name', type=str, required=True, help="matches that used in train.py,evaluate.py")
argparse.add_argument('--trials', type=int, required=True, help="number of trials to display")
argparse.add_argument('--outdir', type=str, default='res/plots', help="save location for results")
argparse.add_argument('--single_run', action='store_true', help="flag to mark single-job submitit runs (ran without -m, multirun)")
argparse.add_argument('--eval_only', action='store_true', help="flag to mark evaulation runs")


TRAIN_METRICS = ['train_rewards', 'train_total_loss', 'train_val_loss', 'train_pol_loss', 'train_ent_loss']
TEST_METRICS = ['eval_rewards', 'eval_accuracy']
VERBOSE = True

TRAIN_METRICS.append('eval_generator_loss') # uncomment for debugging policy/generator net interaction
#TEST_METRICS.append('eval_ys')
#TEST_METRICS.append('eval_logs')

def plot_multi_dim(ax, X, Y, ci, method_name):
    ax[0].plot(X, Y[:,0], label=method_name, linewidth=2)
    for dim in range(min(Y.shape[1], 10)):
        if dim > 0:
            ax[dim].plot(X, Y[:,dim], linewidth=2)

        ax[dim].tick_params(axis='both', which='major', labelsize=FONT_SIZE//2)
        ax[dim].ticklabel_format(style='sci', axis='x', scilimits=(-4,4))
        ax[dim].fill_between(X, Y[:,0] + ci[:,0], Y[:,0] - ci[:,0], alpha=0.25)
        ax[dim].grid(color='grey', linestyle='--', linewidth=0.7, alpha=0.4)



if __name__ == '__main__':
    args = argparse.parse_args()

    out_path = os.path.join(args.outdir, args.run_name.replace(',', '_'))
    os.makedirs(out_path, exist_ok=True)

    loaded_results_dirs = set()
    unloaded_results_dirs = set()
    for run_name in args.run_name.split(','):
        results_path = os.path.join(args.results_dir, run_name) + '/'
        if args.single_run:
            unloaded_results_dirs.update(set([results_path[:-1]]))
        else:
            # skip dummy env for hydra runs
            unloaded_results_dirs.update(set([x[:-1] for x in glob.glob(results_path + "*/") if 'cfg_default' not in x]))

    import os
    a = os.getcwd()
    plot_intervals = dict()
    plot_horizons = dict()
    METRICS = TEST_METRICS if args.eval_only else TRAIN_METRICS + TEST_METRICS
    results = dict()
    results.update({ metric: dict() for metric in METRICS })
    for env_dir in unloaded_results_dirs:
        # load metrics grouped by subdirs in worker dir 
        env_name = os.path.basename(env_dir).capitalize()
        for method_cfg in glob.glob(os.path.join(env_dir, "**/config.yaml"), recursive=True):
            method_dir = os.path.dirname(method_cfg)
            cfg = OmegaConf.load(method_cfg)
            method = os.path.basename(method_dir)
            if not args.single_run:
                method = f"{method}_{os.path.basename(os.path.dirname(method_dir))}"

            method_env = (method, env_name)
            print(f"loading results for {method_env[0]} in {method_env[1]}")
            plot_intervals[method_env] = cfg.test_interval
            plot_horizons[method_env] = cfg.horizon
            for metric in METRICS:
                # stack method trials  (for eval runs, test metrics are stored in last_trial+1.pkl)
                results[metric][method_env] = list()
                min_trial, max_trial = (args.trials, args.trials+1) if args.eval_only else (0,args.trials)
                for t in range(min_trial, max_trial):
                    results[metric][method_env].append(try_load_pkl(os.path.join(method_dir, f"t-{t}_{metric}.pkl")))
            # count as loaded if all trials are loaded
            if all([len(results[m][method_env]) == (max_trial - min_trial) for m in METRICS]):
                loaded_results_dirs.add(env_dir)

    if len(loaded_results_dirs) == 0:
        print("no results loaded")
        exit(-1) 

    print(f"saving plots to {out_path}")
    with open(os.path.join(out_path, "log.txt"), "w") as fl:
        lrd = set([xx for x in loaded_results_dirs for xx in glob.glob(os.path.join(x, "**/config.yaml"), recursive=True)])
        urd = set([xx for x in unloaded_results_dirs for xx in glob.glob(os.path.join(x, "**/config.yaml"), recursive=True)])
        fl.write("loaded results:\n"+"\n".join(lrd))
        fl.write("\nunloaded results:\n" + "\n".join(urd - lrd))

    styles = ['solid', 'dashdot', 'dashed', 'dotted']

    METRICS = TEST_METRICS if args.eval_only else TRAIN_METRICS + TEST_METRICS
    for env_dir in loaded_results_dirs:
        # plot and save each metric
        env_name = os.path.basename(env_dir).capitalize()
        print(f"saving results for {env_name}")

        for metric in METRICS:
            env_results = [(k, to_matrix(results[metric][k])) for k in results[metric].keys() if k[1] == env_name]
            max_vec_size = max([Y.shape[2] if len(Y.shape) > 2 else 1 for _, Y in env_results]) # trials x steps x dims
            max_vec_size = min(max_vec_size, 10) if metric in TEST_METRICS else 1

            fig, ax = plt.subplots(1, max_vec_size, figsize=(5*max_vec_size, 5), sharex=True)
            plt.xlabel("Samples", fontsize=FONT_SIZE)
            plt.ylabel(metric.split('_')[1].replace('_', '\n').capitalize(), fontsize=FONT_SIZE)
            if max_vec_size == 1: ax = np.array(ax)
            ax = ax.reshape(-1)
            for j, (method_env, res) in enumerate(reversed(sorted(env_results, key=lambda x: x[0][0]))):
                if len(res) == 0:
                    print(f"no result for {method_env[1]}")
                    continue
                method_name, env_name = method_env
                method_name = method_name.replace('_', ' ')

                Y, ci, n_samples = mean_ci95(res)
                interval = plot_intervals[method_env] if metric in TEST_METRICS else 1 
                X = np.linspace(0, Y.shape[0] * interval * plot_horizons[method_env], Y.shape[0])
                if VERBOSE:
                    method_name += f" (n={n_samples})"
                if n_samples == 0:
                    continue

                plot_multi_dim(ax, X, Y, ci, method_name.replace(' ', '\n'))

            fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), fancybox=False, ncol=len(env_results), fontsize=FONT_SIZE//2)
            plt.xticks(fontsize=FONT_SIZE//2)
            plt.ticklabel_format(style='sci', axis='x', scilimits=(-4,4))
            if 'accuracy' in metric: plt.ylim(0, 1)
            [spv.set_visible(False) for spv in plt.gca().spines.values()]
            plt.gcf().tight_layout(pad=0.2)
            plt.savefig(os.path.join(out_path, f"{env_name}_{metric}.png"), bbox_inches='tight')
            plt.close()