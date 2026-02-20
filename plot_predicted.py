import os
import glob 
import numpy as np
from omegaconf import OmegaConf
from argparse import ArgumentParser
from common.extra_plot import try_load_pkl, mean_ci95, to_matrix

import matplotlib.pyplot as plt
FONT_SIZE = 28
plt.rcParams.update({"font.size": FONT_SIZE//2})


argparse = ArgumentParser()
argparse.add_argument('--results_dir', type=str, default='res', help="path to output directory")
argparse.add_argument('--run_name', type=str, required=True, help="matches that used in train.py,evaluate.py")
argparse.add_argument('--trials', type=int, required=True, help="number of trials to display")
argparse.add_argument('--outdir', type=str, default='res/plots', help="save location for results")
argparse.add_argument('--seek', type=int, default=10000, help="starting timestep")
argparse.add_argument('--window', type=int, default=10000, help="size of time window before seek location")
argparse.add_argument('--single_run', action='store_true', help="flag to mark single-job submitit runs (ran without -m, multirun)")
argparse.add_argument('--eval_only', action='store_true', help="flag to mark evaulation runs")


GT_METRICS = ['eval_actual', 'eval_predicted']
VERBOSE = True

if __name__ == '__main__':
    args = argparse.parse_args()

    out_path = os.path.join(args.outdir, args.run_name.replace(',', '_'))
    os.makedirs(out_path, exist_ok=True)

    loaded_results_count = 0
    unloaded_results_dirs = list()
    for run_name in args.run_name.split(','):
        results_path = os.path.join(args.results_dir, run_name) + '/'
        if args.single_run:
            unloaded_results_dirs.extend([results_path[:-1]])
        else:
            # skip dummy env for hydra runs
            unloaded_results_dirs.extend([x[:-1] for x in glob.glob(results_path + "*/") if 'cfg_default' not in x])

    import os
    a = os.getcwd()
    plot_intervals = dict()
    plot_horizons = dict()
    results = dict()
    results.update({ metric: dict() for metric in GT_METRICS })
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
            for metric in GT_METRICS:
                # stack method trials  (for eval runs, test metrics are stored in last_trial+1.pkl)
                results[metric][method_env] = list()
                min_trial, max_trial = (args.trials, args.trials+1) if args.eval_only else (0,args.trials)
                for t in range(min_trial, max_trial):
                    results[metric][method_env].append(try_load_pkl(os.path.join(method_dir, f"t-{t}_{metric}.pkl")))
            # count as loaded if all trials are loaded
            loaded_results_count += int(args.eval_only or all([len(results[m][method_env]) == (max_trial - min_trial) for m in GT_METRICS]))

    if loaded_results_count == 0:
        print("no results loaded")
        print(unloaded_results_dirs)
        exit(-1) 

    print(f"saving plots to {out_path}")
    styles = ['solid', 'dashdot', 'dashed', 'dotted']

    for env_dir in unloaded_results_dirs:
        # plot and save each metric
        env_name = os.path.basename(env_dir).capitalize()
        print(f"saving results for {env_name}")

        if not GT_METRICS[0] in results:
            print(f"missing {GT_METRICS}")
            continue 

        env_results = [(k, to_matrix(results[GT_METRICS[0]][k])) for k in results[GT_METRICS[0]].keys() if k[1] == env_name]
        max_vec_size = max([Y.shape[2] if len(Y.shape) > 2 else 1 for _, Y in env_results]) # trials x steps x dims

        Ys = list() # visualise errors from ys reference
        max_vec_size = min(max_vec_size, 5)
        fig, ax = plt.subplots(max_vec_size, len(env_results), figsize=(6*len(env_results),5*max_vec_size))
        if max_vec_size == 1: ax = np.array(ax).reshape(1, -1)
        if len(ax.shape) == 1: ax = np.array(ax).reshape(-1, 1)
        for i, metric in enumerate(sorted(GT_METRICS)):
            for j, method_env in enumerate(sorted([k for k in results[metric].keys() if k[1] == env_name], key=lambda x: x[0])):
                method_name, env_name = method_env
                res = to_matrix(results[metric][method_env])
                if len(res) == 0:
                    print(f"no result for {method_env[1]}")
                    continue

                y, ci, n_samples = mean_ci95(res)
                X = np.linspace(0, y.shape[0], y.shape[0])

                if n_samples == 0:
                    continue

                if i == 0: Ys.append(y)
                Y = y if i == 0 else Ys[j] # set reference

                place = min(Y.shape[0] - 1, min(args.seek, y.shape[0] - 1))
                off = min(args.window,place)
                X = X[place - off:place]

                for dim in range(min(Y.shape[1], max_vec_size)):
                    if i == 0:
                        # plot reference
                        ax[dim, j].plot(X, y[place - off:place,dim], label=metric.lower() if dim == 0 and j == 0 else None, linewidth=3)
                        #
                        #pass # uncomment to view just predictions
                    else:
                        # uncomment to view just predictions
                        #ax[dim, j].plot(X, y[place - off:place,dim], label=metric.lower() if dim == 0 and j == 0 else None, color='r', linewidth=3)
                        #ax[dim, j].set_xlabel("Timesteps", fontsize=FONT_SIZE)
                        #
                        # plot error from reference
                        err = np.abs(Y[place - off:place,dim] - y[place-off:place,dim])
                        ax[dim, j].scatter(X, y[place - off:place,dim], label=metric.lower() if dim == 0 and j == 0 else None, color='r', marker='o', s=6)
                        ax[dim, j].vlines(X, Y[place - off:place,dim], y[place - off:place,dim], color='r', linewidth=1)
                        axe = ax[dim, j].inset_axes([0, -0.45, 1, 0.25], sharex=ax[dim, j])
                        axe.plot(X, err, color='r', linewidth=1)
                        axe.set_ylim(0, np.max(err))
                        axe.set_xlabel("Timesteps", fontsize=FONT_SIZE)

                    ax[dim, j].tick_params(axis='both', which='major', labelsize=int(0.7*FONT_SIZE))
                    ax[dim, j].ticklabel_format(style='sci', axis='x', scilimits=(-4,4))
                ax[0, j].set_title(method_name.replace('_', '\n'))

        # label rows 
        [ax[dim, 0].set_ylabel(f"cue[{dim}]", fontsize=FONT_SIZE) for dim in range(max_vec_size)]
        #[spv.set_visible(False) for iax in ax.ravel() for spv in iax.spines.values()]
        plt.gcf().legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), fancybox=True, ncol=len(env_results), fontsize=int(0.7*FONT_SIZE))
        plt.gcf().tight_layout(pad=0.2)
        plt.savefig(os.path.join(out_path, f"{env_name}_gt_metrics.png"), bbox_inches='tight')
        plt.close()

