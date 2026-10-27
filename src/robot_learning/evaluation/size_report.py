"""Export saved episode-count rollouts and common-holdout command diagnostics."""
import argparse
import csv
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from robot_learning.data.storage import read_json, write_json, sha256
from robot_learning.learning.dataset import load_episode
from robot_learning.learning.fidelity import command_fidelity, relative_command_fidelity
from robot_learning.learning.model import LinearPolicy, MLPPolicy
from .subsystem_report import report, load_evaluation
from .trajectory import compare_traces
from .trajectory_plots import plot_case, plot_overview, COLORS


def flatten(value, prefix=''):
    """Keep every saved scalar, including explicit nulls, in a wide CSV."""
    result = {}
    for key, item in value.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        elif not isinstance(item, list):
            result[name] = item
    return result


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def offline_fidelity(campaign, output):
    """Use the already leakage-checked common evaluation bank and fixed scales."""
    from .cli import validate_common_validation

    comparison = read_json(campaign/'offline_comparison.json')
    split = read_json(comparison['common_split'])
    reference = LinearPolicy.load(Path(comparison['common_split']).parent/'model.npz')
    data = [load_episode(split['dataset'], record)[:2] for record in split['validation']]
    truth = np.concatenate([y for _, y in data])
    results = {}
    for label, source in comparison['results'].items():
        path = Path(source['training_run'])
        manifest = read_json(path/'run.json')
        if (manifest['status'] != 'complete' or
                sha256(path/'model.npz') != source['model_sha256'] or
                sha256(path/'model.npz') != manifest['model_sha256'] or
                sha256(path/'split.json') != manifest['split_sha256']):
            raise ValueError('Model or split changed after the campaign')
        validate_common_validation(split['dataset'], split['validation'], read_json(path/'split.json'))
        cls = LinearPolicy if source['model_info']['kind'] == 'ordinary_least_squares' else MLPPolicy
        policy = cls.load(path/'model.npz')
        predictions = [policy.predict(x) for x, _ in data]
        episodes = [command_fidelity(y, p) for (_, y), p in zip(data, predictions)]
        pooled = np.concatenate(predictions)
        results[label] = {
            'model_sha256': source['model_sha256'],
            'pooled': command_fidelity(truth, pooled),
            'relative_and_stop_pooled': relative_command_fidelity(truth, pooled, reference.y_scale),
            'episode_macro_all_nine_within_tolerance_fraction': np.mean(
                [e['all_nine_within_tolerance_fraction'] for e in episodes], axis=0).tolist(),
            'episode_macro_per_output': {
                name: np.mean([e['per_output'][name]['within_tolerance_fraction'] for e in episodes], axis=0).tolist()
                for name in episodes[0]['per_output']}}
    write_json(output/'common_command_fidelity.json', {
        'scope': 'Same 13 held-out expert-state trajectories; raw predictions, no retraining',
        'normalization': 'Mixed tolerance floors use fixed original 64-training output std',
        'aggregation': 'Relative, stopping and P95 pooled by transition; absolute fractions also episode-macro',
        'no_control_acceptance_threshold_selected': True,
        'common_split_sha256': sha256(comparison['common_split']), 'results': results})
    return results


def plot_fidelity(results, output):
    """Display relative errors and zero commands separately, using existing metrics."""
    import matplotlib.pyplot as plt
    primary_relative_fidelity(results, output)
    counts = sorted(int(label[3:]) for label in results if label.startswith('mlp') and label[3:].isdigit())
    outputs = list(results['mlp64']['pooled']['per_output'])
    exports = []
    for kind in ('mlp', 'linear'):
        fig, axes = plt.subplots(3, 3, figsize=(14, 10), layout='constrained')
        for ax, name in zip(axes.flat, outputs):
            for n in counts:
                label = f'{kind}{n}'
                result = results[label]
                relative = result['relative_and_stop_pooled']
                channel = relative['per_output'][name]
                ax.plot(np.array(relative['relative_tolerances'])*100,
                        channel['nonzero_relative_within_fraction'],
                        marker='o', label=str(n), color=COLORS.get(label, '#9955aa'))
                exports.append({'policy': label, 'output': name,
                                'p95_absolute_error': result['pooled']['per_output'][name]['p95_absolute_error'],
                                **flatten(channel)})
            ax.set(title=name, xlabel='Relative tolerance (%)', ylabel='Fraction within tolerance', ylim=(0, 1))
            ax.grid(alpha=.2)
            ax.legend(title='Episodes', fontsize=7)
        fig.suptitle(f'{kind.upper()}: offline fidelity on nonzero expert commands; same 13 scenarios; pooled rows')
        fig.savefig(output/f'{kind}_relative_command_fidelity.png', dpi=150)
        fig.savefig(output/f'{kind}_relative_command_fidelity.pdf')
        plt.close(fig)
    write_csv(output/'command_tail_and_zero_metrics.csv', exports)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    for n in counts:
        result = results[f'mlp{n}']
        axes[0].plot(result['pooled']['tolerances'], result['episode_macro_all_nine_within_tolerance_fraction'],
                     marker='o', label=str(n), color=COLORS.get(f'mlp{n}', '#9955aa'))
    axes[0].set(title='MLP: all nine commands within tolerance', xlabel='Tolerance (rad/s per wheel; Nm per joint)',
                ylabel='Equal-episode fraction', ylim=(0, 1))
    for kind in ('mlp', 'linear'):
        for side in (0, 1):
            axes[1+side].plot(counts,
                [results[f'{kind}{n}']['relative_and_stop_pooled']['base_stop_prediction_mean_abs_by_side'][side]
                 for n in counts], marker='o', label=kind)
    for ax, side in zip(axes[1:], ('right', 'left')):
        ax.set(title=f'Expert commands stop: predicted {side} wheel', xlabel='Collected episodes', ylabel='Mean absolute command (rad/s)')
        ax.set_xticks(counts)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Offline command proximity and stop-command error; no threshold certifies stable control')
    fig.savefig(output/'command_fidelity_and_stop.png', dpi=150)
    fig.savefig(output/'command_fidelity_and_stop.pdf')
    plt.close(fig)


def primary_relative_fidelity(results, output):
    """Primary command criterion: 5% of nonzero truth, no absolute floor."""
    import matplotlib.pyplot as plt
    counts = sorted(int(label[3:]) for label in results if label.startswith('mlp') and label[3:].isdigit())
    outputs = list(results['mlp64']['pooled']['per_output'])
    rows = []
    for label, result in results.items():
        relative = result['relative_and_stop_pooled']
        index = relative['relative_tolerances'].index(.05)
        for name in outputs:
            channel = relative['per_output'][name]
            rows.append({'policy': label, 'output': name,
                         'fraction_within_5pct_on_nonzero': channel['nonzero_relative_within_fraction'][index],
                         'nonzero_rows': channel['nonzero_rows'], 'zero_rows': channel['zero_rows'],
                         'zero_mean_abs_prediction': channel['zero_command_prediction_mean_abs'],
                         'zero_p95_abs_prediction': channel['zero_command_prediction_p95_abs'],
                         'zero_threshold': relative['zero_reference_threshold_physical_units'],
                         'unit': result['pooled']['per_output'][name]['unit']})
    write_csv(output/'primary_relative_5pct.csv', rows)
    write_json(output/'primary_relative_5pct.json', {
        'criterion': 'abs(prediction-truth) <= 0.05*abs(truth); no additive tolerance',
        'aggregation': 'Pooled transitions on the same 13 expert-state validation episodes, separately per output',
        'zero_handling': 'abs(truth)<=1e-6 excluded from relative denominator; absolute mean/P95 reported separately',
        'control_stability_guarantee': False, 'per_output': rows})
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), layout='constrained')
    for ax, name in zip(axes.flat, outputs):
        for kind in ('mlp', 'linear'):
            values = [next(r['fraction_within_5pct_on_nonzero'] for r in rows
                           if r['policy'] == f'{kind}{n}' and r['output'] == name)
                      for n in counts]
            ax.plot(counts, np.array(values)*100, marker='o', label=kind)
        ax.set(title=name, xlabel='Collected episodes', ylabel='Commands within 5% (%)',
               ylim=(0, 100), xticks=counts)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Primary command fidelity: error <= 5% of absolute expert command; zeros evaluated separately\nSame 13 expert-state scenarios; pooled nonzero rows; no absolute floor or stability guarantee')
    fig.savefig(output/'primary_relative_5pct.png', dpi=150)
    fig.savefig(output/'primary_relative_5pct.pdf')
    plt.close(fig)


def plot_navigation(summary, output):
    import matplotlib.pyplot as plt
    counts = sorted(int(row['policy'][3:]) for row in summary if row['policy'].startswith('mlp') and row['policy'][3:].isdigit())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    fields = [
        ('base_final_goal_distance_xy_median_m', 'Median final base-goal XY distance (m)'),
        ('median_final_goal_yaw_error_deg', 'Median final navigation yaw error (degrees)'),
        ('base_time_weighted_xy_rmse_median_m', 'Median XY trajectory RMSE vs expert (m)'),
        ('base_terminal_speed_xy_median_m_s', 'Median terminal base speed (m/s)'),
        ('mean_wheel_clipping_fraction', 'Mean wheel clipping fraction'),
        ('mean_torque_clipping_fraction', 'Mean torque clipping fraction')]
    for ax, (field, title) in zip(axes.flat, fields):
        for kind in ('mlp', 'linear'):
            points = sorted([r for r in summary if r['policy'].startswith(kind)], key=lambda r: int(r['policy'][len(kind):]))
            ax.plot([int(r['policy'][len(kind):]) for r in points], [r[field] for r in points], marker='o', label=kind)
        expert = next(r for r in summary if r['policy'] == 'expert')
        ax.axhline(expert[field], color=COLORS['expert'], ls=':', label='expert')
        ax.set(title=title, xlabel='Collected episodes', xticks=counts)
        if field in (fields[0][0], fields[2][0], fields[3][0]):
            ax.set_yscale('symlog', linthresh=.001)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Common 13 scenarios; base goal follows expert standoff; extreme distances indicate simulated divergence')
    fig.savefig(output/'navigation_summary.png', dpi=150)
    fig.savefig(output/'navigation_summary.pdf')
    plt.close(fig)


def generate(campaign, output):
    campaign = Path(campaign).resolve()
    output = Path(output).resolve()
    report(campaign/'autonomous', output, plots=False)
    analysis = read_json(output/'trajectory_analysis.json')
    rows = analysis['per_episode']
    write_csv(output/'all_episode_metrics.csv', [flatten(row) for row in rows])
    grouped, _ = load_evaluation(campaign/'autonomous')
    summary = []
    for label, item in analysis['summary'].items():
        selected = [r for r in rows if r['policy'] == label]
        median = lambda getter: float(np.median([getter(r) for r in selected]))
        clipping = np.array([r['saturation']['full_step_clipping_fraction_per_command'] for r in selected])
        summary.append({
            'policy': label, **flatten(item),
            'median_final_goal_yaw_error_deg': median(lambda r: abs(np.rad2deg(r['base_navigation']['base_goal_bearing_error_rad']))),
            'median_yaw_rmse_vs_expert_deg': median(lambda r: np.rad2deg(r['trajectory_vs_expert']['base_yaw_separation_rad']['time_rmse'])),
            'median_minimum_base_goal_distance_m': median(lambda r: r['base_navigation']['minimum_goal_distance_xy_m']),
            'median_average_base_goal_distance_m': median(lambda r: r['base_navigation']['time_average_goal_distance_xy_m']),
            'median_base_path_length_m': median(lambda r: r['base_navigation']['path_length_xy_m']),
            'median_terminal_abs_yaw_rate_rad_s': median(lambda r: r['stop_motion']['base_mean_abs_yaw_rate_rad_s']),
            'mean_wheel_clipping_fraction': float(clipping[:, :4].mean()),
            'mean_torque_clipping_fraction': float(clipping[:, 4:].mean())})
    write_csv(output/'policy_summary.csv', summary)
    write_json(output/'policy_summary.json', summary)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gallery = ['# Episode-count trajectory gallery', '',
               'Saved 40 s rollouts, same 13 validation scenarios. Trace cadence: 100 ms.',
               'Threshold crossings and clipping onsets are first observed samples, not exact onsets.',
               'Command curves compare separate live states, not supervised errors.', '',
               '## Summary', '', '![Navigation and stopping](navigation_summary.png)', '']
    for kind in ('mlp', 'linear'):
        directory = output/kind
        directory.mkdir()
        selected_groups = {eid: {label: pair for label, pair in group.items()
                                if label == 'expert' or label.startswith(kind)}
                           for eid, group in grouped.items()}
        curves = {eid: {label: compare_traces(pair[1], group['expert'][1], pair[0]['dt'])[1]
                        for label, pair in group.items()}
                  for eid, group in selected_groups.items()}
        for eid, group in selected_groups.items():
            plot_case(directory, eid, group, curves[eid], 2.)
        plot_overview(directory, selected_groups, curves,
                      {label: value for label, value in analysis['summary'].items()
                       if label == 'expert' or label.startswith(kind)}, 2.)
        gallery += [f'## {kind.upper()}', '', f'![Overview]({kind}/trajectory_overview.png)', '']
        for eid in selected_groups:
            gallery += [f'### {eid}', '']
            for family in ('base_trajectories', 'trajectory_differences', 'orientation', 'saturation', 'initial_commands'):
                gallery += [f'[{family.replace("_", " ")}]({kind}/{family}_{eid}.png)', '']
    plot_navigation(summary, output)
    with threadpool_limits(limits=1):
        fidelity = offline_fidelity(campaign, output)
    plot_fidelity(fidelity, output)
    gallery += ['## Command fidelity on common expert states', '',
                '![Primary 5% relative criterion](primary_relative_5pct.png)', '',
                '![Absolute tolerance and stopping](command_fidelity_and_stop.png)', '',
                '![MLP relative tolerance](mlp_relative_command_fidelity.png)', '',
                '![OLS relative tolerance](linear_relative_command_fidelity.png)', '']
    (output/'GALLERY.md').write_text('\n'.join(gallery)+'\n')
    (output/'size_report.py').write_bytes(Path(__file__).read_bytes())
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(generate(args.campaign, args.output_dir))
