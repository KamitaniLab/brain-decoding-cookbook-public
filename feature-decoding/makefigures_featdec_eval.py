'''Make figures of feature decoding results.'''


import argparse
import os

import bdpy
from bdpy.fig import makeplots
from bdpy.util import makedir_ifnot
import matplotlib.pyplot as plt
import pandas as pd
import yaml


# Main #######################################################################

def makefigures_featdec_eval(decoding_accuracy_file, output_dir='./figures'):

    perf_df = pd.read_pickle(decoding_accuracy_file)
    print('Loaded {}'.format(decoding_accuracy_file))

    print(perf_df)

    if 'test fmri' in conf:
        subjects = list(conf['test fmri'].keys())
    else:
        subjects = list(conf['fmri'].keys())
    rois = list(conf['rois'].keys())
    features = conf['layers']


    # Creating figures #######################################################

    # Profile correlation
    figs = makeplots(
        perf_df,
        x='layer', x_list=features,
        y='profile correlation',
        subplot='roi', subplot_list=rois,
        figure='subject', figure_list=subjects,
        plot_type='violin',
        horizontal=True,
        x_label='Layer', y_label='Profile correlation',
        title='Subject',
        style='seaborn-bright',
        plot_size_auto=True, plot_size=(4, 0.3), max_col=2,
        y_lim=[-0.6, 1], y_ticks=[-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8],
        chance_level=0, chance_level_style={'color': 'gray', 'linewidth': 1}
    )
    for i, fig in enumerate(figs):
        save_filename = os.path.join(output_dir, 'featdec_profile-correlation_layers_subject-{}.pdf'.format(subjects[i]))
        fig.savefig(save_filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print('Saved {}'.format(save_filename))
        save_filename = os.path.join(output_dir, 'featdec_profile-correlation_layers_subject-{}.png'.format(subjects[i]))
        fig.savefig(save_filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print('Saved {}'.format(save_filename))
        plt.close(fig)

    figs = makeplots(
        perf_df,
        x='roi', x_list=rois,
        y='profile correlation',
        subplot='layer', subplot_list=features,
        figure='subject', figure_list=subjects,
        plot_type='violin',
        horizontal=True,
        x_label='Layer', y_label='Profile correlation',
        title='Subject',
        style='seaborn-bright',
        plot_size_auto=True, plot_size=(4, 0.3), max_col=2,
        y_lim=[-0.6, 1], y_ticks=[-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8],
        chance_level=0, chance_level_style={'color': 'gray', 'linewidth': 1}
    )
    for i, fig in enumerate(figs):
        save_filename = os.path.join(output_dir, 'featdec_profile-correlation_rois_subject-{}.pdf'.format(subjects[i]))
        fig.savefig(save_filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print('Saved {}'.format(save_filename))
        save_filename = os.path.join(output_dir, 'featdec_profile-correlation_rois_subject-{}.png'.format(subjects[i]))
        fig.savefig(save_filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print('Saved {}'.format(save_filename))
        plt.close(fig)

    # Pattern correlation
    figs = makeplots(
        perf_df,
        x='layer', x_list=features,
        y='pattern correlation',
        subplot='roi', subplot_list=rois,
        figure='subject', figure_list=subjects,
        plot_type='swarm+box',
        horizontal=True,
        x_label='Layer', y_label='Pattern correlation',
        title='Subject',
        style='seaborn-bright',
        plot_size_auto=True, plot_size=(4, 0.3), max_col=2,
        y_lim=[-0.6, 1], y_ticks=[-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8],
        chance_level=0, chance_level_style={'color': 'gray', 'linewidth': 1}
    )
    for i, fig in enumerate(figs):
        save_filename = os.path.join(output_dir, 'featdec_pattern-correlation_layers_subject-{}.pdf'.format(subjects[i]))
        fig.savefig(save_filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print('Saved {}'.format(save_filename))
        save_filename = os.path.join(output_dir, 'featdec_pattern-correlation_layers_subject-{}.png'.format(subjects[i]))
        fig.savefig(save_filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print('Saved {}'.format(save_filename))
        plt.close(fig)

    figs = makeplots(
        perf_df,
        x='roi', x_list=rois,
        y='pattern correlation',
        subplot='layer', subplot_list=features,
        figure='subject', figure_list=subjects,
        plot_type='swarm+box',
        horizontal=True,
        x_label='Layer', y_label='Pattern correlation',
        title='Subject',
        style='seaborn-bright',
        plot_size_auto=True, plot_size=(4, 0.3), max_col=2,
        y_lim=[-0.6, 1], y_ticks=[-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8],
        chance_level=0, chance_level_style={'color': 'gray', 'linewidth': 1}
    )
    for i, fig in enumerate(figs):
        save_filename = os.path.join(output_dir, 'featdec_pattern-correlation_rois_subject-{}.pdf'.format(subjects[i]))
        fig.savefig(save_filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print('Saved {}'.format(save_filename))
        save_filename = os.path.join(output_dir, 'featdec_pattern-correlation_rois_subject-{}.png'.format(subjects[i]))
        fig.savefig(save_filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print('Saved {}'.format(save_filename))
        plt.close(fig)

    # Identification
    figs = makeplots(
        perf_df,
        x='layer', x_list=features,
        y='identification accuracy',
        subplot='roi', subplot_list=rois,
        figure='subject', figure_list=subjects,
        plot_type='swarm+box',
        horizontal=True,
        x_label='Layer', y_label='Pairwise identification accuracy',
        title='Subject',
        style='seaborn-bright',
        plot_size_auto=True, plot_size=(4, 0.3), max_col=2,
        y_lim=[0, 1], y_ticks=[0, 0.25, 0.5, 0.75, 1.0],
        chance_level=0.5, chance_level_style={'color': 'gray', 'linewidth': 1}
    )
    for i, fig in enumerate(figs):
        save_filename = os.path.join(output_dir, 'featdec_pairwise-identification_layers_subject-{}.pdf'.format(subjects[i]))
        fig.savefig(save_filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print('Saved {}'.format(save_filename))
        save_filename = os.path.join(output_dir, 'featdec_pairwise-identification_layers_subject-{}.png'.format(subjects[i]))
        fig.savefig(save_filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print('Saved {}'.format(save_filename))
        plt.close(fig)

    figs = makeplots(
        perf_df,
        x='roi', x_list=rois,
        y='identification accuracy',
        subplot='layer', subplot_list=features,
        figure='subject', figure_list=subjects,
        plot_type='swarm+box',
        horizontal=True,
        x_label='Layer', y_label='Pairwise identification accuracy',
        title='Subject',
        style='seaborn-bright',
        plot_size_auto=True, plot_size=(4, 0.3), max_col=2,
        y_lim=[0, 1], y_ticks=[0, 0.25, 0.5, 0.75, 1.0],
        chance_level=0.5, chance_level_style={'color': 'gray', 'linewidth': 1}
    )
    for i, fig in enumerate(figs):
        save_filename = os.path.join(output_dir, 'featdec_pairwise-identification_rois_subject-{}.pdf'.format(subjects[i]))
        fig.savefig(save_filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print('Saved {}'.format(save_filename))
        save_filename = os.path.join(output_dir, 'featdec_pairwise-identification_rois_subject-{}.png'.format(subjects[i]))
        fig.savefig(save_filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print('Saved {}'.format(save_filename))
        plt.close(fig)

    # Creating figures (subject comparison) ##################################

    # Profile correlation
    figs = makeplots(
        perf_df,
        x='subject', x_list=subjects,
        y='profile correlation',
        subplot='layer', subplot_list=features,
        figure='roi', figure_list=rois,
        plot_type='violin',
        horizontal=True,
        x_label='Layer', y_label='Profile correlation',
        title='ROI',
        style='seaborn-bright',
        plot_size_auto=True, plot_size=(4, 0.3), max_col=2,
        y_lim=[-0.6, 1], y_ticks=[-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8],
        chance_level=0, chance_level_style={'color': 'gray', 'linewidth': 1}
    )
    for i, fig in enumerate(figs):
        save_filename = os.path.join(output_dir, 'featdec_profile-correlation_all-subjects_roi-{}.pdf'.format(rois[i]))
        fig.savefig(save_filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print('Saved {}'.format(save_filename))
        save_filename = os.path.join(output_dir, 'featdec_profile-correlation_all-subjects_roi-{}.png'.format(rois[i]))
        fig.savefig(save_filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print('Saved {}'.format(save_filename))
        plt.close(fig)

    # Pattern correlation
    figs = makeplots(
        perf_df,
        x='subject', x_list=subjects,
        y='pattern correlation',
        subplot='layer', subplot_list=features,
        figure='roi', figure_list=rois,
        plot_type='swarm+box',
        horizontal=True,
        x_label='Layer', y_label='Pattern correlation',
        title='ROI',
        style='seaborn-bright',
        plot_size_auto=True, plot_size=(4, 0.3), max_col=2,
        y_lim=[-0.6, 1], y_ticks=[-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8],
        chance_level=0, chance_level_style={'color': 'gray', 'linewidth': 1}
    )
    for i, fig in enumerate(figs):
        save_filename = os.path.join(output_dir, 'featdec_pattern-correlation_all-subjects_roi-{}.pdf'.format(rois[i]))
        fig.savefig(save_filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print('Saved {}'.format(save_filename))
        save_filename = os.path.join(output_dir, 'featdec_pattern-correlation_all-subjects_roi-{}.png'.format(rois[i]))
        fig.savefig(save_filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print('Saved {}'.format(save_filename))
        plt.close(fig)

    # Identification
    figs = makeplots(
        perf_df,
        x='subject', x_list=subjects,
        y='identification accuracy',
        subplot='layer', subplot_list=features,
        figure='roi', figure_list=rois,
        plot_type='swarm+box',
        horizontal=True,
        x_label='Layer', y_label='Pairwise identification accuracy',
        title='ROI',
        style='seaborn-bright',
        plot_size_auto=True, plot_size=(4, 0.3), max_col=2,
        y_lim=[0, 1], y_ticks=[0, 0.25, 0.5, 0.75, 1.0],
        chance_level=0.5, chance_level_style={'color': 'gray', 'linewidth': 1}
    )
    for i, fig in enumerate(figs):
        save_filename = os.path.join(output_dir, 'featdec_pairwise-identification_all-subjects_roi-{}.pdf'.format(rois[i]))
        fig.savefig(save_filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print('Saved {}'.format(save_filename))
        save_filename = os.path.join(output_dir, 'featdec_pairwise-identification_all-subjects_roi-{}.png'.format(rois[i]))
        fig.savefig(save_filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print('Saved {}'.format(save_filename))
        plt.close(fig)

    print('All done')

    return output_dir


# Entry point ################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'conf',
        type=str,
        help='analysis configuration file',
    )
    args = parser.parse_args()

    conf_file = args.conf

    with open(conf_file, 'r') as f:
        conf = yaml.safe_load(f)

    conf.update({
        '__filename__': os.path.splitext(os.path.basename(conf_file))[0]
    })

    if 'analysis name' in conf:
        analysis_name = conf['analysis name']
    else:
        analysis_name = ''

    decoding_accuracy_file = os.path.join(
        conf['decoded feature dir'],
        analysis_name,
        'decoded_features',
        conf['network'],
        'accuracy.pkl.gz'
    )
    if not os.path.exists(decoding_accuracy_file):
        decoding_accuracy_file = os.path.join(
            conf['decoded feature dir'],
            analysis_name,
            conf['network'],
            'accuracy.pkl.gz'
        )

    if 'decoding figure dir' in conf:
        figure_dir = os.path.join(
            conf['decoding figure dir'],
            analysis_name,
            conf['network']
        )
        makedir_ifnot(figure_dir)
    else:
        figure_dir = './'

    makefigures_featdec_eval(
        decoding_accuracy_file,
        output_dir=figure_dir
    )
