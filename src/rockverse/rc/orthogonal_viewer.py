import numpy as np

ORTHOGONAL_VIEWER = {
    'X-ray CT': {
        'figure': {
            'layout': 'compressed'
        },
        'image': {
            'cmap': 'gray',
            'interpolation': 'none'
        },
        'segmentation': {
            'colors': 'tab10',
            'alpha': 0.5,
            'interpolation': 'none'
        },
        'mask': {
            'color': 'k',
            'alpha': 0.75,
            'interpolation': 'none'
        },
        'guide_lines': {
            'linestyle': '--',
            'color': 'r',
            'alpha': 0.75,
            'linewidth': 1
        },
        'histogram_lines': {
            'full': {
                'color': 'k',
                'linewidth': 1,
                'linestyle': '-',
            },
            'phases': {
                'linewidth': 2,
                'linestyle': '-',
            },
            'clim': {
                'color': 'k',
                'linestyle': '--',
                'alpha': 0.75,
            },
        },
    },
    'scalar field': {
        'figure': {
            'layout': 'compressed'
        },
        'image': {
            'cmap': 'summer',
            'interpolation': 'none'
        },
        'segmentation': {
            'colors': 'tab10_r',
            'alpha': 0.5,
            'interpolation': 'none'
        },
        'mask': {
            'color': 'k',
            'alpha': 0.75,
            'interpolation': 'none'
        },
        'guide_lines': {
            'linestyle': '-',
            'color': [0.25, 0.25, 0.25],
            'alpha': 0.75,
            'linewidth': 1
        },
        'histogram_lines': {
            'full': {
                'color': 'k',
                'linewidth': 1,
                'linestyle': '-',
            },
            'phases': {
                'linewidth': 2,
                'linestyle': '-',
            },
            'clim': {
                'color': 'k',
                'linestyle': '--',
                'alpha': 0.5,
            },
        },
    },
}
