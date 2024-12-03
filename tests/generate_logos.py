import rockverse as rv
dpi = 300
for model in (1, 2, 3):
    for color in ('black', 'white'):
        for transparency in (True, False):
            fig = rv.make_logo(model=model, facecolor=color) fig.savefig(f'RockVerse_logo_model{model}_for_{color}_background_facecolor_transparent_{transparency}.png', dpi=dpi, transparent=transparency)  # Save the figure
