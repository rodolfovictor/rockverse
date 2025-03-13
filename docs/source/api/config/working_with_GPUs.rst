Configure GPUs
==============



Get or set the list of selected GPU indices.

These indices set the devices that are allowed to be used by
RockVerse during runtime.

You can set this list at runtime by providing a list of integers in
range('total number of available GPUs').

Examples
--------
    >>> # Get the currently selected GPU indices
    >>> current_selected = config.selected_gpus

    >>> # Set the selected GPUs to the first and second GPUs available
    >>> config['selected_gpus'] = [0, 1]

    >>> # You can use any iterable
    >>> config['selected_gpus'] = (0, 1, 2)
    >>> config['selected_gpus'] = {0, 1, 2}
    >>> config['selected_gpus'] = range(2)


    >>> # Attempting to set selected GPUs to an invalid index will raise an error
    >>> try:
    >>>     config['selected_gpus'] = [0, 5]  # Assuming only 3 GPUs are available
    >>> except RuntimeError as e:
    >>>    print(e)  # Output: GPU device indices must be less than 3.

    >>> # Setting selected GPUs to an empty list means no GPU will be used
    >>> config['selected_gpus'] = []
