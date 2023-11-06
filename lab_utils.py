
def tqdm_lab_fix():
    # from IPython import HTML
    # HTML(tqdm_lab_fix())
    return """
    <style>
    .jp-OutputArea-prompt:empty {
      padding: 0;
      border: 0;
    }
    </style>
    """