# Functions for generating Kaggle submission files.

def update_submission(predictions, new_submission_file,
                      old_submission_file='sampleSubmission.csv',
                      default_value=None):
    """
    Read predicted probabilities for test data files from
    old_submission_file, replace the predictions using the
    dictionary called 'predictions' (with data file names as keys
    and predicted probabilities for the preictal class as values;
    for missing keys, default_value will be used if defined,
    otherwise the predicted probability from the original
    submission file will be used), and write the new predictions
    to new_submission file.
    """
    with open(old_submission_file, 'r') as f:
        old_lines = f.readlines()

    new_lines = [old_lines[0]]
    for line in old_lines[1:]:
        key, old_value = line.strip().split(',')
        if default_value is None:
            default = old_value
        else:
            default = default_value
        new_value = str(predictions.get(key, default))
        new_lines.append(','.join((key, new_value)) + '\n')
        
    with open(new_submission_file, 'w') as f:
        f.writelines(new_lines)

