# Functions for generating Kaggle submission files.

def update_submission(predictions, new_submission_file,
                      old_submission_file='sampleSubmission.csv'):
    """
    Read predicted probabilities for test data files from
    old_submission_file, replace the predictions using the
    dictionary called 'predictions' (with data file names as keys
    and predicted probabilities for the preictal class as values;
    for missing keys, the predicted probability from the original
    submission file will be used), and write the new predictions
    to new_submission file.
    """
    with open(old_submission_file, 'r') as f:
        old_lines = f.readlines()

    new_lines = []
    for line in old_lines:
        key, old_value = line.strip().split(',')
        new_value = str(predictions.get(key, old_value))
        new_lines.append(','.join((key, new_value)) + '\n')
        
    with open(new_submission_file, 'w') as f:
        f.writelines(new_lines)

