import copy
import numpy as np

def main():
    '''Balance classes.  Requires hand tuning.'''
    INPUT_FILE_NAME = 'synthetics_100000.npz'
    OUTPUT_FILE_NAME = 'features_and_labels.npz'
    KEEP_NUMBER = 20000
    KEEP_IDX = [3, 4, 5, 6, 7, 8]
    PIXELS = 64

    data = np.load(INPUT_FILE_NAME)
    frames_bzdip = data['arr_0']
    _frames_moment_scalar_sum = data['arr_1']
    _frames_moment_vector_sum = data['arr_2']
    frames_moment_scalar_sum_labels = data['arr_3']
    _frames_moment_vector_sum_labels = data['arr_4']

    # Balance classes by throwing away tons of data
    _unique, _counts = np.unique(frames_moment_scalar_sum_labels, return_counts=True)
    labels_balanced = np.zeros(len(KEEP_IDX) * KEEP_NUMBER)
    frames_balanced = np.zeros((len(KEEP_IDX) * KEEP_NUMBER, PIXELS, PIXELS))

    for idx, val in enumerate(KEEP_IDX):
        start_idx = idx * KEEP_NUMBER
        end_idx = (idx + 1) * KEEP_NUMBER
        labels_balanced[start_idx:end_idx] = idx # Note shift to new labels
        frames_balanced[start_idx:end_idx, :, :] = frames_bzdip[np.where(frames_moment_scalar_sum_labels == val)][0, 0:KEEP_NUMBER]
        
    labels_balanced = labels_balanced.astype(int)
    shuffled_idx = np.random.permutation(labels_balanced.size)
    labels_balanced = labels_balanced[shuffled_idx]
    frames_balanced = frames_balanced[shuffled_idx, :, :]
    frames_original = copy.deepcopy(frames_balanced)

    # Save to balanced classes and labels to .npz
    np.savez(OUTPUT_FILE_NAME,
            labels_balanced=labels_balanced,
            frames_balanced=frames_balanced,
            frames_original=frames_original)
    print('Saved ' + str(KEEP_NUMBER) + ' balanced and shuffled fields and labels to: ' + OUTPUT_FILE_NAME)

if __name__ == '__main__':
    main()