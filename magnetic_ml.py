import copy
import sys
import keras
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from matplotlib import cm
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


SYNTHETICS_FILE_NAME = 'synthetics_100000.npz'
OUTPUT_FILE_NAME = 'features_and_labels.npz'
KEEP_NUMBER = 20000
KEEP_IDX = [3, 4, 5, 6, 7, 8]
PIXELS = 64
FONT_SIZE = 14


def balance_classes():
    '''Balance classes.  Requires hand tuning.'''

    print('Reading: ' + SYNTHETICS_FILE_NAME)
    data = np.load(SYNTHETICS_FILE_NAME)

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
        sys.stdout.write('\r [ %d'%(idx/len(KEEP_IDX) * 100)+'% ] ')
        sys.stdout.flush()

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
    print('Saved ' +
          str(KEEP_NUMBER) +
          ' balanced and shuffled fields and labels to: '
          + OUTPUT_FILE_NAME)


def build_model():
    '''Build Keras model'''
    input_shape = (PIXELS, PIXELS, 1)
    num_classes = len(KEEP_IDX)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def shape_training_data():
    ''' Format data for Keras training '''
    num_classes = len(KEEP_IDX)
    with np.load('features_and_labels.npz') as data:
        labels_balanced = data['labels_balanced']
        frames_balanced = data['frames_balanced']
        _frames_original = data['frames_original']

    # Normalization and clean up
    frames_balanced = frames_balanced - np.min(frames_balanced) # put min at 0
    frames_balanced = frames_balanced / np.max(frames_balanced) # put max at 1

    # Split between train, test sets, and validation sets
    start_idx_train = 0
    end_idx_train = 10000
    start_idx_test = 10001
    end_idx_test = 20000
    start_idx_validate = 10001
    end_idx_validate = 20000

    x_train = frames_balanced[start_idx_train:end_idx_train, :, :]
    y_train = labels_balanced[start_idx_train:end_idx_train]
    x_test = frames_balanced[start_idx_test:end_idx_test, :, :]
    y_test = labels_balanced[start_idx_test:end_idx_test]
    x_validate = frames_balanced[start_idx_validate:end_idx_validate]
    y_validate = labels_balanced[start_idx_validate:end_idx_validate]

    # Reshape because this is what we do in machine learning
    x_train = x_train.reshape(x_train.shape[0], PIXELS, PIXELS, 1)
    x_test = x_test.reshape(x_test.shape[0], PIXELS, PIXELS, 1)
    x_validate = x_validate.reshape(x_validate.shape[0], PIXELS, PIXELS, 1)

    # Cast
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_validate = x_validate.astype('float32')
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print(x_validate.shape[0], 'validate samples')

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    y_validate = keras.utils.to_categorical(y_validate, num_classes)

    return x_train, y_train, x_test, y_test, x_validate, y_validate


def learn_simple():
    '''https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py'''
    batch_size = 128
    epochs = 2

    model = build_model()
    x_train, y_train, x_test, y_test, x_validate, y_validate = shape_training_data()

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Save everything we need for postprocessing
    model.save('learn_simple.h5')
    np.savez('learn_simple.npz',
             x_train = x_train,
             y_train = y_train,
             x_test = x_test,
             y_test = y_test,
             x_validate = x_validate,
             y_validate = y_validate)


def learn_noisy():
    '''Train over noisy data and assess recoverability'''
    batch_size = 128
    epochs = 2
    noise_levels = 0

    model = build_model()
    x_train, y_train, x_test, y_test, x_validate, y_validate = shape_training_data()

    # Loop over noise
    for i in range(noise_levels):
        # Add noise to x_train and x_validate

        # Train and test
        model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])


def predict_simple():
    '''Predictions/inference from a single run'''
    #TODO: fix variable names.  Plot as a single figure

    model = load_model('learn_simple.h5')

    with np.load('features_and_labels.npz') as data:
        _labels_balanced = data['labels_balanced']
        frames_balanced = data['frames_balanced']
        _frames_original = data['frames_original']

    with np.load('learn_simple.npz') as data:
        x_train = data['x_train']
        y_train = data['y_train']
        _x_test = data['x_test']
        _y_test = data['y_test']
        _x_validate = data['x_validate']
        _y_validate = data['y_validate']

    hist_bins = np.arange(0, len(KEEP_IDX), 1)
    predict_labels = model.predict(x_train)
    true = np.argmax(y_train, axis=1)
    vals = np.argmax(predict_labels, axis=1)
    correlation_nn = np.corrcoef(true, vals)

    simple_metric_max = np.zeros(KEEP_NUMBER)
    simple_metric_sum = np.zeros(KEEP_NUMBER)
    for i in range(0, KEEP_NUMBER):
        simple_metric_sum[i] = np.sum(frames_balanced[i, :, :]**2.0)
        simple_metric_max[i] = np.max(frames_balanced[i, :, :])

    _, bins, _ = plt.hist(simple_metric_max, facecolor='green', bins=len(KEEP_IDX))
    plt.close()
    simple_metric_labels_max = np.digitize(simple_metric_max, bins)
    simple_metric_labels_max = simple_metric_labels_max[0:10000]
    correlation_max = np.corrcoef(simple_metric_labels_max, vals)

    _, bins, _ = plt.hist(simple_metric_sum, facecolor='green', bins=len(KEEP_IDX))
    plt.close()
    simple_metric_labels_sum = np.digitize(simple_metric_sum, bins)
    simple_metric_labels_sum = simple_metric_labels_sum[0:10000]
    correlation_sum = np.corrcoef(simple_metric_labels_sum, vals)

    plt.figure(figsize=(8, 12))
    matplotlib.rc('xtick', labelsize=FONT_SIZE) 
    matplotlib.rc('ytick', labelsize=FONT_SIZE)

    ax = plt.subplot(3, 1, 1)
    ax_nn = plt.hist(vals-true,
             bins=hist_bins,
             edgecolor='red',
             facecolor='red',
             align='left',
             histtype='stepfilled',
             alpha = 0.25,
             linewidth=2.0)
    plt.title('neural network, ' + r'$R = $' \
              + '{:.2f}'.format(correlation_nn[1, 0]),
              fontsize=FONT_SIZE)
    plt.xlabel('label error', fontsize=FONT_SIZE)
    plt.ylabel('$N$', fontsize=FONT_SIZE)
    plt.ylim(0, 10500)

    ax = plt.subplot(3, 1, 2)
    ax_max = plt.hist(np.abs(simple_metric_labels_max-vals),
             bins=hist_bins,
             edgecolor='green',
             facecolor='green',
             align='left',
             histtype='stepfilled',
             alpha=0.25,
             linewidth=2.0)
    plt.title('maximum value, ' + r'$R = $' \
              + '{:.2f}'.format(correlation_max[1, 0]),
              fontsize=FONT_SIZE)
    plt.xlabel('label error', fontsize=FONT_SIZE)
    plt.ylabel('$N$', fontsize=FONT_SIZE)
    plt.ylim(0, 10500)

    ax = plt.subplot(3, 1, 3)
    ax_sum = plt.hist(np.abs(simple_metric_labels_sum-vals),
             bins=hist_bins,
             edgecolor='blue',
             facecolor='blue',
             align='left',
             histtype='stepfilled',
             alpha=0.25,
             linewidth=2.0)
    plt.title('sum of squares, ' + r'$R = $' \
              + '{:.2f}'.format(correlation_sum[1, 0]),
              fontsize=FONT_SIZE)
    plt.xlabel('label error', fontsize=FONT_SIZE)
    plt.ylabel('$N$', fontsize=FONT_SIZE)
    plt.ylim(0, 10500)

    plt.tight_layout()
    plt.show(block=False)


def plot_field():
    '''Simple visualization'''
    idx = 10
    
    with np.load('features_and_labels.npz') as data:
        labels_balanced = data['labels_balanced']
        frames_balanced = data['frames_balanced']
        _frames_original = data['frames_original']

    min_val = np.min(frames_balanced)
    max_val = np.max(frames_balanced)
    cbar_val = np.max(np.array([np.abs(min_val), max_val]))
    scale_factor = np.round(np.log10(cbar_val))
    cbar_val = cbar_val / (10**scale_factor)

    plt.imshow(frames_balanced[idx, :, :] / 10**(scale_factor),
               interpolation='nearest',
               cmap=cm.coolwarm,
               origin='lower')
    plt.clim(-cbar_val, cbar_val)
    plt.plot(np.array([-50, 50, 50, -50, -50])*PIXELS/200 + PIXELS/2,
             np.array([-50, -50, 50, 50, -50])*PIXELS/200 + PIXELS/2,
             '--k',
             linewidth=1.0)
    plt.xticks([0, PIXELS-1], ['-100', '100'])
    plt.yticks([0, PIXELS-1], ['-100', '100'])
    plt.xlabel('$x \; \mathrm{(microns)}$', fontsize=FONT_SIZE)
    plt.ylabel('$y \; \mathrm{(microns)}$', fontsize=FONT_SIZE)
    
    matplotlib.rc('xtick', labelsize=FONT_SIZE) 
    matplotlib.rc('ytick', labelsize=FONT_SIZE) 

    cbar = plt.colorbar(ticks=[-cbar_val, 0, cbar_val])
    exponent_string = '$10^{' + str(scale_factor) + '}$' 
    cbar.ax.set_ylabel(r'field strength (units ' + r'$\times$' + ' ' + exponent_string + ')', fontsize=14, rotation=90)
    cbar.ax.tick_params(labelsize=14) 
    plt.title('$\mathrm{label} \; = \; $' + str(labels_balanced[idx]), fontsize=14)
    plt.show(block=False)


def plot_field_noise():
    '''Simple visualization'''
    idx = 10
    
    with np.load('features_and_labels.npz') as data:
        labels_balanced = data['labels_balanced']
        frames_balanced = data['frames_balanced']
        _frames_original = data['frames_original']

    min_val = np.min(frames_balanced)
    max_val = np.max(frames_balanced)
    cbar_val = np.max(np.array([np.abs(min_val), max_val]))
    scale_factor = np.round(np.log10(cbar_val))
    cbar_val = cbar_val / (10**scale_factor)

    plt.figure(figsize=(10, 3.5))
    plt.subplot(1, 2, 1)
    plt.imshow(frames_balanced[idx, :, :] / 10**(scale_factor),
               interpolation='nearest',
               cmap=cm.coolwarm,
               origin='lower')
    plt.clim(-cbar_val, cbar_val)
    plt.plot(np.array([-50, 50, 50, -50, -50])*PIXELS/200 + PIXELS/2,
             np.array([-50, -50, 50, 50, -50])*PIXELS/200 + PIXELS/2,
             '--k',
             linewidth=1.0)
    plt.xticks([0, PIXELS-1], ['-100', '100'])
    plt.yticks([0, PIXELS-1], ['-100', '100'])
    plt.xlabel('$x \; \mathrm{(microns)}$', fontsize=FONT_SIZE)
    plt.ylabel('$y \; \mathrm{(microns)}$', fontsize=FONT_SIZE)
    
    matplotlib.rc('xtick', labelsize=FONT_SIZE) 
    matplotlib.rc('ytick', labelsize=FONT_SIZE) 

    cbar = plt.colorbar(ticks=[-cbar_val, 0, cbar_val])
    exponent_string = '$10^{' + str(scale_factor) + '}$' 
    cbar.ax.set_ylabel(r'field strength (units ' + r'$\times$' + ' ' + exponent_string + ')', fontsize=14, rotation=90)
    cbar.ax.tick_params(labelsize=14) 
    plt.title('$\mathrm{label} \; = \; $' + str(labels_balanced[idx]), fontsize=14)
    plt.show(block=False)

    plt.subplot(1, 2, 2)
    plt.imshow(frames_balanced[idx, :, :] / 10**(scale_factor),
               interpolation='nearest',
               cmap=cm.coolwarm,
               origin='lower')
    plt.clim(-cbar_val, cbar_val)
    plt.plot(np.array([-50, 50, 50, -50, -50])*PIXELS/200 + PIXELS/2,
             np.array([-50, -50, 50, 50, -50])*PIXELS/200 + PIXELS/2,
             '--k',
             linewidth=1.0)
    plt.xticks([0, PIXELS-1], ['-100', '100'])
    plt.yticks([0, PIXELS-1], ['-100', '100'])
    plt.xlabel('$x \; \mathrm{(microns)}$', fontsize=FONT_SIZE)
    plt.ylabel('$y \; \mathrm{(microns)}$', fontsize=FONT_SIZE)
    
    matplotlib.rc('xtick', labelsize=FONT_SIZE) 
    matplotlib.rc('ytick', labelsize=FONT_SIZE) 

    cbar = plt.colorbar(ticks=[-cbar_val, 0, cbar_val])
    exponent_string = '$10^{' + str(scale_factor) + '}$' 
    cbar.ax.set_ylabel(r'field strength (units ' + r'$\times$' + ' ' + exponent_string + ')', fontsize=14, rotation=90)
    cbar.ax.tick_params(labelsize=14) 
    plt.title('$\mathrm{label} \; = \; $' + str(labels_balanced[idx]), fontsize=14)

    plt.tight_layout()
    plt.show(block=False)


def main():
    # balance_classes()
    # learn_simple()
    # plot_field()
    plot_field_noise()
    # predict_simple()
    # learn_noisy()
    

if __name__ == '__main__':
    main()