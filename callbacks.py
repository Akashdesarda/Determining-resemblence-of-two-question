from keras.callbacks import ModelCheckpoint, TerminateOnNaN, LearningRateScheduler, CSVLogger
from datetime import datetime
now = datetime.now().strftime("%d-%m-%Y")

#TODO: Add mlflow callback
def callbacks():
    """Keras callbacks which include ModelCheckpoint, CSVLogger, TensorBoard, LearningRateScheduler, TerminateOnNaN
    
    Returns
    -------
    list
        all callbacks
    """
    model_checkpoint = ModelCheckpoint(filepath='./assets/weights/dense_512/SimilarityNet-{epoch:02d}-{val_accuracy:.2f}.hdf5',
                save_best_only=True,
                save_weights_only=True,
                verbose=1)

    csv_logger = CSVLogger(filename=f"./assets/logs/logs-{now}.csv",
            append=True)

    def lr_schedule(epoch):
        if epoch < 10:
            return 0.001
        elif epoch < 50:
            return 0.0001
        else:
            return 0.00001
    
    lr_schedular = LearningRateScheduler(lr_schedule, verbose=1)

    terminate_on_nan = TerminateOnNaN()
    
    callbacks_list = [model_checkpoint, csv_logger, lr_schedular, terminate_on_nan]
    return callbacks_list