import mlflow
from keras.callbacks import ModelCheckpoint, TerminateOnNaN, LearningRateScheduler, CSVLogger, Callback
from datetime import datetime
now = datetime.now().strftime("%d-%m-%Y")

class MlflowCallback(Callback):
    
    def on_train_begin(self, logs=None):
        print("[INFO]...Training has begun and tracking with Mlflow")
            
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metric('loss',logs['loss'])
        mlflow.log_metric('val_loss',logs['val_loss'])
        print(f"At end of Epoch {epoch} loss is {logs['loss']:.4f} and val_loss is {logs['val_loss']:.4f}")
        
def callbacks():
    """Keras callbacks which include ModelCheckpoint, CSVLogger, TensorBoard, LearningRateScheduler, TerminateOnNaN
    
    Returns
    -------
    list
        all callbacks
    """
    model_checkpoint = ModelCheckpoint(filepath='./assets/weights/DenseIncrementalSigmoid/exp2/SimilarityNet-epoch:{epoch:02d}-val_acc:{val_accuracy:.2f}.hdf5',
                save_best_only=True,
                save_weights_only=False,
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
    
    mlflow = MlflowCallback()
    
    callbacks_list = [mlflow, model_checkpoint, csv_logger, lr_schedular, terminate_on_nan]
    return callbacks_list