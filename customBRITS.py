from typing import Union, Optional
import torch
from torch.utils.data import DataLoader, ConcatDataset
from pypots.imputation import BRITS

from pypots.imputation.brits.data import DatasetForBRITS
from pypots.data.checking import key_in_data_set

from pypots.utils.logging import logger
from pypots.utils.metrics import calc_mse
from pypots.imputation.base import BaseNNImputer
from typing import Union, Optional
from pypots.nn.functional import autocast, gather_listed_dicts

import numpy as np
import torch
from torch.utils.data import DataLoader
import os
try:
    import nni
except ImportError:
    pass


class CustomBaseNNImputer(BaseNNImputer):
    def _train_model(
        self,
        training_loader: DataLoader,
        val_loader: DataLoader = None,
    ) -> None:
        # each training starts from the very beginning, so reset the loss and model dict here
        self.best_loss = float("inf")
        self.best_model_dict = None

        try:
            training_step = 0
            for epoch in range(1, self.epochs + 1):
                self.model.train()
                epoch_train_loss_collector = []
                for seq_len, training_data_loader in training_loader.items():
                    for idx, data in enumerate(training_data_loader):
                        training_step += 1
                        inputs = self._assemble_input_for_training(data)
                        with autocast(enabled=self.amp_enabled):
                            self.optimizer.zero_grad()
                            results = self.model(inputs, calc_criterion=True)
                            loss = results["loss"].sum()
                            loss.backward()
                            self.optimizer.step()
                        epoch_train_loss_collector.append(loss.item())
                        # epoch_train_loss_collector.append(results["loss"].sum().item())

                        # save training loss logs into the tensorboard file for every step if in need
                        if self.summary_writer is not None:
                            self._save_log_into_tb_file(training_step, "training", results)

                    # mean training loss of the current epoch
                    mean_train_loss = np.mean(epoch_train_loss_collector)

                if val_loader is not None:
                    self.model.eval()
                    val_metric_collector = []
                    with torch.no_grad():
                        for seq_len, val_data_loader in val_loader.items():
                            imputation_loss_collector = []
                            for idx, data in enumerate(val_data_loader):
                                # print(data)
                                inputs = self._assemble_input_for_validating(data)
#                                 results = self.model.forward(inputs, training=False)
                                with autocast(enabled=self.amp_enabled):
                                    results = self.model(inputs, calc_criterion=True)

                                val_metric = results["metric"].sum()
                                val_metric_collector.append(val_metric.detach().item())
                                # imputation_loss_collector.append(imputation_mse)

                    mean_val_loss = np.mean(val_metric_collector)

                    # save validation loss logs into the tensorboard file for every epoch if in need
                    if self.summary_writer is not None:
                        val_loss_dict = {
                            "imputation_loss": mean_val_loss,
                        }
                        self._save_log_into_tb_file(epoch, "validating", val_loss_dict)

                    logger.info(
                        f"Epoch {epoch:03d} - "
                        f"training loss: {mean_train_loss:.4f}, "
                        f"validation loss: {mean_val_loss:.4f}"
                    )
                    mean_loss = mean_val_loss
                else:
                    logger.info(f"Epoch {epoch:03d} - training loss: {mean_train_loss:.4f}")
                    mean_loss = mean_train_loss

                if np.isnan(mean_loss):
                    logger.warning(f"‼️ Attention: got NaN loss in Epoch {epoch}. This may lead to unexpected errors.")

                if mean_loss < self.best_loss:
                    self.best_epoch = epoch
                    self.best_loss = mean_loss
                    self.best_model_dict = self.model.state_dict()
                    self.patience = self.original_patience
                else:
                    self.patience -= 1

                # save the model if necessary
                self._auto_save_model_if_necessary(
                    confirm_saving=self.best_epoch == epoch and self.model_saving_strategy == "better",
                    saving_name=f"{self.__class__.__name__}_epoch{epoch}_loss{mean_loss:.4f}",
                )

                if os.getenv("enable_tuning", False):
                    nni.report_intermediate_result(mean_loss)
                    if epoch == self.epochs - 1 or self.patience == 0:
                        nni.report_final_result(self.best_loss)

                if self.patience == 0:
                    logger.info("Exceeded the training patience. Terminating the training procedure...")
                    break

        except KeyboardInterrupt:  # if keyboard interrupt, only warning
            logger.warning("‼️ Training got interrupted by the user. Exist now ...")
        except Exception as e:  # other kind of exception follows below processing
            logger.error(f"❌ Exception: {e}")
            if self.best_model_dict is None:  # if no best model, raise error
                raise RuntimeError(
                    "Training got interrupted. Model was not trained. Please investigate the error printed above."
                )
            else:
                RuntimeWarning(
                    "Training got interrupted. Please investigate the error printed above.\n"
                    "Model got trained and will load the best checkpoint so far for testing.\n"
                    "If you don't want it, please try fit() again."
                )

        if np.isnan(self.best_loss):
            raise ValueError("Something is wrong. best_loss is Nan after training.")

        logger.info(f"Finished training. The best model is from epoch#{self.best_epoch}.")

class CustomBRITS(CustomBaseNNImputer, BRITS):

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        # Step 1: wrap the input data with classes Dataset and DataLoader
        training_dataloaders = {}
        for seq_len, groups in train_set.items():
            training_set = DatasetForBRITS(groups, return_X_ori=False, return_y=False, file_type=file_type)
            training_loader = DataLoader(
                training_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            training_dataloaders[seq_len] = training_loader
        validation_dataloaders = None
        if val_set is not None:
            validation_dataloaders = {}
            for seq_len, groups in val_set.items():
                if not key_in_data_set("X_ori", groups):
                    raise ValueError("val_set must contain 'X_ori' for model validation.")
                val_set = DatasetForBRITS(groups, return_X_ori=True, return_y=False, file_type=file_type)
                val_loader = DataLoader(
                    val_set,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                )
                validation_dataloaders[seq_len]=val_loader

        # Step 2: train the model and freeze it
        self._train_model(training_dataloaders, validation_dataloaders)
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()  # set the model as eval status to freeze it.

        # Step 3: save the model if necessary
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")

    def predict(
        self,
        test_set: Union[dict, str],
        file_type: str = "hdf5",
        **kwargs
    ) -> dict:
        self.model.eval()  # set the model as eval status to freeze it.
        test_dataloaders = {}
        for seq_len, groups in test_set.items():
            test_set = DatasetForBRITS(groups, return_X_ori=False, return_y=False, file_type=file_type)
            test_loader = DataLoader(
                test_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            test_dataloaders[seq_len]=test_loader

        imputation_results_group = {}
        with torch.no_grad():
            for seq_len, testing_data_loader in test_dataloaders.items():
                imputation_collector = []
                for idx, data in enumerate(testing_data_loader):
                    inputs = self._assemble_input_for_testing(data)
                    with autocast(enabled=self.amp_enabled):
                        results = self.model(inputs, **kwargs)
                    imputation_collector.append(results)
                imputation_results = gather_listed_dicts(imputation_collector)
                imputation_results_group[seq_len] = imputation_results

        
        
        return imputation_results_group

