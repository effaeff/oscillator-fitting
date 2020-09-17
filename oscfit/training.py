"""Learning routine to model the amplitudes of FRFs based on pose and frequency information"""

import numpy as np
from tqdm import tqdm

from pytorchutils.basic_trainer import BasicTrainer
from pytorchutils.globals import DEVICE, torch

class Trainer(BasicTrainer):
    """Wrapper class for training routine."""
    def __init__(self, config, model, preprocessor):
        BasicTrainer.__init__(self, config, model, preprocessor)

    def learn_from_epoch(self):
        """Training method."""
        epoch_loss = 0
        nb_scenarios = 0
        inp_batches, out_batches = self.get_batches_fn()

        for batch_idx in tqdm(range(len(inp_batches))):
            pred_out = self.predict(inp_batches[batch_idx])

            batch_loss = self.loss(
                pred_out,
                torch.Tensor(out_batches[batch_idx]).to(DEVICE)
            )

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            epoch_loss += batch_loss.item()

        epoch_loss /= nb_scenarios
        return epoch_loss

    def predict(self, inp):
        """
        Capsuled prediction method.
        Only single model usage supported for now.
        """
        inp = torch.Tensor(inp).to(DEVICE)
        return self.model(inp)

    def evaluate(self, inp, out):
        """Predition and error estimation for given input and output."""
        with torch.no_grad():
            # Switch to PyTorch's evaluation mode.
            # Some layers, which are used for regularization, e.g., dropout or batch norm layers,
            # behave differently, i.e., are turnd off, in evaluation mode
            # to prevent influencing the prediction accuracy.
            self.model.eval()
            pred_out = self.predict(inp)
             # RMSE is the default accuracy metric
            error = torch.sqrt(
                self.loss(
                    pred_out,
                    torch.Tensor(out).to(DEVICE)
                )
            )
            return pred_out, (error * 100.0)
