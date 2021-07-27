import time, simplejson, gc
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from sklearn.metrics import confusion_matrix
import excursion.models.gp
from excursion_new.plotting_ import *
from excursion.models.gp import get_gp
from excursion.acquisition import *
from excursion.models.fit import fit_hyperparams




def init_gp( testcase, algorithmopts, ninit, device):
    likelihood_type = algorithmopts["likelihood"]["type"]
    modelopts = algorithmopts["model"]["type"]
    kernelopts = algorithmopts["model"]["kernel"]
    prioropts = algorithmopts["model"]["prior"]

    n_dims = testcase.n_dims
    epsilon = float(algorithmopts["likelihood"]["epsilon"])
    dtype = torch.float64

    #
    # TRAIN DATA
    #
    X_grid = torch.Tensor(testcase.X_plot).to(device, dtype)
    init_type = algorithmopts["init_type"]
    noise_dist = MultivariateNormal(torch.zeros(ninit), torch.eye(ninit))

    if init_type == "random":
        indexs = np.random.choice(range(len(X_grid)), size=ninit, replace=False)
        X_init = X_grid[indexs].to(device, dtype)
        noises = epsilon * noise_dist.sample(torch.Size([])).to(device, dtype)
        y_init = testcase.true_functions[0](X_init).to(device, dtype) + noises
    #
    #
    # This is broken
    #
    #
    elif init_type == "worstcase":
        X_init = [X_grid[0]]
        X_init = torch.Tensor(X_init).to(device, dtype)
        noises = epsilon * noise_dist.sample(torch.Size([])).to(device, dtype)
        y_init = testcase.true_functions[0](X_init).to(device, dtype) + noises
    elif init_type == "custom":
        raise NotImplementedError("Not implemented yet")
    else:
        raise RuntimeError("No init data specification found")

    #
    # LIKELIHOOD
    #

    ## has bernoullilikelihood been used? if yes what was result if no then why not?

    if likelihood_type == "GaussianLikelihood":
        if epsilon > 0.0:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise=torch.tensor([epsilon])).to(device, dtype)
        elif epsilon == 0.0:
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=torch.tensor([epsilon])).to(device, dtype)
    else:
        raise RuntimeError("unknown likelihood")

    #
    # GAUSSIAN PROCESS
    #
    if modelopts == "ExactGP" and kernelopts == "RBF":
        model = ExcursionGP(X_init, y_init, likelihood, prioropts).to(device)

    elif modelopts == "GridGP" and kernelopts == "RBF":
        model = ExcursionGP(X_init, y_init, likelihood, prioropts, grid=testcase.rangedef).to(device)

    else:
        raise RuntimeError("unknown gpytorch model")

    # fit
    # print("X_init ", X_init)
    # print("y_init ", y_init)
    model.train()
    likelihood.train()
    fit_hyperparams(model, likelihood)

    return model, likelihood



class ExcursionSetEstimator:
    def __init__(self, testcase, algorithmopts, model, likelihood, device):
        self.x_new = torch.zeros(1, testcase.n_dims, dtype=torch.float64)
        self.y_new = torch.zeros(1, 1, dtype=torch.float64)
        self.acq_values = []

        self.this_iteration = 0
        self.confusion_matrix = []
        self.pct_correct = []
        self.walltime_step = []
        self.walltime_posterior = []
        self.device = device
        self.dtype = torch.float64

        self._acq_type = algorithmopts["acq"]["acq_type"]
        self._X_grid = testcase.X.to(self.device, self.dtype)
        self._epsilon = algorithmopts["likelihood"]["epsilon"]
        self._n_dims = testcase.n_dims

    def get_diagnostics(self, testcase, model, likelihood):
        thresholds = [-np.inf] + testcase.thresholds.tolist() + [np.inf]
        X_eval = testcase.X

        # noise_dist = MultivariateNormal(
        #    torch.zeros(len(X_eval)), torch.eye(len(X_eval))
        # )

        # noise = self._epsilon * noise_dist.sample(torch.Size([])).to(
        #    self.device, self.dtype
        # )

        noise = self._epsilon * Normal(
            torch.tensor([0.0]), torch.tensor([1.0])
        ).rsample(sample_shape=torch.Size([len(X_eval)]))

        # y_true = testcase.true_functions[0](X_eval).to(
        #    self.device, self.dtype
        # ) + noise.to(self.device, self.dtype)

        y_true = testcase.true_functions[0](X_eval).to(self.device, self.dtype)

        model.eval()
        likelihood.eval()
        y_pred = likelihood(model(X_eval.to(self.device, self.dtype))).mean.to(
            self.device, self.dtype
        )

        def label(y):
            for j in range(len(thresholds) - 1):
                if y < thresholds[j + 1] and y >= thresholds[j]:
                    return int(j)

        labels_pred = np.array([label(y) for y in y_pred])
        isnan_vector = np.isnan(labels_pred)

        labels_true = np.array([label(y) for y in y_true])

        # for y in labels_true:
        #    print('label ', y, type(y))

        # force y_true = y_train for those x in dataset

        conf_matrix = confusion_matrix(labels_true, labels_pred)
        self.confusion_matrix.append(conf_matrix)
        pct = np.diag(conf_matrix).sum() * 1.0 / len(X_eval)
        self.pct_correct.append(pct)
        print("pct ", pct)
        return None

    def step(self, testcase, algorithmopts, model, likelihood):
        # track wall time
        start_time = time.process_time()
        self.this_iteration += 1

        # print("Iteration ", self.this_iteration)

        ################################## this should be all one step with output
        ################################## number of batches, ordered max indices in grid

        acq_values_of_grid = self.get_acq_values(model, testcase)
        # print('ACQ VALUES')
        # print(acq_values_of_grid)

        batchgrid = batchGrid(
            acq_values_of_grid,
            device=self.device,
            dtype=self.dtype,
            n_dims=self._n_dims,
        )
        batchgrid.update(acq_values_of_grid, self.device, self.dtype)

        if algorithmopts["acq"]["batch"]:
            batchsize = algorithmopts["acq"]["batchsize"]
            batchtype = algorithmopts["acq"]["batchtype"]
            new_indexs = batchgrid.batch_types[batchtype](
                model,
                testcase,
                batchsize,
                self.device,
                self.dtype,
                likelihood=likelihood,
                algorithmopts=algorithmopts,
                excursion_estimator=self,
            )
            self.x_new = (
                torch.stack([testcase.X[index] for index in new_indexs])
                .to(self.device, self.dtype)
                .reshape(batchsize, self._n_dims)
            )

            # .reshape(batchsize, self._n_dims)

            # self.x_new = (testcase.X[new_indexs]).reshape(batchsize, self._n_dims).to(self.device, self.dtype)

        else:
            new_index = batchgrid.get_first_max_index(
                model, testcase, self.device, self.dtype
            )
            self.x_new = (
                testcase.X[new_index]
                .reshape(1, self._n_dims)
                .to(self.device, self.dtype)
            )

        ##################################

        gc.collect()
        torch.cuda.empty_cache()


        # get y from selected x
        # This is our expensive call to a black_box function

        noise_dist = MultivariateNormal(torch.zeros(1), torch.eye(1))
        noise = self._epsilon * noise_dist.sample(torch.Size([])).to(self.device, self.dtype)
        self.y_new = (testcase.true_functions[0](self.x_new).to(self.device, self.dtype)
                      + noise)

        # track wall time
        end_time = time.process_time() - start_time
        self.walltime_step.append(end_time)

        #print("x_new ", self.x_new.size(), self.x_new)
        #print("y_new ", self.y_new.size(), self.y_new)

        return self.x_new, self.y_new

    def get_acq_values(self, model, testcase):

        thresholds = [-np.inf] + testcase.thresholds.tolist() + [np.inf]

        start_time = time.time()

        acquisition_values_grid = acquisition_functions[self._acq_type](
            model, testcase, thresholds, self.device, self.dtype)

        end_time = time.time() - start_time

        # Used for plotting_. must plot after this call to step
        self.acq_values = acquisition_values_grid
            # print(f"the MES acquistion_values_grid is a: {type(acquisition_values_grid)}")
            # print(f"the MES \"          \" size is: {acquisition_values_grid.size()}")
            # print(f"the MES \"          \" is: {acquisition_values_grid}")
            #

        return acquisition_values_grid

    def update_posterior(self, testcase, algorithmopts, model, likelihood):
        # track wall time
        start_time = time.process_time()
        if self._n_dims == 1:
            inputs_i = torch.cat(
                (model.train_inputs[0], self.x_new), dim=0).flatten()
            targets_i = torch.cat(
                (model.train_targets.flatten(), self.y_new.flatten()), dim=0).flatten()
        else:
            inputs_i = torch.cat(
                (model.train_inputs[0], self.x_new), 0)
            targets_i = torch.cat(
                (model.train_targets, self.y_new), 0).flatten()

        #
        #
        # What is this line doing here?
        # model.set_train_data(inputs=inputs_i, targets=targets_i, strict=False)
        #
        #
        # model = get_gp(
        #    inputs_i, targets_i, likelihood, algorithmopts, testcase, self.device)

        model.set_train_data(inputs=inputs_i, targets=targets_i, strict=False)
        model.train()
        likelihood.train()
        fit_hyperparams(model, likelihood)

        # track wall time
        end_time = time.process_time() - start_time
        self.walltime_posterior.append(end_time)

        return model

    def plot_status(self, testcase, algorithmopts, model, outputfolder):

        if self._n_dims == 1:
            fig = plt.figure()
            plots_1D(
                model,
                testcase,
                acq=self.acq_values,
                acq_type=self._acq_type,
                x_new=self.x_new,
                device=self.device,
                dtype=self.dtype,
            )
            plt.tight_layout()
            figname = (
                outputfolder
                + str(self._n_dims)
                + "D_"
                + str(self.this_iteration)
                + "_"
                + str(self._acq_type)
                + ".png"
            )
            plt.savefig(figname)

        elif self._n_dims == 2:
            fig = plt.figure()
            if algorithmopts["acq"]["batch"]:
                batchsize = algorithmopts["acq"]["batchsize"]
            else:
                batchsize = 1

            plot = plots_2D(
                plt, model, testcase, self.device, self.dtype, batchsize
            )
            plt.tight_layout()
            figname = (
                outputfolder
                + str(self._n_dims)
                + "D_"
                + str(self.this_iteration)
                + "_"
                + str(self._acq_type)
                + ".png"
            )
            plt.savefig(figname)

        elif self._n_dims == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            plot = plots_3D(ax, model, testcase, self.device, self.dtype)
            plt.tight_layout()
            figname = (
                outputfolder
                + str(self._n_dims)
                + "D_"
                + str(self.this_iteration)
                + "_"
                + str(self._acq_type)
                + ".png"
            )
            plt.savefig(figname)

        else:
            pass

    def print_results(self, outputfolder, testcase, algorithmopts):
        print("Printing results...")

        # plot pct_correct
        filename_pct = (
            outputfolder
            + "/pct_correct_"
            + self._acq_type
            + "_"
            + algorithmopts["init_type"]
            + ".txt"
        )
        filename_pct_img = (
            outputfolder
            + "/pct_correct_"
            + self._acq_type
            + "_"
            + algorithmopts["init_type"]
            + ".png"
        )
        plt.clf()
        plt.title("percentage correct classification")
        plt.plot(range(self.this_iteration), self.pct_correct, label=self._acq_type)
        plt.xlabel("iteration")
        plt.ylabel("%")
        plt.legend()
        plt.hlines(y=1, xmax=self.this_iteration, xmin=0, color="grey", linestyle="--")
        plt.savefig(filename_pct_img)

        # tick_marks = np.arange(len(testcase.thresholds) + 2)
        # c = ["c_" + str(j) for j in range(len(testcase.thresholds) + 1)]

        # print pct to file
        with open(outputfolder + "pct_correct.txt", "w") as f:
            simplejson.dump(self.pct_correct, f)

        # confusion matrix plot
        for i in range(self.this_iteration):
            plt.clf()
            plt.title("Confusion matrix iter=" + str(i) + " " + self._acq_type)
            # plt.xticks(tick_marks, c, rotation=45)
            # plt.yticks(tick_marks, c)
            plt.imshow(self.confusion_matrix[i], cmap="binary")
            for i1 in range(self.confusion_matrix[i].shape[0]):
                for i2 in range(self.confusion_matrix[i].shape[1]):
                    plt.text(
                        i1,
                        i2,
                        self.confusion_matrix[i][i1][i2],
                        ha="center",
                        va="center",
                        color="red",
                    )
            plt.tight_layout()
            plt.savefig(outputfolder + "/CF_" + str(i) + ".png")

        # print to file walltimes
        filename_walltime_step = outputfolder + "walltime_step.txt"
        filename_walltime_posterior = outputfolder + "walltime_posterior.txt"
        with open(filename_walltime_step, "w") as g:
            simplejson.dump(self.walltime_step, g)
        with open(filename_walltime_posterior, "w") as h:
            simplejson.dump(self.walltime_posterior, h)

        # save to numpy file
        filename_pct_np = outputfolder + "/pct_correct.npy"
        filename_walltime_np = outputfolder + "/walltime.npy"
        np.save(filename_pct_np, self.pct_correct)
        np.save(filename_walltime_np, self.walltime_step)

        return None
