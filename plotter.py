import matplotlib.pyplot as plt
from sbi.analysis import plot_summary
from model import Model
import numpy as np
from torch import Tensor
from sbi.inference import NPE

class Plotter:
    """
    Make plots to visualize the data, the predictions, etc.
    """

    axis_fontsize = 21
    legend_fontsize = 15
    tick_fontsize = 15     
    data_labels = ["$q^2$", r"$\cos \theta_l$", r"$\cos \theta_d$", r"$\phi$"]
    parameters_labels = ["$C_9$"]   

    @staticmethod
    def plot_a_sample_1D(sample : Tensor, parameter : Tensor, label : str):
        fig, ax = plt.subplots(figsize=(7,4))
        ax.hist(
            sample,
            bins=40, 
            color="blue",
            alpha=0.7,
            label=f"$C_9={parameter.item():.3f}$"
        )
        ax.set_xlabel(label, fontsize=Plotter.axis_fontsize)
        ax.tick_params(labelsize=Plotter.tick_fontsize)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=Plotter.legend_fontsize)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_a_sample(sample : Tensor, parameter : Tensor):
        for i,label in enumerate(Plotter.data_labels):
            Plotter.plot_a_sample_1D(sample[:,i], parameter, label)


    @staticmethod
    def _loss_lot(model : Model, detailed : bool) -> plt.Figure:
        values1 = model.training_loss
        values2 = model.validation_loss
        if detailed:
            values1 = values1[100:] # removes the 100 first epochs to focus on the last small improvements
            values2 = values2[100:]
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(range(1, len(values1)+1), values1, label="Training")
        ax.plot(range(1, len(values2)+1), values2, label="Validation")
        ax.set_xlabel("Epoch", fontsize=Plotter.axis_fontsize)
        ax.set_ylabel("Loss", fontsize=Plotter.axis_fontsize)
        ax.legend(fontsize=Plotter.legend_fontsize)
        ax.tick_params(labelsize=Plotter.tick_fontsize)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    # plot train and validation loss during last training
    @staticmethod
    def plot_loss(model : Model, file : str | None = None):
        fig = Plotter._loss_lot(model, False)
        if file is None: fig.show()
        else: fig.savefig(f"{file}.png")
        if len(model.training_loss) > 110 and len(model.validation_loss) > 110:
            fig2 = Plotter._loss_lot(model, True)
            if file is None: fig2.show()
            else: fig2.savefig(f"{file}_zoom.png")


    @staticmethod
    def plot_a_posterior_parameter(sampled_parameters : Tensor, label : str, true_value : float):
        fig, ax = plt.subplots(figsize=(7,4))
        ax.hist(
            sampled_parameters,
            bins=40,
            density=True,
            alpha=0.6,
            color="green",
            label="posterior"
        )
        ax.set_xlabel(label, fontsize=Plotter.axis_fontsize)
        ax.tick_params(labelsize=Plotter.tick_fontsize)
        ax.grid(True, alpha=0.3)
        ax.axvline(true_value, color="red", linestyle="--", linewidth=2, label="True value")
        ax.legend(fontsize=Plotter.legend_fontsize)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_a_posterior(sampled_parameters : Tensor, true_value : Tensor):
        for i,label in enumerate(Plotter.parameters_labels):
            Plotter.plot_a_posterior_parameter(sampled_parameters[:,i], label, true_value[i])

    
    @staticmethod
    def plot_similar_data_1D(observed_sample : Tensor, similar_data : Tensor, label : str):
        fig, ax = plt.subplots(figsize=(7,4))
        ax.hist(
            observed_sample,
            bins=40, 
            color="red",
            alpha=1,
            label=f"True data",
            density=True
        )
        for i in range(similar_data.shape[0]):
             ax.hist(
                similar_data[i],
                bins=40, 
                alpha=0.3,
                color="blue",
                density=True
            )
        ax.set_xlabel(label, fontsize=Plotter.axis_fontsize)
        ax.tick_params(labelsize=Plotter.tick_fontsize)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=Plotter.legend_fontsize)
        plt.tight_layout()
        plt.show()

    # plot data generated from parameters drawn from the posterior estimation associated with the observed sample
    @staticmethod
    def plot_similar_data(model : Model, observed_sample : Tensor, n_samples : int, n_points : int):
        similar_data, similar_parameters = model.simulate_data_from_predicted_posterior(observed_sample, n_samples, n_points)
        for i,label in enumerate(Plotter.data_labels):
            Plotter.plot_similar_data_1D(observed_sample[:,i], similar_data[:,:,i], label)

    @staticmethod
    def compare_distributions(samples_list : Tensor, parameters_list : Tensor, n_samples_to_plot : int = 5):
        # Select indices to compare (evenly spaced across parameter range)
        n_total = len(parameters_list)
        if n_samples_to_plot > n_total:
            n_samples_to_plot = n_total

        # Sort by parameter value and select evenly spaced samples
        param_values = parameters_list.squeeze().cpu().numpy()
        sorted_indices = np.argsort(param_values)
        selected_indices = sorted_indices[np.linspace(0, n_total-1, n_samples_to_plot, dtype=int)]

        # Create color map
        colors = plt.cm.viridis(np.linspace(0, 1, n_samples_to_plot))

        # Create 4 subplots (one for each observable)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for obs_idx, label in enumerate(Plotter.data_labels):
            ax = axes[obs_idx]

            for i, sample_idx in enumerate(selected_indices):
                sample = samples_list[sample_idx]
                parameter = parameters_list[sample_idx]

                ax.hist(
                    sample[:, obs_idx].cpu().numpy(),
                    bins=50,
                    alpha=0.5,
                    color=colors[i],
                    label=f"$C_9={parameter.item():.2f}$",
                    density=True
                )

            ax.set_xlabel(label, fontsize=Plotter.axis_fontsize)
            ax.set_ylabel("Density", fontsize=Plotter.axis_fontsize)
            ax.tick_params(labelsize=Plotter.tick_fontsize)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=Plotter.legend_fontsize)

        plt.tight_layout()
        plt.show()