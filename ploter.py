import matplotlib.pyplot as plt
from sbi.analysis import plot_summary
from model import Model
import numpy as np

class Ploter:
    # make cool plots and test the performance 
    axis_fontsize = 21 # companion object variables
    legend_fontsize = 15
    tick_fontsize = 15     
    data_labels = ["$q^2$", r"$\cos \theta_l$", r"$\cos \theta_d$", r"$\phi$"]
    parameters_labels = ["$C_9$"]   

    @staticmethod
    def plot_a_sample_1D(sample, parameter, label):
        fig, ax = plt.subplots(figsize=(7,4))
        ax.hist(
            sample,
            bins=40, 
            color="blue",
            alpha=0.7,
            label=f"$C_9={parameter.item():.3f}$"
        )
        ax.set_xlabel(label, fontsize=Ploter.axis_fontsize)
        ax.tick_params(labelsize=Ploter.tick_fontsize)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=Ploter.legend_fontsize)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_a_sample(sample, parameter):
        for i,label in enumerate(Ploter.data_labels):
            Ploter.plot_a_sample_1D(sample[:,i], parameter, label)

    # plot train and validation loss during last training
    @staticmethod
    def plot_loss(neural_network):
        _ = plot_summary(
            neural_network,
            tags=["training_loss", "validation_loss"],
            figsize=(10, 2),
        )

    @staticmethod
    def plot_a_posterior_parameter(sampled_parameters, label, true_value):
        fig, ax = plt.subplots(figsize=(7,4))
        ax.hist(
            sampled_parameters,
            bins=40,
            density=True,
            alpha=0.6,
            color="green",
            label="posterior"
        )
        ax.set_xlabel(label, fontsize=Ploter.axis_fontsize)
        ax.tick_params(labelsize=Ploter.tick_fontsize)
        ax.grid(True, alpha=0.3)
        ax.axvline(true_value, color="red", linestyle="--", linewidth=2, label="True value")
        ax.legend(fontsize=Ploter.legend_fontsize)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_a_posterior(sampled_parameters, true_value):
        for i,label in enumerate(Ploter.parameters_labels):
            Ploter.plot_a_posterior_parameter(sampled_parameters[:,i], label, true_value[i])

    
    @staticmethod
    def plot_similar_data_1D(observed_sample, similar_data, label):
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
        ax.set_xlabel(label, fontsize=Ploter.axis_fontsize)
        ax.tick_params(labelsize=Ploter.tick_fontsize)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=Ploter.legend_fontsize)
        plt.tight_layout()
        plt.show()

    # plot data generated from parameters drawn from the posterior estimation associated with the observed sample
    @staticmethod
    def plot_similar_data(model, observed_sample, n_samples, n_points):
        similar_data, similar_parameters = model.simulate_data_from_predicted_posterior(observed_sample, n_samples, n_points)
        for i,label in enumerate(Ploter.data_labels):
            Ploter.plot_similar_data_1D(observed_sample[:,i], similar_data[:,:,i], label)

    @staticmethod
    def compare_distributions(samples_list, parameters_list, n_samples_to_plot=5):
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

        for obs_idx, label in enumerate(Ploter.data_labels):
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

            ax.set_xlabel(label, fontsize=Ploter.axis_fontsize)
            ax.set_ylabel("Density", fontsize=Ploter.axis_fontsize)
            ax.tick_params(labelsize=Ploter.tick_fontsize)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=Ploter.legend_fontsize)

        plt.tight_layout()
        plt.show()