import matplotlib.pyplot as plt
from sbi.analysis import plot_summary
from sbi_particle_physics.objects.model import Model
import numpy as np
from torch import Tensor
from sbi.inference import NPE
from sbi_particle_physics.config import AXIS_FONTSIZE, BLUE_COLOR, ENCODED_DATA_LABELS, LEGEND_FONTSIZE, RED_COLOR, TICK_FONTSIZE, DATA_LABELS, PARAMETERS_LABEL, GREEN_COLOR, PLOTS_DIR

class Plotter:
    """
    Make plots to visualize the data, the predictions, etc.
    """

    @staticmethod
    def plot_a_sample_1D(sample : Tensor, parameter : Tensor, label : str):
        fig, ax = plt.subplots(figsize=(5.5,4)) # , constrained_layout=True
        labelo = "background" # f"$C_9={parameter.item():.3f}$"
        ax.hist(sample, bins=40, alpha=1,label=labelo, density=True, color=RED_COLOR)
        ax.set_xlabel(label, fontsize=AXIS_FONTSIZE+5, labelpad=0) # , fontweight='bold'
        ax.set_ylabel("Density", fontsize=AXIS_FONTSIZE+2, labelpad=0) # , fontweight='bold'
        ax.tick_params(labelsize=TICK_FONTSIZE-1, width=1.2)
        ax.locator_params(nbins=4)
        ax.grid(True, alpha=0.4, linewidth=0.8)
        leg = ax.legend(fontsize=LEGEND_FONTSIZE+5, frameon=True, framealpha=0.55, borderpad=0.4, labelspacing=0.3)
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor('white')
        plt.savefig(PLOTS_DIR / "viva" / f"background_{label.replace('\\','_')}.pdf")
        plt.show()

    @staticmethod
    def plot_a_sample(sample : Tensor, parameter : Tensor, raw: bool = True):
        lab = DATA_LABELS if raw else ENCODED_DATA_LABELS
        for i,label in enumerate(lab):
            Plotter.plot_a_sample_1D(sample[:,i], parameter, label)


    @staticmethod
    def _loss_lot(model : Model, detailed : bool) -> plt.Figure:
        values1 = model.training_loss
        values2 = model.validation_loss
        if detailed:
            values1 = values1[100:] # removes the 100 first epochs to focus on the last small improvements
            values2 = values2[100:]
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(range(1, len(values1)+1), values1, label="Training loss", color="blue", lw=2.3)
        ax.plot(range(1, len(values2)+1), values2, label="Validation loss", color="red", lw=2.3)
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Computer Modern Roman"]
        plt.rcParams["mathtext.fontset"] = "cm"
        ax.set_xlabel("Epoch", fontsize=AXIS_FONTSIZE-1)
        ax.set_ylabel("Loss", fontsize=AXIS_FONTSIZE-1)
        ax.legend(fontsize=LEGEND_FONTSIZE+1)
        ax.tick_params(labelsize=TICK_FONTSIZE-1)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    # plot train and validation loss during last training
    @staticmethod
    def plot_loss(model : Model, file : str | None = None):
        fig = Plotter._loss_lot(model, False)
        if file is None: fig.show()
        else: fig.savefig(f"{file}.png")
        plt.close(fig)
        if len(model.training_loss) > 110 and len(model.validation_loss) > 110:
            fig2 = Plotter._loss_lot(model, True)
            if file is None: fig2.show()
            else: fig2.savefig(f"{file}_zoom.png")
            plt.close(fig2)



    @staticmethod
    def plot_a_posterior_parameter(sampled_parameters : Tensor, label : str, true_value : float, range : tuple[float,float] = (None, None)):
        fig, ax = plt.subplots(figsize=(5,4), constrained_layout=True)
        ax.hist(
            sampled_parameters,
            bins=40,
            density=True,
            alpha=0.6,
            color=GREEN_COLOR,
            label="Posterior"
        )
        if range[0] is not None and range[1] is not None:
            ax.set_xlim(range[0], range[1])
        ax.axvline(true_value, color="red", linestyle="--", linewidth=2.5, label="True value")
        ax.set_xlabel(label, fontsize=AXIS_FONTSIZE+7, labelpad=0) # , fontweight='bold'
        ax.set_ylabel("Density", fontsize=AXIS_FONTSIZE+3, labelpad=0)  #, fontweight='bold'
        ax.tick_params(labelsize=TICK_FONTSIZE-2, width=1.2)
        ax.locator_params(nbins=4)
        ax.grid(True, alpha=0.4, linewidth=0.8)
        leg = ax.legend(
            fontsize=LEGEND_FONTSIZE+5,
            frameon=True,
            framealpha=0.55,
            handlelength=1.3,
            handleheight=0.6,
            handletextpad=0.4,
            borderpad=0.3,
            labelspacing=0.2,
        )
        leg.get_frame().set_linewidth(0.8)
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor('white')
        plt.savefig(PLOTS_DIR / "viva" / "posterior.pdf")
        plt.show()

    @staticmethod
    def plot_a_posterior(sampled_parameters : Tensor, true_value : Tensor):
        for i,label in enumerate(PARAMETERS_LABEL):
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
        ax.set_xlabel(label, fontsize=AXIS_FONTSIZE)
        ax.tick_params(labelsize=TICK_FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=LEGEND_FONTSIZE)
        plt.tight_layout()
        plt.show()

    # plot data generated from parameters drawn from the posterior estimation associated with the observed sample
    @staticmethod
    def plot_similar_data(model : Model, observed_sample : Tensor, n_samples : int, n_points : int):
        similar_data, similar_parameters = model.simulate_data_from_predicted_posterior(observed_sample, n_samples, n_points)
        for i,label in enumerate(DATA_LABELS):
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
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        axes = axes.flatten()

        for obs_idx, label in enumerate(DATA_LABELS):
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

            ax.set_xlabel(label, fontsize=AXIS_FONTSIZE)
            ax.set_ylabel("Density", fontsize=AXIS_FONTSIZE)
            ax.tick_params(labelsize=TICK_FONTSIZE)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=LEGEND_FONTSIZE)

        plt.tight_layout()
        plt.show()


    @staticmethod
    def poster_plot():
        #plt.rcParams['font.family'] = 'Arial'
        #plt.rcParams['font.weight'] = 'medium'
        #plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["#9e3f00","#1f5d8c", "#2b8c6b","#6a3d9a"])
        fig, ax = plt.subplots(figsize=(5.5,4), constrained_layout=True)
        ax.plot([0,1], [0,1], linestyle="--", label="Ideal posterior", linewidth=2.2)
        ax.plot([0,1], [0.5,0.5], linestyle="--", linewidth=2.2)
        ax.plot([0,1], [1,0], linestyle="--", linewidth=2.2)
        ax.plot([0,1], [0.2,0.7], linestyle="--", linewidth=2.2)
        ax.set_xlabel("x axis", fontsize=AXIS_FONTSIZE, labelpad=0) # , fontweight='bold'
        ax.set_ylabel("y axis", fontsize=AXIS_FONTSIZE, labelpad=0) # , fontweight='bold'
        ax.tick_params(labelsize=TICK_FONTSIZE, width=1.2)
        ax.locator_params(nbins=4)
        ax.grid(True, alpha=0.4, linewidth=0.8)
        leg = ax.legend(fontsize=LEGEND_FONTSIZE, frameon=True, framealpha=0.55, borderpad=0.4, labelspacing=0.3)
        leg.get_frame().set_linewidth(0.7)
        leg.get_frame().set_facecolor('white')
        plt.show()

    # courbes dominantes: #166d12 ou #145f10
    # courbes secondaires: #2f6fa3 ou #255a85
    # troisième courbe : # #6a3d9a ou #5a2d82
    # quatrième courbe : #6a3d9a
