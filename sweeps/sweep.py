import wandb

from examples.envelope_minecart import main as envelope_minecart
from examples.envelope_dst import main as envelope_dst
from examples.gpi_pd_minecart import main as gpi_minecart
from examples.gpi_pd_dst import main as gpi_dst


def train():
    with wandb.init():
        config = wandb.config

        algo = config.algo.lower()
        env = config.env.lower()

        if algo == "envelope" and env == "minecart":
            envelope_minecart(
                total_timesteps=config.total_timesteps,
                seed=config.seed,
                learning_rate=config.learning_rate,
                gradient_updates=config.gradient_updates,
                batch_size=config.batch_size,
                tau=config.tau,
                initial_epsilon=config.initial_epsilon,
                final_epsilon=config.final_epsilon,
                epsilon_decay_fraction=config.epsilon_decay_fraction,
                log=True,
                wandb_mode="online",
            )

        elif algo == "envelope" and env == "dst":
            envelope_dst(
                total_timesteps=config.total_timesteps,
                seed=config.seed,
                learning_rate=config.learning_rate,
                gradient_updates=config.gradient_updates,
                batch_size=config.batch_size,
                tau=config.tau,
                initial_epsilon=config.initial_epsilon,
                final_epsilon=config.final_epsilon,
                epsilon_decay_fraction=config.epsilon_decay_fraction,
                log=True,
                wandb_mode="online",
            )

        elif algo == "gpi" and env == "minecart":
            gpi_minecart(
                total_timesteps=config.total_timesteps,
                seed=config.seed,
                learning_rate=config.learning_rate,
                gradient_updates=config.gradient_updates,
                final_epsilon=config.final_epsilon,
                log=True,
                wandb_mode="online",
            )

        elif algo == "gpi" and env == "dst":
            gpi_dst(
                total_timesteps=config.total_timesteps,
                seed=config.seed,
                learning_rate=config.learning_rate,
                gradient_updates=config.gradient_updates,
                log=True,
                wandb_mode="online",
            )

        else:
            raise ValueError(
                f"Unsupported algorithm/environment combination: "
                f"{algo}/{env}"
            )


if __name__ == "__main__":
    train()