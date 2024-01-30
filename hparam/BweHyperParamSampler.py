from abc import ABC, abstractmethod
import d3rlpy
import optuna
from typing import Any, Dict, Self


class BweHyperParamSampler(ABC):
    """
    Abstract base class for hyperparameter samplers
    """

    @abstractmethod
    def sample_hyperparams(self, trial: optuna.Trial) -> Dict[str, Any]:
        pass

    @classmethod
    def create_sampler(cls, model_name: str) -> Self:
        """
        A fabric to create a hyperparameter sampler for the provided DRL model
        """
        if model_name.lower() == "cql":
            return CQLSampler()
        # ... other algos
        raise ValueError(f"BweHyperParamSampler.create_sampler -- Unknown algorithm: {model_name}")


class CQLSampler(BweHyperParamSampler):
    def sample_hyperparams(self, trial: optuna.Trial) -> Dict[str, Any]:
        # most of the ranges are taken from the original CQL paper: https://ar5iv.labs.arxiv.org/html/2006.04779
        actor_learning_rate = trial.suggest_float("actor_learning_rate", 1e-5, 1e-3)
        critic_learning_rate = trial.suggest_float("critic_learning_rate", 1e-5, 1e-3)
        temp_learning_rate = trial.suggest_float("temp_learning_rate", 1e-5, 1e-3)
        alpha_learning_rate = trial.suggest_float("alpha_learning_rate", 1e-5, 1e-3)

        # NOTE: sample also optimizer factories? Let them be Adam as default for now.
        # For Q-functions, let's select the factory type but use its default parameters for now.
        q_func_factory = self.create_q_func_factory(trial.suggest_categorical("q_func_factory", ["mean", "qr", "iqn"]))

        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
        gamma = trial.suggest_categorical("gamma", [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
        # target network sync update not Polyak update as for DQN
        tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.05, 0.1])
        n_critics = trial.suggest_categorical("n_critics", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # NOTE: initial temperature and alpha are not sampled, set to 1.0 for now

        # these ranges are taken from other papers that use CQL.
        # NOTE: I did not find how people change n_action_samples so left it as default
        # langrangian threshold
        alpha_threshold = trial.suggest_categorical("alpha_threshold", [2.0, 5.0, 10.0])
        # weight for conservative loss
        conservative_weight = trial.suggest_categorical("conservative_weight", [1.0, 5.0, 10.0])
        soft_q_backup = trial.suggest_categorical("soft_q_backup", [True, False])

        # network
        is_ac_same = trial.suggest_categorical("is_ac_same", [True, False])
        if is_ac_same:
            actor_encoder_factory = critic_encoder_factory = self.suggest_net_encoder_factory(trial)
        else:
            actor_encoder_factory = self.suggest_net_encoder_factory(trial)
            critic_encoder_factory = self.suggest_net_encoder_factory(trial)

        return {
            "actor_learning_rate": actor_learning_rate,
            "critic_learning_rate": critic_learning_rate,
            "temp_learning_rate": temp_learning_rate,
            "alpha_learning_rate": alpha_learning_rate,
            "q_func_factory": q_func_factory,
            "batch_size": batch_size,
            "gamma": gamma,
            "tau": tau,
            "n_critics": n_critics,
            "alpha_threshold": alpha_threshold,
            "conservative_weight": conservative_weight,
            "soft_q_backup": soft_q_backup,
            "actor_encoder_factory": actor_encoder_factory,
            "critic_encoder_factory": critic_encoder_factory,
        }

    # FIXME: move to optuna.utils submodule later
    def suggest_net_encoder_factory(self, trial: optuna.Trial) -> d3rlpy.models.encoders.EncoderFactory:
        hidden_layer_num = trial.suggest_categorical("hidden_layer_num", [1, 2, 3, 5, 10])
        hidden_layer_size = trial.suggest_categorical("hidden_layer_size", [32, 64, 128, 256, 512, 1024])
        # swish is softmax as far as I got it
        activation_function = trial.suggest_categorical(
            "activation_function", ["relu", "tanh", "swish", "gelu", "geglu"]
        )
        use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
        dropout_rate = trial.suggest_categorical("dropout_rate", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        return d3rlpy.models.encoders.VectorEncoderFactory(
            hidden_units=[hidden_layer_size] * hidden_layer_num,
            activation=activation_function,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            exclude_last_activation=False,  # don't exclude last activation.. why to do it?..
        )

    def create_q_func_factory(self, factory_name: str) -> d3rlpy.models.q_functions.QFunctionFactory:
        if factory_name == "mean":
            return d3rlpy.models.q_functions.MeanQFunctionFactory()
        elif factory_name == "qr":
            return d3rlpy.models.q_functions.QRQFunctionFactory()
        elif factory_name == "iqn":
            return d3rlpy.models.q_functions.IQNQFunctionFactory()
        else:
            raise ValueError(f"Unknown q function factory name: {factory_name}")
