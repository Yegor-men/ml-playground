"""
Online CartPole experiment with episodic resets.

Only one-step graphs are retained: the critic prediction and next-observation
prediction wait for the next tick's targets. The GRU hidden state persists
within an episode, is zeroed on reset, and is detached before every encoder
call, so there is no BPTT/replay.
"""

import math
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

OBS_SIZE = 4
ACTION_SIZE = 1

# Experiment configuration
NUM_STEPS = 20_000
HIDDEN_SIZE = 64
LATENT_SIZE = 16
MODEL_WIDTH = 64
FORCE_MAGNITUDE = 10.0
THETA_LIMIT_DEGREES = 12.0
ACTOR_INITIAL_PREFERENCE = 0.0
ENCODER_LEARNING_RATE = 1e-4
ACTOR_LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 3e-4
PREDICTOR_LEARNING_RATE = 3e-4
LOCAL_REGULARIZATION_STRENGTH = 1.0
COUNTERFACTUAL_STRENGTH = 1.0
COUNTERFACTUAL_LOSS_WEIGHT = 0.25
GRAD_CLIP = 1.0
EMA_DECAY = 0.99
LOG_INTERVAL = 0
RENDER = True
RENDER_INTERVAL = 2
RENDER_HISTORY = 1_200
RENDER_FPS = 30.0
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _module_local_parameter_matrix(module: nn.Module) -> torch.Tensor | None:
    if isinstance(module, nn.Linear):
        matrix = module.weight.flatten(start_dim=1)
        if module.bias is not None:
            matrix = torch.cat([matrix, module.bias[:, None]], dim=1)
        return matrix

    if isinstance(module, nn.GRUCell):
        parts = [
            module.weight_ih.flatten(start_dim=1),
            module.weight_hh.flatten(start_dim=1),
        ]
        if module.bias:
            parts.extend([module.bias_ih[:, None], module.bias_hh[:, None]])
        return torch.cat(parts, dim=1)

    return None


def iter_local_parameter_matrices(
        model: nn.Module,
) -> Iterator[tuple[str, torch.Tensor]]:
    for name, module in model.named_modules():
        matrix = _module_local_parameter_matrix(module)
        if matrix is not None and matrix.size(1) > 1:
            yield name, matrix


def local_weight_variance_loss(model: nn.Module) -> torch.Tensor:
    variances = [
        matrix.var(dim=1, unbiased=False)
        for _, matrix in iter_local_parameter_matrices(model)
    ]
    if variances:
        return torch.cat(variances).mean()

    parameter = next(model.parameters(), None)
    if parameter is None:
        return torch.tensor(0.0)
    return parameter.new_zeros(())


def combined_local_weight_variance_loss(models: list[nn.Module]) -> torch.Tensor:
    losses = [local_weight_variance_loss(model) for model in models]
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


@torch.no_grad()
def combined_local_weight_variance_value(models: list[nn.Module]) -> float:
    return float(combined_local_weight_variance_loss(models).detach().cpu())


@dataclass
class StepSnapshot:
    step: int
    episode: int
    x: float
    theta: float
    action: float
    action_preference: float
    true_badness: float
    predicted_badness: float
    predicted_left_badness: float
    predicted_right_badness: float
    badness_ema: float
    critic_loss_ema: float
    predictor_loss_ema: float
    local_variance: float
    terminated: bool


@dataclass
class PendingGraphs:
    critic_chosen_badness: torch.Tensor
    critic_unchosen_badness: torch.Tensor
    predicted_obs: torch.Tensor


@dataclass
class TrainMetrics:
    critic_loss: float = 0.0
    counterfactual_loss: float = 0.0
    predictor_loss: float = 0.0
    regularization_loss: float = 0.0


class EpisodicCartPole:
    """
    CartPole dynamics with full-force left/right actions and Gym-style resets.

    The equations follow Gymnasium CartPole, but the action is represented as
    -1/+1 so the actor can use a full-force deterministic policy.
    """

    def __init__(
            self,
            seed: int,
            force_mag: float = 10.0,
            theta_limit_degrees: float = 12.0,
    ):
        if not 0.0 < theta_limit_degrees < 90.0:
            raise ValueError("THETA_LIMIT_DEGREES must be between 0 and 90.")
        if force_mag <= 0.0:
            raise ValueError("FORCE_MAGNITUDE must be positive.")

        self.rng = random.Random(seed)
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = force_mag
        self.tau = 0.02

        self.x_limit = 2.4
        self.theta_limit = math.radians(theta_limit_degrees)
        self.x_dot_limit = 8.0
        self.theta_dot_limit = 10.0
        self.max_raw_badness = 25.0

        minimum_force = self.minimum_recovery_force_at_limit()
        if self.force_mag <= minimum_force:
            raise ValueError(
                f"FORCE_MAGNITUDE={force_mag:g} is too weak for "
                f"THETA_LIMIT_DEGREES={theta_limit_degrees:g}; use a value "
                f"above {minimum_force:.2f}."
            )

        self.x = 0.0
        self.x_dot = 0.0
        self.theta = 0.0
        self.theta_dot = 0.0

    def reset(self) -> list[float]:
        self.x = self.rng.uniform(-0.05, 0.05)
        self.x_dot = self.rng.uniform(-0.05, 0.05)
        self.theta = self.rng.uniform(-0.05, 0.05)
        self.theta_dot = self.rng.uniform(-0.05, 0.05)
        return self.observation()

    def observation(self) -> list[float]:
        return [
            self._clamp(self.x / self.x_limit, -1.0, 1.0),
            self._clamp(self.x_dot / self.x_dot_limit, -1.0, 1.0),
            self._clamp(self.theta / self.theta_limit, -1.0, 1.0),
            self._clamp(self.theta_dot / self.theta_dot_limit, -1.0, 1.0),
        ]

    def step(self, action_unit: float) -> tuple[list[float], float, bool]:
        action_unit = 1.0 if action_unit >= 0.0 else -1.0
        force = self.force_mag * action_unit

        costheta = math.cos(self.theta)
        sintheta = math.sin(self.theta)
        temp = (
                       force + self.polemass_length * self.theta_dot * self.theta_dot * sintheta
               ) / self.total_mass
        theta_acc = (self.gravity * sintheta - costheta * temp) / (
                self.length
                * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )
        x_acc = temp - self.polemass_length * theta_acc * costheta / self.total_mass

        self.x += self.tau * self.x_dot
        self.x_dot += self.tau * x_acc
        self.theta += self.tau * self.theta_dot
        self.theta_dot += self.tau * theta_acc

        terminated = self.terminated()
        return self.observation(), self.badness(action_unit, terminated), terminated

    def badness(self, action_unit: float, terminated: bool = False) -> float:
        if terminated:
            return 1.0

        theta_frac = abs(self.theta) / self.theta_limit
        x_frac = abs(self.x) / self.x_limit
        x_dot_frac = abs(self.x_dot) / self.x_dot_limit
        theta_dot_frac = abs(self.theta_dot) / self.theta_dot_limit

        raw = (
                8.0 * self._barrier(theta_frac)
                + 2.0 * self._barrier(x_frac)
                + 0.15 * x_dot_frac * x_dot_frac
                + 0.10 * theta_dot_frac * theta_dot_frac
                + 0.01 * action_unit * action_unit
        )
        return min(raw / self.max_raw_badness, 1.0)

    def terminated(self) -> bool:
        return (
                not all(
                    math.isfinite(v)
                    for v in (self.x, self.x_dot, self.theta, self.theta_dot)
                )
                or abs(self.x) > self.x_limit
                or abs(self.theta) > self.theta_limit
        )

    def state_summary(self) -> str:
        return (
            f"x={self.x:+.2f} xdot={self.x_dot:+.2f} "
            f"theta={math.degrees(self.theta):+.1f}deg "
            f"thetadot={self.theta_dot:+.2f}"
        )

    def angular_acceleration_at_limit(self) -> float:
        theta = self.theta_limit
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = self.force_mag / self.total_mass
        return (self.gravity * sintheta - costheta * temp) / (
                self.length
                * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )

    def minimum_recovery_force_at_limit(self) -> float:
        return self.total_mass * self.gravity * math.tan(self.theta_limit)

    @staticmethod
    def _barrier(fraction: float) -> float:
        fraction = min(abs(fraction), 0.999)
        return fraction * fraction / max(1.0 - fraction * fraction, 0.02)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))


class LiveCartPoleWindow:
    def __init__(
            self,
            env: EpisodicCartPole,
            history_size: int,
            fps: float,
    ):
        import matplotlib
        from matplotlib.backends import BackendFilter, backend_registry

        interactive_backends = {
            backend.lower()
            for backend in backend_registry.list_builtin(BackendFilter.INTERACTIVE)
        }
        if matplotlib.get_backend().lower() not in interactive_backends:
            try:
                matplotlib.use("TkAgg")
            except Exception:
                pass

        import matplotlib.pyplot as plt
        from matplotlib import patches

        if matplotlib.get_backend().lower() not in interactive_backends:
            raise RuntimeError(
                "Matplotlib is using a non-interactive backend and cannot open a "
                "live window. Configure a GUI backend such as TkAgg, or set "
                "RENDER = False."
            )

        self.env = env
        self.plt = plt
        self.history_size = history_size
        self.min_frame_period = 1.0 / fps if fps > 0 else 0.0
        self.next_frame_time = time.monotonic()
        self.closed = False

        self.steps: deque[int] = deque(maxlen=history_size)
        self.true_badness: deque[float] = deque(maxlen=history_size)
        self.badness_ema: deque[float] = deque(maxlen=history_size)
        self.pred_badness: deque[float] = deque(maxlen=history_size)
        self.pred_left_badness: deque[float] = deque(maxlen=history_size)
        self.pred_right_badness: deque[float] = deque(maxlen=history_size)
        self.critic_loss: deque[float] = deque(maxlen=history_size)
        self.predictor_loss: deque[float] = deque(maxlen=history_size)
        self.local_variance: deque[float] = deque(maxlen=history_size)
        self.actions: deque[float] = deque(maxlen=history_size)
        self.action_preferences: deque[float] = deque(maxlen=history_size)

        plt.ion()
        self.fig = plt.figure(figsize=(11, 7))
        self.fig.canvas.manager.set_window_title("Online CartPole")
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        grid = self.fig.add_gridspec(2, 1, height_ratios=[2.0, 1.2])
        self.ax_world = self.fig.add_subplot(grid[0])
        self.ax_plot = self.fig.add_subplot(grid[1])

        self.ax_world.set_xlim(-env.x_limit - 0.9, env.x_limit + 0.9)
        self.ax_world.set_ylim(-0.55, 1.35)
        self.ax_world.set_aspect("equal", adjustable="box")
        self.ax_world.set_xticks([])
        self.ax_world.set_yticks([])
        self.ax_world.axhline(0.0, color="#444444", linewidth=2)
        self.ax_world.axvline(-env.x_limit, color="#999999", linewidth=1, linestyle=":")
        self.ax_world.axvline(env.x_limit, color="#999999", linewidth=1, linestyle=":")

        self.cart_width = 0.42
        self.cart_height = 0.22
        self.pole_length = 1.0
        self.pivot_y = self.cart_height / 2.0
        self.cart = patches.Rectangle(
            (-self.cart_width / 2.0, -self.cart_height / 2.0),
            self.cart_width,
            self.cart_height,
            linewidth=1.5,
            edgecolor="#222222",
            facecolor="#4C78A8",
        )
        self.ax_world.add_patch(self.cart)
        (self.pole_line,) = self.ax_world.plot([], [], linewidth=6, color="#F58518")
        (self.force_line,) = self.ax_world.plot([], [], linewidth=3, color="#54A24B")
        self.info = self.ax_world.text(
            0.02,
            0.96,
            "",
            transform=self.ax_world.transAxes,
            va="top",
            ha="left",
            family="monospace",
            fontsize=10,
        )

        (self.true_line,) = self.ax_plot.plot([], [], label="badness", color="#E45756")
        (self.ema_line,) = self.ax_plot.plot([], [], label="badness ema", color="#B279A2")
        (self.pred_line,) = self.ax_plot.plot(
            [], [], label="chosen pred", color="#4C78A8"
        )
        (self.pred_left_line,) = self.ax_plot.plot(
            [], [], label="left pred", color="#9D755D"
        )
        (self.pred_right_line,) = self.ax_plot.plot(
            [], [], label="right pred", color="#BAB0AC"
        )
        (self.actor_line,) = self.ax_plot.plot(
            [], [], label="action", color="#54A24B"
        )
        (self.preference_line,) = self.ax_plot.plot(
            [], [], label="action pref", color="#EECA3B"
        )
        (self.critic_loss_line,) = self.ax_plot.plot(
            [], [], label="critic loss ema", color="#F58518"
        )
        (self.predictor_loss_line,) = self.ax_plot.plot(
            [], [], label="predictor loss ema", color="#72B7B2"
        )
        (self.local_variance_line,) = self.ax_plot.plot(
            [], [], label="local variance", color="#B0B0B0"
        )
        self.ax_plot.set_xlabel("step")
        self.ax_plot.set_ylim(-1.05, 1.05)
        self.ax_plot.grid(True, alpha=0.25)
        self.ax_plot.legend(loc="upper left", ncols=3, fontsize=8)
        self.fig.tight_layout()

    def update(self, snapshot: StepSnapshot) -> bool:
        if self.closed or not self.plt.fignum_exists(self.fig.number):
            self.closed = True
            return False

        self.steps.append(snapshot.step)
        self.true_badness.append(snapshot.true_badness)
        self.badness_ema.append(snapshot.badness_ema)
        self.pred_badness.append(snapshot.predicted_badness)
        self.pred_left_badness.append(snapshot.predicted_left_badness)
        self.pred_right_badness.append(snapshot.predicted_right_badness)
        self.critic_loss.append(snapshot.critic_loss_ema)
        self.predictor_loss.append(snapshot.predictor_loss_ema)
        self.local_variance.append(snapshot.local_variance)
        self.actions.append(snapshot.action)
        self.action_preferences.append(snapshot.action_preference)

        cart_x = snapshot.x
        self.cart.set_xy((cart_x - self.cart_width / 2.0, -self.cart_height / 2.0))
        pivot_x = cart_x
        pole_x = pivot_x + self.pole_length * math.sin(snapshot.theta)
        pole_y = self.pivot_y + self.pole_length * math.cos(snapshot.theta)
        self.pole_line.set_data([pivot_x, pole_x], [self.pivot_y, pole_y])
        self.force_line.set_data(
            [cart_x, cart_x + 0.7 * snapshot.action],
            [-0.35, -0.35],
        )
        self.force_line.set_color("#54A24B" if snapshot.action >= 0 else "#E45756")

        self.info.set_text(
            f"step              {snapshot.step}\n"
            f"episode           {snapshot.episode}\n"
            f"x                 {snapshot.x:+.3f}\n"
            f"theta             {math.degrees(snapshot.theta):+.2f} deg\n"
            f"action            {snapshot.action:+.3f}\n"
            f"action pref       {snapshot.action_preference:+.3f}\n"
            f"badness           {snapshot.true_badness:.4f}\n"
            f"chosen pred       {snapshot.predicted_badness:.4f}\n"
            f"left/right pred   {snapshot.predicted_left_badness:.4f} / "
            f"{snapshot.predicted_right_badness:.4f}\n"
            f"critic loss ema   {snapshot.critic_loss_ema:.5f}\n"
            f"predictor loss ema {snapshot.predictor_loss_ema:.5f}\n"
            f"local variance    {snapshot.local_variance:.6f}\n"
            f"terminated        {snapshot.terminated}"
        )

        steps = list(self.steps)
        self.true_line.set_data(steps, list(self.true_badness))
        self.ema_line.set_data(steps, list(self.badness_ema))
        self.pred_line.set_data(steps, list(self.pred_badness))
        self.pred_left_line.set_data(steps, list(self.pred_left_badness))
        self.pred_right_line.set_data(steps, list(self.pred_right_badness))
        self.actor_line.set_data(steps, list(self.actions))
        self.preference_line.set_data(steps, list(self.action_preferences))
        self.critic_loss_line.set_data(steps, list(self.critic_loss))
        self.predictor_loss_line.set_data(steps, list(self.predictor_loss))
        self.local_variance_line.set_data(steps, list(self.local_variance))

        if steps:
            self.ax_plot.set_xlim(max(1, steps[-1] - self.history_size), steps[-1] + 1)
        max_positive = max(
            1.0,
            *self.true_badness,
            *self.badness_ema,
            *self.pred_badness,
            *self.pred_left_badness,
            *self.pred_right_badness,
            *self.critic_loss,
            *self.predictor_loss,
            *self.local_variance,
        )
        self.ax_plot.set_ylim(-1.05, max(1.05, max_positive * 1.1))

        now = time.monotonic()
        pause_for = max(0.001, self.next_frame_time - now)
        self.fig.canvas.draw_idle()
        self.plt.pause(pause_for)
        self.next_frame_time = max(
            time.monotonic(),
            self.next_frame_time + self.min_frame_period,
        )
        return not self.closed

    def wait_until_closed(self) -> None:
        if self.closed or not self.plt.fignum_exists(self.fig.number):
            return
        self.fig.canvas.manager.set_window_title("Online CartPole - finished")
        self.plt.ioff()
        self.plt.show()

    def _on_close(self, _event) -> None:
        self.closed = True


class Encoder(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, latent_size: int):
        super().__init__()
        self.gru = nn.GRUCell(obs_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, latent_size),
            nn.Tanh(),
        )

    def forward(
            self, obs: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.gru(obs, hidden)
        latent = self.to_latent(self.norm(hidden))
        return hidden, latent


class Actor(nn.Module):
    def __init__(self, latent_size: int, width: int, initial_preference: float):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(latent_size, width),
            nn.SiLU(),
            nn.Linear(width, width, bias=False),
            nn.SiLU(),
        )
        self.logits = nn.Linear(width, ACTION_SIZE)

        nn.init.zeros_(self.logits.weight)
        nn.init.constant_(self.logits.bias, initial_preference)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(latent)
        preference = torch.tanh(self.logits(x))
        hard_action = torch.where(
            preference >= 0.0,
            torch.ones_like(preference),
            -torch.ones_like(preference),
        )
        soft_action = preference
        action = hard_action.detach() + soft_action - soft_action.detach()
        return action, preference


class Critic(nn.Module):
    def __init__(self, latent_size: int, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size + ACTION_SIZE, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(torch.cat([latent, action], dim=-1)))


class Predictor(nn.Module):
    def __init__(self, latent_size: int, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size + ACTION_SIZE, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, OBS_SIZE),
            nn.Tanh(),
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([latent, action], dim=-1))


def make_obs_tensor(obs: list[float], device: torch.device) -> torch.Tensor:
    return torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


def update_pending_graphs(
        pending: PendingGraphs | None,
        target_obs: torch.Tensor,
        target_badness: float | None,
        encoder: Encoder,
        predictor: Predictor,
        critic: Critic,
        encoder_optimizer: torch.optim.Optimizer,
        predictor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        grad_clip: float,
        local_regularization_strength: float,
        counterfactual_strength: float,
        counterfactual_loss_weight: float,
) -> TrainMetrics:
    metrics = TrainMetrics()
    if pending is None or target_badness is None:
        return metrics

    critic_optimizer.zero_grad(set_to_none=True)
    target_badness_tensor = torch.tensor(
        [[target_badness]], dtype=torch.float32, device=target_obs.device
    )
    critic_regularization = local_weight_variance_loss(critic)
    chosen_loss = F.mse_loss(
        pending.critic_chosen_badness, target_badness_tensor
    )
    prediction_error = (
            target_badness_tensor - pending.critic_chosen_badness.detach()
    )
    counterfactual_target = (
            target_badness_tensor - counterfactual_strength * prediction_error
    ).clamp(0.0, 1.0)
    counterfactual_loss = F.mse_loss(
        pending.critic_unchosen_badness, counterfactual_target
    )
    critic_loss = chosen_loss + counterfactual_loss_weight * counterfactual_loss
    critic_loss = (
            critic_loss + local_regularization_strength * critic_regularization
    )
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
    critic_optimizer.step()

    encoder_optimizer.zero_grad(set_to_none=True)
    predictor_optimizer.zero_grad(set_to_none=True)
    predictor_regularization = combined_local_weight_variance_loss(
        [encoder, predictor]
    )
    predictor_loss = F.mse_loss(pending.predicted_obs, target_obs)
    predictor_loss = (
            predictor_loss + local_regularization_strength * predictor_regularization
    )
    predictor_loss.backward()
    nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
    nn.utils.clip_grad_norm_(predictor.parameters(), grad_clip)
    encoder_optimizer.step()
    predictor_optimizer.step()

    metrics.critic_loss = float(critic_loss.detach().cpu())
    metrics.counterfactual_loss = float(counterfactual_loss.detach().cpu())
    metrics.predictor_loss = float(predictor_loss.detach().cpu())
    metrics.regularization_loss = float(
        (critic_regularization + predictor_regularization).detach().cpu()
    )
    return metrics


def choose_action_and_update_actor(
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
        encoder: Encoder,
        actor: Actor,
        critic: Critic,
        actor_optimizer: torch.optim.Optimizer,
        grad_clip: float,
        local_regularization_strength: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
    float,
    float,
    float,
]:
    hidden_state = hidden_state.detach()
    # The world model is only trained by the predictor loss, not the critic.
    with torch.no_grad():
        _, latent = encoder(obs, hidden_state)

    _, action_preference = actor(latent)
    critic_latent = latent.detach()
    left_action = -torch.ones_like(action_preference)
    right_action = torch.ones_like(action_preference)
    predicted_left_badness = critic(critic_latent, left_action)
    predicted_right_badness = critic(critic_latent, right_action)
    choose_right = predicted_right_badness.detach() <= predicted_left_badness.detach()
    action = torch.where(choose_right, right_action, left_action)
    predicted_badness = torch.where(
        choose_right, predicted_right_badness, predicted_left_badness
    )
    unchosen_badness = torch.where(
        choose_right, predicted_left_badness, predicted_right_badness
    )

    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss = F.mse_loss(action_preference, action.detach())
    actor_regularization = local_weight_variance_loss(actor)
    actor_loss = actor_loss + local_regularization_strength * actor_regularization
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
    actor_optimizer.step()

    return (
        action.detach(),
        action_preference.detach(),
        predicted_badness,
        unchosen_badness,
        float(predicted_badness.detach().cpu()),
        float(predicted_left_badness.detach().cpu()),
        float(predicted_right_badness.detach().cpu()),
        float(actor_regularization.detach().cpu()),
    )


def make_predictor_graph(
        obs: torch.Tensor,
        hidden_state: torch.Tensor,
        action: torch.Tensor,
        encoder: Encoder,
        predictor: Predictor,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden, latent = encoder(obs, hidden_state.detach())
    predicted_obs = predictor(latent, action.detach())
    return hidden.detach(), predicted_obs


def train() -> None:
    if LOCAL_REGULARIZATION_STRENGTH < 0.0:
        raise ValueError("LOCAL_REGULARIZATION_STRENGTH must be non-negative.")
    if COUNTERFACTUAL_STRENGTH < 0.0:
        raise ValueError("COUNTERFACTUAL_STRENGTH must be non-negative.")
    if COUNTERFACTUAL_LOSS_WEIGHT < 0.0:
        raise ValueError("COUNTERFACTUAL_LOSS_WEIGHT must be non-negative.")

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    env = EpisodicCartPole(
        seed=SEED,
        force_mag=FORCE_MAGNITUDE,
        theta_limit_degrees=THETA_LIMIT_DEGREES,
    )
    encoder = Encoder(OBS_SIZE, HIDDEN_SIZE, LATENT_SIZE).to(DEVICE)
    actor = Actor(
        latent_size=LATENT_SIZE,
        width=MODEL_WIDTH,
        initial_preference=ACTOR_INITIAL_PREFERENCE,
    ).to(DEVICE)
    critic = Critic(LATENT_SIZE, MODEL_WIDTH).to(DEVICE)
    predictor = Predictor(LATENT_SIZE, MODEL_WIDTH).to(DEVICE)

    encoder_optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=ENCODER_LEARNING_RATE
    )
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=ACTOR_LEARNING_RATE)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=CRITIC_LEARNING_RATE)
    predictor_optimizer = torch.optim.AdamW(
        predictor.parameters(), lr=PREDICTOR_LEARNING_RATE
    )

    obs = env.reset()
    hidden_state = torch.zeros(1, HIDDEN_SIZE, device=DEVICE)
    pending: PendingGraphs | None = None
    target_badness: float | None = None
    reset_next_step = False
    episode = 1

    badness_ema = 0.0
    critic_loss_ema = 0.0
    predictor_loss_ema = 0.0
    action_ema = 0.0
    ema_decay = EMA_DECAY
    render_interval = max(1, RENDER_INTERVAL)
    window = (
        LiveCartPoleWindow(
            env=env,
            history_size=RENDER_HISTORY,
            fps=RENDER_FPS,
        )
        if RENDER
        else None
    )

    progress = tqdm(range(1, NUM_STEPS + 1), desc="ONLINE")
    for step in progress:
        obs_tensor = make_obs_tensor(obs, DEVICE)

        metrics = update_pending_graphs(
            pending=pending,
            target_obs=obs_tensor,
            target_badness=target_badness,
            encoder=encoder,
            predictor=predictor,
            critic=critic,
            encoder_optimizer=encoder_optimizer,
            predictor_optimizer=predictor_optimizer,
            critic_optimizer=critic_optimizer,
            grad_clip=GRAD_CLIP,
            local_regularization_strength=LOCAL_REGULARIZATION_STRENGTH,
            counterfactual_strength=COUNTERFACTUAL_STRENGTH,
            counterfactual_loss_weight=COUNTERFACTUAL_LOSS_WEIGHT,
        )
        pending = None

        if reset_next_step:
            obs = env.reset()
            obs_tensor = make_obs_tensor(obs, DEVICE)
            hidden_state = torch.zeros_like(hidden_state)
            target_badness = None
            reset_next_step = False
            episode += 1

        hidden_before_obs = hidden_state.detach()
        (
            action,
            action_preference,
            critic_badness,
            critic_unchosen_badness,
            predicted_badness,
            predicted_left_badness,
            predicted_right_badness,
            _actor_update_regularization,
        ) = choose_action_and_update_actor(
            obs=obs_tensor,
            hidden_state=hidden_before_obs,
            encoder=encoder,
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            grad_clip=GRAD_CLIP,
            local_regularization_strength=LOCAL_REGULARIZATION_STRENGTH,
        )

        hidden_state, predicted_obs = make_predictor_graph(
            obs=obs_tensor,
            hidden_state=hidden_before_obs,
            action=action,
            encoder=encoder,
            predictor=predictor,
        )
        pending = PendingGraphs(
            critic_chosen_badness=critic_badness,
            critic_unchosen_badness=critic_unchosen_badness,
            predicted_obs=predicted_obs,
        )

        action_value = float(action.squeeze().cpu())
        action_preference_value = float(action_preference.squeeze().cpu())
        obs, target_badness, terminated = env.step(action_value)

        current_badness = target_badness
        badness_ema = ema_decay * badness_ema + (1.0 - ema_decay) * current_badness
        critic_loss_ema = (
                ema_decay * critic_loss_ema + (1.0 - ema_decay) * metrics.critic_loss
        )
        predictor_loss_ema = (
                ema_decay * predictor_loss_ema + (1.0 - ema_decay) * metrics.predictor_loss
        )
        action_ema = ema_decay * action_ema + (1.0 - ema_decay) * action_value
        local_variance = combined_local_weight_variance_value(
            [encoder, actor, critic, predictor]
        )
        progress.set_postfix(
            badness=f"{badness_ema:.3f}",
            critic=f"{critic_loss_ema:.4f}",
            episode=episode,
        )

        if window is not None and (
                step == 1 or step % render_interval == 0 or terminated
        ):
            keep_running = window.update(
                StepSnapshot(
                    step=step,
                    episode=episode,
                    x=env.x,
                    theta=env.theta,
                    action=action_value,
                    action_preference=action_preference_value,
                    true_badness=current_badness,
                    predicted_badness=predicted_badness,
                    predicted_left_badness=predicted_left_badness,
                    predicted_right_badness=predicted_right_badness,
                    badness_ema=badness_ema,
                    critic_loss_ema=critic_loss_ema,
                    predictor_loss_ema=predictor_loss_ema,
                    local_variance=local_variance,
                    terminated=terminated,
                )
            )
            if not keep_running:
                break

        if terminated:
            reset_next_step = True

        if LOG_INTERVAL > 0 and (
                step % LOG_INTERVAL == 0 or step == 1
        ):
            tqdm.write(
                f"step={step:>6} "
                f"badness_ema={badness_ema:.4f} "
                f"critic_loss_ema={critic_loss_ema:.5f} "
                f"predictor_loss_ema={predictor_loss_ema:.5f} "
                f"pred_badness={predicted_badness:.4f} "
                f"pred_l={predicted_left_badness:.4f} "
                f"pred_r={predicted_right_badness:.4f} "
                f"action_ema={action_ema:+.4f} "
                f"action_pref={action_preference_value:+.3f} "
                f"local_var={local_variance:.6f} "
                f"episode={episode} "
                f"terminated={terminated} "
                f"{env.state_summary()}"
            )

    if window is not None and not window.closed:
        window.wait_until_closed()


def main() -> None:
    train()


if __name__ == "__main__":
    main()
