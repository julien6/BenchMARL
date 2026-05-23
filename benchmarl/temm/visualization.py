#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from .types import TEMMDiagnostics, TEMMResult


class TEMMVisualizer:
    """Build interactive organizational visualizations from TEMM diagnostics."""

    def __init__(self, result: TEMMResult, diagnostics: TEMMDiagnostics):
        self.result = result
        self.diagnostics = diagnostics

    def build_figures(self) -> Dict[str, object]:
        try:
            import plotly.graph_objects as go
        except ImportError:
            return {}

        return {
            "figure_fit_summary": self.fit_summary(go),
            "figure_role_projection_pca": self.role_projection_pca(go),
            "figure_goal_projection_pca": self.goal_projection_pca(go),
            "figure_role_behavior_heatmap": self.role_behavior_heatmap(go),
            "figure_role_mission_matrix": self.role_mission_matrix(go),
            "figure_mission_graph": self.mission_graph(go),
        }

    def fit_summary(self, go):
        labels = ["Structural fit", "Functional fit", "Organizational fit"]
        values = [
            self.result.fit.structural,
            self.result.fit.functional,
            self.result.fit.organizational,
        ]
        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=values,
                    text=[f"{value:.3f}" for value in values],
                    textposition="auto",
                )
            ]
        )
        fig.update_layout(
            title="TEMM organizational fit summary",
            yaxis=dict(range=[0, 1], title="fit"),
            xaxis_title="metric",
        )
        return fig

    def role_projection_pca(self, go):
        points = _pca2(self.diagnostics.role_embeddings)
        labels = self.diagnostics.role_labels
        hover = [
            f"{traj_id}<br>agent={agent}<br>episode={episode}"
            for traj_id, agent, episode in zip(
                self.diagnostics.role_trajectory_ids,
                self.diagnostics.role_agent_names,
                self.diagnostics.role_episode_indices,
            )
        ]
        return _scatter_by_label(
            go,
            points,
            labels,
            hover,
            title="Latent role space projection (PCA)",
            x_title="PC1",
            y_title="PC2",
        )

    def goal_projection_pca(self, go):
        points = _pca2(self.diagnostics.goal_embeddings)
        labels = self.diagnostics.goal_labels
        hover = [
            f"episode={episode}<br>t={time}"
            for episode, time in zip(
                self.diagnostics.goal_episode_indices,
                self.diagnostics.goal_time_indices,
            )
        ]
        return _scatter_by_label(
            go,
            points,
            labels,
            hover,
            title="Goal latent space projection (PCA)",
            x_title="PC1",
            y_title="PC2",
        )

    def role_behavior_heatmap(self, go):
        role_ids = sorted(self.diagnostics.role_action_histograms)
        action_labels = self.diagnostics.action_bin_labels
        values = [
            self.diagnostics.role_action_histograms[role_id]
            for role_id in role_ids
        ]
        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=values or [[0.0]],
                    x=action_labels or ["-"],
                    y=role_ids or ["-"],
                    colorscale="Viridis",
                    colorbar=dict(title="frequency"),
                )
            ]
        )
        fig.update_layout(
            title="Role behavior heatmap",
            xaxis_title="action bin",
            yaxis_title="role",
        )
        return fig

    def role_mission_matrix(self, go):
        roles = self.diagnostics.role_mission_roles
        missions = self.diagnostics.role_mission_missions
        matrix = self.diagnostics.role_mission_matrix
        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=matrix or [[0.0]],
                    x=missions or ["-"],
                    y=roles or ["-"],
                    colorscale="Blues",
                    zmin=0,
                    zmax=1,
                    colorbar=dict(title="support"),
                )
            ]
        )
        fig.update_layout(
            title="Role-mission matrix",
            xaxis_title="mission",
            yaxis_title="role",
        )
        return fig

    def mission_graph(self, go):
        missions = self.result.missions
        goals = sorted({goal for mission in missions for goal in mission.goals})
        node_labels = [mission.id for mission in missions] + goals
        if not node_labels:
            node_labels = ["no mission"]
        positions = _circle_positions(len(node_labels))
        node_index = {label: index for index, label in enumerate(node_labels)}

        edge_x = []
        edge_y = []
        for mission in missions:
            for goal in mission.goals:
                if mission.id not in node_index or goal not in node_index:
                    continue
                x0, y0 = positions[node_index[mission.id]]
                x1, y1 = positions[node_index[goal]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

        node_x = [position[0] for position in positions]
        node_y = [position[1] for position in positions]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=1, color="#667085"),
                hoverinfo="none",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=node_labels,
                textposition="top center",
                marker=dict(size=16, color="#1f77b4"),
                hovertext=node_labels,
                hoverinfo="text",
                showlegend=False,
            )
        )
        fig.update_layout(
            title="Mission graph",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        return fig

    def write_html(self, output_dir: Path) -> Dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name, figure in self.build_figures().items():
            path = output_dir / f"{name}.html"
            figure.write_html(str(path), include_plotlyjs="cdn")
            paths[name] = path
        return paths


def _pca2(values) -> torch.Tensor:
    if not values:
        return torch.zeros((0, 2))
    x = torch.tensor(values, dtype=torch.float)
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)
    if x.shape[0] == 1:
        return torch.zeros((1, 2))
    x = x - x.mean(0, keepdim=True)
    try:
        _, _, v = torch.pca_lowrank(x, q=min(2, x.shape[0], x.shape[1]))
        projected = x @ v[:, : min(2, v.shape[1])]
    except RuntimeError:
        projected = torch.zeros((x.shape[0], 0))
    if projected.shape[1] < 2:
        pad = torch.zeros((projected.shape[0], 2 - projected.shape[1]))
        projected = torch.cat([projected, pad], dim=1)
    return projected[:, :2]


def _scatter_by_label(go, points, labels, hover, title, x_title, y_title):
    fig = go.Figure()
    unique_labels = sorted(set(labels)) if labels else ["-"]
    for label in unique_labels:
        indices = [index for index, item in enumerate(labels) if item == label]
        if not indices and label == "-":
            x = [0.0]
            y = [0.0]
            text = ["no data"]
        else:
            x = points[indices, 0].tolist()
            y = points[indices, 1].tolist()
            text = [hover[index] for index in indices]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=label,
                text=text,
                hoverinfo="text+x+y",
                marker=dict(size=9, opacity=0.8),
            )
        )
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    return fig


def _circle_positions(n_items: int) -> list[tuple[float, float]]:
    if n_items <= 1:
        return [(0.0, 0.0)]
    angles = torch.linspace(0, 2 * torch.pi, n_items + 1)[:-1]
    return [(float(torch.cos(angle)), float(torch.sin(angle))) for angle in angles]
