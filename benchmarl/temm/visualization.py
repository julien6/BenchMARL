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
            "figure_role_prototype_timelines": self.role_prototype_timelines(go),
            "figure_goal_prototype_timelines": self.goal_prototype_timelines(go),
            "figure_symbolic_rule_table": self.symbolic_rule_table(go),
            "figure_agent_role_episode_map": self.agent_role_episode_map(go),
            "figure_role_hierarchy_tree": self.role_hierarchy_tree(go),
            "figure_goal_mission_hierarchy_tree": self.goal_mission_hierarchy_tree(go),
            "figure_role_distance_heatmap": self.role_distance_heatmap(go),
            "figure_goal_distance_heatmap": self.goal_distance_heatmap(go),
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

    def role_prototype_timelines(self, go):
        fig = go.Figure()
        for role_id, prototype in self.diagnostics.role_prototypes.items():
            timeline = prototype.get("timeline", [])
            if not timeline:
                continue
            x = [step["t"] for step in timeline]
            y = [role_id for _ in timeline]
            text = [
                f"t={step['t']}<br>obs={step['observation']}<br>"
                f"action={step['action']}<br>delta={step['delta']}<br>"
                f"reward={step['reward']:.3f}"
                for step in timeline
            ]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers+lines",
                    name=role_id,
                    text=text,
                    hoverinfo="text",
                )
            )
        if not fig.data:
            fig.add_trace(go.Scatter(x=[0], y=["no role"], mode="markers"))
        fig.update_layout(
            title="Prototype role timelines",
            xaxis_title="time",
            yaxis_title="role",
        )
        return fig

    def goal_prototype_timelines(self, go):
        rows = []
        for goal_id, prototype in self.diagnostics.goal_prototypes.items():
            plan = prototype.get("plan", []) or ["-"]
            for index, token in enumerate(plan):
                rows.append(
                    [
                        goal_id,
                        prototype.get("episode", -1),
                        prototype.get("time", -1),
                        index,
                        token,
                    ]
                )
        if not rows:
            rows = [["-", -1, -1, 0, "-"]]
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=["goal", "episode", "time", "plan_step", "token"]
                    ),
                    cells=dict(values=_columns(rows)),
                )
            ]
        )
        fig.update_layout(title="Prototype goal timelines")
        return fig

    def symbolic_rule_table(self, go):
        rows = []
        for role in self.result.roles:
            rows.append(
                [
                    "role",
                    role.id,
                    " -> ".join(role.representative_pattern) or "-",
                ]
            )
        for goal in self.result.goals:
            rows.append(
                [
                    "goal",
                    goal.id,
                    " -> ".join(goal.representative_plan) or "-",
                ]
            )
        if not rows:
            rows = [["-", "-", "-"]]
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(values=["type", "id", "symbolic_rule"]),
                    cells=dict(values=_columns(rows)),
                )
            ]
        )
        fig.update_layout(title="Extracted symbolic rules")
        return fig

    def agent_role_episode_map(self, go):
        agents = sorted(set(self.diagnostics.role_agent_names))
        episodes = sorted(set(self.diagnostics.role_episode_indices))
        role_ids = sorted(set(self.diagnostics.role_labels))
        role_to_value = {role: index + 1 for index, role in enumerate(role_ids)}
        matrix = [[0 for _ in episodes] for _ in agents]
        text = [["-" for _ in episodes] for _ in agents]
        agent_index = {agent: index for index, agent in enumerate(agents)}
        episode_index = {episode: index for index, episode in enumerate(episodes)}
        for agent, episode, role in zip(
            self.diagnostics.role_agent_names,
            self.diagnostics.role_episode_indices,
            self.diagnostics.role_labels,
        ):
            row = agent_index[agent]
            col = episode_index[episode]
            matrix[row][col] = role_to_value.get(role, 0)
            text[row][col] = role
        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=matrix or [[0]],
                    x=episodes or ["-"],
                    y=agents or ["-"],
                    text=text or [["-"]],
                    hovertemplate="agent=%{y}<br>episode=%{x}<br>role=%{text}<extra></extra>",
                    colorscale="Turbo",
                    colorbar=dict(title="role id"),
                )
            ]
        )
        fig.update_layout(
            title="Agent-role map by episode",
            xaxis_title="episode",
            yaxis_title="agent",
        )
        return fig

    def role_hierarchy_tree(self, go):
        return _edge_tree(
            go,
            self.diagnostics.role_hierarchy_edges,
            fallback_nodes=self.diagnostics.role_distance_labels,
            title="Approximate role hierarchy",
        )

    def goal_mission_hierarchy_tree(self, go):
        nodes = [mission.id for mission in self.result.missions] + [
            goal.id for goal in self.result.goals
        ]
        return _edge_tree(
            go,
            self.diagnostics.goal_hierarchy_edges,
            fallback_nodes=nodes,
            title="Goal-mission hierarchy",
        )

    def role_distance_heatmap(self, go):
        return _distance_heatmap(
            go,
            self.diagnostics.role_distance_matrix,
            self.diagnostics.role_distance_labels,
            "Role prototype distance matrix",
        )

    def goal_distance_heatmap(self, go):
        return _distance_heatmap(
            go,
            self.diagnostics.goal_distance_matrix,
            self.diagnostics.goal_distance_labels,
            "Goal prototype distance matrix",
        )

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


def _columns(rows):
    return [list(column) for column in zip(*rows)]


def _distance_heatmap(go, matrix, labels, title):
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=matrix or [[0.0]],
                x=labels or ["-"],
                y=labels or ["-"],
                colorscale="Magma",
                zmin=0,
                zmax=1,
                colorbar=dict(title="distance"),
            )
        ]
    )
    fig.update_layout(title=title, xaxis_title="item", yaxis_title="item")
    return fig


def _edge_tree(go, edges, fallback_nodes, title):
    node_labels = sorted(
        set(fallback_nodes)
        | {edge["parent"] for edge in edges}
        | {edge["child"] for edge in edges}
    )
    if not node_labels:
        node_labels = ["no data"]
    positions = _tree_positions(node_labels, edges)
    edge_x = []
    edge_y = []
    edge_text = []
    for edge in edges:
        parent = edge["parent"]
        child = edge["child"]
        if parent not in positions or child not in positions:
            continue
        x0, y0 = positions[parent]
        x1, y1 = positions[child]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(f"{parent} -> {child}: {edge.get('support', 0):.3f}")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1, color="#667085"),
            text=edge_text,
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[positions[label][0] for label in node_labels],
            y=[positions[label][1] for label in node_labels],
            mode="markers+text",
            text=node_labels,
            textposition="top center",
            marker=dict(size=16, color="#1570ef"),
            hovertext=node_labels,
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _tree_positions(node_labels, edges):
    children = {}
    has_parent = set()
    for edge in edges:
        children.setdefault(edge["parent"], []).append(edge["child"])
        has_parent.add(edge["child"])
    roots = [node for node in node_labels if node not in has_parent] or node_labels[:1]
    levels = {}
    queue = [(root, 0) for root in roots]
    while queue:
        node, level = queue.pop(0)
        if node in levels and levels[node] <= level:
            continue
        levels[node] = level
        for child in children.get(node, []):
            queue.append((child, level + 1))
    for node in node_labels:
        levels.setdefault(node, 0)
    by_level = {}
    for node, level in levels.items():
        by_level.setdefault(level, []).append(node)
    positions = {}
    for level, nodes in by_level.items():
        nodes = sorted(nodes)
        for index, node in enumerate(nodes):
            x = index - (len(nodes) - 1) / 2
            y = -level
            positions[node] = (x, y)
    return positions
