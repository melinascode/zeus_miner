#!/usr/bin/env python3
"""Create per-lead CNN diagnostic plots, tables, and a Cursor canvas."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


VARIABLES = {
    "2m_temperature": ("2m temperature", "K"),
    "100m_u_component_of_wind": ("100m u-wind", "m/s"),
    "100m_v_component_of_wind": ("100m v-wind", "m/s"),
    "surface_solar_radiation_downwards": ("SSRD", "W/m²"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--canvas", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = json.loads(args.input.read_text())
    cycle = result["cycles"][0]
    diagnostics = cycle["lead_diagnostics"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_per_lead_csv(
        args.output_dir / f"{cycle['cycle']}_per_lead_metrics.csv",
        cycle,
    )
    write_summary_csv(
        args.output_dir / f"{cycle['cycle']}_summary.csv",
        cycle,
    )
    write_plot(
        args.output_dir / f"{cycle['cycle']}_per_lead_iwrmse_iwmae.png",
        cycle,
    )
    args.canvas.write_text(build_canvas(cycle, diagnostics))


def write_per_lead_csv(path: Path, cycle: dict) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "cycle",
                "lead_hour",
                "variable",
                "model",
                "iwRMSE",
                "iwMAE",
                "combined_error",
            ]
        )
        for lead in cycle["lead_diagnostics"]:
            for variable in VARIABLES:
                for model in ("raw_gfs", "cnn_corrected"):
                    metrics = lead["variables"][variable][model]
                    writer.writerow(
                        [
                            cycle["cycle"],
                            lead["lead_hour"],
                            variable,
                            model,
                            metrics["rmse"],
                            metrics["mae"],
                            metrics["combined_error"],
                        ]
                    )


def write_summary_csv(path: Path, cycle: dict) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "cycle",
                "variable",
                "unit",
                "raw_iwRMSE",
                "cnn_iwRMSE",
                "iwRMSE_skill_percent",
                "raw_iwMAE",
                "cnn_iwMAE",
                "iwMAE_skill_percent",
                "raw_combined",
                "cnn_combined",
                "combined_skill_percent",
            ]
        )
        for variable, (_, unit) in VARIABLES.items():
            result = cycle["variables"][variable]
            raw = result["raw_gfs"]
            cnn = result["cnn_corrected"]
            writer.writerow(
                [
                    cycle["cycle"],
                    variable,
                    unit,
                    raw["rmse"],
                    cnn["rmse"],
                    skill(raw["rmse"], cnn["rmse"]),
                    raw["mae"],
                    cnn["mae"],
                    skill(raw["mae"], cnn["mae"]),
                    raw["combined_error"],
                    cnn["combined_error"],
                    result["combined_error_skill"] * 100.0,
                ]
            )


def write_plot(path: Path, cycle: dict) -> None:
    leads = [row["lead_hour"] for row in cycle["lead_diagnostics"]]
    figure, axes = plt.subplots(4, 2, figsize=(18, 19), sharex=True)
    for row_index, (variable, (label, unit)) in enumerate(VARIABLES.items()):
        for column_index, (metric, metric_label) in enumerate(
            (("rmse", "iwRMSE"), ("mae", "iwMAE"))
        ):
            axis = axes[row_index, column_index]
            raw = [
                row["variables"][variable]["raw_gfs"][metric]
                for row in cycle["lead_diagnostics"]
            ]
            cnn = [
                row["variables"][variable]["cnn_corrected"][metric]
                for row in cycle["lead_diagnostics"]
            ]
            axis.plot(leads, raw, label="Raw GFS", linewidth=1.5)
            axis.plot(leads, cnn, label="Lead-aware CNN", linewidth=1.5)
            axis.set_title(f"{label}: {metric_label}")
            axis.set_ylabel(f"{metric_label} ({unit})")
            axis.set_xlim(0, 360)
            axis.set_xticks([0, 48, 96, 144, 192, 240, 288, 336, 360])
            axis.grid(alpha=0.25)
            axis.legend()
            if row_index == len(VARIABLES) - 1:
                axis.set_xlabel("Lead hour")
    figure.suptitle(
        f"Raw GFS vs lead-aware CNN — {cycle['cycle']} — leads 0–360",
        fontsize=16,
    )
    figure.text(
        0.5,
        0.005,
        "Validator-faithful latitude and Europe/Germany weighted metrics; "
        "361 hourly leads.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.98))
    figure.savefig(path, dpi=170)
    plt.close(figure)


def build_canvas(cycle: dict, diagnostics: list[dict]) -> str:
    plot_diagnostics = diagnostics[::3]
    leads = [str(row["lead_hour"]) for row in plot_diagnostics]
    chart_data = {}
    summary_rows = []
    per_lead_wins = 0
    total_per_lead = len(diagnostics) * len(VARIABLES)
    for variable, (label, unit) in VARIABLES.items():
        chart_data[variable] = {
            model_metric: [
                row["variables"][variable][model][metric]
                for row in plot_diagnostics
            ]
            for model_metric, model, metric in (
                ("rawRmse", "raw_gfs", "rmse"),
                ("cnnRmse", "cnn_corrected", "rmse"),
                ("rawMae", "raw_gfs", "mae"),
                ("cnnMae", "cnn_corrected", "mae"),
            )
        }
        result = cycle["variables"][variable]
        raw = result["raw_gfs"]
        cnn = result["cnn_corrected"]
        summary_rows.append(
            [
                label,
                unit,
                f"{raw['rmse']:.3f}",
                f"{cnn['rmse']:.3f}",
                f"{skill(raw['rmse'], cnn['rmse']):.2f}%",
                f"{raw['mae']:.3f}",
                f"{cnn['mae']:.3f}",
                f"{skill(raw['mae'], cnn['mae']):.2f}%",
                f"{result['combined_error_skill'] * 100.0:.2f}%",
            ]
        )
        per_lead_wins += sum(
            row["variables"][variable]["cnn_corrected"]["combined_error"]
            < row["variables"][variable]["raw_gfs"]["combined_error"]
            for row in diagnostics
        )

    best = max(
        cycle["variables"].items(),
        key=lambda item: item[1]["combined_error_skill"],
    )
    chart_blocks = []
    for variable, (label, unit) in VARIABLES.items():
        chart_blocks.append(
            f"""
      <Card>
        <CardHeader>{label}: per-lead iwRMSE and iwMAE</CardHeader>
        <CardBody>
          <LineChart
            categories={{leads}}
            series={{[
              {{ name: "Raw GFS iwRMSE", data: chartData[{json.dumps(variable)}].rawRmse }},
              {{ name: "CNN iwRMSE", data: chartData[{json.dumps(variable)}].cnnRmse, tone: "success" }},
              {{ name: "Raw GFS iwMAE", data: chartData[{json.dumps(variable)}].rawMae }},
              {{ name: "CNN iwMAE", data: chartData[{json.dumps(variable)}].cnnMae, tone: "info" }},
            ]}}
            height={{350}}
            showValues={{false}}
            showHoverGuide
          />
          <Text tone="secondary">
            X-axis: lead hour (0–360). Y-axis: weighted error ({unit}).
            Display sampled every 3 hours for interactive rendering; the PNG
            and CSV contain all 361 leads. Hover for exact displayed values.
            Source: diagnostic JSON, cycle {cycle['cycle']}.
          </Text>
        </CardBody>
      </Card>"""
        )

    return f'''import {{
  Callout,
  Card,
  CardBody,
  CardHeader,
  Code,
  Grid,
  H1,
  H2,
  LineChart,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
}} from "cursor/canvas";

const leads = {json.dumps(leads)};
const chartData = {json.dumps(chart_data)};
const summaryRows = {json.dumps(summary_rows)};

export default function CNNDiagnosticCycle() {{
  const theme = useHostTheme();
  return (
    <Stack
      gap={{20}}
      style={{{{
        padding: 24,
        maxWidth: 1260,
        color: theme.text.primary,
        background: theme.bg.editor,
      }}}}
    >
      <Stack gap={{8}}>
        <H1>360-hour CNN diagnostic: {cycle['cycle']}</H1>
        <Text tone="secondary">
          Raw GFS versus lead-aware gated residual CNN across all 361 hourly
          leads, scored with validator-faithful geographic weighting.
        </Text>
        <Row gap={{8}} style={{{{ flexWrap: "wrap" }}}}>
          <Pill tone="success">CNN wins all 4 aggregate variable scores</Pill>
          <Pill tone="neutral">Combined = (iwRMSE + iwMAE) / 2</Pill>
          <Pill tone="neutral">Europe/Germany regime</Pill>
        </Row>
      </Stack>

      <Callout tone="warning" title="Diagnostic evidence only">
        This cycle was previously inspected and explicitly excluded from model
        selection. It is useful for diagnosis, but it is not a locked,
        benchmark-valid estimate of future performance.
      </Callout>

      <Grid columns={{4}} gap={{12}}>
        <Stat label="Lead hours" value="0–360" />
        <Stat label="Aggregate wins" value="4/4" tone="success" />
        <Stat
          label="Per-lead combined wins"
          value="{per_lead_wins}/{total_per_lead}"
          tone="success"
        />
        <Stat
          label="Best combined skill"
          value="{best[1]['combined_error_skill'] * 100.0:.2f}%"
          tone="success"
        />
      </Grid>

      <H2>Full-cycle aggregate table</H2>
      <Table
        headers={{[
          "Variable", "Unit", "Raw iwRMSE", "CNN iwRMSE", "RMSE skill",
          "Raw iwMAE", "CNN iwMAE", "MAE skill", "Combined skill",
        ]}}
        rows={{summaryRows}}
      />
      <Text tone="secondary">
        Aggregate iwRMSE is computed from the full space-time squared-error
        kernel, not by averaging the 361 per-lead RMSE values.
      </Text>

      {''.join(chart_blocks)}

      <Callout tone="success" title="Result">
        The CNN reduces aggregate iwRMSE, iwMAE, and combined error for all four
        variables on this diagnostic bundle. SSRD has the largest combined
        reduction at <Code>{best[1]['combined_error_skill'] * 100.0:.2f}%</Code>.
      </Callout>

      <Text tone="secondary">
        Exact data: <Code>diagnostic_20260709T000000Z.json</Code>, checkpoint
        SHA256 <Code>b25f165acfb0444e…</Code>.
      </Text>
    </Stack>
  );
}}
'''


def skill(raw: float, corrected: float) -> float:
    return (raw - corrected) / raw * 100.0


if __name__ == "__main__":
    main()
