"""Painel exploratório de produção e clima do milho no Brasil."""

from __future__ import annotations

from datetime import date
from html import escape
from io import BytesIO
import json
import math
from pathlib import Path
import re
import zipfile

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    LongTable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
    Flowable,
)


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
CLIMATE_DIR = PROCESSED / "clima_uf_safra"
EVALUATION_DIR = CLIMATE_DIR / "avaliacao"

UF_NAMES = {
    "AC": "Acre",
    "AL": "Alagoas",
    "AM": "Amazonas",
    "AP": "Amapá",
    "BA": "Bahia",
    "CE": "Ceará",
    "DF": "Distrito Federal",
    "ES": "Espírito Santo",
    "GO": "Goiás",
    "MA": "Maranhão",
    "MG": "Minas Gerais",
    "MS": "Mato Grosso do Sul",
    "MT": "Mato Grosso",
    "PA": "Pará",
    "PB": "Paraíba",
    "PE": "Pernambuco",
    "PI": "Piauí",
    "PR": "Paraná",
    "RJ": "Rio de Janeiro",
    "RN": "Rio Grande do Norte",
    "RO": "Rondônia",
    "RR": "Roraima",
    "RS": "Rio Grande do Sul",
    "SC": "Santa Catarina",
    "SE": "Sergipe",
    "SP": "São Paulo",
    "TO": "Tocantins",
}
SEASON_NAMES = {"1a": "1ª safra", "2a": "2ª safra"}
CLIMATE_COMPARISONS = {
    "Chuva acumulada no ciclo (mm)": (
        "chuva_acumulada_ciclo_mm",
        "cobertura_temporal_chuva_ciclo_pct",
    ),
    "Temperatura média do ar no ciclo (°C)": (
        "temp_ar_media_c_ciclo",
        "cobertura_temporal_temp_ar_media_c_ciclo_pct",
    ),
    "Temperatura mínima média no ciclo (°C)": (
        "temp_ar_min_c_ciclo",
        "cobertura_temporal_temp_ar_min_c_ciclo_pct",
    ),
    "Temperatura máxima média no ciclo (°C)": (
        "temp_ar_max_c_ciclo",
        "cobertura_temporal_temp_ar_max_c_ciclo_pct",
    ),
    "Umidade relativa média no ciclo (%)": (
        "umidade_relativa_media_pct_ciclo",
        "cobertura_temporal_umidade_relativa_media_pct_ciclo_pct",
    ),
    "Radiação solar acumulada no ciclo (kJ/m²)": (
        "radiacao_acumulada_ciclo_kj_m2",
        "cobertura_temporal_radiacao_ciclo_pct",
    ),
    "Velocidade média do vento no ciclo (m/s)": (
        "velocidade_vento_media_ms_ciclo",
        "cobertura_temporal_velocidade_vento_media_ms_ciclo_pct",
    ),
}
MAP_AGRICULTURE_METRICS = {
    "Produção (t)": "producao_t",
    "Área plantada (ha)": "area_plantada_ha",
    "Rendimento médio (kg/ha)": "rendimento_kg_ha",
}
MAP_COVERAGE_METRIC = "Cobertura climática (%)"
MAP_REFERENCE_MODES = {
    "Média do período selecionado": "current",
    "Variação em relação ao período anterior de igual duração (%)": "previous",
    "Variação em relação à média histórica anterior (%)": "historical",
}
MAP_GEOJSON_PATH = ROOT / "assets" / "brazil-states.geojson"
METRIC_NAMES = {
    "referencia_ano_anterior": "Referência: rendimento do ano anterior",
    "random_forest_clima": "Random Forest + clima",
}
PLOT_TEMPLATE = "plotly_white"
PLOT_COLORS = {"1a": "#238636", "2a": "#1f6feb"}


def read_semicolon_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {path}")
    return pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        na_values=["", "...", "..", "X"],
        keep_default_na=True,
    )


def convert_numeric(
    frame: pd.DataFrame, columns: tuple[str, ...] | list[str]
) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


@st.cache_data(show_spinner="Carregando as bases processadas...")
def load_data() -> dict[str, pd.DataFrame]:
    agriculture = read_semicolon_csv(
        PROCESSED / "ibge_milho_uf_ano_safra.csv"
    )
    agriculture = agriculture.loc[agriculture["nivel"] == "UF"].copy()
    agriculture["ano"] = pd.to_numeric(agriculture["ano"], errors="coerce")
    agriculture = convert_numeric(
        agriculture,
        [
            "area_plantada_ha",
            "area_colhida_ha",
            "producao_t",
            "rendimento_kg_ha",
        ],
    )
    agriculture["uf"] = agriculture["codigo_ibge"].astype(str).str.zfill(2).map(
        {
            "11": "RO",
            "12": "AC",
            "13": "AM",
            "14": "RR",
            "15": "PA",
            "16": "AP",
            "17": "TO",
            "21": "MA",
            "22": "PI",
            "23": "CE",
            "24": "RN",
            "25": "PB",
            "26": "PE",
            "27": "AL",
            "28": "SE",
            "29": "BA",
            "31": "MG",
            "32": "ES",
            "33": "RJ",
            "35": "SP",
            "41": "PR",
            "42": "SC",
            "43": "RS",
            "50": "MS",
            "51": "MT",
            "52": "GO",
            "53": "DF",
        }
    )
    if agriculture["uf"].isna().any():
        raise ValueError("Há códigos de UF não reconhecidos na base do IBGE.")

    monthly = read_semicolon_csv(
        CLIMATE_DIR / "clima_uf_ano_safra_mensal.csv"
    )
    cycle = read_semicolon_csv(
        CLIMATE_DIR / "clima_uf_ano_safra_ciclo.csv"
    )
    monthly = convert_numeric(
        monthly,
        [
            "ano",
            "mes",
            "area_plantada_ha_ponderacao",
            "area_plantada_ha_ibge",
            "producao_t",
            "rendimento_kg_ha",
            "chuva_mm",
            "area_coberta_chuva_mm_pct",
            "temp_ar_media_c",
            "area_coberta_temp_ar_media_c_pct",
            "temp_ar_min_c",
            "temp_ar_max_c",
            "umidade_relativa_media_pct",
            "area_coberta_umidade_relativa_media_pct",
            "temp_orvalho_media_c",
            "radiacao_global_kj_m2",
            "velocidade_vento_media_ms",
        ],
    )
    cycle = convert_numeric(
        cycle,
        [
            "ano",
            "area_plantada_ha",
            "area_colhida_ha",
            "producao_t",
            "rendimento_kg_ha",
            "chuva_acumulada_ciclo_mm",
            "chuva_parcial_ciclo_mm",
            "cobertura_temporal_chuva_ciclo_pct",
            "temp_ar_media_c_ciclo",
            "temp_ar_media_c_plantio",
            "temp_ar_min_c_ciclo",
            "temp_ar_max_c_ciclo",
            "umidade_relativa_media_pct_ciclo",
            "cobertura_temporal_temp_ar_media_c_ciclo_pct",
            "cobertura_temporal_temp_ar_min_c_ciclo_pct",
            "cobertura_temporal_temp_ar_max_c_ciclo_pct",
            "cobertura_temporal_umidade_relativa_media_pct_ciclo_pct",
            "radiacao_acumulada_ciclo_kj_m2",
            "cobertura_temporal_radiacao_ciclo_pct",
            "velocidade_vento_media_ms_ciclo",
            "cobertura_temporal_velocidade_vento_media_ms_ciclo_pct",
        ],
    )

    predictions = read_semicolon_csv(
        EVALUATION_DIR / "estimativas_teste_por_uf_safra.csv"
    )
    predictions = convert_numeric(
        predictions,
        [
            "ano",
            "area_plantada_ha",
            "producao_observada_t",
            "producao_estimada_modelo_t",
            "producao_estimada_referencia_t",
            "rendimento_observado_kg_ha",
            "erro_modelo_t",
            "erro_absoluto_modelo_t",
            "erro_referencia_t",
            "erro_absoluto_referencia_t",
        ],
    )
    national_predictions = read_semicolon_csv(
        EVALUATION_DIR / "estimativas_teste_brasil_safra.csv"
    )
    national_predictions = convert_numeric(
        national_predictions,
        [
            "ano",
            "producao_observada_t",
            "producao_estimada_modelo_t",
            "producao_estimada_referencia_t",
            "erro_modelo_t",
            "erro_referencia_t",
        ],
    )
    metrics = read_semicolon_csv(EVALUATION_DIR / "metricas_teste.csv")
    metrics = convert_numeric(
        metrics,
        [
            "n_observacoes",
            "mae_producao_t",
            "rmse_producao_t",
            "wape_producao_pct",
            "erro_medio_producao_t",
            "mae_rendimento_kg_ha",
            "rmse_rendimento_kg_ha",
            "wape_rendimento_pct",
            "r2_producao",
        ],
    )
    return {
        "agriculture": agriculture,
        "monthly": monthly,
        "cycle": cycle,
        "predictions": predictions,
        "national_predictions": national_predictions,
        "metrics": metrics,
    }


def production_millions(value: float) -> str:
    return f"{value / 1_000_000:,.2f} mi t".replace(",", "X").replace(".", ",").replace("X", ".")


def integer_format(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".")


def format_hover_data(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["safra_nome"] = result["safra"].map(SEASON_NAMES)
    return result


def style_plot(figure: go.Figure, height: int = 400) -> go.Figure:
    figure.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        margin=dict(l=10, r=10, t=48, b=10),
        legend_title_text="",
        hoverlabel=dict(bgcolor="white"),
    )
    return figure


def format_report_value(value: object) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, float):
        return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return str(value)


def report_table(frame: pd.DataFrame) -> LongTable:
    rows = [[str(column) for column in frame.columns]]
    rows.extend(
        [[format_report_value(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    )
    page_width, _ = landscape(letter)
    available_width = page_width - 30 * mm
    column_width = available_width / max(1, len(frame.columns))
    table = LongTable(
        rows,
        colWidths=[column_width] * len(frame.columns),
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f6feb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("LEADING", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d0d7de")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


class PlotlyChartFlowable(Flowable):
    """Draw a Plotly chart as vector shapes in the PDF without image conversion."""

    def __init__(self, figure: go.Figure, height: float = 78 * mm) -> None:
        super().__init__()
        self.figure = figure
        self.height = height
        self.width = 0

    def wrap(self, available_width: float, available_height: float) -> tuple[float, float]:
        self.width = available_width
        return self.width, self.height

    def draw(self) -> None:
        canvas = self.canv
        plot_left, plot_bottom = 58, 28
        traces = [
            trace
            for trace in self.figure.data
            if getattr(trace, "type", "") in {"scatter", "bar"}
        ]
        if not traces:
            return
        legend_width = 112 if len(traces) > 1 else 0
        plot_width = max(80, self.width - plot_left - 14 - legend_width)
        plot_height = max(45, self.height - plot_bottom - 18)

        def values(sequence: object) -> list[object]:
            if sequence is None:
                return []
            if hasattr(sequence, "tolist"):
                sequence = sequence.tolist()
            if isinstance(sequence, (str, bytes)):
                return [sequence]
            try:
                return list(sequence)  # type: ignore[arg-type]
            except TypeError:
                return [sequence]

        orientations = [getattr(trace, "orientation", None) for trace in traces]
        horizontal = any(orientation == "h" for orientation in orientations)
        all_x = [value for trace in traces for value in values(trace.x)]
        all_y = [value for trace in traces for value in values(trace.y)]
        category_values = all_y if horizontal else all_x
        category_axis = horizontal or any(
            isinstance(value, str) for value in category_values
        )
        if category_axis:
            categories = list(dict.fromkeys(str(value) for value in category_values))
            category_positions = {
                category: index for index, category in enumerate(categories)
            }
            category_count = max(1, len(categories))
            numeric_x = all_x if horizontal else all_y
            numeric_values = [
                float(value)
                for value in numeric_x
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            ]
            value_min = min(0.0, min(numeric_values, default=0.0))
            value_max = max(numeric_values, default=1.0)
            if value_max <= value_min:
                value_max = value_min + 1
            value_pad = (value_max - value_min) * 0.05
            value_min -= value_pad
            value_max += value_pad
        else:
            categories = []
            category_positions = {}
            numeric_x_values = [
                float(value)
                for value in all_x
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            ]
            numeric_y_values = [
                float(value)
                for value in all_y
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            ]
            x_min, x_max = min(numeric_x_values, default=0), max(numeric_x_values, default=1)
            y_min, y_max = min(numeric_y_values, default=0), max(numeric_y_values, default=1)
            if x_max <= x_min:
                x_max = x_min + 1
            if y_max <= y_min:
                y_max = y_min + 1
            x_pad, y_pad = (x_max - x_min) * 0.04, (y_max - y_min) * 0.06
            x_min, x_max = x_min - x_pad, x_max + x_pad
            y_min, y_max = y_min - y_pad, y_max + y_pad

        def to_color(value: object, fallback: str) -> colors.Color:
            try:
                return colors.HexColor(str(value))
            except (ValueError, TypeError):
                return colors.HexColor(fallback)

        palette = ["#238636", "#1f6feb", "#a371f7", "#d29922", "#f85149", "#39c5cf"]
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d0d7de"))
        canvas.setLineWidth(0.5)
        canvas.line(plot_left, plot_bottom, plot_left, plot_bottom + plot_height)
        canvas.line(plot_left, plot_bottom, plot_left + plot_width, plot_bottom)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#57606a"))
        if category_axis:
            tick_count = min(category_count, 8)
            tick_indices = sorted(
                {
                    round(index * (category_count - 1) / max(1, tick_count - 1))
                    for index in range(tick_count)
                }
            )
            for index in tick_indices:
                coordinate = plot_bottom + (index + 0.5) * plot_height / category_count
                if horizontal:
                    coordinate = plot_bottom + (category_count - 1 - index + 0.5) * plot_height / category_count
                    label = categories[index]
                    canvas.drawRightString(plot_left - 4, coordinate - 2, label[:18])
                else:
                    coordinate = plot_left + (index + 0.5) * plot_width / category_count
                    label = categories[index]
                    canvas.saveState()
                    canvas.translate(coordinate, plot_bottom - 4)
                    canvas.rotate(35)
                    canvas.drawString(0, 0, label[:12])
                    canvas.restoreState()
        else:
            for index in range(5):
                ratio = index / 4
                x_value = x_min + ratio * (x_max - x_min)
                y_value = y_min + ratio * (y_max - y_min)
                x_position = plot_left + ratio * plot_width
                y_position = plot_bottom + ratio * plot_height
                canvas.setFillColor(colors.HexColor("#57606a"))
                canvas.drawCentredString(x_position, plot_bottom - 12, f"{x_value:,.0f}")
                canvas.drawRightString(plot_left - 5, y_position - 2, f"{y_value:,.0f}")
                canvas.setStrokeColor(colors.HexColor("#eaeef2"))
                canvas.line(plot_left, y_position, plot_left + plot_width, y_position)

        stacked_bars: dict[str, float] = {}
        bar_trace_count = sum(getattr(trace, "type", "") == "bar" for trace in traces)
        trace_colors: list[colors.Color] = []
        for trace_index, trace in enumerate(traces):
            x_values, y_values = values(trace.x), values(trace.y)
            if len(x_values) != len(y_values):
                continue
            color_value = getattr(getattr(trace, "marker", None), "color", None)
            if isinstance(color_value, (list, tuple)) and color_value:
                color_value = color_value[0]
            if color_value is None:
                color_value = getattr(getattr(trace, "line", None), "color", None)
            color = to_color(color_value, palette[trace_index % len(palette)])
            trace_colors.append(color)
            canvas.setStrokeColor(color)
            canvas.setFillColor(color)
            if getattr(trace, "type", "") == "bar":
                orientation = getattr(trace, "orientation", None)
                is_horizontal = orientation == "h"
                for x_value, y_value in zip(x_values, y_values):
                    category = str(y_value if is_horizontal else x_value)
                    try:
                        value = float(x_value if is_horizontal else y_value)
                    except (TypeError, ValueError):
                        continue
                    if category not in category_positions:
                        continue
                    category_index = category_positions[category]
                    if is_horizontal:
                        row_height = plot_height / category_count
                        bar_height = row_height * 0.7 / max(1, bar_trace_count)
                        offset = (trace_index - (bar_trace_count - 1) / 2) * bar_height
                        start_value = stacked_bars.get(category, 0.0) if self.figure.layout.barmode == "stack" else 0.0
                        end_value = start_value + value
                        start_x = plot_left + (start_value - value_min) / (value_max - value_min) * plot_width
                        end_x = plot_left + (end_value - value_min) / (value_max - value_min) * plot_width
                        y = plot_bottom + (category_count - 1 - category_index + 0.5) * row_height + offset
                        canvas.rect(start_x, y - bar_height / 2, max(0.5, end_x - start_x), bar_height, fill=1, stroke=0)
                        if self.figure.layout.barmode == "stack":
                            stacked_bars[category] = end_value
                    else:
                        row_width = plot_width / category_count
                        bar_width = row_width * 0.7 / max(1, bar_trace_count)
                        offset = (trace_index - (bar_trace_count - 1) / 2) * bar_width
                        start_value = stacked_bars.get(category, 0.0) if self.figure.layout.barmode == "stack" else 0.0
                        end_value = start_value + value
                        y0 = plot_bottom + (start_value - value_min) / (value_max - value_min) * plot_height
                        y1 = plot_bottom + (end_value - value_min) / (value_max - value_min) * plot_height
                        x = plot_left + (category_index + 0.5) * row_width + offset
                        canvas.rect(x - bar_width / 2, y0, bar_width, max(0.5, y1 - y0), fill=1, stroke=0)
                        if self.figure.layout.barmode == "stack":
                            stacked_bars[category] = end_value
            else:
                points: list[tuple[float, float]] = []
                for x_value, y_value in zip(x_values, y_values):
                    try:
                        numeric_y = float(y_value)
                        if category_axis:
                            if str(x_value) not in category_positions:
                                continue
                            x_coordinate = plot_left + (category_positions[str(x_value)] + 0.5) * plot_width / category_count
                            y_coordinate = plot_bottom + (numeric_y - value_min) / (value_max - value_min) * plot_height
                        else:
                            numeric_x = float(x_value)
                            x_coordinate = plot_left + (numeric_x - x_min) / (x_max - x_min) * plot_width
                            y_coordinate = plot_bottom + (numeric_y - y_min) / (y_max - y_min) * plot_height
                        if math.isfinite(x_coordinate) and math.isfinite(y_coordinate):
                            points.append((x_coordinate, y_coordinate))
                    except (TypeError, ValueError):
                        continue
                mode = getattr(trace, "mode", "") or "lines"
                if "lines" in mode and len(points) > 1:
                    canvas.setLineWidth(1.2)
                    canvas.setDash(getattr(getattr(trace, "line", None), "dash", None) == "dash" and [4, 2] or [])
                    path = canvas.beginPath()
                    path.moveTo(*points[0])
                    for point in points[1:]:
                        path.lineTo(*point)
                    canvas.drawPath(path, stroke=1, fill=0)
                    canvas.setDash([])
                if "markers" in mode or "markers" not in mode and "lines" not in mode:
                    for point in points:
                        canvas.circle(point[0], point[1], 2.4, fill=1, stroke=0)

        if len(traces) > 1:
            legend_x = plot_left + plot_width + 9
            legend_y = self.height - 23
            canvas.setFont("Helvetica", 6.5)
            for index, trace in enumerate(traces[:8]):
                color = trace_colors[index]
                y = legend_y - index * 10
                canvas.setStrokeColor(color)
                canvas.setFillColor(color)
                canvas.line(legend_x, y + 2, legend_x + 10, y + 2)
                canvas.circle(legend_x + 5, y + 2, 1.8, fill=1, stroke=0)
                canvas.setFillColor(colors.HexColor("#57606a"))
                name = str(getattr(trace, "name", "") or f"Série {index + 1}")
                canvas.drawString(legend_x + 14, y, name[:18])
            if len(traces) > 8:
                canvas.setFillColor(colors.HexColor("#57606a"))
                canvas.drawString(legend_x + 14, legend_y - 80, f"+ {len(traces) - 8} séries")

        title = getattr(self.figure.layout.title, "text", None)
        if title:
            canvas.setFont("Helvetica-Bold", 9)
            canvas.setFillColor(colors.HexColor("#1f2328"))
            canvas.drawString(plot_left, self.height - 11, str(title)[:100])
        x_title = getattr(self.figure.layout.xaxis.title, "text", None)
        y_title = getattr(self.figure.layout.yaxis.title, "text", None)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#57606a"))
        if x_title:
            canvas.drawCentredString(plot_left + plot_width / 2, 1, str(x_title)[:70])
        if y_title:
            canvas.saveState()
            canvas.translate(8, plot_bottom + plot_height / 2)
            canvas.rotate(90)
            canvas.drawCentredString(0, 0, str(y_title)[:55])
            canvas.restoreState()
        canvas.restoreState()


def create_report_pdf(
    page_title: str,
    filters: list[str],
    sections: list[tuple[str, pd.DataFrame | str | go.Figure]],
) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=17 * mm,
        bottomMargin=17 * mm,
        title=f"Relatório — {page_title}",
        author="Dashboard do milho",
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            alignment=TA_LEFT,
            textColor=colors.HexColor("#1f2328"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSection",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#1f6feb"),
            spaceBefore=12,
            spaceAfter=6,
        )
    )
    story = [
        Paragraph(escape(page_title), styles["ReportTitle"]),
        Paragraph(f"Gerado em {date.today().strftime('%d/%m/%Y')}", styles["Normal"]),
        Spacer(1, 5 * mm),
        Paragraph("<b>Filtros selecionados</b>", styles["Heading3"]),
        Paragraph("<br/>".join(escape(item) for item in filters), styles["Normal"]),
        Spacer(1, 4 * mm),
    ]
    for section_title, content in sections:
        story.append(Paragraph(escape(section_title), styles["ReportSection"]))
        if isinstance(content, pd.DataFrame):
            if content.empty:
                story.append(Paragraph("Sem dados disponíveis para estes filtros.", styles["Normal"]))
            else:
                story.append(report_table(content))
        elif isinstance(content, go.Figure):
            story.append(PlotlyChartFlowable(content))
        else:
            story.append(Paragraph(escape(content), styles["Normal"]))
        story.append(Spacer(1, 3 * mm))

    def add_page_number(canvas: object, doc: SimpleDocTemplate) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#57606a"))
        canvas.drawString(15 * mm, 9 * mm, "Projeto acadêmico · estimativas exploratórias")
        canvas.drawRightString(
            landscape(letter)[0] - 15 * mm,
            9 * mm,
            f"Página {doc.page}",
        )
        canvas.restoreState()

    document.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return buffer.getvalue()


def exportable_tables(
    sections: list[tuple[str, pd.DataFrame | str | go.Figure]],
) -> list[tuple[str, pd.DataFrame]]:
    return [
        (title, content)
        for title, content in sections
        if isinstance(content, pd.DataFrame) and not content.empty
    ]


def export_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return normalized or "tabela"


def csv_tables_archive(
    tables: list[tuple[str, pd.DataFrame]], filters: list[str]
) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "filtros.txt",
            ("\n".join(filters) + "\n").encode("utf-8"),
        )
        for index, (title, frame) in enumerate(tables, start=1):
            filename = f"{index:02d}_{export_name(title)}.csv"
            csv_data = frame.to_csv(
                index=False, sep=";", decimal=","
            ).encode("utf-8-sig")
            archive.writestr(filename, csv_data)
    return buffer.getvalue()


def build_page_report(
    page: str,
    data: dict[str, pd.DataFrame],
    years: tuple[int, int],
    seasons: list[str],
) -> tuple[str, list[str], list[tuple[str, pd.DataFrame | str | go.Figure]]]:
    filters = [
        f"Período: {years[0]}–{years[1]}",
        f"Safras: {', '.join(SEASON_NAMES[season] for season in seasons)}",
    ]
    sections: list[tuple[str, pd.DataFrame | str | go.Figure]] = []
    if page == "Brasil":
        selected = data["agriculture"].loc[
            data["agriculture"]["ano"].between(years[0], years[1])
            & data["agriculture"]["safra"].isin(seasons)
        ]
        annual = (
            selected.groupby(["ano", "safra"], as_index=False)
            .agg(
                producao_t=("producao_t", "sum"),
                area_plantada_ha=("area_plantada_ha", "sum"),
                rendimento_kg_ha=("rendimento_kg_ha", "mean"),
            )
            .sort_values(["ano", "safra"])
        )
        annual["Safra"] = annual.pop("safra").map(SEASON_NAMES)
        annual = annual.rename(
            columns={
                "ano": "Ano",
                "producao_t": "Produção (t)",
                "area_plantada_ha": "Área plantada (ha)",
                "rendimento_kg_ha": "Rendimento médio simples (kg/ha)",
            }
        )
        sections.append(("Histórico nacional por ano e safra", annual))
        chart_data = selected.groupby(["ano", "safra"], as_index=False).agg(
            producao_t=("producao_t", "sum"),
            area_plantada_ha=("area_plantada_ha", "sum"),
        )
        chart_data["Safra"] = chart_data["safra"].map(SEASON_NAMES)
        production_chart = px.line(
            chart_data,
            x="ano",
            y="producao_t",
            color="Safra",
            markers=True,
            labels={"ano": "Ano", "producao_t": "Produção (t)", "Safra": "Safra"},
            title="Produção nacional por safra",
        )
        area_chart = px.line(
            chart_data,
            x="ano",
            y="area_plantada_ha",
            color="Safra",
            markers=True,
            labels={"ano": "Ano", "area_plantada_ha": "Área plantada (ha)", "Safra": "Safra"},
            title="Área plantada nacional por safra",
        )
        sections.extend(
            [
                ("Gráfico: produção nacional", production_chart),
                ("Gráfico: área plantada", area_chart),
            ]
        )
        if not selected.empty:
            latest_year = int(selected["ano"].max())
            latest = selected.loc[selected["ano"] == latest_year]
            by_state = (
                latest.groupby(["uf", "safra"], as_index=False)
                .agg(producao_t=("producao_t", "sum"))
                .sort_values("producao_t", ascending=False)
            )
            by_state["Estado"] = by_state["uf"].map(UF_NAMES)
            by_state["Safra"] = by_state.pop("safra").map(SEASON_NAMES)
            by_state = by_state.rename(columns={"producao_t": "Produção (t)"})
            sections.append((f"Produção por estado em {latest_year}", by_state[["Estado", "Safra", "Produção (t)"]]))
            state_chart = px.bar(
                by_state,
                x="Produção (t)",
                y="Estado",
                color="Safra",
                barmode="stack",
                orientation="h",
                title=f"Produção por estado em {latest_year}",
            )
            sections.append(("Gráfico: produção por estado", state_chart))
        title = "Milho no Brasil"
    elif page == "Clima e produção por estado":
        cycle = data["cycle"]
        available_states = sorted(cycle["uf"].dropna().unique())
        selected_state = st.session_state.get(
            "report_state_uf", available_states[0] if available_states else ""
        )
        selected_season = st.session_state.get(
            "report_safra", seasons[0] if seasons else "1a"
        )
        selected_climate = st.session_state.get(
            "report_climate_indicator", list(CLIMATE_COMPARISONS)[0]
        )
        selected_outcome = st.session_state.get(
            "report_climate_outcome", "Produção (t)"
        )
        climate_column, coverage_column = CLIMATE_COMPARISONS[selected_climate]
        outcome_column = (
            "producao_t" if selected_outcome == "Produção (t)" else "rendimento_kg_ha"
        )
        filters.extend(
            [
                f"Estado: {UF_NAMES.get(selected_state, selected_state)} ({selected_state})",
                f"Safra: {SEASON_NAMES.get(selected_season, selected_season)}",
                f"Indicador climático: {selected_climate}",
                f"Comparação: {selected_outcome}",
            ]
        )
        state = cycle.loc[
            cycle["uf"].eq(selected_state)
            & cycle["safra"].eq(selected_season)
            & cycle["ano"].between(years[0], years[1])
        ].sort_values("ano")
        columns = {
            "ano": "Ano",
            "area_plantada_ha": "Área plantada (ha)",
            "producao_t": "Produção (t)",
            "rendimento_kg_ha": "Rendimento (kg/ha)",
            climate_column: selected_climate,
            coverage_column: "Cobertura climática (%)",
            "situacao_chuva_ciclo": "Situação da chuva no ciclo",
            "dados_agricolas_preliminares": "Dados preliminares",
        }
        report_columns = [
            column
            for column in [
                "ano",
                "area_plantada_ha",
                outcome_column,
                climate_column,
                coverage_column,
                "situacao_chuva_ciclo",
                "dados_agricolas_preliminares",
            ]
            if column in state.columns
        ]
        comparison_data = state.dropna(subset=[climate_column, outcome_column])
        state_table_columns = [
            "ano",
            "area_plantada_ha",
            "area_colhida_ha",
            "producao_t",
            "rendimento_kg_ha",
            "chuva_acumulada_ciclo_mm",
            "chuva_parcial_ciclo_mm",
            "situacao_chuva_ciclo",
            "cobertura_temporal_chuva_ciclo_pct",
            "temp_ar_media_c_ciclo",
            "dados_agricolas_preliminares",
        ]
        state_table_columns = [
            column for column in state_table_columns if column in state.columns
        ]
        sections.append(
            (
                "Indicadores anuais por estado",
                state[state_table_columns].rename(
                    columns={
                        "ano": "Ano",
                        "area_plantada_ha": "Área plantada (ha)",
                        "area_colhida_ha": "Área colhida (ha)",
                        "producao_t": "Produção (t)",
                        "rendimento_kg_ha": "Rendimento (kg/ha)",
                        "chuva_acumulada_ciclo_mm": "Chuva acumulada no ciclo (mm)",
                        "chuva_parcial_ciclo_mm": "Chuva parcial no ciclo (mm)",
                        "situacao_chuva_ciclo": "Situação da chuva no ciclo",
                        "cobertura_temporal_chuva_ciclo_pct": "Cobertura da chuva no ciclo (%)",
                        "temp_ar_media_c_ciclo": "Temperatura média no ciclo (°C)",
                        "dados_agricolas_preliminares": "Dados preliminares",
                    }
                ),
            )
        )
        sections.append(
            (
                "Comparação anual entre clima e produção/rendimento",
                comparison_data[report_columns].rename(columns=columns),
            )
        )
        if not comparison_data.empty:
            scatter_data = comparison_data.copy()
            scatter_data["ano_grafico"] = scatter_data["ano"].astype(int).astype(str)
            climate_chart = px.scatter(
                scatter_data,
                x=climate_column,
                y=outcome_column,
                color="ano_grafico",
                labels={
                    climate_column: selected_climate,
                    outcome_column: selected_outcome,
                    "ano_grafico": "Ano",
                },
                title=f"{selected_climate} × {selected_outcome}",
            )
            production_chart = px.line(
                state,
                x="ano",
                y="producao_t",
                markers=True,
                labels={"ano": "Ano", "producao_t": "Produção (t)"},
                title=f"Produção de milho — {UF_NAMES.get(selected_state, selected_state)}",
            )
            yield_chart = px.line(
                state,
                x="ano",
                y="rendimento_kg_ha",
                markers=True,
                labels={"ano": "Ano", "rendimento_kg_ha": "Rendimento (kg/ha)"},
                title="Rendimento médio",
            )
            sections.extend(
                [
                    ("Gráfico: produção observada", production_chart),
                    ("Gráfico: rendimento médio", yield_chart),
                    ("Gráfico: clima versus produção/rendimento", climate_chart),
                ]
            )
        chosen_year = st.session_state.get(
            "report_climate_year",
            int(state["ano"].max()) if not state.empty else years[1],
        )
        monthly = data["monthly"].loc[
            data["monthly"]["uf"].eq(selected_state)
            & data["monthly"]["ano"].eq(chosen_year)
            & data["monthly"]["safra"].eq(selected_season)
        ].sort_values("mes")
        filters.append(f"Ano da série mensal: {chosen_year}")
        if not monthly.empty:
            monthly_columns = {
                "mes": "Mês",
                "chuva_mm": "Precipitação (mm)",
                "area_coberta_chuva_mm_pct": "Cobertura da área com chuva (%)",
                "temp_ar_media_c": "Temperatura média (°C)",
                "area_coberta_temp_ar_media_c_pct": "Cobertura da área com temperatura (%)",
                "umidade_relativa_media_pct": "Umidade relativa média (%)",
            }
            columns_to_show = [column for column in monthly_columns if column in monthly.columns]
            sections.append(
                ("Série climática mensal", monthly[columns_to_show].rename(columns=monthly_columns))
            )
            monthly_chart = go.Figure()
            monthly_chart.add_trace(
                go.Bar(
                    x=monthly["mes"],
                    y=monthly["chuva_mm"],
                    name="Precipitação mensal completa",
                    marker_color="#238636",
                )
            )
            monthly_chart.update_layout(
                title="Chuva mensal estimada · média espacial ponderada",
                xaxis_title="Mês (UTC)",
                yaxis_title="Precipitação (mm)",
            )
            sections.append(("Gráfico: chuva mensal", monthly_chart))
        title = f"Clima e produção — {UF_NAMES.get(selected_state, selected_state)}"
    elif page == "Mapa interativo por UF":
        selected_metric = st.session_state.get(
            "map_metric", next(iter(MAP_AGRICULTURE_METRICS))
        )
        selected_coverage = st.session_state.get(
            "map_coverage_indicator", next(iter(CLIMATE_COMPARISONS))
        )
        selected_reference = st.session_state.get(
            "map_reference_mode", "Média do período selecionado"
        )
        if selected_reference not in MAP_REFERENCE_MODES:
            selected_reference = "Média do período selecionado"
        reference_mode = MAP_REFERENCE_MODES[selected_reference]
        map_frame = build_state_map_data(
            data,
            years,
            seasons,
            selected_metric,
            selected_coverage,
            reference_mode,
        )
        filters.extend(
            [
                f"Período: {years[0]}–{years[1]}",
                f"Safras: {', '.join(SEASON_NAMES[season] for season in seasons)}",
                f"Indicador: {selected_metric}",
                f"Visualização: {selected_reference}",
                f"Indicador de cobertura: {selected_coverage}",
            ]
        )
        sections.append(
            (
                "Indicadores do mapa por Unidade da Federação",
                map_report_table(map_frame, selected_metric),
            )
        )
        sections.append(
            (
                "Mapa interativo por UF",
                build_state_choropleth(
                    map_frame,
                    selected_metric,
                    reference_mode,
                    years,
                    seasons,
                ),
            )
        )
        title = "Mapa interativo por UF"
    elif page == "Avaliação":
        period = st.session_state.get(
            "report_evaluation_period", "2020–2024 · sem ano preliminar"
        )
        period_key = (
            "teste_2020_2024_nao_preliminar"
            if period.startswith("2020–2024")
            else "teste_2020_2025"
        )
        selected_state = st.session_state.get(
            "report_evaluation_state",
            sorted(data["predictions"]["uf"].dropna().unique())[0],
        )
        predictions = data["predictions"].loc[
            data["predictions"]["ano"].le(2024)
            if period_key == "teste_2020_2024_nao_preliminar"
            else data["predictions"].index == data["predictions"].index
        ]
        metrics = data["metrics"].loc[data["metrics"]["periodo"].eq(period_key)].copy()
        metrics["modelo"] = metrics["modelo"].map(METRIC_NAMES)
        metrics["granularidade"] = metrics["granularidade"].map(
            {"Brasil_ano_safra": "Brasil por ano/safra", "UF_safra": "UF por safra"}
        )
        metrics = metrics[
            [
                "granularidade",
                "modelo",
                "n_observacoes",
                "mae_producao_t",
                "rmse_producao_t",
                "wape_producao_pct",
                "r2_producao",
            ]
        ].rename(
            columns={
                "granularidade": "Granularidade",
                "modelo": "Método",
                "n_observacoes": "Observações",
                "mae_producao_t": "MAE (t)",
                "rmse_producao_t": "RMSE (t)",
                "wape_producao_pct": "WAPE (%)",
                "r2_producao": "R²",
            }
        )
        national = data["national_predictions"].copy()
        if period_key == "teste_2020_2024_nao_preliminar":
            national = national.loc[national["ano"].le(2024)]
        state_predictions = predictions.loc[predictions["uf"].eq(selected_state)].copy()
        filters = [
            f"Período de avaliação: {period}",
            f"Estado selecionado: {UF_NAMES.get(selected_state, selected_state)} ({selected_state})",
        ]
        national["safra"] = national["safra"].map(SEASON_NAMES)
        state_predictions["safra"] = state_predictions["safra"].map(SEASON_NAMES)
        sections.extend(
            [
                ("Métricas de avaliação", metrics),
                (
                    "Estimativas nacionais por ano e safra",
                    national[
                        [
                            "ano",
                            "safra",
                            "producao_observada_t",
                            "producao_estimada_modelo_t",
                            "producao_estimada_referencia_t",
                        ]
                    ].rename(
                        columns={
                            "ano": "Ano",
                            "safra": "Safra",
                            "producao_observada_t": "Produção observada (t)",
                            "producao_estimada_modelo_t": "Random Forest (t)",
                            "producao_estimada_referencia_t": "Referência ano anterior (t)",
                        }
                    ),
                ),
                (
                    f"Estimativas por estado — {UF_NAMES.get(selected_state, selected_state)}",
                    state_predictions[
                        [
                            "ano",
                            "safra",
                            "producao_observada_t",
                            "producao_estimada_modelo_t",
                            "producao_estimada_referencia_t",
                            "erro_modelo_t",
                        ]
                    ].rename(
                        columns={
                            "ano": "Ano",
                            "safra": "Safra",
                            "producao_observada_t": "Produção observada (t)",
                            "producao_estimada_modelo_t": "Random Forest (t)",
                            "producao_estimada_referencia_t": "Referência ano anterior (t)",
                            "erro_modelo_t": "Erro do modelo (t)",
                        }
                    ),
                ),
            ]
        )
        national_chart = px.line(
            national,
            x="ano",
            y=[
                "producao_observada_t",
                "producao_estimada_modelo_t",
                "producao_estimada_referencia_t",
            ],
            color="safra",
            line_dash="variable",
            markers=True,
            labels={
                "ano": "Ano",
                "value": "Produção (t)",
                "variable": "Série",
                "safra": "Safra",
            },
            title="Produção nacional: observado e estimado",
        )
        state_chart = px.line(
            state_predictions,
            x="ano",
            y=[
                "producao_observada_t",
                "producao_estimada_modelo_t",
                "producao_estimada_referencia_t",
            ],
            color="safra",
            line_dash="variable",
            markers=True,
            labels={
                "ano": "Ano",
                "value": "Produção (t)",
                "variable": "Série",
                "safra": "Safra",
            },
            title=f"Comparação por safra — {UF_NAMES.get(selected_state, selected_state)}",
        )
        prediction_scatter = px.scatter(
            predictions,
            x="producao_observada_t",
            y="producao_estimada_modelo_t",
            color="ano",
            symbol="safra",
            labels={
                "producao_observada_t": "Produção observada (t)",
                "producao_estimada_modelo_t": "Produção estimada (t)",
                "ano": "Ano",
                "safra": "Safra",
            },
            title="Estimado × observado por UF/safra",
        )
        sections.extend(
            [
                ("Gráfico: estimativas nacionais", national_chart),
                ("Gráfico: estimativas do estado", state_chart),
                ("Gráfico: estimado versus observado", prediction_scatter),
            ]
        )
        title = "Avaliação das estimativas"
    else:
        title = "Metodologia e qualidade"
        filters = ["Conteúdo metodológico geral; filtros de período e safra não se aplicam."]
        sections.extend(
            [
                (
                    "Produção agrícola",
                    "Dados da PAM/IBGE, tabela 839, normalizados por UF, ano e primeira/segunda safra. "
                    "O ano de 2025 é preliminar; valores indisponíveis permanecem ausentes.",
                ),
                (
                    "Clima e calendário",
                    "Observações horárias do INMET foram agregadas por estação e dia. "
                    "Os indicadores estimam o clima municipal com base nas três estações mais próximas "
                    "e agregam municípios por UF/safra, ponderando pela área plantada. "
                    "As janelas de plantio e duração do ciclo são estimativas exploratórias por macrorregião.",
                ),
                (
                    "Limites de interpretação",
                    "Os pontos municipais são referências geográficas, não polígonos das lavouras. "
                    "Clima incompleto não deve ser interpretado como total. As comparações são exploratórias "
                    "e não implicam causalidade. O modelo usa clima observado durante o ciclo e área plantada "
                    "observada do próprio ano; não representa previsão antecipada.",
                ),
            ]
        )
    return title, filters, sections


def page_header(title: str, description: str) -> None:
    st.title(title)
    st.caption(description)


def show_data_warning(frame: pd.DataFrame) -> None:
    if "dados_preliminares" in frame.columns and (
        frame["dados_preliminares"] == "sim"
    ).any():
        st.warning("O período selecionado inclui dados preliminares do IBGE para 2025.")


@st.cache_data(show_spinner=False)
def load_state_geojson() -> dict:
    with MAP_GEOJSON_PATH.open(encoding="utf-8") as geojson_file:
        return json.load(geojson_file)


def build_state_map_data(
    data: dict[str, pd.DataFrame],
    years: tuple[int, int],
    seasons: list[str],
    metric_name: str,
    coverage_name: str,
    reference_mode: str,
) -> pd.DataFrame:
    is_coverage = metric_name == MAP_COVERAGE_METRIC
    is_climate = metric_name in CLIMATE_COMPARISONS
    source_name = "cycle" if is_climate or is_coverage else "agriculture"
    source = data[source_name]
    metric_column = (
        CLIMATE_COMPARISONS[coverage_name][1]
        if is_coverage
        else CLIMATE_COMPARISONS[metric_name][0]
        if is_climate
        else MAP_AGRICULTURE_METRICS[metric_name]
    )
    coverage_column = (
        CLIMATE_COMPARISONS[coverage_name][1]
        if is_coverage or not is_climate
        else CLIMATE_COMPARISONS[metric_name][1]
    )
    period_length = years[1] - years[0] + 1

    def period_values(
        period: tuple[int, int],
        column: str,
        period_source: pd.DataFrame | None = None,
        *,
        climate_values: bool | None = None,
    ) -> pd.DataFrame:
        rows_source = source if period_source is None else period_source
        is_agriculture = (
            source_name == "agriculture"
            if climate_values is None
            else not climate_values
        )
        rows = rows_source.loc[
            rows_source["ano"].between(period[0], period[1])
            & rows_source["safra"].isin(seasons)
        ].copy()
        if rows.empty:
            return pd.DataFrame(columns=["uf", "valor"])
        if is_agriculture:
            annual = rows.groupby(["uf", "ano"], as_index=False).agg(
                area_plantada_ha=("area_plantada_ha", lambda values: values.sum(min_count=1)),
                producao_t=("producao_t", lambda values: values.sum(min_count=1)),
            )
            if column == "rendimento_kg_ha":
                annual["valor"] = (
                    annual["producao_t"] * 1000
                    / annual["area_plantada_ha"].where(
                        annual["area_plantada_ha"].gt(0)
                    )
                )
            else:
                annual["valor"] = annual[column]
            return annual.groupby("uf", as_index=False)["valor"].mean()

        values = pd.to_numeric(rows[column], errors="coerce")
        weights = pd.to_numeric(
            rows["area_plantada_ha"], errors="coerce"
        ).where(lambda weight: weight.gt(0) & values.notna())
        weighted_values = rows.assign(
            _valor=values,
            _peso=weights,
            _valor_ponderado=values * weights,
        )
        return (
            weighted_values.groupby("uf", as_index=False)
            .agg(
                _total_ponderado=("_valor_ponderado", "sum"),
                _total_peso=("_peso", "sum"),
                _media=("_valor", "mean"),
            )
            .assign(
                valor=lambda group: group["_total_ponderado"].div(
                    group["_total_peso"].where(group["_total_peso"].gt(0))
                ).fillna(group["_media"])
            )[["uf", "valor"]]
        )

    current = period_values(years, metric_column)
    coverage = period_values(
        years,
        coverage_column,
        data["cycle"],
        climate_values=True,
    ).rename(
        columns={"valor": "cobertura_climatica_pct"}
    )
    result = current.merge(coverage, on="uf", how="left")
    reference_label = "média do período selecionado"
    if reference_mode == "previous":
        reference_period = (years[0] - period_length, years[0] - 1)
        reference_label = (
            f"período anterior ({reference_period[0]}–{reference_period[1]})"
        )
        reference = period_values(reference_period, metric_column)
        result = result.merge(
            reference.rename(columns={"valor": "valor_referencia"}),
            on="uf",
            how="left",
        )
    elif reference_mode == "historical":
        available_years = source.loc[source["safra"].isin(seasons), "ano"].dropna()
        if available_years.empty or years[0] <= int(available_years.min()):
            reference = pd.DataFrame(columns=["uf", "valor"])
            reference_label = "sem anos anteriores disponíveis"
        else:
            reference_period = (int(available_years.min()), years[0] - 1)
            reference_label = (
                f"média histórica ({reference_period[0]}–{reference_period[1]})"
            )
            reference = period_values(reference_period, metric_column)
        result = result.merge(
            reference.rename(columns={"valor": "valor_referencia"}),
            on="uf",
            how="left",
        )
    else:
        result["valor_referencia"] = float("nan")

    preliminary_column = (
        "dados_agricolas_preliminares"
        if "dados_agricolas_preliminares" in data["agriculture"]
        else "dados_preliminares"
    )
    preliminary_rows = data["agriculture"].loc[
        data["agriculture"]["ano"].between(years[0], years[1])
        & data["agriculture"]["safra"].isin(seasons)
    ]
    preliminary = (
        preliminary_rows.groupby("uf")[preliminary_column]
        .apply(lambda values: "sim" if values.eq("sim").any() else "não")
        .rename("dados_agricolas_preliminares")
    )
    result = result.merge(preliminary, on="uf", how="left")
    result["dados_agricolas_preliminares"] = result[
        "dados_agricolas_preliminares"
    ].fillna("não")

    result["variacao_pct"] = (
        (result["valor"] - result["valor_referencia"])
        / result["valor_referencia"].where(result["valor_referencia"].ne(0))
        * 100
    )
    result["valor_mapa"] = (
        result["valor"] if reference_mode == "current" else result["variacao_pct"]
    )
    result["estado"] = result["uf"].map(UF_NAMES)
    result["periodo"] = f"{years[0]}–{years[1]}"
    result["safras"] = ", ".join(SEASON_NAMES[season] for season in seasons)
    result["valor_nome"] = metric_name
    result["referencia_nome"] = reference_label
    return result


def build_state_choropleth(
    frame: pd.DataFrame,
    metric_name: str,
    reference_mode: str,
    years: tuple[int, int],
    seasons: list[str],
) -> go.Figure:
    geojson = load_state_geojson()
    if reference_mode == "current":
        color_title = metric_name
        title = (
            f"{metric_name} por UF · média de {years[0]}–{years[1]} · "
            f"{', '.join(SEASON_NAMES[season] for season in seasons)}"
        )
        color_scale = "Viridis"
    else:
        color_title = "Variação (%)"
        title = (
            f"Variação de {metric_name.lower()} · período {years[0]}–{years[1]} · "
            f"{', '.join(SEASON_NAMES[season] for season in seasons)}"
        )
        color_scale = "RdBu"
    hover_data = {
        "uf": True,
        "periodo": True,
        "safras": True,
        "valor": ":,.2f",
        "valor_referencia": ":,.2f",
        "variacao_pct": ":+.1f",
        "cobertura_climatica_pct": ":.1f",
        "valor_mapa": False,
        "estado": False,
        "valor_nome": False,
        "referencia_nome": True,
    }
    if "dados_agricolas_preliminares" in frame:
        hover_data["dados_agricolas_preliminares"] = True
    figure = px.choropleth(
        frame,
        geojson=geojson,
        locations="uf",
        featureidkey="properties.sigla",
        color="valor_mapa",
        color_continuous_scale=color_scale,
        hover_name="estado",
        hover_data=hover_data,
        labels={
            "uf": "UF",
            "periodo": "Período",
            "safras": "Safras",
            "valor": metric_name,
            "valor_referencia": "Valor de referência",
            "variacao_pct": "Variação (%)",
            "referencia_nome": "Referência",
            "cobertura_climatica_pct": "Cobertura climática (%)",
            "dados_agricolas_preliminares": "Dados agrícolas preliminares",
            "valor_mapa": color_title,
        },
        title=title,
    )
    if reference_mode != "current":
        figure.update_coloraxes(cmid=0)
    if metric_name == MAP_COVERAGE_METRIC and reference_mode == "current":
        figure.update_coloraxes(cmin=0, cmax=100)
    figure.update_geos(
        fitbounds="locations",
        visible=False,
        showcountries=False,
        showcoastlines=False,
        projection_type="mercator",
    )
    figure.update_layout(
        template=PLOT_TEMPLATE,
        height=620,
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        coloraxis_colorbar_title=color_title,
    )
    return figure


def map_report_table(frame: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    columns = {
        "estado": "Estado",
        "uf": "UF",
        "periodo": "Período",
        "safras": "Safras",
        "valor": metric_name,
        "valor_referencia": "Valor de referência",
        "variacao_pct": "Variação (%)",
        "referencia_nome": "Referência",
        "cobertura_climatica_pct": "Cobertura climática (%)",
        "dados_agricolas_preliminares": "Dados agrícolas preliminares",
    }
    available = [column for column in columns if column in frame]
    return frame[available].sort_values("estado").rename(columns=columns)


def interactive_map_page(
    data: dict[str, pd.DataFrame], years: tuple[int, int], seasons: list[str]
) -> None:
    page_header(
        "Mapa interativo por Unidade da Federação",
        "O mapa respeita o período e as safras selecionados na barra lateral. "
        "Os indicadores agrícolas são médias anuais; os climáticos são médias dos ciclos ponderadas pela área plantada.",
    )
    metric_options = [
        *MAP_AGRICULTURE_METRICS,
        *CLIMATE_COMPARISONS,
        MAP_COVERAGE_METRIC,
    ]
    metric_name = st.selectbox("Indicador exibido no mapa", metric_options, key="map_metric")
    coverage_name = st.selectbox(
        "Indicador climático usado para avaliar a cobertura",
        list(CLIMATE_COMPARISONS),
        key="map_coverage_indicator",
    )
    reference_options = list(MAP_REFERENCE_MODES)
    if metric_name == MAP_COVERAGE_METRIC:
        reference_options = [reference_options[0]]
    reference_name = st.radio(
        "Visualização",
        reference_options,
        horizontal=True,
        key="map_reference_mode",
    )
    reference_mode = MAP_REFERENCE_MODES[reference_name]
    frame = build_state_map_data(
        data,
        years,
        seasons,
        metric_name,
        coverage_name,
        reference_mode,
    )
    if frame.empty:
        st.info("Não há registros para a UF, ano e safra selecionados.")
        return

    available_states = int(frame["valor"].notna().sum())
    mean_coverage = frame["cobertura_climatica_pct"].mean()
    total_states = len(UF_NAMES)
    metric_col, coverage_col, missing_col = st.columns(3)
    metric_col.metric("UFs com indicador", f"{available_states} de {total_states}")
    coverage_col.metric(
        "Cobertura média",
        f"{mean_coverage:.1f}%" if pd.notna(mean_coverage) else "Sem dados",
    )
    missing_col.metric("UFs sem cobertura", str(frame["cobertura_climatica_pct"].isna().sum()))
    if metric_name in CLIMATE_COMPARISONS or metric_name == MAP_COVERAGE_METRIC:
        st.caption(
            "A cobertura mostra a proporção temporal de dados climáticos válidos no ciclo estimado. "
            "Ela é um indicador de completude, não de qualidade agronômica."
        )
    else:
        st.caption(
            f"A cobertura climática exibida corresponde a “{coverage_name}” e serve como contexto "
            "para os dados agrícolas."
        )

    figure = build_state_choropleth(
        frame, metric_name, reference_mode, years, seasons
    )
    st.plotly_chart(figure, use_container_width=True)
    if reference_mode != "current" and frame["valor_referencia"].notna().sum() == 0:
        st.warning(
            "Não há valores de referência anteriores disponíveis para este recorte. "
            "Amplie o período ou escolha uma comparação com anos históricos disponíveis."
        )
    if years[1] >= 2025 and (
        frame.get("dados_agricolas_preliminares", pd.Series(dtype=str)) == "sim"
    ).any():
        st.warning("O período selecionado inclui dados agrícolas preliminares de 2025.")
    if metric_name == "Chuva acumulada no ciclo (mm)":
        st.info(
            "A chuva acumulada só é apresentada quando o ciclo tem cobertura suficiente; "
            "valores ausentes podem representar ciclos com dados parciais."
        )

    report_table = map_report_table(frame, metric_name)
    st.subheader("Dados por UF")
    st.dataframe(report_table, use_container_width=True, hide_index=True)
    with st.expander("Como interpretar o mapa"):
        st.write(
            "Passe o cursor sobre uma UF para consultar o valor médio, a referência "
            "e a variação percentual quando houver comparação. Produção e área correspondem "
            "à média anual dos totais das safras selecionadas; rendimento é ponderado pela "
            "área. Os indicadores climáticos são médias dos ciclos ponderadas pela área "
            "plantada. Áreas sem cor não têm valor para o recorte escolhido."
        )


def overview_page(data: dict[str, pd.DataFrame], years: tuple[int, int], seasons: list[str]) -> None:
    page_header(
        "Milho no Brasil",
        "Panorama de produção, área e rendimento da PAM/IBGE, com clima agregado por área produtora.",
    )
    agriculture = data["agriculture"]
    selected = agriculture.loc[
        agriculture["ano"].between(years[0], years[1])
        & agriculture["safra"].isin(seasons)
    ].copy()
    if selected.empty:
        st.info("Não há registros para os filtros selecionados.")
        return

    latest_year = int(selected["ano"].max())
    latest = selected.loc[selected["ano"] == latest_year]
    total_production = latest["producao_t"].sum(min_count=1)
    total_area = latest["area_plantada_ha"].sum(min_count=1)
    mean_yield = (
        latest["rendimento_kg_ha"].mean()
        if latest["rendimento_kg_ha"].notna().any()
        else float("nan")
    )
    first_safra_share = (
        latest.loc[latest["safra"] == "1a", "producao_t"].sum()
        / total_production
        * 100
        if total_production
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Produção {latest_year}", production_millions(total_production))
    col2.metric("Área plantada", f"{integer_format(total_area)} ha")
    col3.metric(
        "Rendimento médio simples",
        f"{integer_format(mean_yield)} kg/ha" if pd.notna(mean_yield) else "—",
    )
    col4.metric("Participação 1ª safra", f"{first_safra_share:.1f}%")
    if latest_year == 2025:
        st.caption("O dado agrícola de 2025 é preliminar conforme a exportação do IBGE.")

    st.subheader("Histórico nacional")
    national = agriculture.loc[
        (agriculture["uf"].notna())
        & agriculture["ano"].between(years[0], years[1])
        & agriculture["safra"].isin(seasons)
    ]
    production = (
        national.groupby(["ano", "safra"], as_index=False)
        .agg(
            producao_t=("producao_t", "sum"),
            area_plantada_ha=("area_plantada_ha", "sum"),
            rendimento_kg_ha=("rendimento_kg_ha", "mean"),
        )
        .sort_values("ano")
    )
    production["safra_nome"] = production["safra"].map(SEASON_NAMES)
    left, right = st.columns(2)
    fig_production = px.line(
        production,
        x="ano",
        y="producao_t",
        color="safra_nome",
        markers=True,
        labels={"ano": "Ano", "producao_t": "Produção (t)", "safra_nome": "Safra"},
        color_discrete_map={"1ª safra": PLOT_COLORS["1a"], "2ª safra": PLOT_COLORS["2a"]},
        title="Produção por safra",
    )
    fig_production.update_yaxes(tickformat=",.0f")
    left.plotly_chart(style_plot(fig_production), use_container_width=True)

    fig_area = px.line(
        production,
        x="ano",
        y="area_plantada_ha",
        color="safra_nome",
        markers=True,
        labels={"ano": "Ano", "area_plantada_ha": "Área plantada (ha)", "safra_nome": "Safra"},
        color_discrete_map={"1ª safra": PLOT_COLORS["1a"], "2ª safra": PLOT_COLORS["2a"]},
        title="Área plantada por safra",
    )
    fig_area.update_yaxes(tickformat=",.0f")
    right.plotly_chart(style_plot(fig_area), use_container_width=True)

    st.subheader(f"Produção por estado — {latest_year}")
    by_state = (
        latest.groupby(["uf", "safra"], as_index=False)
        .agg(producao_t=("producao_t", "sum"))
        .sort_values("producao_t")
    )
    by_state["estado"] = by_state["uf"].map(UF_NAMES)
    by_state["safra_nome"] = by_state["safra"].map(SEASON_NAMES)
    chart = px.bar(
        by_state,
        x="producao_t",
        y="estado",
        color="safra_nome",
        barmode="stack",
        orientation="h",
        labels={"producao_t": "Produção (t)", "estado": "", "safra_nome": "Safra"},
        color_discrete_map={"1ª safra": PLOT_COLORS["1a"], "2ª safra": PLOT_COLORS["2a"]},
    )
    chart.update_xaxes(tickformat=",.0f")
    st.plotly_chart(style_plot(chart, height=720), use_container_width=True)

    with st.expander("Como ler esta página"):
        st.write(
            "Os totais nacionais são calculados somando as UFs da tabela normalizada. "
            "A média de rendimento exibida é simples entre registros UF/safra; não "
            "substitui o rendimento oficial nacional ponderado."
        )


def state_page(data: dict[str, pd.DataFrame], years: tuple[int, int], seasons: list[str]) -> None:
    page_header(
        "Clima e produção por estado",
        "Compare a produção ou o rendimento do milho com indicadores climáticos estimados para o ciclo da safra.",
    )
    cycle = data["cycle"].loc[
        data["cycle"]["ano"].between(years[0], years[1])
        & data["cycle"]["safra"].isin(seasons)
    ].copy()
    if cycle.empty:
        st.info("Não há registros para os filtros selecionados.")
        return
    states = sorted(cycle["uf"].dropna().unique())
    selected_state = st.selectbox(
        "Estado",
        states,
        format_func=lambda uf: f"{UF_NAMES.get(uf, uf)} ({uf})",
        key="report_state_uf",
    )
    selected_season = st.selectbox(
        "Safra",
        [season for season in ("1a", "2a") if season in seasons],
        format_func=lambda season: SEASON_NAMES[season],
        key="report_safra",
    )
    state = cycle.loc[
        (cycle["uf"] == selected_state) & (cycle["safra"] == selected_season)
    ].sort_values("ano")
    if state.empty:
        st.info("Não há registros do estado para os filtros selecionados.")
        return

    latest = state.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Produção", production_millions(latest["producao_t"]) if pd.notna(latest["producao_t"]) else "—")
    c2.metric("Área plantada", f"{integer_format(latest['area_plantada_ha'])} ha" if pd.notna(latest["area_plantada_ha"]) else "—")
    c3.metric("Rendimento", f"{integer_format(latest['rendimento_kg_ha'])} kg/ha" if pd.notna(latest["rendimento_kg_ha"]) else "—")
    c4.metric("Ano", f"{int(latest['ano'])}" + (" · preliminar" if latest["dados_agricolas_preliminares"] == "sim" else ""))

    left, right = st.columns(2)
    fig_production = px.line(
        state,
        x="ano",
        y="producao_t",
        markers=True,
        labels={"ano": "Ano", "producao_t": "Produção (t)"},
        title=f"Produção de milho — {UF_NAMES.get(selected_state, selected_state)}",
    )
    fig_production.update_traces(line_color=PLOT_COLORS[selected_season])
    fig_production.update_yaxes(tickformat=",.0f")
    left.plotly_chart(style_plot(fig_production), use_container_width=True)

    fig_yield = px.line(
        state,
        x="ano",
        y="rendimento_kg_ha",
        markers=True,
        labels={"ano": "Ano", "rendimento_kg_ha": "Rendimento (kg/ha)"},
        title="Rendimento médio",
    )
    fig_yield.update_traces(line_color="#a371f7")
    fig_yield.update_yaxes(tickformat=",.0f")
    right.plotly_chart(style_plot(fig_yield), use_container_width=True)

    st.subheader("Comparar produção e clima do ciclo")
    st.caption(
        "Cada ponto representa um ano. Escolha o indicador climático no eixo horizontal "
        "e a medida agrícola no eixo vertical."
    )
    climate_label, outcome_label = st.columns(2)
    climate_name = climate_label.selectbox(
        "Indicador climático",
        list(CLIMATE_COMPARISONS),
        key="report_climate_indicator",
    )
    outcome_name = outcome_label.selectbox(
        "Comparar com",
        ["Produção (t)", "Rendimento (kg/ha)"],
        key="report_climate_outcome",
    )
    climate_column, coverage_column = CLIMATE_COMPARISONS[climate_name]
    outcome_column = (
        "producao_t" if outcome_name == "Produção (t)" else "rendimento_kg_ha"
    )
    climate_data = state.dropna(subset=[climate_column, outcome_column]).copy()
    if not climate_data.empty:
        climate_data["ano_grafico"] = climate_data["ano"].astype(int).astype(str)
        if coverage_column in climate_data:
            climate_data["cobertura_pct"] = climate_data[coverage_column]
        hover_data = {
            "ano_grafico": True,
            "producao_t": ":,.0f",
            "rendimento_kg_ha": ":,.0f",
        }
        if "cobertura_pct" in climate_data:
            hover_data["cobertura_pct"] = ":.1f"
        scatter = px.scatter(
            climate_data,
            x=climate_column,
            y=outcome_column,
            color="ano_grafico",
            hover_data=hover_data,
            labels={
                climate_column: climate_name,
                outcome_column: outcome_name,
                "ano_grafico": "Ano",
                "producao_t": "Produção (t)",
                "rendimento_kg_ha": "Rendimento (kg/ha)",
                "cobertura_pct": "Cobertura climática (%)",
            },
        )
        st.plotly_chart(style_plot(scatter), use_container_width=True)
        st.caption(
            f"{len(climate_data)} anos com dados disponíveis neste recorte. "
            "A cobertura indica a parcela dos meses esperados do ciclo com dados climáticos. "
            "Relação exploratória, não implica causalidade."
        )
    else:
        st.info("Não há pares de dados climáticos e agrícolas disponíveis neste recorte.")

    chosen_year = st.selectbox(
        "Ano para ver a série climática mensal",
        sorted(state["ano"].dropna().astype(int).unique(), reverse=True),
        key="report_climate_year",
    )
    monthly = data["monthly"].loc[
        (data["monthly"]["uf"] == selected_state)
        & (data["monthly"]["ano"] == chosen_year)
        & (data["monthly"]["safra"] == selected_season)
    ].sort_values("mes")
    if not monthly.empty:
        rain_fig = go.Figure()
        rain_fig.add_trace(
            go.Bar(
                x=monthly["mes"],
                y=monthly["chuva_mm"],
                name="Precipitação mensal completa",
                marker_color="#238636",
                customdata=monthly[["area_coberta_chuva_mm_pct"]],
                hovertemplate="Mês %{x}<br>Chuva: %{y:.1f} mm<br>Cobertura: %{customdata[0]:.1f}%<extra></extra>",
            )
        )
        rain_fig.update_layout(
            title="Chuva mensal estimada · média espacial ponderada",
            xaxis_title="Mês (UTC)",
            yaxis_title="Precipitação (mm)",
        )
        rain_fig.update_xaxes(dtick=1)
        st.plotly_chart(style_plot(rain_fig), use_container_width=True)
        st.caption(
            "O total mensal só aparece quando pelo menos 80% dos dias têm 24 horas "
            "válidas. A cobertura é a parcela da área municipal plantada com estimativa disponível."
        )
    show_data_warning(state)
    with st.expander("Tabela de indicadores por ano"):
        columns = [
            "ano",
            "area_plantada_ha",
            "area_colhida_ha",
            "producao_t",
            "rendimento_kg_ha",
            "chuva_acumulada_ciclo_mm",
            "chuva_parcial_ciclo_mm",
            "situacao_chuva_ciclo",
            "cobertura_temporal_chuva_ciclo_pct",
            "temp_ar_media_c_ciclo",
            "dados_agricolas_preliminares",
        ]
        st.dataframe(state[columns], use_container_width=True, hide_index=True)


def evaluation_page(data: dict[str, pd.DataFrame]) -> None:
    page_header(
        "Avaliação das estimativas",
        "Teste temporal: treino até 2019; avaliação em anos posteriores. Estimativa retrospectiva com clima observado.",
    )
    period = st.radio(
        "Período de avaliação",
        ["2020–2024 · sem ano preliminar", "2020–2025 · inclui ano preliminar"],
        horizontal=True,
        key="report_evaluation_period",
    )
    period_key = (
        "teste_2020_2024_nao_preliminar"
        if period.startswith("2020–2024")
        else "teste_2020_2025"
    )
    metrics = data["metrics"].loc[data["metrics"]["periodo"] == period_key]
    national_metrics = metrics.loc[
        metrics["granularidade"] == "Brasil_ano_safra"
    ].set_index("modelo")
    state_metrics = metrics.loc[
        metrics["granularidade"] == "UF_safra"
    ].set_index("modelo")
    baseline = national_metrics.loc["referencia_ano_anterior"]
    forest = national_metrics.loc["random_forest_clima"]
    col1, col2, col3 = st.columns(3)
    col1.metric("WAPE Brasil · Random Forest", f"{forest['wape_producao_pct']:.2f}%")
    col2.metric("WAPE Brasil · referência", f"{baseline['wape_producao_pct']:.2f}%")
    col3.metric(
        "WAPE UF/safra · Random Forest",
        f"{state_metrics.loc['random_forest_clima', 'wape_producao_pct']:.2f}%",
    )
    st.info(
        "O modelo prevê rendimento e converte para toneladas usando a área plantada "
        "observada no próprio ano. O clima observado inclui todo o ciclo da safra; "
        "esta avaliação não representa uma previsão disponível antes da colheita."
    )

    model_display = metrics.copy()
    model_display["modelo"] = model_display["modelo"].map(METRIC_NAMES)
    model_display["granularidade"] = model_display["granularidade"].map(
        {"Brasil_ano_safra": "Brasil por ano/safra", "UF_safra": "UF por safra"}
    )
    st.subheader("Métricas de erro")
    st.dataframe(
        model_display[
            [
                "granularidade",
                "modelo",
                "n_observacoes",
                "mae_producao_t",
                "rmse_producao_t",
                "wape_producao_pct",
                "r2_producao",
            ]
        ].rename(
            columns={
                "granularidade": "Granularidade",
                "modelo": "Método",
                "n_observacoes": "Observações",
                "mae_producao_t": "MAE (t)",
                "rmse_producao_t": "RMSE (t)",
                "wape_producao_pct": "WAPE (%)",
                "r2_producao": "R²",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "MAE e RMSE por UF dão grande peso aos estados produtores. WAPE mede o erro "
        "absoluto total em relação à produção observada."
    )

    predictions = data["predictions"].copy()
    if period_key == "teste_2020_2024_nao_preliminar":
        predictions = predictions.loc[predictions["ano"] <= 2024]
    national = data["national_predictions"].copy()
    if period_key == "teste_2020_2024_nao_preliminar":
        national = national.loc[national["ano"] <= 2024]
    left, right = st.columns(2)
    fig_national = px.line(
        national,
        x="ano",
        y=[
            "producao_observada_t",
            "producao_estimada_modelo_t",
            "producao_estimada_referencia_t",
        ],
        color="safra",
        line_dash="variable",
        markers=True,
        labels={
            "ano": "Ano",
            "value": "Produção (t)",
            "variable": "Série",
            "safra": "Safra",
        },
        title="Produção nacional: observado e estimado",
    )
    fig_national.update_yaxes(tickformat=",.0f")
    left.plotly_chart(style_plot(fig_national), use_container_width=True)

    selected_state = st.selectbox(
        "UF para comparar estimativa e observado",
        sorted(predictions["uf"].dropna().unique()),
        format_func=lambda uf: f"{UF_NAMES.get(uf, uf)} ({uf})",
        key="report_evaluation_state",
    )
    state_predictions = predictions.loc[predictions["uf"] == selected_state]
    fig_state = px.line(
        state_predictions,
        x="ano",
        y=[
            "producao_observada_t",
            "producao_estimada_modelo_t",
            "producao_estimada_referencia_t",
        ],
        color="safra",
        line_dash="variable",
        markers=True,
        labels={
            "ano": "Ano",
            "value": "Produção (t)",
            "variable": "Série",
            "safra": "Safra",
        },
        title=f"Comparação por safra — {UF_NAMES.get(selected_state, selected_state)}",
    )
    fig_state.update_yaxes(tickformat=",.0f")
    right.plotly_chart(style_plot(fig_state), use_container_width=True)

    chart = px.scatter(
        predictions,
        x="producao_observada_t",
        y="producao_estimada_modelo_t",
        color="ano",
        symbol="safra",
        hover_data=["uf", "erro_modelo_t", "dados_agricolas_preliminares"],
        labels={
            "producao_observada_t": "Produção observada (t)",
            "producao_estimada_modelo_t": "Produção estimada (t)",
            "ano": "Ano",
            "safra": "Safra",
            "uf": "UF",
            "erro_modelo_t": "Erro (t)",
            "dados_agricolas_preliminares": "Dados preliminares",
        },
        title="Estimado × observado por UF/safra",
    )
    max_value = max(
        predictions["producao_observada_t"].max(),
        predictions["producao_estimada_modelo_t"].max(),
    )
    chart.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=max_value,
        y1=max_value,
        line=dict(color="#6e7781", dash="dash"),
    )
    chart.update_xaxes(tickformat=",.0f")
    chart.update_yaxes(tickformat=",.0f")
    st.plotly_chart(style_plot(chart, height=500), use_container_width=True)

    with st.expander("Ver estimativas e erros"):
        st.dataframe(
            predictions.sort_values(["ano", "uf", "safra"]),
            use_container_width=True,
            hide_index=True,
        )


def about_page() -> None:
    page_header(
        "Metodologia e qualidade",
        "Fontes, escolhas de processamento e limites para interpretar os gráficos.",
    )
    st.markdown(
        """
        **Produção agrícola.** PAM/IBGE, tabela 839, em base normalizada por UF, ano e
        primeira/segunda safra. O último ano da exportação (2025) está identificado
        como preliminar. Valores indisponíveis permanecem ausentes.

        **Clima.** Observações horárias do INMET foram agregadas por estação e dia.
        Os indicadores mensais e sazonais estimam clima municipal a partir das três
        estações mais próximas e agregam municípios por UF/safra, ponderando pela
        área plantada de milho. Os pontos municipais são referências geográficas,
        não polígonos das lavouras.

        **Calendário.** A janela de plantio e duração do ciclo são estimativas
        exploratórias por macrorregião, não datas observadas ou calendário técnico
        oficial. Precipitação de ciclo incompleta é mostrada separadamente e não
        deve ser interpretada como total.

        **Modelo.** Random Forest estima rendimento; a produção estimada é calculada
        usando a área plantada observada do mesmo ano. Clima observado durante a
        safra só permite uma análise retrospectiva, não uma previsão antecipada.
        Avaliação temporal: treino até 2019 e teste em 2020–2025, também exibido sem
        2025 preliminar.
        """
    )
    st.link_button(
        "Documentação detalhada dos dados e processamento",
        "https://www.gov.br/conab/pt-br/atuacao/informacoes-agropecuarias/safras/safra-de-graos",
    )


def main() -> None:
    st.set_page_config(
        page_title="Milho · Produção e clima",
        page_icon="🌽",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.7rem; padding-bottom: 3rem;}
        [data-testid="stMetric"] {
            background: #f6f8fa; border: 1px solid #d8dee4;
            padding: 1rem; border-radius: 0.75rem; color: #24292f;
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] {color: #24292f !important;}
        [data-testid="stMetricValue"] {font-size: 1.6rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.title("🌽 Milho")
    st.sidebar.caption("Produção, clima e estimativas")
    try:
        data = load_data()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        st.error(
            f"Não foi possível carregar os dados: {error}\n\n"
            "Execute os scripts de preparação descritos nos README da pasta "
            "`data\\processed`."
        )
        st.stop()

    years_available = sorted(
        int(year) for year in data["agriculture"]["ano"].dropna().unique()
    )
    year_range = st.sidebar.slider(
        "Período",
        min_value=years_available[0],
        max_value=years_available[-1],
        value=(years_available[0], years_available[-1]),
    )
    selected_seasons = st.sidebar.multiselect(
        "Safras",
        options=["1a", "2a"],
        default=["1a", "2a"],
        format_func=lambda season: SEASON_NAMES[season],
    )
    if not selected_seasons:
        st.warning("Selecione pelo menos uma safra na barra lateral.")
        st.stop()

    page = st.sidebar.radio(
        "Navegação",
        [
            "Brasil",
            "Mapa interativo por UF",
            "Clima e produção por estado",
            "Avaliação",
            "Metodologia",
        ],
    )
    if page == "Brasil":
        overview_page(data, year_range, selected_seasons)
    elif page == "Mapa interativo por UF":
        interactive_map_page(data, year_range, selected_seasons)
    elif page == "Clima e produção por estado":
        state_page(data, year_range, selected_seasons)
    elif page == "Avaliação":
        evaluation_page(data)
    else:
        about_page()
    report_title, report_filters, report_sections = build_page_report(
        page, data, year_range, selected_seasons
    )
    report_pdf = create_report_pdf(report_title, report_filters, report_sections)
    report_slug = {
        "Brasil": "brasil",
        "Mapa interativo por UF": "mapa_uf",
        "Clima e produção por estado": "clima_estado",
        "Avaliação": "avaliacao",
        "Metodologia": "metodologia",
    }[page]
    tables = exportable_tables(report_sections)
    if tables:
        st.sidebar.subheader("Exportar dados filtrados")
        st.sidebar.caption(
            f"{len(tables)} tabela(s) desta página, com os filtros selecionados."
        )
        st.sidebar.download_button(
            "Baixar tabelas CSV (ZIP)",
            data=csv_tables_archive(tables, report_filters),
            file_name=f"dados_milho_{report_slug}_{date.today():%Y%m%d}.zip",
            mime="application/zip",
            use_container_width=True,
        )
    st.sidebar.download_button(
        "Baixar relatório desta página (PDF)",
        data=report_pdf,
        file_name=f"relatorio_milho_{report_slug}_{date.today():%Y%m%d}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
    st.sidebar.divider()
    st.sidebar.caption("Projeto acadêmico · estimativas exploratórias")


if __name__ == "__main__":
    main()
