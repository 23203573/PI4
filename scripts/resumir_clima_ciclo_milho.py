#!/usr/bin/env python3
"""Resume clima mensal nas janelas estimadas de ciclo do milho."""

from __future__ import annotations

import argparse
import calendar
import csv
import math
from pathlib import Path


CLIMATE_FIELDS = {
    "chuva_mm": ("area_coberta_chuva_mm_pct", "sum"),
    "temp_ar_media_c": ("area_coberta_temp_ar_media_c_pct", "mean"),
    "temp_ar_min_c": ("area_coberta_temp_ar_min_c_pct", "min"),
    "temp_ar_max_c": ("area_coberta_temp_ar_max_c_pct", "max"),
    "temp_orvalho_media_c": ("area_coberta_temp_orvalho_media_c_pct", "mean"),
    "radiacao_global_kj_m2": (
        "area_coberta_radiacao_global_kj_m2_pct",
        "sum",
    ),
    "umidade_relativa_media_pct": (
        "area_coberta_umidade_relativa_media_pct",
        "mean",
    ),
    "velocidade_vento_media_ms": (
        "area_coberta_velocidade_vento_media_ms_pct",
        "mean",
    ),
    "rajada_vento_max_ms": ("area_coberta_rajada_vento_max_ms_pct", "max"),
}
MINIMUM_COVERAGE_PCT = 80.0
UF_BY_CODE = {
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


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file, delimiter=";"))


def climate_year_month(
    crop_year: int, start_year_offset: int, start_month: int, offset: int
) -> tuple[int, int]:
    absolute_month = start_month - 1 + offset
    year = crop_year + start_year_offset + absolute_month // 12
    return year, absolute_month % 12 + 1


def monthly_climate_index(
    rows: list[dict[str, str]],
) -> dict[tuple[str, int, str, int], dict[str, str]]:
    index = {}
    for row in rows:
        key = (row["uf"], int(row["ano"]), row["safra"], int(row["mes"]))
        if key in index:
            raise ValueError(f"Registro mensal duplicado: {key}.")
        index[key] = row
    return index


def summarize_metric(
    cycle_rows: list[tuple[int, int, dict[str, str] | None]],
    feature: str,
    operation: str,
) -> dict[str, str]:
    coverage_field, _ = CLIMATE_FIELDS[feature]
    observations = []
    covered_days = 0.0
    expected_days = 0
    observed_months = 0
    all_months_valid = True
    partial_sum = 0.0
    for climate_year, climate_month, row in cycle_rows:
        month_days = calendar.monthrange(climate_year, climate_month)[1]
        expected_days += month_days
        if row is None:
            all_months_valid = False
            continue
        value_text = row.get(feature, "")
        coverage_text = row.get(coverage_field, "")
        coverage = float(coverage_text) if coverage_text else 0.0
        covered_days += month_days * coverage / 100.0
        if not value_text:
            all_months_valid = False
            continue
        value = float(value_text)
        if not math.isfinite(value):
            raise ValueError(f"Valor não finito em {feature}: {value_text!r}.")
        if coverage < MINIMUM_COVERAGE_PCT:
            all_months_valid = False
            continue
        observations.append((value, month_days, coverage))
        observed_months += 1
        if operation == "sum":
            partial_sum += value

    temporal_coverage = 100.0 * covered_days / expected_days if expected_days else 0.0
    if operation == "sum":
        complete_sum = (
            f"{partial_sum:.3f}"
            if all_months_valid and observed_months == len(cycle_rows)
            else ""
        )
        return {
            "valor": complete_sum,
            "parcial": f"{partial_sum:.3f}" if observed_months else "",
            "meses_validos": str(observed_months),
            "cobertura_temporal_pct": f"{temporal_coverage:.2f}",
        }

    if not observations:
        value = ""
    elif operation == "mean":
        denominator = sum(days * coverage for _, days, coverage in observations)
        value = (
            f"{sum(x * days * coverage for x, days, coverage in observations) / denominator:.3f}"
            if denominator
            else ""
        )
    elif operation == "min":
        value = f"{min(x for x, _, _ in observations):.3f}"
    elif operation == "max":
        value = f"{max(x for x, _, _ in observations):.3f}"
    else:
        raise ValueError(f"Operação de agregação não reconhecida: {operation}")

    if temporal_coverage < MINIMUM_COVERAGE_PCT:
        value = ""
    return {
        "valor": value,
        "parcial": value,
        "meses_validos": str(observed_months),
        "cobertura_temporal_pct": f"{temporal_coverage:.2f}",
    }


def summarize(
    monthly_rows: list[dict[str, str]],
    crop_rows: list[dict[str, str]],
    calendar_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    monthly = monthly_climate_index(monthly_rows)
    calendar_index = {
        (row["uf"], row["safra"]): row for row in calendar_rows
    }
    if len(calendar_index) != 54:
        raise ValueError(
            f"Esperadas 54 janelas UF/safra no calendário; recebidas {len(calendar_index)}."
        )

    output = []
    for crop in crop_rows:
        key = (crop["uf"], crop["ano"], crop["safra"])
        calendar_row = calendar_index.get((crop["uf"], crop["safra"]))
        if not calendar_row:
            raise ValueError(f"Calendário ausente para UF/safra {key}.")
        start_month = int(calendar_row["mes_inicio"])
        start_offset = int(calendar_row["ano_offset_inicio"])
        cycle_length = int(calendar_row["duracao_ciclo_meses"])
        planting_length = int(calendar_row["meses_janela_plantio"])
        months = []
        for offset in range(cycle_length):
            climate_year, climate_month = climate_year_month(
                int(crop["ano"]),
                start_offset,
                start_month,
                offset,
            )
            month_row = monthly.get(
                (crop["uf"], climate_year, crop["safra"], climate_month)
            )
            if month_row is None:
                other_season = "2a" if crop["safra"] == "1a" else "1a"
                month_row = monthly.get(
                    (crop["uf"], climate_year, other_season, climate_month)
                )
            months.append((climate_year, climate_month, month_row))

        planting_months = months[:planting_length]
        cycle_values = {}
        for feature, (_, operation) in CLIMATE_FIELDS.items():
            cycle_values[feature] = summarize_metric(months, feature, operation)
            cycle_values[f"plantio_{feature}"] = summarize_metric(
                planting_months, feature, operation
            )

        start_year = int(crop["ano"]) + start_offset
        cycle_end_year, cycle_end_month = climate_year_month(
            int(crop["ano"]), start_offset, start_month, cycle_length - 1
        )
        result: dict[str, object] = {
            "uf": crop["uf"],
            "ano": crop["ano"],
            "safra": crop["safra"],
            "macrorregiao_calendario": calendar_row["macrorregiao"],
            "metodo_calendario": calendar_row["metodo"],
            "inicio_ciclo_mes": start_month,
            "inicio_ciclo_ano": start_year,
            "fim_ciclo_mes": cycle_end_month,
            "fim_ciclo_ano": cycle_end_year,
            "meses_janela_plantio": planting_length,
            "meses_ciclo_esperados": cycle_length,
            "meses_com_registro_climatico_mensal": sum(
                row is not None for _, _, row in months
            ),
            "dados_agricolas_preliminares": crop["dados_agricolas_preliminares"],
            "area_plantada_ha": crop["area_plantada_ha"],
            "area_colhida_ha": crop["area_colhida_ha"],
            "producao_t": crop["producao_t"],
            "rendimento_kg_ha": crop["rendimento_kg_ha"],
        }
        for feature in CLIMATE_FIELDS:
            if feature == "chuva_mm":
                result["chuva_acumulada_ciclo_mm"] = cycle_values[feature]["valor"]
                result["chuva_parcial_ciclo_mm"] = cycle_values[feature]["parcial"]
                result["meses_chuva_validos_ciclo"] = cycle_values[feature]["meses_validos"]
                result["cobertura_temporal_chuva_ciclo_pct"] = cycle_values[feature]["cobertura_temporal_pct"]
                result["situacao_chuva_ciclo"] = (
                    "completo"
                    if cycle_values[feature]["valor"]
                    else "parcial"
                    if cycle_values[feature]["parcial"]
                    else "sem_dados"
                )
                result["chuva_janela_plantio_mm"] = cycle_values[f"plantio_{feature}"]["valor"]
                result["chuva_parcial_janela_plantio_mm"] = cycle_values[f"plantio_{feature}"]["parcial"]
                result["meses_chuva_validos_plantio"] = cycle_values[f"plantio_{feature}"]["meses_validos"]
                result["cobertura_temporal_chuva_plantio_pct"] = cycle_values[f"plantio_{feature}"]["cobertura_temporal_pct"]
                result["situacao_chuva_plantio"] = (
                    "completo"
                    if cycle_values[f"plantio_{feature}"]["valor"]
                    else "parcial"
                    if cycle_values[f"plantio_{feature}"]["parcial"]
                    else "sem_dados"
                )
            elif feature == "radiacao_global_kj_m2":
                result["radiacao_acumulada_ciclo_kj_m2"] = cycle_values[feature]["valor"]
                result["radiacao_parcial_ciclo_kj_m2"] = cycle_values[feature]["parcial"]
                result["meses_radiacao_validos_ciclo"] = cycle_values[feature]["meses_validos"]
                result["cobertura_temporal_radiacao_ciclo_pct"] = cycle_values[feature]["cobertura_temporal_pct"]
                result["radiacao_janela_plantio_kj_m2"] = cycle_values[f"plantio_{feature}"]["valor"]
                result["radiacao_parcial_janela_plantio_kj_m2"] = cycle_values[f"plantio_{feature}"]["parcial"]
                result["meses_radiacao_validos_plantio"] = cycle_values[f"plantio_{feature}"]["meses_validos"]
                result["cobertura_temporal_radiacao_plantio_pct"] = cycle_values[f"plantio_{feature}"]["cobertura_temporal_pct"]
            else:
                result[f"{feature}_ciclo"] = cycle_values[feature]["valor"]
                result[f"{feature}_plantio"] = cycle_values[f"plantio_{feature}"]["valor"]
                result[f"meses_{feature}_validos_ciclo"] = cycle_values[feature]["meses_validos"]
                result[f"cobertura_temporal_{feature}_ciclo_pct"] = cycle_values[feature]["cobertura_temporal_pct"]
                result[f"meses_{feature}_validos_plantio"] = cycle_values[f"plantio_{feature}"]["meses_validos"]
                result[f"cobertura_temporal_{feature}_plantio_pct"] = cycle_values[f"plantio_{feature}"]["cobertura_temporal_pct"]
        output.append(result)
    return output


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    default_dir = repository_root / "data" / "processed" / "clima_uf_safra"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--monthly",
        type=Path,
        default=default_dir / "clima_uf_ano_safra_mensal.csv",
    )
    parser.add_argument(
        "--agriculture",
        type=Path,
        default=repository_root / "data" / "processed" / "ibge_milho_uf_ano_safra.csv",
    )
    parser.add_argument(
        "--calendar",
        type=Path,
        default=default_dir / "calendario_milho_macroregiao.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_dir / "clima_uf_ano_safra_ciclo.csv",
    )
    args = parser.parse_args()
    monthly_rows = load_csv(args.monthly)
    crop_rows = []
    for source_row in load_csv(args.agriculture):
        if source_row["nivel"] != "UF":
            continue
        uf_code = source_row["codigo_ibge"].zfill(2)
        if uf_code not in UF_BY_CODE:
            raise ValueError(f"Código de UF não reconhecido: {source_row['codigo_ibge']}.")
        crop_rows.append(
            {
                **source_row,
                "uf": UF_BY_CODE[uf_code],
                "dados_agricolas_preliminares": source_row["dados_preliminares"],
            }
        )
    calendar_rows = load_csv(args.calendar)
    result = summarize(monthly_rows, crop_rows, calendar_rows)
    fields = list(result[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(result)
    print(f"Arquivo gerado: {args.output}")
    print(f"Registros UF/ano/safra: {len(result)}")
    print(
        "Ciclos sem todos os meses de chuva observados: "
        f"{sum(not row.get('chuva_acumulada_ciclo_mm') for row in result)}"
    )


if __name__ == "__main__":
    main()
