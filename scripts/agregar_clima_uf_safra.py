#!/usr/bin/env python3
"""Estima clima mensal de UFs por safra, ponderado pela área municipal plantada."""

from __future__ import annotations

import argparse
import calendar
import csv
import gzip
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


EARTH_RADIUS_KM = 6371.0088
NEAREST_STATIONS = 3
MONTH_COVERAGE_THRESHOLD = 0.80
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
SEASONS = (("1a", 1), ("2a", 2))
MISSING_MARKERS = {"", "...", "..", "X"}

DAILY_VARIABLES = (
    ("chuva_mm", "precipitacao_mm"),
    ("temp_ar_media_c", "temp_ar_media_c"),
    ("temp_ar_min_c", "temp_ar_min_c"),
    ("temp_ar_max_c", "temp_ar_max_c"),
    ("temp_orvalho_media_c", "temp_orvalho_media_c"),
    ("radiacao_global_kj_m2", "radiacao_global_kj_m2"),
    ("umidade_relativa_media_pct", "umidade_relativa_media_pct"),
    ("velocidade_vento_media_ms", "velocidade_vento_media_ms"),
    ("rajada_vento_max_ms", "rajada_vento_max_ms"),
)
HOUR_FIELDS = {
    "temp_ar_media_c": "horas_temp_ar_validas",
    "temp_ar_min_c": "horas_temp_ar_validas",
    "temp_ar_max_c": "horas_temp_ar_validas",
    "temp_orvalho_media_c": "horas_orvalho_validas",
    "radiacao_global_kj_m2": "horas_radiacao_validas",
    "umidade_relativa_media_pct": "horas_umidade_validas",
    "velocidade_vento_media_ms": "horas_vento_validas",
}
DAILY_COVERAGE_MINIMUM_HOURS = 18
CLIMATE_FIELDS = tuple(field for field, _ in DAILY_VARIABLES)


def parse_sidra_area(value: str) -> int | None:
    value = value.strip()
    if value == "-":
        return 0
    if value in MISSING_MARKERS:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Área SIDRA inesperada: {value!r}") from exc


def parse_measurement(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        number = float(value.replace(",", "."))
    except ValueError as exc:
        raise ValueError(f"Valor climático inesperado: {value!r}") from exc
    if not math.isfinite(number):
        return None
    return number


def coverage_column(climate_field: str) -> str:
    base_name = (
        climate_field[:-4] if climate_field.endswith("_pct") else climate_field
    )
    return f"area_coberta_{base_name}_pct"


def load_municipal_coordinates(
    path: Path,
) -> dict[str, tuple[str, float, float]]:
    coordinates = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            code = row["codigo_ibge"].strip().zfill(7)
            uf_code = row.get("codigo_uf", "").strip().zfill(2)
            if uf_code not in UF_BY_CODE:
                raise ValueError(f"Código de UF inválido para o município {code}.")
            coordinates[code] = (
                UF_BY_CODE[uf_code],
                float(row["latitude"]),
                float(row["longitude"]),
            )
    if not coordinates:
        raise ValueError(f"Nenhuma coordenada municipal encontrada em {path}.")
    return coordinates


def load_planting_areas(
    path: Path, coordinates: dict[str, tuple[str, float, float]]
) -> tuple[
    dict[tuple[int, str], dict[str, tuple[str, int]]],
    dict[int, str],
]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file, delimiter=";", quotechar='"'))
    sections = [
        index
        for index, row in enumerate(rows)
        if row and row[0].startswith("Vari")
    ]
    if len(sections) != 4 or "Área plantada" not in rows[sections[0]][0]:
        raise ValueError("Esperada exportação SIDRA com os quatro indicadores da tabela 839.")

    start = sections[0]
    year_headers = rows[start + 2]
    season_headers = rows[start + 3]
    if len(year_headers) != 70 or len(season_headers) != 72:
        raise ValueError("Cabeçalho de área plantada fora do formato SIDRA esperado.")

    preliminary_years = {}
    if any("RESULTADOS PRELIMINARES" in cell for row in rows for cell in row):
        latest_year = max(int(year_headers[column]) for column in range(3, 70, 3))
        preliminary_years[latest_year] = "sim"

    planting: dict[tuple[int, str], dict[str, tuple[str, int]]] = defaultdict(dict)
    for column in range(3, 70, 3):
        year = int(year_headers[column])
        if (
            "1" not in season_headers[column + 1]
            or "2" not in season_headers[column + 2]
        ):
            raise ValueError(f"Safras não identificadas nas colunas do ano {year}.")
        for season, offset in SEASONS:
            key = (year, season)
            for row in rows[start + 4 : sections[1]]:
                if len(row) != 72 or row[0] != "MU":
                    continue
                code = row[1].strip().zfill(7)
                area = parse_sidra_area(row[column + offset])
                if area is None or area <= 0:
                    continue
                if code not in coordinates:
                    raise ValueError(
                        f"Município com área plantada sem coordenada de referência: {code}."
                    )
                uf, _, _ = coordinates[code]
                planting[key][code] = (uf, area)
    return dict(planting), preliminary_years


def load_state_crop_targets(
    path: Path,
) -> dict[tuple[str, int, str], dict[str, str]]:
    targets = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file, delimiter=";"):
            if row["nivel"] != "UF":
                continue
            uf_code = row["codigo_ibge"].zfill(2)
            if uf_code not in UF_BY_CODE:
                raise ValueError(
                    f"Código de UF inválido nos indicadores do IBGE: {row['codigo_ibge']}."
                )
            key = (UF_BY_CODE[uf_code], int(row["ano"]), row["safra"])
            if key in targets:
                raise ValueError(f"Chave UF/ano/safra duplicada nos dados do IBGE: {key}.")
            targets[key] = {
                "area_plantada_ha_ibge": row["area_plantada_ha"],
                "area_colhida_ha": row["area_colhida_ha"],
                "producao_t": row["producao_t"],
                "rendimento_kg_ha": row["rendimento_kg_ha"],
                "dados_agricolas_preliminares": row["dados_preliminares"],
            }
    return targets


def unit_sphere_coordinates(
    locations: list[tuple[float, float]],
) -> np.ndarray:
    latitudes = np.radians(np.asarray([point[0] for point in locations]))
    longitudes = np.radians(np.asarray([point[1] for point in locations]))
    cos_latitude = np.cos(latitudes)
    return np.column_stack(
        (
            cos_latitude * np.cos(longitudes),
            cos_latitude * np.sin(longitudes),
            np.sin(latitudes),
        )
    )


def nearest_station_weights(
    municipality_codes: list[str],
    coordinates: dict[str, tuple[str, float, float]],
    station_locations: dict[str, tuple[float, float]],
) -> dict[str, list[tuple[str, float, float]]]:
    station_codes = sorted(station_locations)
    if not station_codes:
        raise ValueError("O ano não tem coordenadas de estações meteorológicas.")

    municipality_xyz = unit_sphere_coordinates(
        [(coordinates[code][1], coordinates[code][2]) for code in municipality_codes]
    )
    station_xyz = unit_sphere_coordinates(
        [station_locations[code] for code in station_codes]
    )
    result = {}
    for start in range(0, len(municipality_codes), 256):
        stop = min(start + 256, len(municipality_codes))
        delta = (
            municipality_xyz[start:stop, None, :]
            - station_xyz[None, :, :]
        )
        squared_chords = np.einsum("ijk,ijk->ij", delta, delta)
        count = min(NEAREST_STATIONS, len(station_codes))
        nearest = np.argpartition(squared_chords, count - 1, axis=1)[:, :count]
        nearest_distances = np.sqrt(
            np.take_along_axis(squared_chords, nearest, axis=1)
        )
        order = np.argsort(nearest_distances, axis=1)
        nearest = np.take_along_axis(nearest, order, axis=1)
        nearest_distances = np.take_along_axis(nearest_distances, order, axis=1)

        for offset, code in enumerate(municipality_codes[start:stop]):
            distances = (
                2.0
                * EARTH_RADIUS_KM
                * np.arcsin(np.minimum(nearest_distances[offset] / 2.0, 1.0))
            )
            if distances[0] < 0.001:
                raw_weights = np.zeros(count, dtype=float)
                raw_weights[0] = 1.0
            else:
                raw_weights = 1.0 / np.maximum(distances, 1.0) ** 2
            raw_weights /= raw_weights.sum()
            result[code] = [
                (station_codes[int(index)], float(distance), float(weight))
                for index, distance, weight in zip(
                    nearest[offset], distances, raw_weights
                )
            ]
    return result


def monthly_station_climate(path: Path, year: int) -> dict[tuple[str, int], dict[str, float | None]]:
    daily: dict[tuple[str, int], dict[str, object]] = {}
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file, delimiter=";"):
            month = int(row["data"][5:7])
            key = (row["codigo_estacao"], month)
            group = daily.setdefault(
                key,
                {
                    "precipitation_sum": 0.0,
                    "precipitation_days": 0,
                    "variable_sum": defaultdict(float),
                    "variable_min": {},
                    "variable_max": {},
                    "variable_days": defaultdict(int),
                },
            )
            if row["precipitacao_mm"]:
                group["precipitation_sum"] = (
                    float(group["precipitation_sum"])
                    + float(row["precipitacao_mm"])
                )
                group["precipitation_days"] = int(group["precipitation_days"]) + 1

            for field, _ in DAILY_VARIABLES[1:]:
                if field in HOUR_FIELDS:
                    hour_field = HOUR_FIELDS[field]
                    if int(row[hour_field] or 0) < DAILY_COVERAGE_MINIMUM_HOURS:
                        continue
                value = parse_measurement(row[field])
                if value is None:
                    continue
                group["variable_days"][field] = (
                    int(group["variable_days"][field]) + 1
                )
                if field == "temp_ar_min_c":
                    group["variable_min"][field] = min(
                        value, float(group["variable_min"].get(field, value))
                    )
                elif field in {"temp_ar_max_c", "rajada_vento_max_ms"}:
                    group["variable_max"][field] = max(
                        value, float(group["variable_max"].get(field, value))
                    )
                elif field == "radiacao_global_kj_m2":
                    group["variable_sum"][field] += value
                else:
                    group["variable_sum"][field] += value

    station_monthly = {}
    for (station, month), group in daily.items():
        days_in_month = calendar.monthrange(year, month)[1]
        minimum_days = math.ceil(days_in_month * MONTH_COVERAGE_THRESHOLD)
        values: dict[str, float | None] = {"chuva_mm": None}
        if int(group["precipitation_days"]) >= minimum_days:
            values["chuva_mm"] = float(group["precipitation_sum"])

        for field, _ in DAILY_VARIABLES[1:]:
            values[field] = None
            valid_days = int(group["variable_days"][field])
            if valid_days < minimum_days:
                continue
            if field == "temp_ar_min_c":
                values[field] = float(group["variable_min"][field])
            elif field in {"temp_ar_max_c", "rajada_vento_max_ms"}:
                values[field] = float(group["variable_max"][field])
            elif field == "radiacao_global_kj_m2":
                values[field] = float(group["variable_sum"][field])
            else:
                values[field] = float(group["variable_sum"][field]) / valid_days
        station_monthly[(station, month)] = values
    return station_monthly


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_year(
    year: int,
    planting_by_season: dict[tuple[int, str], dict[str, tuple[str, int]]],
    coordinates: dict[str, tuple[str, float, float]],
    crop_targets: dict[tuple[str, int, str], dict[str, str]],
    manifest_path: Path,
    climate_path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    with manifest_path.open(encoding="utf-8-sig", newline="") as file:
        manifest_rows = [
            row for row in csv.DictReader(file, delimiter=";") if int(row["ano"]) == year
        ]
    if not manifest_rows:
        raise ValueError(f"Nenhum arquivo de estação no manifesto para {year}.")

    station_locations = {}
    for row in manifest_rows:
        code = row["codigo_estacao"]
        station_locations[code] = (float(row["latitude"]), float(row["longitude"]))
    available_station_codes = set(station_locations)

    seasons = {
        season: areas
        for (area_year, season), areas in planting_by_season.items()
        if area_year == year
    }
    productive_municipalities = sorted(
        {
            code
            for areas in seasons.values()
            for code, (_, area) in areas.items()
            if area > 0
        }
    )
    neighbors = nearest_station_weights(
        productive_municipalities, coordinates, station_locations
    )
    station_monthly = monthly_station_climate(climate_path, year)

    season_area: dict[tuple[str, str], int] = defaultdict(int)
    season_uf_municipalities: dict[tuple[str, str], set[str]] = defaultdict(set)
    feature_sums: dict[tuple[str, str, int, str], float] = defaultdict(float)
    feature_area: dict[tuple[str, str, int, str], int] = defaultdict(int)
    uf_area: dict[tuple[str, str], int] = defaultdict(int)
    uf_nearest_distance: dict[tuple[str, str], float] = defaultdict(float)
    uf_near_area: dict[tuple[str, str], int] = defaultdict(int)

    assignment_rows = []
    for season, areas in sorted(seasons.items()):
        for municipality, (uf, area) in sorted(areas.items()):
            season_area[(uf, season)] += area
            season_uf_municipalities[(uf, season)].add(municipality)
            if municipality not in neighbors:
                raise ValueError(f"Sem estações próximas para o município {municipality}.")
            nearest = neighbors[municipality]
            uf_area[(uf, season)] += area
            uf_nearest_distance[(uf, season)] += area * nearest[0][1]
            if nearest[0][1] <= 100:
                uf_near_area[(uf, season)] += area
            assignment = {
                "uf": uf,
                "ano": year,
                "safra": season,
                "codigo_municipio": municipality,
                "area_plantada_ha": area,
            }
            for index in range(NEAREST_STATIONS):
                if index < len(nearest):
                    station, distance, weight = nearest[index]
                    assignment[f"codigo_estacao_{index + 1}"] = station
                    assignment[f"distancia_{index + 1}_km"] = f"{distance:.3f}"
                    assignment[f"peso_geografico_{index + 1}"] = f"{weight:.8f}"
                else:
                    assignment[f"codigo_estacao_{index + 1}"] = ""
                    assignment[f"distancia_{index + 1}_km"] = ""
                    assignment[f"peso_geografico_{index + 1}"] = ""
            assignment_rows.append(assignment)

            for month in range(1, 13):
                for climate_field in CLIMATE_FIELDS:
                    weighted_value = 0.0
                    usable_weight = 0.0
                    for station, _, spatial_weight in nearest:
                        if station not in available_station_codes:
                            continue
                        station_features = station_monthly.get((station, month))
                        if not station_features:
                            continue
                        station_value = station_features.get(climate_field)
                        if station_value is None:
                            continue
                        weighted_value += spatial_weight * station_value
                        usable_weight += spatial_weight
                    if usable_weight:
                        estimate = weighted_value / usable_weight
                        key = (uf, season, month, climate_field)
                        feature_sums[key] += area * estimate
                        feature_area[key] += area

    output = []
    for (uf, season), total_area in sorted(season_area.items()):
        municipality_count = len(season_uf_municipalities[(uf, season)])
        nearest_distance = (
            uf_nearest_distance[(uf, season)] / total_area if total_area else None
        )
        near_area_share = (
            100.0 * uf_near_area[(uf, season)] / total_area if total_area else None
        )
        for month in range(1, 13):
            row: dict[str, object] = {
                "uf": uf,
                "ano": year,
                "safra": season,
                "mes": month,
                "area_plantada_ha_ponderacao": total_area,
                "area_municipal_sobre_uf_pct": "",
                "divergencia_area_ponderacao_vs_ibge_ha": "",
                "municipios_produtores": municipality_count,
                "distancia_estacao_mais_proxima_media_km": (
                    f"{nearest_distance:.3f}" if nearest_distance is not None else ""
                ),
                "area_com_estacao_em_ate_100km_pct": (
                    f"{near_area_share:.2f}" if near_area_share is not None else ""
                ),
            }
            target = crop_targets.get((uf, year, season), {})
            row.update(
                {
                    field: target.get(field, "")
                    for field in (
                        "area_plantada_ha_ibge",
                        "area_colhida_ha",
                        "producao_t",
                        "rendimento_kg_ha",
                        "dados_agricolas_preliminares",
                    )
                }
            )
            state_planted_area = parse_sidra_area(
                target.get("area_plantada_ha_ibge", "")
            )
            if state_planted_area is not None:
                row["divergencia_area_ponderacao_vs_ibge_ha"] = (
                    total_area - state_planted_area
                )
                if state_planted_area > 0:
                    row["area_municipal_sobre_uf_pct"] = (
                        f"{100.0 * total_area / state_planted_area:.3f}"
                    )
            for climate_field in CLIMATE_FIELDS:
                key = (uf, season, month, climate_field)
                covered_area = feature_area[key]
                row[climate_field] = (
                    f"{feature_sums[key] / covered_area:.3f}"
                    if covered_area
                    else ""
                )
                coverage_field = coverage_column(climate_field)
                row[coverage_field] = (
                    f"{100.0 * covered_area / total_area:.2f}"
                    if total_area
                    else ""
                )
                row[f"municipios_com_{climate_field}"] = sum(
                    1
                    for municipality, (municipality_uf, area) in seasons[season].items()
                    if municipality_uf == uf
                    and area > 0
                    and any(
                        station_monthly.get((station, month), {}).get(climate_field)
                        is not None
                        for station, _, _ in neighbors[municipality]
                    )
                )
            output.append(row)
    return output, assignment_rows


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agriculture",
        type=Path,
        default=repository_root / "data" / "raw" / "agricultura" / "milho3.csv",
    )
    parser.add_argument(
        "--coordinates",
        type=Path,
        default=repository_root
        / "data"
        / "raw"
        / "geografia"
        / "municipios_centroides.csv",
    )
    parser.add_argument(
        "--crop-targets",
        type=Path,
        default=repository_root
        / "data"
        / "processed"
        / "ibge_milho_uf_ano_safra.csv",
        help="Base normalizada do passo 1 com indicadores-alvo por UF/ano/safra.",
    )
    parser.add_argument(
        "--station-manifest",
        type=Path,
        default=repository_root
        / "data"
        / "processed"
        / "inmet_diario"
        / "inmet_manifest_estacao_ano.csv",
    )
    parser.add_argument(
        "--climate-dir",
        type=Path,
        default=repository_root / "data" / "processed" / "inmet_diario",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root / "data" / "processed" / "clima_uf_safra",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        type=int,
        help="Anos a processar; por padrão, os anos disponíveis de 2003 a 2025.",
    )
    args = parser.parse_args()

    coordinates = load_municipal_coordinates(args.coordinates)
    planting, preliminary_years = load_planting_areas(args.agriculture, coordinates)
    crop_targets = load_state_crop_targets(args.crop_targets)
    years = args.years or sorted({year for year, _ in planting})
    result_rows = []
    assignment_rows = []
    for year in years:
        climate_path = args.climate_dir / f"inmet_diario_{year}.csv.gz"
        if not climate_path.is_file():
            raise FileNotFoundError(f"Partição climática ausente: {climate_path}")
        year_rows, year_assignments = aggregate_year(
            year,
            planting,
            coordinates,
            crop_targets,
            args.station_manifest,
            climate_path,
        )
        result_rows.extend(year_rows)
        assignment_rows.extend(year_assignments)
        print(
            f"{year}: {len(year_rows)} agregados UF-safra-mês, "
            f"{len(year_assignments)} pesos municipais"
        )

    if not result_rows:
        raise ValueError("Nenhum agregado climático foi criado.")

    area_audit = {}
    for row in result_rows:
        key = (row["uf"], row["ano"], row["safra"])
        area_audit.setdefault(key, row["divergencia_area_ponderacao_vs_ibge_ha"])
    differences = [
        (key, value)
        for key, value in area_audit.items()
        if value not in ("", None, 0, "0")
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    climate_fields = [
        "uf",
        "ano",
        "safra",
        "mes",
        "area_plantada_ha_ponderacao",
        "area_plantada_ha_ibge",
        "area_municipal_sobre_uf_pct",
        "divergencia_area_ponderacao_vs_ibge_ha",
        "area_colhida_ha",
        "producao_t",
        "rendimento_kg_ha",
        "dados_agricolas_preliminares",
        "municipios_produtores",
        "distancia_estacao_mais_proxima_media_km",
        "area_com_estacao_em_ate_100km_pct",
    ]
    for climate_field in CLIMATE_FIELDS:
        climate_fields.extend(
            (
                climate_field,
                coverage_column(climate_field),
                f"municipios_com_{climate_field}",
            )
        )
    write_csv(
        args.output_dir / "clima_uf_ano_safra_mensal.csv",
        climate_fields,
        result_rows,
    )

    assignment_fields = [
        "uf",
        "ano",
        "safra",
        "codigo_municipio",
        "area_plantada_ha",
    ]
    for index in range(1, NEAREST_STATIONS + 1):
        assignment_fields.extend(
            (
                f"codigo_estacao_{index}",
                f"distancia_{index}_km",
                f"peso_geografico_{index}",
            )
        )
    assignment_path = args.output_dir / "pesos_estacao_municipio_ano_safra.csv.gz"
    with gzip.open(assignment_path, "wt", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=assignment_fields, delimiter=";")
        writer.writeheader()
        writer.writerows(assignment_rows)

    print(f"Clima mensal por UF/safra: {args.output_dir / 'clima_uf_ano_safra_mensal.csv'}")
    print(f"Pesos espaciais: {assignment_path}")
    print(f"Total: {len(result_rows)} linhas agregadas e {len(assignment_rows)} linhas municipais.")
    print(
        "Ano(s) de dados agrícolas preliminares: "
        + (", ".join(str(year) for year in sorted(preliminary_years)) or "nenhum")
    )
    if differences:
        examples = ", ".join(
            f"{uf}/{year}/{season}: {difference} ha"
            for (uf, year, season), difference in sorted(
                differences, key=lambda item: abs(int(item[1])), reverse=True
            )[:5]
        )
        print(
            f"Divergência de soma das áreas municipais vs. UF: {len(differences)} "
            f"combinações; maiores diferenças: {examples}"
        )


if __name__ == "__main__":
    main()
