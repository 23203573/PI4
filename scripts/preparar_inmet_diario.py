#!/usr/bin/env python3
"""Normaliza os CSVs horários do INMET em partições diárias por ano."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import math
import re
import unicodedata
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from zipfile import ZipFile


DAILY_FIELDS = (
    "data",
    "ano",
    "codigo_estacao",
    "nome_estacao",
    "uf",
    "regiao",
    "latitude",
    "longitude",
    "altitude_m",
    "observacoes_horarias",
    "horas_precipitacao_validas",
    "precipitacao_mm",
    "precipitacao_mm_parcial",
    "horas_temp_ar_validas",
    "temp_ar_media_c",
    "temp_ar_min_c",
    "temp_ar_max_c",
    "horas_orvalho_validas",
    "temp_orvalho_media_c",
    "horas_radiacao_validas",
    "radiacao_global_kj_m2",
    "horas_umidade_validas",
    "umidade_relativa_media_pct",
    "horas_vento_validas",
    "velocidade_vento_media_ms",
    "rajada_vento_max_ms",
)
MANIFEST_FIELDS = (
    "ano",
    "arquivo_origem",
    "codigo_estacao",
    "nome_estacao",
    "uf",
    "regiao",
    "latitude",
    "longitude",
    "altitude_m",
    "metadados_incompletos",
    "data_inicial",
    "data_final",
    "linhas_origem",
    "horas_unicas",
    "horas_esperadas_intervalo",
    "timestamps_duplicados",
    "horas_precipitacao_validas",
    "horas_precipitacao_ausentes",
    "dias_com_precipitacao_completa",
    "cobertura_horaria_pct",
    "cobertura_precipitacao_pct",
)
VARIABLES = {
    "precipitacao": 2,
    "temp_ar": 7,
    "temp_orvalho": 8,
    "radiacao": 6,
    "umidade": 15,
    "vento": 18,
    "rajada": 17,
}
MISSING_TEXT = {"", "-9999", "-9999.0", "-9999,0"}
STATION_CODE = re.compile(r"(?:^|_)([A-Z]\d{3})(?:_|$)", re.IGNORECASE)
DATE_PATTERN = re.compile(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$")
TIME_PATTERN = re.compile(r"(\d{2}):?(\d{2})")


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"[^A-Z0-9]+", " ", without_marks.upper()).strip()


def parse_number(value: str) -> float | None:
    normalized = value.strip()
    if normalized in MISSING_TEXT:
        return None
    try:
        number = float(normalized.replace(",", "."))
    except ValueError as exc:
        raise ValueError(f"Valor numérico inesperado: {value!r}") from exc
    if not math.isfinite(number) or number == -9999:
        return None
    return number


def parse_timestamp(day_text: str, hour_text: str) -> datetime:
    match = DATE_PATTERN.fullmatch(day_text.strip())
    if not match:
        raise ValueError(f"Data inesperada: {day_text!r}")
    observation_date = date(*(int(part) for part in match.groups()))
    time_match = TIME_PATTERN.search(hour_text)
    if not time_match:
        raise ValueError(f"Hora inesperada: {hour_text!r}")
    hour, minute = (int(part) for part in time_match.groups())
    if hour == 24 and minute == 0:
        observation_date += timedelta(days=1)
        hour = 0
    if hour > 23 or minute > 59:
        raise ValueError(f"Hora fora do intervalo: {hour_text!r}")
    return datetime(observation_date.year, observation_date.month, observation_date.day, hour, minute)


def station_metadata(
    metadata_rows: list[list[str]], filename: str
) -> tuple[dict[str, str], str]:
    metadata: dict[str, str] = {}
    for row in metadata_rows:
        if len(row) < 2:
            continue
        key = normalize_text(row[0])
        value = row[1].strip()
        if key.startswith("REGIAO"):
            metadata["regiao"] = value
        elif key == "UF":
            metadata["uf"] = value
        elif key.startswith("ESTACAO"):
            metadata["nome_estacao"] = value
        elif key.startswith("CODIGO"):
            metadata["codigo_estacao"] = value
        elif key == "LATITUDE":
            metadata["latitude"] = value
        elif key == "LONGITUDE":
            metadata["longitude"] = value
        elif key == "ALTITUDE":
            metadata["altitude_m"] = value

    code_match = STATION_CODE.search(filename)
    metadata.setdefault(
        "codigo_estacao", code_match.group(1).upper() if code_match else ""
    )
    metadata.setdefault("nome_estacao", Path(filename).stem)
    for key in ("uf", "regiao", "latitude", "longitude", "altitude_m"):
        metadata.setdefault(key, "")
    if not metadata["codigo_estacao"]:
        raise ValueError(f"Código da estação não encontrado em {filename!r}.")
    issues = []
    for key in ("latitude", "longitude", "altitude_m"):
        try:
            number = parse_number(metadata[key])
        except ValueError:
            number = None
        if number is None:
            if metadata[key]:
                issues.append(f"{key}={metadata[key]}")
            metadata[key] = ""
        else:
            metadata[key] = f"{number:g}"
    return metadata, "|".join(issues)


def empty_day() -> dict[str, object]:
    return {
        "observacoes": 0,
        "horas_precipitacao_validas": 0,
        "precipitacao_mm_parcial": 0.0,
        "horas_temp_ar_validas": 0,
        "temp_ar_soma": 0.0,
        "temp_ar_min": None,
        "temp_ar_max": None,
        "horas_orvalho_validas": 0,
        "temp_orvalho_soma": 0.0,
        "horas_radiacao_validas": 0,
        "radiacao_global_kj_m2": 0.0,
        "horas_umidade_validas": 0,
        "umidade_relativa_soma": 0.0,
        "horas_vento_validas": 0,
        "velocidade_vento_soma": 0.0,
        "rajada_vento_max": None,
    }


def add_value(
    day_data: dict[str, object],
    row: list[str],
    variable: str,
    count_key: str,
    aggregate_key: str,
) -> float | None:
    value = parse_number(row[VARIABLES[variable]])
    if value is not None:
        day_data[count_key] = int(day_data[count_key]) + 1
        day_data[aggregate_key] = float(day_data[aggregate_key]) + value
    return value


def summarize_station_file(
    archive: ZipFile, filename: str, year: int
) -> tuple[list[dict[str, object]], dict[str, object]]:
    with archive.open(filename) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="")
        reader = csv.reader(text, delimiter=";")
        metadata_rows = [next(reader) for _ in range(8)]
        header = next(reader)
        if len(header) != 20 or not header[0].strip().lower().startswith("data"):
            raise ValueError(f"Cabeçalho horário inesperado em {filename!r}.")
        metadata, metadata_issues = station_metadata(metadata_rows, filename)
        days: dict[date, dict[str, object]] = defaultdict(empty_day)
        seen_timestamps: set[datetime] = set()
        source_rows = 0
        duplicate_timestamps = 0

        for row in reader:
            if not row or not any(value.strip() for value in row):
                continue
            if len(row) < 19:
                raise ValueError(
                    f"Linha com {len(row)} colunas (esperadas 20) em {filename!r}."
                )
            source_rows += 1
            try:
                timestamp = parse_timestamp(row[0], row[1])
            except ValueError as exc:
                raise ValueError(f"{filename}: {exc}") from exc
            if timestamp in seen_timestamps:
                duplicate_timestamps += 1
                continue
            seen_timestamps.add(timestamp)

            day_data = days[timestamp.date()]
            day_data["observacoes"] = int(day_data["observacoes"]) + 1

            precipitation = add_value(
                day_data,
                row,
                "precipitacao",
                "horas_precipitacao_validas",
                "precipitacao_mm_parcial",
            )
            air_temp = add_value(
                day_data, row, "temp_ar", "horas_temp_ar_validas", "temp_ar_soma"
            )
            if air_temp is not None:
                current_min = day_data["temp_ar_min"]
                current_max = day_data["temp_ar_max"]
                day_data["temp_ar_min"] = (
                    air_temp if current_min is None else min(float(current_min), air_temp)
                )
                day_data["temp_ar_max"] = (
                    air_temp if current_max is None else max(float(current_max), air_temp)
                )
            add_value(
                day_data,
                row,
                "temp_orvalho",
                "horas_orvalho_validas",
                "temp_orvalho_soma",
            )
            add_value(
                day_data,
                row,
                "radiacao",
                "horas_radiacao_validas",
                "radiacao_global_kj_m2",
            )
            add_value(
                day_data,
                row,
                "umidade",
                "horas_umidade_validas",
                "umidade_relativa_soma",
            )
            add_value(
                day_data,
                row,
                "vento",
                "horas_vento_validas",
                "velocidade_vento_soma",
            )
            gust = parse_number(row[VARIABLES["rajada"]])
            if gust is not None:
                current_gust = day_data["rajada_vento_max"]
                day_data["rajada_vento_max"] = (
                    gust if current_gust is None else max(float(current_gust), gust)
                )

    if not days:
        raise ValueError(f"Nenhuma observação horária em {filename!r}.")

    first_date, last_date = min(days), max(days)
    expected_hours = ((last_date - first_date).days + 1) * 24
    daily_rows = []
    complete_precip_days = 0
    for observation_date in sorted(days):
        data = days[observation_date]
        observations = int(data["observacoes"])
        precipitation_count = int(data["horas_precipitacao_validas"])
        precipitation_sum = float(data["precipitacao_mm_parcial"])
        is_complete = observations == 24 and precipitation_count == 24
        complete_precip_days += int(is_complete)

        output_row: dict[str, object] = {
            "data": observation_date.isoformat(),
            "ano": observation_date.year,
            **metadata,
            "observacoes_horarias": observations,
            "horas_precipitacao_validas": precipitation_count,
            "precipitacao_mm": f"{precipitation_sum:.3f}" if is_complete else "",
            "precipitacao_mm_parcial": f"{precipitation_sum:.3f}",
        }
        for count_field, average_field, count_key, sum_key in (
            (
                "horas_temp_ar_validas",
                "temp_ar_media_c",
                "horas_temp_ar_validas",
                "temp_ar_soma",
            ),
            (
                "horas_orvalho_validas",
                "temp_orvalho_media_c",
                "horas_orvalho_validas",
                "temp_orvalho_soma",
            ),
            (
                "horas_umidade_validas",
                "umidade_relativa_media_pct",
                "horas_umidade_validas",
                "umidade_relativa_soma",
            ),
            (
                "horas_vento_validas",
                "velocidade_vento_media_ms",
                "horas_vento_validas",
                "velocidade_vento_soma",
            ),
        ):
            count = int(data[count_key])
            output_row[count_field] = count
            output_row[average_field] = f"{float(data[sum_key]) / count:.3f}" if count else ""
        output_row["temp_ar_min_c"] = (
            f"{float(data['temp_ar_min']):.3f}" if data["temp_ar_min"] is not None else ""
        )
        output_row["temp_ar_max_c"] = (
            f"{float(data['temp_ar_max']):.3f}" if data["temp_ar_max"] is not None else ""
        )
        output_row["horas_radiacao_validas"] = int(data["horas_radiacao_validas"])
        output_row["radiacao_global_kj_m2"] = (
            f"{float(data['radiacao_global_kj_m2']):.3f}"
            if int(data["horas_radiacao_validas"])
            else ""
        )
        output_row["rajada_vento_max_ms"] = (
            f"{float(data['rajada_vento_max']):.3f}"
            if data["rajada_vento_max"] is not None
            else ""
        )
        daily_rows.append(output_row)

    valid_precipitation_hours = sum(
        int(data["horas_precipitacao_validas"]) for data in days.values()
    )
    unique_hours = len(seen_timestamps)
    manifest = {
        "ano": year,
        "arquivo_origem": filename,
        **metadata,
        "metadados_incompletos": metadata_issues,
        "data_inicial": first_date.isoformat(),
        "data_final": last_date.isoformat(),
        "linhas_origem": source_rows,
        "horas_unicas": unique_hours,
        "horas_esperadas_intervalo": expected_hours,
        "timestamps_duplicados": duplicate_timestamps,
        "horas_precipitacao_validas": valid_precipitation_hours,
        "horas_precipitacao_ausentes": unique_hours - valid_precipitation_hours,
        "dias_com_precipitacao_completa": complete_precip_days,
        "cobertura_horaria_pct": (
            f"{100 * unique_hours / expected_hours:.2f}" if expected_hours else ""
        ),
        "cobertura_precipitacao_pct": (
            f"{100 * valid_precipitation_hours / expected_hours:.2f}"
            if expected_hours
            else ""
        ),
    }
    return daily_rows, manifest


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def write_state_year_coverage(
    output_path: Path, station_manifest: list[dict[str, object]]
) -> None:
    grouped: dict[tuple[str, int], dict[str, object]] = {}
    for row in station_manifest:
        key = (str(row["uf"]), int(row["ano"]))
        data = grouped.setdefault(
            key,
            {
                "uf": key[0],
                "ano": key[1],
                "estacoes": set(),
                "arquivos_estacao": 0,
                "horas_unicas": 0,
                "horas_esperadas_intervalo": 0,
                "horas_precipitacao_validas": 0,
            },
        )
        data["estacoes"].add(str(row["codigo_estacao"]))
        data["arquivos_estacao"] = int(data["arquivos_estacao"]) + 1
        for field in (
            "horas_unicas",
            "horas_esperadas_intervalo",
            "horas_precipitacao_validas",
        ):
            data[field] = int(data[field]) + int(row[field])

    rows = []
    for key in sorted(grouped):
        data = grouped[key]
        expected = int(data["horas_esperadas_intervalo"])
        valid = int(data["horas_precipitacao_validas"])
        rows.append(
            {
                **data,
                "estacoes": len(data["estacoes"]),
                "cobertura_precipitacao_pct": (
                    f"{100 * valid / expected:.2f}" if expected else ""
                ),
            }
        )
    write_csv(
        output_path,
        (
            "uf",
            "ano",
            "estacoes",
            "arquivos_estacao",
            "horas_unicas",
            "horas_esperadas_intervalo",
            "horas_precipitacao_validas",
            "cobertura_precipitacao_pct",
        ),
        rows,
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root / "data" / "raw" / "inmet" / "dadoshistoricos",
        help="Diretório dos ZIPs anuais do INMET.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "processed" / "inmet_diario",
        help="Diretório das partições diárias compactadas e manifestos.",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        type=int,
        help="Anos a processar; por padrão, todos os ZIPs 2003–2025 encontrados.",
    )
    args = parser.parse_args()

    archives = (
        [(year, args.input_dir / f"{year}.zip") for year in args.years]
        if args.years
        else [
            (int(path.stem), path)
            for path in sorted(args.input_dir.glob("*.zip"))
            if path.stem.isdigit() and 2003 <= int(path.stem) <= 2025
        ]
    )
    if not archives:
        raise FileNotFoundError(f"Nenhum ZIP anual encontrado em {args.input_dir}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    station_manifest: list[dict[str, object]] = []
    total_daily_rows = 0
    for year, archive_path in archives:
        if not archive_path.is_file():
            raise FileNotFoundError(f"Arquivo INMET não encontrado: {archive_path}")
        temporary_output = args.output_dir / f"inmet_diario_{year}.csv.gz.tmp"
        final_output = args.output_dir / f"inmet_diario_{year}.csv.gz"
        station_count = 0
        daily_count = 0
        try:
            with ZipFile(archive_path) as archive, gzip.open(
                temporary_output, "wt", encoding="utf-8", newline=""
            ) as output_file:
                writer = csv.DictWriter(output_file, fieldnames=DAILY_FIELDS, delimiter=";")
                writer.writeheader()
                entries = sorted(
                    item.filename
                    for item in archive.infolist()
                    if not item.is_dir() and item.filename.lower().endswith(".csv")
                )
                if not entries:
                    raise ValueError(f"Nenhum CSV de estação encontrado em {archive_path}.")
                for filename in entries:
                    daily_rows, manifest_row = summarize_station_file(
                        archive, filename, year
                    )
                    writer.writerows(daily_rows)
                    station_manifest.append(manifest_row)
                    station_count += 1
                    daily_count += len(daily_rows)
            temporary_output.replace(final_output)
        except Exception:
            temporary_output.unlink(missing_ok=True)
            raise
        total_daily_rows += daily_count
        print(
            f"{year}: {station_count} arquivos de estação, "
            f"{daily_count} registros diários -> {final_output}"
        )

    write_csv(args.output_dir / "inmet_manifest_estacao_ano.csv", MANIFEST_FIELDS, station_manifest)
    write_state_year_coverage(
        args.output_dir / "inmet_cobertura_uf_ano.csv", station_manifest
    )
    print(
        f"Concluído: {len(archives)} anos, {len(station_manifest)} arquivos de estação, "
        f"{total_daily_rows} registros diários."
    )


if __name__ == "__main__":
    main()
