#!/usr/bin/env python3
"""Normaliza a exportação da tabela 839 do SIDRA para Brasil/UF, ano e safra."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


VARIABLES = (
    ("area_plantada_ha", "Área plantada"),
    ("area_colhida_ha", "Área colhida"),
    ("producao_t", "Quantidade produzida"),
    ("rendimento_kg_ha", "Rendimento médio da produção"),
)
SAFRAS = (("1a", 1), ("2a", 2))
MISSING_MARKERS = {"", "...", "..", "X"}


def parse_value(value: str) -> int | None:
    value = value.strip()
    if value == "-":
        return 0
    if value in MISSING_MARKERS:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Valor SIDRA inesperado: {value!r}") from exc


def parse_export(source: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    with source.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file, delimiter=";", quotechar='"'))

    sections = [
        index
        for index, row in enumerate(rows)
        if row and row[0].startswith("Vari")
    ]
    if len(sections) != len(VARIABLES):
        raise ValueError(
            f"Esperadas {len(VARIABLES)} seções de variáveis; "
            f"encontradas {len(sections)}."
        )

    values_by_key: dict[
        tuple[str, str, str, int, str], dict[str, int | None]
    ] = {}
    names_by_level: dict[tuple[str, str], str] = {}
    additive_mismatches = 0
    years_seen: set[int] = set()

    for section_number, start in enumerate(sections):
        end = sections[section_number + 1] if section_number + 1 < len(sections) else len(rows)
        variable, expected_heading = VARIABLES[section_number]
        heading = rows[start][0]
        if expected_heading not in heading:
            raise ValueError(
                f"Seção inesperada: esperado {expected_heading!r}, recebido {heading!r}."
            )

        year_headers = rows[start + 2]
        product_headers = rows[start + 3]
        if len(year_headers) != 70 or len(product_headers) != 72:
            raise ValueError(f"Cabeçalho da variável {variable} fora do formato esperado.")

        start_column = 3
        years: list[int] = []
        while start_column < len(year_headers):
            try:
                year = int(year_headers[start_column])
            except ValueError as exc:
                raise ValueError(
                    f"Ano inválido na seção {variable}: {year_headers[start_column]!r}."
                ) from exc
            if product_headers[start_column + 1] == "Total":
                raise ValueError(f"Colunas de safra inesperadas no ano {year}.")
            first_label = product_headers[start_column + 1]
            second_label = product_headers[start_column + 2]
            if "1" not in first_label or "2" not in second_label:
                raise ValueError(f"Safras não identificadas corretamente no ano {year}.")
            years.append(year)
            years_seen.add(year)

            for row in rows[start + 4 : end]:
                if len(row) != 72 or row[0] not in {"BR", "UF"}:
                    continue
                level, code, name = row[:3]
                location_key = (level, code)
                previous_name = names_by_level.setdefault(location_key, name)
                if previous_name != name:
                    raise ValueError(f"Nome inconsistente para código {code}: {name!r}.")

                total = parse_value(row[start_column])
                first_value = parse_value(row[start_column + 1])
                second_value = parse_value(row[start_column + 2])
                if (
                    variable in {"area_plantada_ha", "area_colhida_ha", "producao_t"}
                    and total is not None
                    and first_value is not None
                    and second_value is not None
                    and total != first_value + second_value
                ):
                    additive_mismatches += 1

                for safra, offset in SAFRAS:
                    key = (level, code, name, year, safra)
                    record_values = values_by_key.setdefault(key, {})
                    value = first_value if offset == 1 else second_value
                    record_values[variable] = value

            start_column += 3
        if years != list(range(min(years), max(years) + 1)):
            raise ValueError(f"Anos ausentes ou fora de ordem na seção {variable}.")

    if not years_seen:
        raise ValueError("Nenhum ano foi encontrado na exportação.")

    has_preliminary_note = any(
        "RESULTADOS PRELIMINARES" in cell
        for row in rows
        for cell in row
    )
    latest_year = max(years_seen)
    output: list[dict[str, object]] = []
    for key, values in values_by_key.items():
        level, code, name, year, safra = key
        record: dict[str, object] = {
            "nivel": "Brasil" if level == "BR" else "UF",
            "codigo_ibge": code,
            "localidade": name,
            "ano": year,
            "safra": safra,
        }
        for variable, _ in VARIABLES:
            record[variable] = values.get(variable)
        record["dados_preliminares"] = (
            "sim" if has_preliminary_note and year == latest_year else "nao"
        )
        output.append(record)

    output.sort(
        key=lambda row: (
            0 if row["nivel"] == "Brasil" else 1,
            str(row["codigo_ibge"]),
            int(row["ano"]),
            str(row["safra"]),
        )
    )
    stats = {
        "anos": len(years_seen),
        "brasil_ufs": len(names_by_level),
        "inconsistencias_total_vs_safras": additive_mismatches,
    }
    return output, stats


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "data" / "raw" / "agricultura" / "milho3.csv",
        help="Exportação original da tabela 839 do SIDRA.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "data" / "processed" / "ibge_milho_uf_ano_safra.csv",
        help="Caminho para o CSV normalizado.",
    )
    args = parser.parse_args()

    records, stats = parse_export(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "nivel",
        "codigo_ibge",
        "localidade",
        "ano",
        "safra",
        *(variable for variable, _ in VARIABLES),
        "dados_preliminares",
    ]
    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(records)

    print(f"Arquivo gerado: {args.output}")
    print(f"Registros: {len(records)}")
    print(f"Anos: {stats['anos']}; localidades Brasil/UF: {stats['brasil_ufs']}")
    print(
        "Divergências entre totais e safras (área/produção): "
        f"{stats['inconsistencias_total_vs_safras']}"
    )


if __name__ == "__main__":
    main()
