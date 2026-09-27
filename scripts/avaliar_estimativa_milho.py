#!/usr/bin/env python3
"""Avalia estimativa retrospectiva de produção de milho por UF e safra."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TRAIN_END_YEAR = 2019
TEST_START_YEAR = 2020
TEST_END_YEAR = 2025
CATEGORICAL_FEATURES = ("uf", "safra")
NUMERIC_FEATURES = (
    "ano",
    "rendimento_ano_anterior_kg_ha",
    "chuva_acumulada_ciclo_mm",
    "cobertura_temporal_chuva_ciclo_pct",
    "chuva_janela_plantio_mm",
    "cobertura_temporal_chuva_plantio_pct",
    "temp_ar_media_c_ciclo",
    "temp_ar_media_c_plantio",
    "temp_ar_min_c_ciclo",
    "temp_ar_max_c_ciclo",
    "temp_orvalho_media_c_ciclo",
    "umidade_relativa_media_pct_ciclo",
    "velocidade_vento_media_ms_ciclo",
)
OUTPUT_FIELDS = (
    "uf",
    "ano",
    "safra",
    "area_plantada_ha",
    "area_colhida_ha",
    "producao_observada_t",
    "rendimento_observado_kg_ha",
    "rendimento_ano_anterior_kg_ha",
    "rendimento_estimado_modelo_kg_ha",
    "producao_estimada_modelo_t",
    "erro_modelo_t",
    "erro_absoluto_modelo_t",
    "rendimento_estimado_referencia_kg_ha",
    "producao_estimada_referencia_t",
    "erro_referencia_t",
    "erro_absoluto_referencia_t",
    "chuva_acumulada_ciclo_mm",
    "chuva_parcial_ciclo_mm",
    "situacao_chuva_ciclo",
    "cobertura_temporal_chuva_ciclo_pct",
    "temp_ar_media_c_ciclo",
    "temp_ar_media_c_plantio",
    "dados_agricolas_preliminares",
    "metodo_calendario",
)
METRIC_FIELDS = (
    "granularidade",
    "periodo",
    "modelo",
    "n_observacoes",
    "anos_avaliados",
    "mae_producao_t",
    "rmse_producao_t",
    "wape_producao_pct",
    "erro_medio_producao_t",
    "mae_rendimento_kg_ha",
    "rmse_rendimento_kg_ha",
    "wape_rendimento_pct",
    "r2_producao",
    "preliminares_incluidos",
)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file, delimiter=";"))


def number(value: str) -> float | None:
    if not value.strip():
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def build_observations(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows_by_key: dict[tuple[str, int, str], dict[str, str]] = {}
    for row in rows:
        key = (row["uf"], int(row["ano"]), row["safra"])
        if key in rows_by_key:
            raise ValueError(f"Duplicata na base de safra por UF/ano: {key}.")
        rows_by_key[key] = row

    observations = []
    for key, row in rows_by_key.items():
        uf, year, season = key
        area = number(row["area_plantada_ha"])
        production = number(row["producao_t"])
        yield_value = number(row["rendimento_kg_ha"])
        observation: dict[str, object] = {
            "uf": uf,
            "ano": year,
            "safra": season,
            "area": area,
            "production": production,
            "yield": yield_value,
            "source": row,
        }
        previous = rows_by_key.get((uf, year - 1, season))
        observation["lag_yield"] = (
            number(previous["rendimento_kg_ha"]) if previous else None
        )
        observation["lag_year"] = year - 1
        observations.append(observation)

    observations.sort(key=lambda item: (item["ano"], item["uf"], item["safra"]))
    return observations


def create_training_features(
    observation: dict[str, object],
) -> dict[str, object]:
    source = observation["source"]
    assert isinstance(source, dict)
    features: dict[str, object] = {
        "uf": observation["uf"],
        "safra": observation["safra"],
        "ano": observation["ano"],
        "rendimento_ano_anterior_kg_ha": observation["lag_yield"],
    }
    for field in NUMERIC_FEATURES[2:]:
        features[field] = number(source.get(field, ""))
    return features


def create_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURES),
            ),
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                list(NUMERIC_FEATURES),
            ),
        ],
        remainder="drop",
    )
    estimator = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline((("preprocess", preprocessor), ("model", estimator)))


def metric_row(
    observations: list[dict[str, object]],
    estimate_field: str,
    granularity: str,
    period: str,
    model: str,
    preliminaries_included: bool,
) -> dict[str, object]:
    if not observations:
        raise ValueError(f"Nenhuma observação para calcular métricas: {period} / {model}.")
    actual = np.asarray([float(row["production"]) for row in observations])
    predicted = np.asarray([float(row[estimate_field]) for row in observations])
    production_error = predicted - actual
    residual_sum_squares = float(np.sum(production_error**2))
    total_sum_squares = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares
        else None
    )
    valid_yield_rows = [
        row
        for row in observations
        if row.get("yield") is not None and row.get(f"{model}_yield") is not None
    ]
    if valid_yield_rows:
        actual_yield = np.asarray(
            [float(row["yield"]) for row in valid_yield_rows]
        )
        predicted_yield = np.asarray(
            [float(row[f"{model}_yield"]) for row in valid_yield_rows]
        )
        yield_error = predicted_yield - actual_yield
        yield_mae = f"{np.mean(np.abs(yield_error)):.3f}"
        yield_rmse = f"{np.sqrt(np.mean(yield_error**2)):.3f}"
        yield_wape = (
            f"{100 * np.sum(np.abs(yield_error)) / np.sum(actual_yield):.3f}"
            if np.sum(actual_yield)
            else ""
        )
    else:
        yield_mae = ""
        yield_rmse = ""
        yield_wape = ""
    return {
        "granularidade": granularity,
        "periodo": period,
        "modelo": model,
        "n_observacoes": len(observations),
        "anos_avaliados": ",".join(
            str(year) for year in sorted({int(row["ano"]) for row in observations})
        ),
        "mae_producao_t": f"{np.mean(np.abs(production_error)):.3f}",
        "rmse_producao_t": f"{np.sqrt(np.mean(production_error**2)):.3f}",
        "wape_producao_pct": (
            f"{100 * np.sum(np.abs(production_error)) / np.sum(actual):.3f}"
            if np.sum(actual)
            else ""
        ),
        "erro_medio_producao_t": f"{np.mean(production_error):.3f}",
        "mae_rendimento_kg_ha": yield_mae,
        "rmse_rendimento_kg_ha": yield_rmse,
        "wape_rendimento_pct": yield_wape,
        "r2_producao": f"{r2:.5f}" if r2 is not None else "",
        "preliminares_incluidos": "sim" if preliminaries_included else "nao",
    }


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    observations: list[dict[str, object]],
    brazil_targets: list[dict[str, str]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    Pipeline,
    int,
]:
    train = [
        row
        for row in observations
        if row["ano"] <= TRAIN_END_YEAR
        and row["ano"] >= min(int(item["ano"]) for item in observations) + 1
        and row["area"] is not None
        and float(row["area"]) > 0
        and row["yield"] is not None
        and row["source"]["dados_agricolas_preliminares"] != "sim"
    ]
    test = [
        row
        for row in observations
        if TEST_START_YEAR <= int(row["ano"]) <= TEST_END_YEAR
        and row["area"] is not None
        and row["production"] is not None
        and row["yield"] is not None
    ]
    if not train or not test:
        raise ValueError("Treino ou teste vazio; confira cobertura das séries agrícolas.")

    model = create_model()
    model.fit(
        pd.DataFrame([create_training_features(row) for row in train]),
        np.asarray([float(row["yield"]) for row in train]),
    )
    training_yield_by_season = {}
    for season in ("1a", "2a"):
        values = [float(row["yield"]) for row in train if row["safra"] == season]
        if not values:
            raise ValueError(f"Safra sem observações de treino: {season}.")
        training_yield_by_season[season] = float(np.median(values))

    prediction_records: list[dict[str, object]] = []
    test_results = []
    model_predictions = model.predict(
        pd.DataFrame([create_training_features(row) for row in test])
    )
    for row, model_prediction in zip(test, model_predictions):
        area = float(row["area"])
        actual_production = float(row["production"])
        actual_yield = float(row["yield"])
        lag_yield = row["lag_yield"]
        reference_yield = (
            float(lag_yield)
            if lag_yield is not None
            else training_yield_by_season[str(row["safra"])]
        )
        if area == 0:
            reference_yield = 0.0
            predicted_yield = 0.0
        else:
            predicted_yield = max(0.0, float(model_prediction))
        predicted_production = predicted_yield * area / 1000.0
        reference_production = reference_yield * area / 1000.0
        enriched = {
            **row,
            "model_production": predicted_production,
            "model_yield": predicted_yield,
            "random_forest_clima_yield": predicted_yield,
            "reference_production": reference_production,
            "reference_yield": reference_yield,
            "referencia_ano_anterior_yield": reference_yield,
            "actual_production": actual_production,
            "actual_yield": actual_yield,
        }
        test_results.append(enriched)
        source = row["source"]
        prediction_records.append(
            {
                "uf": row["uf"],
                "ano": row["ano"],
                "safra": row["safra"],
                "area_plantada_ha": f"{area:.0f}",
                "area_colhida_ha": source["area_colhida_ha"],
                "producao_observada_t": f"{actual_production:.3f}",
                "rendimento_observado_kg_ha": f"{actual_yield:.3f}",
                "rendimento_ano_anterior_kg_ha": (
                    f"{float(lag_yield):.3f}" if lag_yield is not None else ""
                ),
                "rendimento_estimado_modelo_kg_ha": f"{predicted_yield:.9f}",
                "producao_estimada_modelo_t": f"{predicted_production:.3f}",
                "erro_modelo_t": f"{predicted_production - actual_production:.3f}",
                "erro_absoluto_modelo_t": f"{abs(predicted_production - actual_production):.3f}",
                "rendimento_estimado_referencia_kg_ha": f"{reference_yield:.9f}",
                "producao_estimada_referencia_t": f"{reference_production:.3f}",
                "erro_referencia_t": f"{reference_production - actual_production:.3f}",
                "erro_absoluto_referencia_t": f"{abs(reference_production - actual_production):.3f}",
                **{
                    field: source.get(field, "")
                    for field in (
                        "chuva_acumulada_ciclo_mm",
                        "chuva_parcial_ciclo_mm",
                        "situacao_chuva_ciclo",
                        "cobertura_temporal_chuva_ciclo_pct",
                        "temp_ar_media_c_ciclo",
                        "temp_ar_media_c_plantio",
                        "dados_agricolas_preliminares",
                        "metodo_calendario",
                    )
                },
            }
        )

    national_results: list[dict[str, object]] = []
    national_groups: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in test_results:
        national_groups.setdefault((int(row["ano"]), str(row["safra"])), []).append(row)
    for (year, season), group in sorted(national_groups.items()):
        actual = sum(float(row["production"]) for row in group)
        model_prediction = sum(float(row["model_production"]) for row in group)
        baseline_prediction = sum(float(row["reference_production"]) for row in group)
        brazil_source = next(
            (
                source
                for source in brazil_targets
                if source["ano"] == str(year) and source["safra"] == season
            ),
            None,
        )
        if brazil_source and number(brazil_source["producao_t"]) is not None:
            actual = float(number(brazil_source["producao_t"]))
        national_results.append(
            {
                "ano": year,
                "safra": season,
                "producao_observada_t": f"{actual:.3f}",
                "producao_estimada_modelo_t": f"{model_prediction:.3f}",
                "erro_modelo_t": f"{model_prediction - actual:.3f}",
                "erro_absoluto_modelo_t": f"{abs(model_prediction - actual):.3f}",
                "producao_estimada_referencia_t": f"{baseline_prediction:.3f}",
                "erro_referencia_t": f"{baseline_prediction - actual:.3f}",
                "erro_absoluto_referencia_t": f"{abs(baseline_prediction - actual):.3f}",
                "ufs_avaliadas": len(group),
                "dados_agricolas_preliminares": (
                    "sim"
                    if any(
                        row["source"]["dados_agricolas_preliminares"] == "sim"
                        for row in group
                    )
                    else "nao"
                ),
            }
        )

    metrics = []
    for included_preliminaries, period in (
        (True, "teste_2020_2025"),
        (False, "teste_2020_2024_nao_preliminar"),
    ):
        state_rows = [
            row
            for row in test_results
            if included_preliminaries
            or row["source"]["dados_agricolas_preliminares"] != "sim"
        ]
        for name, field in (
            ("referencia_ano_anterior", "reference_production"),
            ("random_forest_clima", "model_production"),
        ):
            metrics.append(
                metric_row(
                    state_rows,
                    field,
                    "UF_safra",
                    period,
                    name,
                    included_preliminaries,
                )
            )

        national_rows = [
            row
            for row in national_results
            if included_preliminaries or row["dados_agricolas_preliminares"] != "sim"
        ]
        for name, field in (
            ("referencia_ano_anterior", "reference_production"),
            ("random_forest_clima", "model_production"),
        ):
            adapted = [
                {
                    "uf": "Brasil",
                    "ano": row["ano"],
                    "safra": row["safra"],
                    "production": float(row["producao_observada_t"]),
                    "model_production": float(row["producao_estimada_modelo_t"]),
                    "reference_production": float(row["producao_estimada_referencia_t"]),
                    "yield": None,
                    "model_yield": None,
                    "reference_yield": None,
                }
                for row in national_rows
            ]
            metrics.append(
                metric_row(
                    adapted,
                    field,
                    "Brasil_ano_safra",
                    period,
                    name,
                    included_preliminaries,
                )
            )
    return prediction_records, national_results, metrics, model, len(train)


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    default_dir = repository_root / "data" / "processed" / "clima_uf_safra"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=default_dir / "clima_uf_ano_safra_ciclo.csv",
    )
    parser.add_argument(
        "--agriculture",
        type=Path,
        default=repository_root / "data" / "processed" / "ibge_milho_uf_ano_safra.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=default_dir / "avaliacao")
    args = parser.parse_args()

    brazil_targets = [
        row
        for row in load_rows(args.agriculture)
        if row["nivel"] == "Brasil"
    ]
    observations = build_observations(load_rows(args.input))
    predictions, national, metrics, model, training_count = evaluate(
        observations, brazil_targets
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "estimativas_teste_por_uf_safra.csv",
        OUTPUT_FIELDS,
        predictions,
    )
    write_csv(
        args.output_dir / "estimativas_teste_brasil_safra.csv",
        (
            "ano",
            "safra",
            "producao_observada_t",
            "producao_estimada_modelo_t",
            "erro_modelo_t",
            "erro_absoluto_modelo_t",
            "producao_estimada_referencia_t",
            "erro_referencia_t",
            "erro_absoluto_referencia_t",
            "ufs_avaliadas",
            "dados_agricolas_preliminares",
        ),
        national,
    )
    write_csv(args.output_dir / "metricas_teste.csv", METRIC_FIELDS, metrics)
    model_path = args.output_dir / "modelo_rendimento_random_forest.joblib"
    joblib.dump(model, model_path)

    print(f"Anos de treino: até {TRAIN_END_YEAR}; linhas de treino: {training_count}")
    print(
        f"Anos de teste: {TEST_START_YEAR}-{TEST_END_YEAR}; "
        f"estimativas UF/safra: {len(predictions)}"
    )
    print(f"Versão scikit-learn: {sklearn.__version__}")
    for row in metrics:
        print(
            f"{row['granularidade']} / {row['periodo']} / {row['modelo']}: "
            f"MAE={row['mae_producao_t']} t, WAPE={row['wape_producao_pct']}%"
        )
    print(f"Resultados: {args.output_dir}")
    print(f"Modelo: {model_path}")


if __name__ == "__main__":
    main()
