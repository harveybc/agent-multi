#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import sys
import warnings
import random
from statsmodels.tsa.arima.model import ARIMA
from deap import base, creator, tools
from tqdm import tqdm

def eval_arima(individual, series):
    p, d, q = individual
    # Penaliza si se pasan parámetros negativos
    if p < 0 or d < 0 or q < 0:
        return (1e6,),
    try:
        model = ARIMA(series, order=(int(p), int(d), int(q))).fit()
        aic = model.aic
    except Exception as e:
        aic = 1e6  # Penalización alta en caso de error
    return (aic,)

def main():
    parser = argparse.ArgumentParser(
        description="Optimiza los parámetros ARIMA usando DEAP para minimizar el AIC, usando un subconjunto de la serie si se desea."
    )
    parser.add_argument("csv_file", type=str, help="Ruta al archivo CSV con la serie de tiempo.")
    parser.add_argument("--column", type=str, default="CLOSE", help="Nombre de la columna con la serie de tiempo (por defecto 'CLOSE').")
    parser.add_argument("--p_max", type=int, default=10, help="Valor máximo para p (orden autorregresivo).")
    parser.add_argument("--d_max", type=int, default=5, help="Valor máximo para d (diferenciación).")
    parser.add_argument("--q_max", type=int, default=10, help="Valor máximo para q (orden de media móvil).")
    parser.add_argument("--pop_size", type=int, default=10, help="Tamaño de la población.")
    parser.add_argument("--ngen", type=int, default=10, help="Número de generaciones.")
    parser.add_argument("--max_steps", type=int, default=6300, help="Máximo número de filas a usar durante el ajuste de ARIMA.")
    args = parser.parse_args()

    # Leer el CSV y seleccionar la columna de la serie
    try:
        df = pd.read_csv(args.csv_file)
    except Exception as e:
        sys.exit(f"Error al leer el CSV: {e}")

    if args.column not in df.columns:
        sys.exit(f"La columna '{args.column}' no se encuentra. Columnas disponibles: {list(df.columns)}")
    series = df[args.column]

    if series.isnull().any():
        sys.exit("La serie contiene valores nulos. Por favor, limpia los datos y vuelve a intentarlo.")

    try:
        series = pd.to_numeric(series)
    except Exception as e:
        sys.exit(f"Error al convertir la serie a valores numéricos: {e}")

    # Limitar la serie al número de filas especificado en max_steps (si se define)
    if args.max_steps is not None:
        series = series.iloc[:args.max_steps]
        print(f"Se utilizarán las primeras {args.max_steps} filas de la serie.")

    warnings.filterwarnings("ignore")

    # Configuración de DEAP: minimizamos el AIC
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    # Generadores de atributos: números enteros aleatorios dentro del rango definido
    toolbox.register("attr_p", random.randint, 0, args.p_max)
    toolbox.register("attr_d", random.randint, 0, args.d_max)
    toolbox.register("attr_q", random.randint, 0, args.q_max)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_p, toolbox.attr_d, toolbox.attr_q), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Función de evaluación
    def eval_func(individual):
        return eval_arima(individual, series)
    toolbox.register("evaluate", eval_func)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutUniformInt, low=[0, 0, 0], up=[args.p_max, args.d_max, args.q_max], indpb=0.5)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Inicialización de la población
    population = toolbox.population(n=args.pop_size)
    hof = tools.HallOfFame(1)

    # Estadísticas para cada generación
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    print("Iniciando optimización evolutiva de parámetros ARIMA...\n")
    ngen = args.ngen
    cxpb = 0.5
    mutpb = 0.2

    # Evaluación inicial
    print("Evaluando población inicial:")
    for ind in tqdm(population, desc="Evaluando genomas", leave=True):
        ind.fitness.values = toolbox.evaluate(ind)
    record = stats.compile(population)
    best_genome = tools.selBest(population, 1)[0]
    hof.update(population)
    print(f"Generación 0: Mejor AIC = {record['min']:.2f}, Promedio AIC = {record['avg']:.2f}")

    # Evolución generacional
    for gen in range(1, ngen + 1):
        print(f"\n=== Generación {gen} de {ngen} ===")
        # Selección
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        # Cruce
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        # Mutación
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluación de los individuos sin fitness asignado
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        print(f"Evaluando {len(invalid_ind)} genomas sin fitness...")
        for ind in tqdm(invalid_ind, desc="Evaluando genomas", leave=False):
            ind.fitness.values = toolbox.evaluate(ind)

        # Reemplazo de la población
        population[:] = offspring
        hof.update(population)
        record = stats.compile(population)
        best_genome = tools.selBest(population, 1)[0]
        # Información detallada de la generación actual
        print(f"Generación {gen}: Mejor AIC = {record['min']:.2f}, Promedio AIC = {record['avg']:.2f}, Máximo AIC = {record['max']:.2f}")
        print(f"Mejor genoma en esta generación: ARIMA({int(best_genome[0])},{int(best_genome[1])},{int(best_genome[2])})")
    
    # Resultados finales
    best = hof[0]
    best_aic = eval_arima(best, series)[0]
    print("\n=== Resultado Final ===")
    print(f"Mejores parámetros encontrados: ARIMA({int(best[0])},{int(best[1])},{int(best[2])}) con AIC = {best_aic:.2f}")

if __name__ == "__main__":
    main()
