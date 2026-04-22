import backtrader as bt
import pandas as pd
import datetime
import random
import time
import multiprocessing
import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms

# Import your strategy; make sure your strategy file (for example, forex_dynamic_volume_strategy.py)
# has been modified to accept the extra parameters: tp_multiplier, sl_multiplier, lower_rr_threshold, and upper_rr_threshold.
from heuristic_strategy import HeuristicStrategy

# Evaluation function: run a backtest with a given set of parameters and return the profit.
def evaluate_strategy(individual):
    # Unpack the individual
    profit_threshold, tp_multiplier, sl_multiplier, rel_volume, lower_rr, upper_rr = individual
    
    cerebro = bt.Cerebro()
    cerebro.addstrategy(HeuristicStrategy,
                        profit_threshold=profit_threshold,
                        tp_multiplier=tp_multiplier,
                        sl_multiplier=sl_multiplier,
                        rel_volume=rel_volume,
                        lower_rr_threshold=lower_rr,
                        upper_rr_threshold=upper_rr,
                        date_start=datetime.datetime(2014, 1, 1),
                        date_end=datetime.datetime(2015, 1, 1),
                        min_drawdown_pips=10)
    
    data = bt.feeds.GenericCSVData(
        dataname='../trading-signal/output.csv',
        dtformat=('%Y-%m-%d %H:%M:%S'),
        datetime=0,
        time=-1,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=-1,
        openinterest=-1,
        timeframe=bt.TimeFrame.Minutes,
        compression=60,
        fromdate=datetime.datetime(2014, 1, 1),
        todate=datetime.datetime(2015, 1, 1)
    )
    
    cerebro.adddata(data)
    cerebro.broker.setcash(10000.0)
    cerebro.addsizer(bt.sizers.FixedSize, stake=1)
    
    start_time = time.time()
    cerebro.run()
    end_time = time.time()
    final_value = cerebro.broker.getvalue()
    profit = final_value - 10000.0
    # We print the runtime for reference (you can comment it out if desired).
    #print(f"Evaluated individual {individual} in {end_time - start_time:.2f} sec, Profit: {profit:.2f}")
    return (profit,)

# Setup DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # maximize profit
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Define attribute generators for each hyperparameter.
toolbox.register("attr_profit_threshold", random.uniform, 1, 20)
toolbox.register("attr_tp_multiplier", random.uniform, 0.8, 1.2)
toolbox.register("attr_sl_multiplier", random.uniform, 1.5, 3.0)
toolbox.register("attr_rel_volume", random.uniform, 0.01, 0.1)
toolbox.register("attr_lower_rr", random.uniform, 0.3, 1.0)
toolbox.register("attr_upper_rr", random.uniform, 1.5, 3.0)

# Create an individual from these attributes.
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_profit_threshold,
                  toolbox.attr_tp_multiplier,
                  toolbox.attr_sl_multiplier,
                  toolbox.attr_rel_volume,
                  toolbox.attr_lower_rr,
                  toolbox.attr_upper_rr), n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register genetic operators.
toolbox.register("evaluate", evaluate_strategy)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1.0, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(42)
    population = toolbox.population(n=20)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 100  # Crossover prob, mutation prob, number of generations
    print("Starting Genetic Algorithm Optimization")
    
    # Optionally use multiprocessing
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    print("  Evaluated %i individuals" % len(population))
    
    for gen in range(1, NGEN + 1):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        population[:] = offspring
        fits = [ind.fitness.values[0] for ind in population]
        print(f"Generation {gen}: Max Profit = {max(fits):.2f}, Avg Profit = {sum(fits)/len(fits):.2f}")
    
    best_ind = tools.selBest(population, 1)[0]
    print("Best parameter set found:")
    print(f"  profit_threshold = {best_ind[0]:.2f}")
    print(f"  tp_multiplier    = {best_ind[1]:.2f}")
    print(f"  sl_multiplier    = {best_ind[2]:.2f}")
    print(f"  rel_volume       = {best_ind[3]:.3f}")
    print(f"  lower_rr_thresh  = {best_ind[4]:.2f}")
    print(f"  upper_rr_thresh  = {best_ind[5]:.2f}")
    print("With profit: {:.2f}".format(best_ind.fitness.values[0]))
    
    pool.close()

if __name__ == '__main__':
    main()
